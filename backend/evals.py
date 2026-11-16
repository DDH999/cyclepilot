# -*- coding: utf-8 -*-
"""
QuantPilot · 评估门禁层 (Evaluation Harness)
============================================================
这是整个系统唯一决定"能不能上钱"的模块。

设计原则（违反任何一条，AUC 就会是假的）：
  1. 特征只用 t 时刻及之前的信息；标签用 t+1..t+h 的收益
  2. 时间序列 walk-forward + 清洗(purge) 标签重叠区，禁止随机切分
  3. 每个指标同时给出"随机切分"对照，用来量化泄漏幅度
  4. 必须通过洗牌标签检验（AUC 应回落到 0.5 附近）
  5. 必须通过"故意泄漏"检验（注入未来信息后 AUC 应飙升，证明检测器有效）
  6. 多重检验惩罚：Deflated Sharpe / PBO(CSCV)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ── 环境适配：本沙箱禁止命名管道，joblib 线程池会创建 Pipe 而报 WinError 5 ──
# 必须在 import sklearn.ensemble 之前把 Parallel 替换为串行执行器。
import joblib as _jl


class _SequentialParallel:
    def __init__(self, n_jobs=None, backend=None, **kw):
        pass

    def __call__(self, iterable):
        out = []
        for item in iterable:
            if callable(item):
                out.append(item())
            else:
                fn, args, kwargs = item
                out.append(fn(*args, **kwargs))
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_jl.Parallel = _SequentialParallel
try:
    import sklearn.utils.parallel as _sup
    _sup.Parallel = _SequentialParallel
except Exception:
    pass

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────
# 1. 特征工程（全部 shift 安全，无未来函数）
# ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    r1 = c.pct_change(fill_method=None)
    f = pd.DataFrame(index=d.index)
    f["date"] = d["date"]

    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 20, 60):
        f[f"vol_{n}"] = r1.rolling(n).std()
    for n in (5, 20, 60, 120):
        f[f"ma_gap_{n}"] = c / c.rolling(n).mean() - 1
        f[f"ma_cross_{n}"] = c.rolling(n).mean() / c.rolling(n * 2).mean() - 1

    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["hl_range"] = (h - l) / c
    f["gap"] = o / c.shift() - 1

    f["vol_z_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["amt_z_20"] = (d["amount"] - d["amount"].rolling(20).mean()) / d["amount"].rolling(20).std()
    f["vol_ratio"] = v / v.rolling(60).mean()

    f["skew_20"] = r1.rolling(20).skew()
    f["kurt_20"] = r1.rolling(20).kurt()
    f["autocorr_5"] = r1.rolling(60).apply(lambda x: pd.Series(x).autocorr(5) if len(x) > 6 else np.nan,
                                           raw=False)
    f["slope_20"] = f["ma_gap_20"].diff(5)
    f["dd_60"] = c / c.rolling(60).max() - 1
    f["up_days_20"] = (r1 > 0).rolling(20).mean()
    f["ret_skew_ratio"] = f["ret_20"] / (f["vol_20"] * np.sqrt(20))
    f["park_vol"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)))
    f["vix_proxy"] = f["vol_20"] / f["vol_60"]
    return f.drop(columns=["date"])


FEATURE_COLS = None


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].reset_index(drop=True)
    r1 = c.pct_change(fill_method=None)
    t = pd.DataFrame(index=df.index)
    fwd1 = c.shift(-1) / c - 1
    fwd5 = c.shift(-5) / c - 1
    fwd20 = c.shift(-20) / c - 1
    t["y_dir_1d"] = (fwd1 > 0).astype(float)
    t["y_dir_5d"] = (fwd5 > 0).astype(float)

    # 未来20日已实现波动率（在 t 时刻不可知，作为标签）
    fwdvol = r1.rolling(20).std().shift(-20)
    trailvol = r1.rolling(20).std()
    thresh = trailvol.rolling(60).median().shift(1)
    t["y_vol_regime"] = (fwdvol > thresh).astype(float)
    t.loc[fwdvol.isna() | thresh.isna(), "y_vol_regime"] = np.nan

    t["y_tail_5d"] = (fwd5 < -1.5 * (trailvol.shift(1) * np.sqrt(5))).astype(float)
    t.loc[fwd5.isna(), "y_tail_5d"] = np.nan

    t["fwd1"] = fwd1
    t["fwd5"] = fwd5
    t["fwd20"] = fwd20
    return t


# ─────────────────────────────────────────────────────────────
# 2. Purged Walk-Forward
# ─────────────────────────────────────────────────────────────
def purged_walk_forward(n: int, n_folds: int = 6, horizon: int = 1,
                        start_frac: float = 0.40, min_train: int = 400):
    """扩展窗口 + 清洗：训练样本的标签窗口不得与测试窗口重叠。"""
    start = max(int(n * start_frac), min_train)
    if start >= n - 40:
        raise ValueError(f"样本不足: n={n}, start={start}")
    fold = (n - start) // n_folds
    if fold < 20:
        raise ValueError(f"每折样本不足: fold={fold}")
    for k in range(n_folds):
        a = start + k * fold
        b = a + fold if k < n_folds - 1 else n
        tr_end = a - horizon          # 关键：purge
        if tr_end < min_train:
            continue
        yield np.arange(0, tr_end), np.arange(a, b)


def _model():
    return HistGradientBoostingClassifier(
        max_iter=250, max_depth=4, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=40,
        early_stopping=False, random_state=42)


def _fit_eval(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    m = _model().fit(Xtr, ytr)
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(Xte)[:, 1]
    else:
        p = m.decision_function(Xte)
    return {
        "auc": roc_auc_score(yte, p),
        "pr_auc": average_precision_score(yte, p),
        "brier": brier_score_loss(yte, p),
        "n_te": int(len(yte)),
        "pos_rate": float(np.mean(yte)),
    }


# ─────────────────────────────────────────────────────────────
# 3. 单任务评估
# ─────────────────────────────────────────────────────────────
TASKS = {
    "y_dir_1d":     ("日频涨跌方向", 1,   "0.50-0.56"),
    "y_dir_5d":     ("5日涨跌方向",  5,   "0.52-0.58"),
    "y_vol_regime": ("波动率状态",   20,  "0.65-0.75"),
    "y_tail_5d":    ("5日尾部风险事件", 5, "0.65-0.80"),
}


def evaluate(df: pd.DataFrame, task: str, universe_shift: int = 0) -> dict:
    label, horizon, expect = TASKS[task]
    X = build_features(df)
    T = build_targets(df)
    y = T[task]

    if universe_shift:            # 故意注入未来信息，用于验证检测器
        X = X.copy()
        X["__leak__"] = T["fwd1"].shift(-universe_shift)

    mask = y.notna() & X.notna().any(axis=1)
    Xv = X[mask].reset_index(drop=True)
    yv = y[mask].reset_index(drop=True).astype(float)
    Xv = Xv.replace([np.inf, -np.inf], np.nan)
    n = len(Xv)

    folds = []
    for tr, te in purged_walk_forward(n, horizon=horizon):
        r = _fit_eval(Xv.iloc[tr], yv.iloc[tr], Xv.iloc[te], yv.iloc[te])
        if r:
            folds.append(r)
    if not folds:
        return {"task": task, "label": label, "error": "no valid folds"}

    aucs = np.array([f["auc"] for f in folds])
    return {
        "task": task, "label": label, "horizon": horizon, "expected": expect,
        "n": n,
        "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
        "pr_auc_mean": float(np.mean([f["pr_auc"] for f in folds])),
        "brier_mean": float(np.mean([f["brier"] for f in folds])),
        "n_folds": len(folds),
        "pos_rate": float(yv.mean()),
        "per_fold": [round(float(a), 4) for a in aucs],
    }


# ─────────────────────────────────────────────────────────────
# 4. 泄漏审计
# ─────────────────────────────────────────────────────────────
def leakage_audit(df: pd.DataFrame, task: str = "y_dir_5d") -> dict:
    """
    三个对照实验，量化"你的 AUC 里有多少是假的"：
      A. 随机切分（错误做法）      -> 通常会虚高
      B. 洗牌标签（应回落 0.5）    -> 检验流程本身是否泄漏
      C. 注入未来信息（应飙升）    -> 检验检测器灵敏度
    """
    label, horizon, _ = TASKS[task]
    X = build_features(df)
    T = build_targets(df)
    y = T[task]
    mask = y.notna() & X.notna().any(axis=1)
    Xv = X[mask].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    yv = y[mask].reset_index(drop=True).astype(float).values

    # A. 错误做法：随机切分
    idx = RNG.permutation(len(Xv))
    cut = int(len(idx) * 0.7)
    a = _fit_eval(Xv.iloc[idx[:cut]], yv[idx[:cut]], Xv.iloc[idx[cut:]], yv[idx[cut:]])

    # 正确做法：purged WF
    fold_aucs = []
    for tr, te in purged_walk_forward(len(Xv), horizon=horizon):
        r = _fit_eval(Xv.iloc[tr], yv[tr], Xv.iloc[te], yv[te])
        if r:
            fold_aucs.append(r["auc"])
    wf_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    # B. 洗牌标签
    ys = yv.copy()
    RNG.shuffle(ys)
    fold_aucs_s = []
    for tr, te in purged_walk_forward(len(Xv), horizon=horizon):
        r = _fit_eval(Xv.iloc[tr], ys[tr], Xv.iloc[te], ys[te])
        if r:
            fold_aucs_s.append(r["auc"])
    shuf_auc = float(np.mean(fold_aucs_s)) if fold_aucs_s else float("nan")

    # C. 注入未来信息
    Xl = Xv.copy()
    Xl["__leak_fwd1__"] = T["fwd1"][mask].reset_index(drop=True)
    fold_aucs_l = []
    for tr, te in purged_walk_forward(len(Xl), horizon=horizon):
        r = _fit_eval(Xl.iloc[tr], yv[tr], Xl.iloc[te], yv[te])
        if r:
            fold_aucs_l.append(r["auc"])
    leak_auc = float(np.mean(fold_aucs_l)) if fold_aucs_l else float("nan")

    return {
        "task": task, "label": label,
        "random_split_auc": float(a["auc"]) if a else None,
        "purged_wf_auc": wf_auc,
        "inflation_from_random_split": (float(a["auc"]) - wf_auc) if a else None,
        "shuffled_label_auc": shuf_auc,
        "shuffle_ok": bool(abs(shuf_auc - 0.5) < 0.06) if not np.isnan(shuf_auc) else None,
        "leak_injected_auc": leak_auc,
        "leak_detector_ok": bool(leak_auc > wf_auc + 0.08) if not np.isnan(leak_auc) else None,
    }


# ─────────────────────────────────────────────────────────────
# 5. 多重检验惩罚
# ─────────────────────────────────────────────────────────────
def deflated_sharpe(sr: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Bailey & López de Prado (2014). sr 为（非年化）夏普。"""
    if n_obs < 2 or n_trials < 1:
        return {"dsr": None}
    e = 0.5772156649
    z = stats.norm.ppf(1 - 1.0 / max(n_trials, 2))
    sr0 = (1 - e) * z + e * stats.norm.ppf(1 - 1.0 / max(n_trials, 2) * np.e)
    sr0 = sr0 / np.sqrt(max(n_obs - 1, 1))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-9))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom)
    return {"sr": float(sr), "sr_benchmark": float(sr0), "dsr": float(dsr),
            "n_trials": n_trials, "n_obs": n_obs}


def pbo_cscv(perf: np.ndarray, n_blocks: int = 10) -> dict:
    """
    Probability of Backtest Overfitting via CSCV.
    perf: (T, N) 矩阵，N 个策略配置的逐期收益。
    """
    T, N = perf.shape
    if T < n_blocks * 4 or N < 2:
        return {"pbo": None, "reason": "样本或配置不足"}
    edges = np.linspace(0, T, n_blocks + 1, dtype=int)
    blocks = [perf[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    from itertools import combinations
    half = n_blocks // 2
    ranks = []
    for comb in combinations(range(n_blocks), half):
        is_idx = list(comb)
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        IS = np.vstack([blocks[i] for i in is_idx]).mean(axis=0)
        OOS = np.vstack([blocks[i] for i in oos_idx]).mean(axis=0)
        best = int(np.argmax(IS))
        r = stats.rankdata(OOS)[best] / N        # 0..1
        ranks.append(r)
    ranks = np.array(ranks)
    return {"pbo": float(np.mean(ranks < 0.5)), "n_combinations": int(len(ranks)),
            "median_oos_rank_of_is_best": float(np.median(ranks))}


def ic_decay(feature: pd.Series, fwd: pd.Series, max_lag: int = 10) -> list[dict]:
    out = []
    for lag in range(1, max_lag + 1):
        f = feature.shift(lag)
        m = f.notna() & fwd.notna()
        if m.sum() < 60:
            continue
        out.append({"lag": lag, "ic": float(f[m].corr(fwd[m], method="spearman"))})
    return out
