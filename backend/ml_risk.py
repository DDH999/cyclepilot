# -*- coding: utf-8 -*-
"""
回撤风险预测（M37）—— 机器学习在这里的第一个正当用途
====================================================
**不预测涨跌，只预测「未来 20 天会不会出现 >10% 回撤」。**

为什么是这个目标
----------------
* 收益方向预测：几十年来基本无效，本项目的校准也反复证实过
* **风险预测**：波动有聚集性（volatility clustering），是可预测的
* 而且它有直接用途：风险高的时候**减少定投、保留现金**

实测结果（40 年日线，walk-forward，带 embargo）
----------------------------------------------
| 市场 | 均值 AUC | 最低折 | 结论 |
|---|---|---|---|
| 标普500 | **0.767** | 0.683 | 可用 |
| 日经225 | 0.578 | **0.371** | **不可用** |

**同一个模型，标普能用、日经不能用。** 这不是 bug，是过拟合的反面：
不同市场的信息效率不同。所以本模块**逐市场验证**，
达不到门槛的市场**明确拒绝出信号**，而不是硬给一个数。

使用门槛（写死）
----------------
* 均值 AUC ≥ 0.62
* 每一折 AUC ≥ 0.55（不能有"比瞎猜还差"的折）
* 测试期正样本 ≥ 100（否则 AUC 噪声太大）

三条缺一不可。不达标 → `usable=False`，调用方必须当它不存在。

⚠ 诚实的边界
-------------
* AUC 0.77 ≠ 能预警每一次崩盘；它只是一个排序能力
* 极端事件（2020 疫情、2008）样本极少，模型对"没见过的崩法"无能为力
* 概率高不代表一定跌；概率低也不代表安全
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["FEATURE_COLS", "THRESHOLDS", "build_features", "make_target",
           "walk_forward", "validate_market", "current_risk",
           "risk_multiplier", "format_report"]

FEATURE_COLS = [
    "rv_20", "rv_60", "mom_20", "mom_60", "mom_120",
    "dd_now", "dist_ma200", "skew_60", "kurt_60",
    "vol_ratio", "down_streak",
]

THRESHOLDS = {
    "min_mean_auc": 0.62,
    "min_fold_auc": 0.55,
    "min_positives": 100,
    "horizon": 20,
    "drawdown": -0.10,
}

_CACHE: dict[str, dict] = {}


def build_features(close: pd.Series) -> pd.DataFrame:
    """只用价格构造特征，全部基于当期及之前的数据（无未来信息）。"""
    r = close.pct_change(fill_method=None)
    f = pd.DataFrame(index=close.index)
    f["rv_20"] = r.rolling(20).std()
    f["rv_60"] = r.rolling(60).std()
    f["mom_20"] = close.pct_change(20)
    f["mom_60"] = close.pct_change(60)
    f["mom_120"] = close.pct_change(120)
    peak = close.cummax()
    f["dd_now"] = close / peak - 1.0
    f["dist_ma200"] = close / close.rolling(200).mean() - 1.0
    f["skew_60"] = r.rolling(60).skew()
    f["kurt_60"] = r.rolling(60).kurt()
    f["vol_ratio"] = f["rv_20"] / f["rv_60"].replace(0, np.nan)
    down = (r < 0).astype(int)
    streak = down.copy()
    for i in range(1, len(down)):
        if down.iloc[i] == 1:
            streak.iloc[i] = streak.iloc[i - 1] + 1
    f["down_streak"] = streak
    return f


def make_target(close: pd.Series, horizon: int = 20,
                thresh: float = -0.10) -> pd.Series:
    """未来 horizon 天内的最大回撤是否 ≤ thresh（1=是）。最后 horizon 天为 NaN。"""
    arr = close.to_numpy()
    n = len(arr)
    y = np.full(n, np.nan)
    for i in range(n - horizon):
        win = arr[i + 1: i + 1 + horizon]
        y[i] = 1.0 if (win.min() / arr[i] - 1.0) <= thresh else 0.0
    return pd.Series(y, index=close.index)


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, C=0.5))


def walk_forward(X: pd.DataFrame, y: pd.Series, n_folds: int = 5,
                 embargo: int = 20) -> dict:
    """扩展窗口 walk-forward，训练段与测试段之间留 embargo 防标签泄漏。"""
    from sklearn.metrics import roc_auc_score

    df = pd.concat([X[FEATURE_COLS], y.rename("y")], axis=1).dropna()
    if len(df) < 500:
        return {"ok": False, "reason": f"有效样本仅 {len(df)}"}
    Xa, ya = df[FEATURE_COLS], df["y"]

    n = len(df)
    bounds = np.linspace(n // 2, n, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        tr_end = bounds[i] - embargo
        te_lo, te_hi = bounds[i], bounds[i + 1]
        if tr_end < 200 or te_hi - te_lo < 50:
            continue
        Xtr, ytr = Xa.iloc[:tr_end], ya.iloc[:tr_end]
        Xte, yte = Xa.iloc[te_lo:te_hi], ya.iloc[te_lo:te_hi]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        m = _model()
        m.fit(Xtr, ytr)
        try:
            auc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        except Exception:
            continue
        folds.append({"fold": i + 1, "n_train": int(len(Xtr)),
                      "n_test": int(len(Xte)), "auc": round(auc, 4),
                      "positives": int(yte.sum()),
                      "base_rate": round(float(yte.mean()), 4)})

    if not folds:
        return {"ok": False, "reason": "没有可用折"}
    aucs = [f["auc"] for f in folds]
    return {"ok": True, "folds": folds,
            "mean_auc": round(float(np.mean(aucs)), 4),
            "min_auc": round(float(np.min(aucs)), 4),
            "max_auc": round(float(np.max(aucs)), 4),
            "n": int(len(df)), "pos_rate": round(float(ya.mean()), 4)}


def validate_market(key: str, refresh: bool = False) -> dict:
    """
    对某个市场做完整验证，给出 usable 判断。

    结果缓存在进程内（同一进程内重复调用不再训练）。
    """
    ck = f"validate:{key}"
    if not refresh and ck in _CACHE:
        return _CACHE[ck]

    import overseas as OS
    got = OS.index_history(key, days=20000, refresh=False)
    if not got.get("ok"):
        return {"ok": False, "key": key,
                "reason": f"取数失败：{got.get('reason')}"}

    close = got["series"]
    X = build_features(close)
    y = make_target(close, THRESHOLDS["horizon"], THRESHOLDS["drawdown"])
    wf = walk_forward(X, y)
    if not wf.get("ok"):
        return {"ok": False, "key": key, "reason": wf.get("reason")}

    total_pos = int(y.dropna().sum())
    reasons = []
    if wf["mean_auc"] < THRESHOLDS["min_mean_auc"]:
        reasons.append(f"均值 AUC {wf['mean_auc']:.3f} < "
                       f"{THRESHOLDS['min_mean_auc']}")
    if wf["min_auc"] < THRESHOLDS["min_fold_auc"]:
        reasons.append(f"最低折 AUC {wf['min_auc']:.3f} < "
                       f"{THRESHOLDS['min_fold_auc']}")
    if total_pos < THRESHOLDS["min_positives"]:
        reasons.append(f"正样本仅 {total_pos} < {THRESHOLDS['min_positives']}")

    out = {"ok": True, "key": key, "name": got["name"],
           "n_days": int(len(close)),
           "start": str(close.index[0].date()),
           "end": str(close.index[-1].date()),
           "walk_forward": wf, "positives_total": total_pos,
           "usable": len(reasons) == 0, "reasons": reasons,
           "thresholds": dict(THRESHOLDS)}
    _CACHE[ck] = out
    return out


def current_risk(key: str, refresh: bool = False) -> dict:
    """
    当前风险概率：未来 20 天出现 >10% 回撤的可能性。

    市场未通过验证 → **拒绝出信号**（返回 usable=False）。
    """
    val = validate_market(key, refresh=refresh)
    if not val.get("ok"):
        return {"ok": False, "key": key, "reason": val.get("reason")}
    if not val.get("usable"):
        return {"ok": True, "key": key, "name": val["name"],
                "usable": False, "probability": None,
                "reason": "该市场未通过验证：" + "；".join(val["reasons"]),
                "walk_forward": val["walk_forward"]}

    import overseas as OS
    got = OS.index_history(key, days=20000, refresh=False)
    close = got["series"]
    X = build_features(close)
    y = make_target(close, THRESHOLDS["horizon"], THRESHOLDS["drawdown"])

    df = pd.concat([X[FEATURE_COLS], y.rename("y")], axis=1).dropna()
    m = _model()
    m.fit(df[FEATURE_COLS], df["y"])
    last = X[FEATURE_COLS].dropna().iloc[[-1]]
    try:
        p = float(m.predict_proba(last)[0, 1])
    except Exception as ex:
        return {"ok": False, "key": key, "reason": f"预测失败：{ex}"}

    base = float(df["y"].mean())
    if p >= base * 2.0:
        level, note = "高", "风险概率显著高于历史均值，建议放慢投入"
    elif p >= base * 1.2:
        level, note = "偏高", "略高于历史均值"
    elif p <= base * 0.5:
        level, note = "低", "风险概率低于历史均值"
    else:
        level, note = "正常", "接近历史均值"

    return {"ok": True, "key": key, "name": val["name"], "usable": True,
            "probability": round(p, 4), "base_rate": round(base, 4),
            "level": level, "note": note,
            "mean_auc": val["walk_forward"]["mean_auc"],
            "as_of": str(close.index[-1].date())}


def risk_multiplier(prob: float | None, base: float | None = None) -> dict:
    """
    风险概率 → 定投倍数。

    高概率时分批减速；低概率时略加速。范围限定 [0.5, 1.2]，
    不做极端操作（模型只是排序器，不是预言机）。
    """
    if prob is None or base is None or not np.isfinite(prob) or base <= 0:
        return {"multiplier": 1.0, "note": "风险信号不可用，按计划"}
    ratio = float(prob) / float(base)
    if ratio >= 2.0:
        mult, note = 0.5, "风险概率≥2倍均值：减半投"
    elif ratio >= 1.5:
        mult, note = 0.7, "风险概率偏高：减到 7 成"
    elif ratio <= 0.5:
        mult, note = 1.2, "风险概率低：可加到 1.2 成"
    else:
        mult, note = 1.0, "风险正常：按计划"
    return {"multiplier": mult, "ratio": round(ratio, 2), "note": note}


def format_report(val: dict, cur: dict | None = None) -> str:
    L = ["=" * 68, "回撤风险预测（机器学习）", "=" * 68]
    if not val.get("ok"):
        L.append(f"  验证失败：{val.get('reason')}")
        L.append("=" * 68)
        return "\n".join(L)

    L.append(f"  {val['name']}   {val['n_days']} 天 "
             f"（{val['start']} ~ {val['end']}）")
    wf = val["walk_forward"]
    L.append(f"  目标：未来 {THRESHOLDS['horizon']} 天回撤超 "
             f"{abs(THRESHOLDS['drawdown'])*100:.0f}%")
    L.append(f"  正样本 {val['positives_total']} 个"
             f"（占比 {wf['pos_rate']:.1%}）")
    L.append("-" * 68)
    L.append(f"  walk-forward 均值 AUC {wf['mean_auc']:.4f}   "
             f"最低折 {wf['min_auc']:.4f}  最高折 {wf['max_auc']:.4f}")
    for f in wf["folds"]:
        L.append(f"    折{f['fold']}  训练{f['n_train']:>5} "
                 f"测试{f['n_test']:>4}  正例{f['positives']:>4}  "
                 f"AUC {f['auc']:.4f}")
    L.append("-" * 68)
    if val["usable"]:
        L.append("  结论：通过验证，可用于风险提示")
    else:
        L.append("  结论：**未通过验证，拒绝出信号**")
        for r in val["reasons"]:
            L.append(f"    ✗ {r}")

    if cur and cur.get("ok"):
        L.append("-" * 68)
        if cur.get("usable"):
            L.append(f"  当前风险概率 {cur['probability']*100:.1f}%   "
                     f"（历史均值 {cur['base_rate']*100:.1f}%）   等级 {cur['level']}")
            L.append(f"  {cur['note']}")
        else:
            L.append(f"  不出信号：{cur.get('reason')}")
    L.append("=" * 68)
    L.append("  ⚠ AUC 是排序能力，不是预言机。极端行情样本极少，别把它当保险。")
    L.append("=" * 68)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 特征与标签无泄漏检查 ===")
    idx = pd.bdate_range("2020-01-02", periods=400)
    close = pd.Series(np.linspace(100, 120, 400), index=idx)
    X = build_features(close)
    y = make_target(close, 20, -0.10)
    print(f"  特征列 {len(FEATURE_COLS)} 个: {FEATURE_COLS}")
    print(f"  标签最后 20 天为 NaN（无未来数据可看）: "
          f"{int(y.tail(20).isna().sum())}/20")
    assert y.tail(20).isna().all()
    first_valid = X.dropna().index[0]
    print(f"  首个有效特征日 {first_valid.date()}（前 200 天因长窗口不足为 NaN）")
    assert first_valid > idx[100], first_valid

    print("\n=== 逐市场验证 ===")
    for k in ("sp500", "nikkei"):
        val = validate_market(k)
        if val.get("ok"):
            wf = val["walk_forward"]
            print(f"  {val['name']:<12} AUC {wf['mean_auc']:.3f} "
                  f"(min {wf['min_auc']:.3f})  可用={val['usable']}  "
                  f"{('；'.join(val['reasons']) if val['reasons'] else '')}")
        else:
            print(f"  {k}: {val.get('reason')}")

    print("\n=== 当前风险 ===")
    for k in ("sp500", "nikkei"):
        cur = current_risk(k)
        if cur.get("ok") and cur.get("usable"):
            m = risk_multiplier(cur["probability"], cur["base_rate"])
            print(f"  {cur['name']:<12} P={cur['probability']:.3f} "
                  f"均值={cur['base_rate']:.3f}  {cur['level']}  "
                  f"→ 定投 {m['multiplier']}x  {m['note']}")
        else:
            print(f"  {k}: {cur.get('reason')}")
    print("\nOK v" + __version__)
