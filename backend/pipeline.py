# -*- coding: utf-8 -*-
"""
QuantPilot · 主流水线
真实数据 -> 特征 -> 池化模型(purged WF) -> 泄漏审计 -> 策略族 -> PBO/DSR -> 33 图
输出: reports/summary.json + reports/charts/*.png
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import datafeed
from evals import (_fit_eval, deflated_sharpe, ic_decay, pbo_cscv)
import charts

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)


def feats(d: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    r1 = c.pct_change(fill_method=None)
    f = pd.DataFrame(index=d.index)
    for n in (1, 5, 10, 20, 60, 120, 250):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 60, 120, 250):
        f[f"vol_{n}"] = r1.rolling(n).std()
    f["vr_5_20"] = f["vol_5"] / f["vol_20"]
    f["vr_20_60"] = f["vol_20"] / f["vol_60"]
    f["vr_60_120"] = f["vol_60"] / f["vol_120"]
    f["vol_pct_250"] = f["vol_20"].rolling(250).rank(pct=True)
    f["vol_pct_60"] = f["vol_20"].rolling(60).rank(pct=True)
    f["absret_20"] = r1.abs().rolling(20).mean()
    f["absret_5"] = r1.abs().rolling(5).mean()
    f["absret_ratio"] = f["absret_5"] / f["absret_20"]
    lhl = np.log(h / l) ** 2
    f["park_20"] = np.sqrt(lhl.rolling(20).mean() / (4 * np.log(2)))
    f["park5_park20"] = np.sqrt(lhl.rolling(5).mean()) / np.sqrt(lhl.rolling(20).mean())
    f["hl_20"] = ((h - l) / c).rolling(20).mean()
    f["dd_60"] = c / c.rolling(60).max() - 1
    f["ma_gap_20"] = c / c.rolling(20).mean() - 1
    f["ma_gap_60"] = c / c.rolling(60).mean() - 1
    f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["up_days_20"] = (r1 > 0).rolling(20).mean()
    return f


def target_vol(d: pd.DataFrame) -> pd.Series:
    """未来20日波动率是否高于过去250日的长期中位数（稳定口径）"""
    r1 = d["close"].pct_change(fill_method=None)
    fwdvol = r1.rolling(20).std().shift(-20)
    base = r1.rolling(20).std().rolling(250).median().shift(1)
    y = (fwdvol > base).astype(float)
    y[fwdvol.isna() | base.isna()] = np.nan
    return y


def target_dir(d: pd.DataFrame) -> pd.Series:
    fwd = d["close"].shift(-1) / d["close"] - 1
    return (fwd > 0).astype(float)


def _model_fit(X, y):
    from evals import _model
    return _model().fit(X, y)


def pooled_eval(panel: dict, yfun, purge=20, n_folds=6, start_frac=0.40, shuffle=False,
                random_split=False, add_leak=False):
    rows = []
    for s, d in panel.items():
        F = feats(d)
        F["_y"] = yfun(d)
        F["_fwd1"] = d["close"].shift(-1) / d["close"] - 1
        F["_sym"] = s
        F["_date"] = d["date"]
        rows.append(F)
    P = pd.concat(rows, ignore_index=True)

    mkt = P.groupby("_date").agg(mkt_vol20=("vol_20", "mean"),
                                 mkt_ret20=("ret_20", "mean"),
                                 mkt_vr=("vr_20_60", "mean"),
                                 mkt_dd=("dd_60", "mean"),
                                 mkt_volpct=("vol_pct_250", "mean")).reset_index()
    P = P.merge(mkt, on="_date", how="left")

    fc = [c for c in P.columns if not c.startswith("_")]
    d0 = P.dropna(subset=["_y"]).sort_values("_date").reset_index(drop=True)
    X = d0[fc].replace([np.inf, -np.inf], np.nan)
    y = d0["_y"].astype(float)
    if add_leak:
        # 强泄漏源：注入"近似标签"（真实标签翻转15%）。用于验证检测器能否抓住明显泄漏。
        yy = y.values.copy()
        flip = np.random.default_rng(3).random(len(yy)) < 0.15
        yy[flip] = 1.0 - yy[flip]
        X = X.copy()
        X["__leak_label__"] = yy
    if shuffle:
        y = y.sample(frac=1.0, random_state=11).reset_index(drop=True)
    dates = pd.to_datetime(d0["_date"]).values
    n = len(d0)

    # 错误做法对照：随机 70/30 切分（无视时间顺序）
    rnd_auc = None
    if random_split:
        perm = np.random.default_rng(5).permutation(n)
        cut = int(n * 0.70)
        rr = _fit_eval(X.iloc[perm[:cut]], y.iloc[perm[:cut]],
                       X.iloc[perm[cut:]], y.iloc[perm[cut:]])
        rnd_auc = float(rr["auc"]) if rr else None

    start = int(n * start_frac)
    fold = (n - start) // n_folds

    res, last = [], None
    for k in range(n_folds):
        a = start + k * fold
        b = a + fold if k < n_folds - 1 else n
        safe_pos = max(0, a - purge)
        cut = dates[safe_pos]
        tr = np.where(dates < cut)[0]
        te = np.arange(a, b)
        if len(tr) < 500 or len(te) < 30:
            continue
        r = _fit_eval(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
        if not r:
            continue
        res.append(r)
        m = _model_fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        last = (X.iloc[te].reset_index(drop=True), y.iloc[te].values, p, m)
    if not res:
        return None
    aucs = np.array([r["auc"] for r in res])
    return {
        "n": int(n), "n_feat": len(fc), "n_folds": len(res),
        "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
        "pr_auc_mean": float(np.mean([r["pr_auc"] for r in res])),
        "brier_mean": float(np.mean([r["brier"] for r in res])),
        "pos_rate": float(y.mean()),
        "per_fold": [round(float(a), 4) for a in aucs],
        "random_split_auc": rnd_auc,
        "last": last,
    }


def strategy_family(panel: dict, lookbacks=(20, 40, 60, 120), topns=(3, 5)):
    """动量 Top-N 等权多头族（逐日调仓，向量化）。为 PBO/DSR 提供多配置收益矩阵。"""
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).sort_index()
    px = px.ffill()
    rets = px.pct_change(fill_method=None)
    cols, series = [], []
    for lb in lookbacks:
        mom = px.pct_change(lb)
        rank = mom.rank(axis=1, ascending=False)
        for tn in topns:
            sel = (rank <= tn)
            cnt = sel.sum(axis=1).replace(0, np.nan)
            w = sel.div(cnt, axis=0)
            port = (w.shift(1) * rets).sum(axis=1, min_count=1).dropna()
            if len(port) < 250:
                continue
            cols.append(f"mom{lb}_top{tn}")
            series.append(port)
    perf = pd.DataFrame(dict(zip(cols, series))).dropna(how="all")
    perf = perf.dropna(axis=1, how="all").fillna(0.0)
    return perf.loc[:, perf.abs().sum() > 1e-9]


def main():
    t0 = time.time()
    print("[1/6] 加载真实行情 ...")
    uni = {s: "cn" for s in datafeed.UNIVERSE_CN}
    uni.update({s: "us" for s in ["AAPL", "NVDA", "MSFT", "TSLA"]})
    panel = datafeed.get_panel(uni)
    panel = datafeed.sanitize_panel(panel)
    print(f"      清洗后 {len(panel)} 个标的: {', '.join(panel)}")

    print("[2/6] 池化波动率状态模型 (purged WF) ...")
    vol = pooled_eval(panel, target_vol, purge=20, random_split=True)
    print(f"      AUC = {vol['auc_mean']:.4f} +- {vol['auc_std']:.4f}  折={vol['per_fold']}")
    print(f"      (对照) 随机切分 AUC = {vol['random_split_auc']}")

    print("[3/6] 洗牌对照 + 方向基准 + 注入泄漏 ...")
    vol_shuf = pooled_eval(panel, target_vol, purge=20, shuffle=True)
    dir_ = pooled_eval(panel, target_dir, purge=1)
    vol_leak = pooled_eval(panel, target_vol, purge=20, add_leak=True)
    print(f"      洗牌 AUC = {vol_shuf['auc_mean']:.4f} (应约 0.5)")
    print(f"      方向 AUC = {dir_['auc_mean']:.4f} (应约 0.51-0.53)")
    print(f"      注入近似标签(15%噪声)后 AUC = {vol_leak['auc_mean']:.4f} (应≈0.97，证明检测器有效)")

    print("[4/6] 策略族 + PBO + DSR ...")
    perf = strategy_family(panel)
    pbo = pbo_cscv(perf.values)
    sharpe = (perf.mean() / perf.std().replace(0, np.nan)).dropna()
    if not len(sharpe):
        raise RuntimeError("strategy family empty")
    bi = sharpe.idxmax()
    r = perf[bi].dropna()
    sr = float(r.mean() / r.std())
    dsr = deflated_sharpe(sr, n_trials=len(perf.columns), n_obs=len(r),
                          skew=float(r.skew()), kurt=float(r.kurtosis() + 3))
    print(f"      策略族 {len(perf.columns)} 配置 | PBO={pbo.get('pbo')} | 最佳={bi} | DSR={dsr.get('dsr')}")

    print("[5/6] 生成图表 ...")
    Xte, yte, pte, model = vol["last"]
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xte), size=min(600, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte.iloc[sub], yte[sub], n_repeats=2,
                                random_state=0, scoring="roc_auc")
    imp = dict(zip(Xte.columns, pi.importances_mean))

    c = panel[datafeed.UNIVERSE_CN[0]]
    dd = c.set_index("date")["close"]
    icd = ic_decay(dd.pct_change(20), dd.pct_change(1).shift(-1), max_lag=10)

    made = []
    def add(p):
        if p:
            made.append(p)

    try:
        add(charts.c01_price_ma(c, "601919")); add(charts.c02_cum_return(c, "601919"))
        add(charts.c03_drawdown(c, "601919")); add(charts.c04_rolling_vol(c, "601919"))
        add(charts.c05_vol_term(c, "601919")); add(charts.c06_return_dist(c, "601919"))
        add(charts.c07_qq(c, "601919")); add(charts.c08_acf(c, "601919"))
        add(charts.c09_monthly_heat(c, "601919")); add(charts.c10_annual(c, "601919"))
        add(charts.c11_corr(panel)); add(charts.c12_roll_corr(panel)); add(charts.c13_pca(panel))
        perf20 = {s: float(d.set_index("date")["close"].pct_change(20).iloc[-1] * 100) for s, d in panel.items()}
        add(charts.c14_sector_rank(perf20)); add(charts.c15_beta(panel)); add(charts.c16_rolling_beta(panel))
        add(charts.c17_auc_folds(vol)); add(charts.c18_roc(yte, pte)); add(charts.c19_pr(yte, pte))
        add(charts.c20_calibration(yte, pte)); add(charts.c21_score_dist(yte, pte))
        add(charts.c22_confusion(yte, pte)); add(charts.c23_importance(imp))
        add(charts.c24_ic_decay(icd)); add(charts.c25_regime_timeline(c, "601919"))
        add(charts.c26_tail_risk(c, "601919")); add(charts.c27_sharpe_roll(c, "601919"))
        add(charts.c28_equity_vs_bench(panel)); add(charts.c29_volume(c, "601919"))
        add(charts.c30_overfit_pbo(pbo)); add(charts.c31_deflated(dsr))
        aud = {"random_split_auc": vol.get("random_split_auc"),
               "purged_wf_auc": vol["auc_mean"],
               "shuffled_label_auc": vol_shuf["auc_mean"],
               "leak_injected_auc": vol_leak["auc_mean"]}
        add(charts.c32_leak_audit(aud)); add(charts.c33_alpha_corr(panel))
    except Exception:
        import traceback; traceback.print_exc()
    print(f"      成功生成 {len(made)} 张")

    print("[6/6] 写汇总 ...")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "universe": list(panel), "n_symbols": len(panel),
        "tasks": {
            "vol_regime": {k: vol[k] for k in ("n", "n_feat", "n_folds", "auc_mean", "auc_std",
                                               "auc_min", "auc_max", "pr_auc_mean",
                                               "brier_mean", "pos_rate", "per_fold")},
            "direction_1d": {k: dir_[k] for k in ("n", "n_folds", "auc_mean", "auc_std", "per_fold")},
            "shuffled_control": {"auc_mean": vol_shuf["auc_mean"], "auc_std": vol_shuf["auc_std"]},
            "random_split_control": {"auc": vol.get("random_split_auc")},
            "leak_injected": {"auc": vol_leak["auc_mean"]},
        },
        "strategy": {"n_configs": int(len(perf.columns)), "best": bi,
                     "best_sr_daily": round(sr, 4),
                     "best_ann_ret": round(float(r.mean() * 252), 4),
                     "best_ann_vol": round(float(r.std() * np.sqrt(252)), 4),
                     "pbo": pbo, "dsr": dsr},
        "charts": len(made),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (REP / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:1800])
    return summary


if __name__ == "__main__":
    main()
