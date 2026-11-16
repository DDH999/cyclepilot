# -*- coding: utf-8 -*-
"""QuantPilot · 用通过 FDR 的因子构建组合并验证"""
from __future__ import annotations
import sys, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, strategies as S, factors as FX
from backtest import Backtest, Costs, benchmark_ew, attribution

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
SPLIT = "2023-01-01"

# 只保留 FDR 通过且彼此不重复的因子
KEEP = ["lowvol_60", "max_ret_20", "ind_rel_lowvol", "turnover_stab", "liquidity", "min_ret_20"]


def load_panel():
    import hashlib
    cc = ROOT / ".cache"
    codes = sorted({m.group(1) for f in cc.glob("*.csv")
                    if (m := re.search(r"_cn_(\d{6})$", f.stem))})
    newest = int(max((f.stat().st_mtime for f in cc.glob("*.csv")), default=0))
    sig = f"{len(codes)}|2016-01-01|{newest}"
    cp = cc / f"panel_{hashlib.md5(sig.encode()).hexdigest()[:12]}.pkl"
    return pd.read_pickle(cp) if cp.exists() else None


def stats(nav):
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    tot = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = float((nav / nav.cummax() - 1).min())
    return {"cagr": float(cagr), "sharpe": float(r.mean() * 252 / vol) if vol > 0 else np.nan,
            "mdd": dd, "calmar": float(cagr / abs(dd)) if dd < 0 else None, "total": float(tot)}


def main():
    t0 = time.time()
    panel = load_panel()
    if panel is None:
        print("无面板缓存，请先跑 run_strategy"); return
    px = FX._px(panel)
    reb = S.rebalance_dates(px.index, "M")
    ind = {}
    try:
        import sector
        im = sector.load_industries()
        ind = {c: im.get(c, "未知") for c in panel}
    except Exception:
        ind = {c: "未知" for c in panel}
    print(f"[1/4] {len(panel)} 只 / {len(reb)} 调仓日")

    print("[2/4] 构建 FDR 存活因子 ...")
    amt = FX._amt(panel)
    volu = FX._vol(panel)
    r = px.pct_change(fill_method=None)
    vol60 = r.rolling(60).std()
    ind_s = pd.Series({c: ind.get(c, "未知") for c in px.columns})

    # 只算通过 FDR 的因子；方向严格按 factor_lab 实测 IC 的符号。
    # 注意：liquidity / amihud 的 IC 为负（越流动、越不流动 → 未来收益越低），
    # 组合中必须取反，否则方向做反。
    parts = {}
    parts["lowvol_60"] = (-vol60).reindex(reb)                       # IC  +
    parts["max_ret_20"] = (-r.rolling(20).max()).reindex(reb)        # IC  +
    parts["min_ret_20"] = (r.rolling(20).min()).reindex(reb)         # IC  +
    parts["turnover_stab"] = (-volu.rolling(20).std() /
                              volu.rolling(60).mean()).reindex(reb)  # IC  +
    parts["liquidity"] = (-np.log1p(amt.rolling(20).mean())).reindex(reb)  # IC  −，取反
    amihud_raw = (-(r.abs() / amt.replace(0, np.nan)).rolling(20).mean() * 1e10)
    parts["amihud"] = (-amihud_raw).reindex(reb)                     # IC  −，取反
    rows_rl = {}
    for d in reb:
        if d in vol60.index:
            s = vol60.loc[d]
            med = s.groupby(ind_s.reindex(s.index)).transform("median")
            rows_rl[d] = -(s - med)
    parts["ind_rel_lowvol"] = pd.DataFrame(rows_rl).T.reindex(columns=px.columns)  # IC +
    print(f"      {len(parts)} 个因子（方向已按 IC 校正）: {list(parts)}")

    print("[3/4] 回测 ...")
    bt = Backtest(panel, capital=1_000_000, costs=Costs(), stop_loss=0.15,
                  dd_brake=0.15, delist_recovery=0.30)
    idx = datafeed.load_index("000300"); s = idx.set_index("date")["close"]
    bnav = s / s.iloc[0]

    comp = S.composite([parts[k] for k in parts], [1.0] * len(parts))
    z = S.xs_zscore(S.winsorize_rows(comp))
    w = S.to_weights_buffered(z, reb, top_n=30, buffer=0.5,
                              industry=ind, max_per_ind=3)
    w = S.regime_overlay(S.apply_vol_target(w, px, target_vol=0.15), s)
    res = bt.run(w, max_positions=30)
    nav = res["nav"]
    f_all, f_is, f_oos = stats(nav), stats(nav.loc[:SPLIT]), stats(nav.loc[SPLIT:])
    at = attribution(nav, bnav.reindex(nav.index).ffill())

    print(f"\n[4/4] 结果（FDR 存活因子合成）")
    print(f"  全样本   年化{f_all['cagr']*100:6.1f}%  Sharpe{f_all['sharpe']:5.2f}  "
          f"回撤{f_all['mdd']*100:6.1f}%  Calmar{f_all['calmar'] or 0:5.2f}")
    print(f"  样本内   年化{f_is['cagr']*100:6.1f}%  Sharpe{f_is['sharpe']:5.2f}  回撤{f_is['mdd']*100:6.1f}%")
    print(f"  样本外   年化{f_oos['cagr']*100:6.1f}%  Sharpe{f_oos['sharpe']:5.2f}  回撤{f_oos['mdd']*100:6.1f}%")
    print(f"  Beta{at.get('beta',0):5.2f}  超额年化{(at.get('alpha_ann') or 0)*100:5.1f}%  IR{at.get('information_ratio',0):5.2f}")
    print(f"  年换手 {res['ann_turnover']:.0f}%  退市核销 {res['n_writeoffs']}  止损 {res['n_stops']}  制动 {res['n_brakes']}")

    navdf = pd.DataFrame({"FDR因子合成": nav, "基准_000300": bnav.reindex(nav.index).ffill()})
    navdf.to_csv(REP / "combo_nav.csv", encoding="utf-8")
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "factors": list(parts),
           "full": f_all, "is": f_is, "oos": f_oos,
           "beta": at.get("beta"), "alpha_ann": at.get("alpha_ann"),
           "ir": at.get("information_ratio"),
           "ann_turnover": res["ann_turnover"],
           "n_writeoffs": res["n_writeoffs"], "n_stops": res["n_stops"],
           "n_brakes": res["n_brakes"], "elapsed": round(time.time() - t0, 1)}
    (REP / "combo_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n保存 combo_nav.csv + combo_result.json")
    print("下一步: python backend/validate.py --nav combo_nav.csv")


if __name__ == "__main__":
    main()
