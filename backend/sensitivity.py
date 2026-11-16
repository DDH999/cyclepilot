# -*- coding: utf-8 -*-
"""QuantPilot · 参数敏感性网格 —— 检验结论是否依赖特定参数"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, strategies as S, factors as FX
from backtest import Backtest, Costs

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
SPLIT = "2023-01-01"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_panel():
    import hashlib
    cc = ROOT / ".cache"
    codes = sorted({m.group(1) for f in cc.glob("*.csv")
                    if (m := re.search(r"_cn_(\d{6})$", f.stem))})
    newest = int(max((f.stat().st_mtime for f in cc.glob("*.csv")), default=0))
    sig = f"{len(codes)}|2016-01-01|{newest}"
    cp = cc / f"panel_{hashlib.md5(sig.encode()).hexdigest()[:12]}.pkl"
    return pd.read_pickle(cp) if cp.exists() else None


def st(nav):
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    tot = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = float((nav / nav.cummax() - 1).min())
    return {"cagr": float(cagr), "sharpe": float(r.mean() * 252 / vol) if vol > 0 else np.nan,
            "mdd": dd}


def main():
    t0 = time.time()
    panel = load_panel()
    if panel is None:
        print("无面板缓存"); return
    px = FX._px(panel)
    reb = S.rebalance_dates(px.index, "M")
    ind = {}
    try:
        import sector
        im = sector.load_industries()
        ind = {c: im.get(c, "未知") for c in panel}
    except Exception:
        ind = {c: "未知" for c in panel}

    # 因子只算一次
    amt, volu = FX._amt(panel), FX._vol(panel)
    r = px.pct_change(fill_method=None)
    vol60 = r.rolling(60).std()
    ind_s = pd.Series({c: ind.get(c, "未知") for c in px.columns})
    parts = [(-vol60).reindex(reb), (-r.rolling(20).max()).reindex(reb),
             r.rolling(20).min().reindex(reb),
             (-volu.rolling(20).std() / volu.rolling(60).mean()).reindex(reb),
             np.log1p(amt.rolling(20).mean()).reindex(reb)]
    rows = {}
    for d in reb:
        if d in vol60.index:
            s = vol60.loc[d]
            rows[d] = -(s - s.groupby(ind_s.reindex(s.index)).transform("median"))
    parts.append(pd.DataFrame(rows).T.reindex(columns=px.columns))
    comp = S.composite(parts, [1.0] * len(parts))
    z = S.xs_zscore(S.winsorize_rows(comp))

    idx = datafeed.load_index("000300")
    bench = idx.set_index("date")["close"]
    bt = Backtest(panel, capital=1_000_000, costs=Costs(), stop_loss=0.15,
                  dd_brake=0.15, delist_recovery=0.30)
    print(f"[1/2] 因子就绪 {time.time()-t0:.0f}s，开始网格 ...")

    grid_top = [20, 30, 50]
    grid_vt = [0.10, 0.15, 0.20]
    rec = {}
    for tn in grid_top:
        w0 = S.to_weights_buffered(z, reb, top_n=tn, buffer=0.5,
                                   industry=ind, max_per_ind=3)
        for vt in grid_vt:
            w = S.regime_overlay(S.apply_vol_target(w0, px, target_vol=vt), bench)
            res = bt.run(w, max_positions=tn)
            if "error" in res:
                continue
            nav = res["nav"]
            f, i_, o = st(nav), st(nav.loc[:SPLIT]), st(nav.loc[SPLIT:])
            rec[f"{tn}_{vt}"] = {"top": tn, "vt": vt, "full": f, "is": i_, "oos": o,
                                 "turnover": res["ann_turnover"]}
            print(f"  top={tn:3d} vt={vt:.2f}  全样本 Sharpe{f['sharpe']:5.2f} "
                  f"年化{f['cagr']*100:5.1f}%  回撤{f['mdd']*100:6.1f}%  "
                  f"OOS Sharpe{o['sharpe']:5.2f}")

    # 热力图
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    hm = np.array([[rec[f"{tn}_{vt}"]["full"]["sharpe"] for vt in grid_vt] for tn in grid_top])
    im = ax.imshow(hm, cmap="RdYlGn", aspect="auto",
                   vmin=min(0, hm.min()), vmax=max(0.1, hm.max()))
    ax.set_xticks(range(len(grid_vt))); ax.set_xticklabels([f"VT={v}" for v in grid_vt], color="#e6e6e6")
    ax.set_yticks(range(len(grid_top))); ax.set_yticklabels([f"Top{v}" for v in grid_top], color="#e6e6e6")
    for i in range(len(grid_top)):
        for j in range(len(grid_vt)):
            ax.text(j, i, f"{hm[i,j]:.2f}", ha="center", va="center", fontsize=12,
                    color="black", fontweight="bold")
    ax.set_title("参数敏感性：全样本 Sharpe（越大越绿）", color="#e6e6e6", fontsize=12, fontweight="bold")
    for s in ax.spines.values():
        s.set_color("#2a2f3a")
    fig.colorbar(im, ax=ax)
    fig.savefig(REP / "charts" / "54_param_sensitivity.png", dpi=115,
                bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)

    vals = [v["full"]["sharpe"] for v in rec.values()]
    print(f"\n[2/2] 敏感性: Sharpe 范围 {min(vals):.2f} ~ {max(vals):.2f}, "
          f"中位 {np.median(vals):.2f}, 标准差 {np.std(vals):.3f}")
    print(f"      正 Sharpe 占比 {sum(1 for v in vals if v>0)/len(vals):.0%}")
    json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "grid": rec,
               "summary": {"min": float(min(vals)), "max": float(max(vals)),
                           "median": float(np.median(vals)), "std": float(np.std(vals))}},
              open(REP / "sensitivity.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"      保存 sensitivity.json + charts/54_param_sensitivity.png")
    print(f"      总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
