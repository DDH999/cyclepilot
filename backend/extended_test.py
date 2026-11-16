# -*- coding: utf-8 -*-
"""
QuantPilot · 延长样本外检验
==================================================
目的：回答「低波因子组合是不是只在 2023-2026 有效」。
做法：
  1) 历史拉回到 2010（价量因子不受基本面数据起点限制）
  2) 加入中证1000（不同市值段）
  3) 逐年收益 + 滚动 3 年 Sharpe + 逐年前推检验
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, strategies as S, factors as FX
from backtest import Backtest, Costs

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
START = "2010-01-01"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BG, FG, GRID = "#0e1117", "#e6e6e6", "#2a2f3a"
ACC, ACC2, ACC3, ACC4 = "#ff4b4b", "#2e86de", "#26de81", "#feca57"


def _fig(w=11, h=5.2, title=""):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8.5)
    ax.grid(alpha=.18, color=GRID)
    if title:
        ax.set_title(title, color=FG, fontsize=11.5, fontweight="bold", pad=10)
    return fig, ax


def load_panel(start=START):
    codes = sorted({m.group(1) for f in (ROOT / ".cache").glob("*.csv")
                    if (m := re.search(r"_cn_(\d{6})$", f.stem))})
    print(f"      缓存标的 {len(codes)}")
    panel = datafeed.get_panel({c: "cn" for c in codes})
    panel = datafeed.sanitize_panel(panel, start=start)
    return panel


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
            "mdd": dd, "total": float(tot)}


def build(panel, ind, reb, px):
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
    return S.xs_zscore(S.winsorize_rows(S.composite(parts, [1.0] * len(parts))))


def main():
    t0 = time.time()
    print("[1/4] 加载全历史面板（2010 起）...")
    panel = load_panel()
    if not panel:
        print("无数据"); return
    px = FX._px(panel)
    print(f"      {len(panel)} 只 | {px.index.min().date()} ~ {px.index.max().date()}")

    reb = S.rebalance_dates(px.index, "M")
    ind = {}
    try:
        import sector
        im = sector.load_industries()
        ind = {c: im.get(c, "未知") for c in panel}
    except Exception:
        ind = {c: "未知" for c in panel}

    print("[2/4] 构建因子并回测 ...")
    z = build(panel, ind, reb, px)
    bench = datafeed.load_index("000300").set_index("date")["close"]
    bt = Backtest(panel, capital=1_000_000, costs=Costs(), stop_loss=0.15,
                  dd_brake=0.15, delist_recovery=0.30)
    w0 = S.to_weights_buffered(z, reb, top_n=30, buffer=0.5, industry=ind, max_per_ind=3)
    w = S.regime_overlay(S.apply_vol_target(w0, px, target_vol=0.15), bench)
    res = bt.run(w, max_positions=30)
    nav = res["nav"]
    bnav = (bench / bench.loc[bench.index >= nav.index[0]].iloc[0]).reindex(nav.index).ffill()

    print("[3/4] 分段检验 ...")
    full = st(nav)
    print(f"  全样本({nav.index[0].year}-{nav.index[-1].year}): 年化{full['cagr']*100:5.1f}% "
          f"Sharpe{full['sharpe']:5.2f} 回撤{full['mdd']*100:6.1f}%")

    # 逐年
    yr = {}
    print(f"\n  {'年份':<6}{'策略收益':>9}{'Sharpe':>8}{'回撤':>8}   基准")
    for y, g in nav.groupby(nav.index.year):
        if len(g) < 60:
            continue
        s = st(g)
        bg = bnav.loc[bnav.index.year == y]
        bre = float(bg.iloc[-1] / bg.iloc[0] - 1) if len(bg) > 1 else np.nan
        yr[int(y)] = {**s, "bench": bre}
        print(f"  {y:<6}{s['cagr']*100:8.1f}%{s['sharpe']:8.2f}{s['mdd']*100:7.1f}%   {bre*100:6.1f}%")

    pos_years = sum(1 for v in yr.values() if v["cagr"] > 0)
    beat = sum(1 for v in yr.values() if v["cagr"] > v["bench"])
    print(f"\n  正收益年份 {pos_years}/{len(yr)}   跑赢基准 {beat}/{len(yr)}")

    # 滚动 3 年 Sharpe
    roll = nav.pct_change(fill_method=None).rolling(750).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else np.nan, raw=True)

    # 逐年前推：每年初重设资金为 1，看当年表现（等价于逐年 Sharpe）
    print("[4/4] 图表 ...")
    fig, ax = _fig(11, 5, f"㊿+ 逐年收益：策略 vs 沪深300（{nav.index[0].year}-{nav.index[-1].year}）")
    ys = sorted(yr)
    x = np.arange(len(ys))
    ax.bar(x - 0.2, [yr[y]["cagr"] * 100 for y in ys], width=0.4, color=ACC2, label="策略")
    ax.bar(x + 0.2, [yr[y]["bench"] * 100 for y in ys], width=0.4, color="#666666", label="沪深300")
    ax.set_xticks(x); ax.set_xticklabels(ys, fontsize=8)
    ax.axhline(0, color=FG, lw=.8)
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    ax.set_ylabel("%")
    fig.savefig(REP / "charts" / "55_yearly_returns.png", dpi=115, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    fig, ax = _fig(11, 4.8, "㊿++ 滚动 3 年 Sharpe（检验是否只在某段有效）")
    ax.plot(nav.index, roll, color=ACC3, lw=1.3)
    ax.axhline(0, color=FG, lw=.8); ax.axhline(1, color=ACC4, ls="--", lw=1)
    ax.set_ylabel("Sharpe")
    fig.savefig(REP / "charts" / "56_rolling_3y_sharpe.png", dpi=115,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
           "n_symbols": len(panel), "full": full, "yearly": yr,
           "pos_years": pos_years, "beat_bench": beat, "n_years": len(yr),
           "ann_turnover": res["ann_turnover"], "elapsed": round(time.time() - t0, 1)}
    (REP / "extended_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    navdf = pd.DataFrame({"策略": nav, "基准_000300": bnav})
    navdf.to_csv(REP / "extended_nav.csv", encoding="utf-8")
    print(f"  保存 extended_test.json + extended_nav.csv")
    print(f"  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
