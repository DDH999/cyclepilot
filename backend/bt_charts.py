# -*- coding: utf-8 -*-
"""QuantPilot · 回测专用图表（竞赛级展示）"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
OUT = REP / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BG, FG, GRID = "#0e1117", "#e6e6e6", "#2a2f3a"
ACC, ACC2, ACC3, ACC4 = "#ff4b4b", "#2e86de", "#26de81", "#feca57"
PAL = ["#2e86de", "#26de81", "#feca57", "#ff4b4b", "#a55eea", "#00d2d3", "#ff9f43", "#e84393"]


def _fig(w=11, h=5.4, title=""):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8.5)
    ax.grid(alpha=.18, color=GRID)
    if title:
        ax.set_title(title, color=FG, fontsize=11.5, fontweight="bold", pad=10)
    return fig, ax


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=115, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


def main():
    nav = pd.read_csv(REP / "nav_curves.csv", index_col=0, parse_dates=True)
    nav = nav / nav.iloc[0]          # 统一归一到回测起点 = 1.0
    res = json.loads((REP / "backtest_results.json").read_text(encoding="utf-8"))
    S = res["strategies"]
    made = []

    # 1 净值曲线
    fig, ax = _fig(12, 5.8, "㉞ 策略净值曲线 vs 双基准（对数坐标）")
    best = sorted(S, key=lambda k: -S[k].get("sharpe", -9))[:6]
    for i, c in enumerate(best):
        ax.plot(nav.index, nav[c], lw=1.5, color=PAL[i % len(PAL)], label=c)
    for b, ls in [(f"基准_{res['benchmark']}", "--"), ("等权票池", ":")]:
        if b in nav:
            ax.plot(nav.index, nav[b], lw=1.8, color=FG, ls=ls, label=b, alpha=.8)
    ax.set_yscale("log")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3, ncol=2)
    made.append(_save(fig, "34_bt_equity.png"))

    # 2 回撤
    fig, ax = _fig(12, 4.6, "㉟ 回撤曲线对比")
    for i, c in enumerate(best[:4]):
        dd = nav[c] / nav[c].cummax() - 1
        ax.plot(nav.index, dd * 100, lw=1.2, color=PAL[i % len(PAL)], label=c)
    ax.set_ylabel("回撤 %")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3, ncol=2)
    made.append(_save(fig, "35_bt_drawdown.png"))

    # 3 风险收益散点
    fig, ax = _fig(9, 6, "㊱ 风险-收益散点（Sharpe vs 最大回撤）")
    for n, r in S.items():
        sz = 90 + 260 * max(r.get("cagr", 0), 0)
        col = ACC3 if r.get("sharpe", 0) >= 0.8 else ACC4 if r.get("sharpe", 0) >= 0.6 else ACC
        ax.scatter(abs(r.get("max_drawdown", 0)) * 100, r.get("sharpe", 0), s=sz,
                   color=col, alpha=.72, edgecolor=FG, linewidth=.4)
    for n, r in S.items():
        if r.get("sharpe", 0) >= 0.85 or abs(r.get("max_drawdown", 0)) < 0.2:
            ax.annotate(n, (abs(r["max_drawdown"]) * 100, r["sharpe"]),
                        fontsize=7.4, color=FG, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("最大回撤 %"); ax.set_ylabel("Sharpe")
    made.append(_save(fig, "36_bt_scatter.png"))

    # 4 滚动 Sharpe
    if best:
        c = best[0]
        r = nav[c].pct_change(fill_method=None)
        sh = r.rolling(250).mean() / r.rolling(250).std() * np.sqrt(252)
        fig, ax = _fig(11, 4.4, f"㊲ {c} 滚动 250 日 Sharpe")
        ax.plot(nav.index, sh, color=ACC2, lw=1.3)
        ax.axhline(0, color=FG, lw=.6)
        ax.axhline(1, color=ACC3, ls="--", lw=1, label="Sharpe=1")
        ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
        made.append(_save(fig, "37_bt_rollsharpe.png"))

        # 5 月度热力图
        d = nav[c].resample("ME").last().pct_change(fill_method=None).dropna()
        m = pd.DataFrame({"y": d.index.year, "mo": d.index.month, "r": d.values * 100})
        pv = m.pivot_table(index="y", columns="mo", values="r")
        fig, ax = plt.subplots(figsize=(11, max(3.2, len(pv) * .42)), facecolor=BG)
        ax.set_facecolor(BG)
        vmax = np.nanmax(np.abs(pv.values))
        im = ax.imshow(pv.values, cmap="RdYlGn", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index, color=FG, fontsize=8)
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([f"{c}月" for c in pv.columns], color=FG, fontsize=8)
        ax.set_title(f"㊳ {c} 月度收益热力图 %", color=FG, fontsize=11.5, fontweight="bold", pad=10)
        cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(colors=FG)
        for s in ax.spines.values():
            s.set_color(GRID)
        made.append(_save(fig, "38_bt_monthlyheat.png"))

    # 6 年度收益对比
    yrs = sorted({d.year for d in nav.index})
    fig, ax = _fig(12, 5, "㊴ 年度收益：Top 策略 vs 指数基准")
    w = 0.8 / (min(len(best), 4) + 1)
    for i, c in enumerate(best[:4]):
        ann = nav[c].resample("YE").last().pct_change(fill_method=None)
        vals = [float(ann.get(pd.Timestamp(f"{y}-12-31"), np.nan)) * 100 for y in yrs]
        ax.bar(np.arange(len(yrs)) + i * w, vals, width=w, color=PAL[i % len(PAL)], label=c)
    bm = f"基准_{res['benchmark']}"
    if bm in nav:
        ann = nav[bm].resample("YE").last().pct_change(fill_method=None)
        ax.plot(np.arange(len(yrs)), [float(ann.get(pd.Timestamp(f"{y}-12-31"), np.nan)) * 100
                                      for y in yrs], color=FG, marker="o", ms=3,
                lw=1.5, label=bm)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs, fontsize=8)
    ax.set_ylabel("%"); ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3, ncol=2)
    made.append(_save(fig, "39_bt_annual.png"))

    # 7 超额累计
    bm = f"基准_{res['benchmark']}"
    if bm in nav and best:
        fig, ax = _fig(11, 4.6, "㊵ 相对指数超额累计（策略净值/基准净值）")
        for i, c in enumerate(best[:4]):
            ex = nav[c] / nav[bm]
            ax.plot(nav.index, ex, lw=1.4, color=PAL[i % len(PAL)], label=c)
        ax.axhline(1, color=FG, ls="--", lw=1)
        ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3, ncol=2)
        made.append(_save(fig, "40_bt_excess.png"))

    # 8 换手 vs 成本敏感性
    fig, ax = _fig(9.5, 5.2, "㊶ 换手率与年化收益（成本敏感度提示）")
    for n, r in S.items():
        ax.scatter(r.get("ann_turnover", 0), r.get("cagr", 0) * 100, s=80,
                   color=ACC3 if "风控" in n else ACC2, alpha=.75, edgecolor=FG, linewidth=.4)
    ax.set_xlabel("年化换手 %"); ax.set_ylabel("年化收益 %")
    ax.text(.02, .95, "绿=含风控层，蓝=裸策略", transform=ax.transAxes, color=FG, fontsize=9, va="top")
    made.append(_save(fig, "41_bt_turnover.png"))

    print("生成", len(made), "张回测图表:")
    for p in made:
        print("  ", p.name)
    return made


if __name__ == "__main__":
    main()
