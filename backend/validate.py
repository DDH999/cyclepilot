# -*- coding: utf-8 -*-
"""
QuantPilot · 最终诚实检验
=========================================
回答竞赛评委一定会问的那个问题：
  "你 14 个策略里挑出来的最好的那个，是真的好，还是你挑出来的？"

三重检验：
  1. PBO (CSCV) — 样本内最优在样本外排名靠后的概率
  2. Deflated Sharpe — 对"试了很多次"做惩罚后的真实置信度
  3. 样本内选优 -> 样本外验证 — 模拟真实决策流程
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evals import pbo_cscv, deflated_sharpe

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
SPLIT = "2023-01-01"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BG, FG, GRID = "#0e1117", "#e6e6e6", "#2a2f3a"
ACC, ACC2, ACC3, ACC4 = "#ff4b4b", "#2e86de", "#26de81", "#feca57"


def _fig(w=10, h=5.2, title=""):
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
    p = REP / "charts" / name
    fig.savefig(p, dpi=115, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


def stats_of(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    tot = float((1 + r).prod() - 1)
    cagr = (1 + tot) ** (1 / yrs) - 1
    vol = float(r.std() * np.sqrt(252))
    sh = float(r.mean() * 252 / vol) if vol > 0 else np.nan
    nav = (1 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    return {"total": tot, "cagr": cagr, "sharpe": sh, "mdd": mdd,
            "calmar": cagr / abs(mdd) if mdd < 0 else np.nan}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", default="nav_curves.csv", help="净值文件（reports/ 下）")
    ap.add_argument("--out", default="validation.json")
    ap.add_argument("--trials", type=int, default=0,
                    help="多重检验次数（默认=策略数；因子从 N 个里筛出则填 N）")
    a = ap.parse_args()
    nav = pd.read_csv(REP / a.nav, index_col=0, parse_dates=True)
    res = json.loads((REP / "backtest_results.json").read_text(encoding="utf-8")) \
        if (REP / "backtest_results.json").exists() else {}
    bm_cols = [c for c in nav.columns if c.startswith("基准") or c == "等权票池"]
    strat_cols = [c for c in nav.columns if c not in bm_cols]
    rets = nav[strat_cols].pct_change(fill_method=None).dropna()
    print(f"策略 {len(strat_cols)} 个 × {len(rets)} 天\n")

    # ── 1. PBO (CSCV)
    print("[1/3] PBO 过拟合概率 (CSCV, 10 blocks)")
    if len(strat_cols) < 3:
        pbo = {"pbo": None, "reason": "策略数<3，PBO 不适用（单一规则无选择过程）"}
        print("      跳过：策略数<3，PBO 不适用")
        verdict_pbo = "跳过（单一规则，无选择过程）"
    else:
        pbo = pbo_cscv(rets.values, n_blocks=10)
        print(f"      PBO = {pbo.get('pbo'):.3f}  组合数={pbo.get('n_combinations')}  "
              f"IS最优的OOS中位排名={pbo.get('median_oos_rank_of_is_best'):.2f}")
        verdict_pbo = ("通过（<0.5，策略族整体可复现）" if pbo.get("pbo") < 0.5
                       else "不通过（≥0.5，样本内挑优不可复现）")

    # ── 2. Deflated Sharpe（对全部变体数做惩罚）
    full = {c: stats_of(nav[c].pct_change(fill_method=None)) for c in strat_cols}
    best_full = max(full, key=lambda k: full[k].get("sharpe", -9))
    r_all = nav[best_full].pct_change(fill_method=None).dropna()
    sr_d = float(r_all.mean() / r_all.std())
    n_trials = a.trials or len(strat_cols)
    dsr = deflated_sharpe(sr_d, n_trials=n_trials, n_obs=len(r_all),
                          skew=float(r_all.skew()), kurt=float(r_all.kurtosis() + 3))
    print(f"[2/3] Deflated Sharpe：全样本最优={best_full}  多重检验次数={n_trials}")
    print(f"      原始年化SR={sr_d*np.sqrt(252):.2f}  基准SR={dsr['sr_benchmark']*np.sqrt(252):.2f}  "
          f"DSR={dsr['dsr']:.3f}（>0.95 才算显著）")

    # ── 3. 样本内选优 -> 样本外验证
    print(f"[3/3] 样本内(≤2022)选优 -> 样本外(≥{SPLIT})验证")
    is_r = nav.loc[:SPLIT, strat_cols].pct_change(fill_method=None).dropna()
    oos_r = nav.loc[SPLIT:, strat_cols].pct_change(fill_method=None).dropna()
    is_s = {c: stats_of(is_r[c]) for c in strat_cols}
    oos_s = {c: stats_of(oos_r[c]) for c in strat_cols}
    pick_by_is = max(strat_cols, key=lambda c: is_s[c].get("sharpe", -9))
    pick_by_full = best_full
    same = pick_by_is == pick_by_full
    print(f"      IS 选出的最优 : {pick_by_is}")
    print(f"        IS  Sharpe={is_s[pick_by_is]['sharpe']:+.2f}  年化={is_s[pick_by_is]['cagr']*100:.1f}%")
    print(f"        OOS Sharpe={oos_s[pick_by_is]['sharpe']:+.2f}  年化={oos_s[pick_by_is]['cagr']*100:.1f}%  "
          f"回撤={oos_s[pick_by_is]['mdd']*100:.1f}%")
    print(f"      全样本最优是否等于 IS 最优: {'是（无选择偏差迹象）' if same else '否（存在选择偏差）'}")

    # 排名稳定性
    is_rank = pd.Series({c: is_s[c].get("sharpe", -9) for c in strat_cols}).rank(ascending=False)
    oos_rank = pd.Series({c: oos_s[c].get("sharpe", -9) for c in strat_cols}).rank(ascending=False)
    rho = float(stats.spearmanr(is_rank.values, oos_rank.values).statistic)
    print(f"      IS 与 OOS 的 Sharpe 排名相关 (Spearman ρ) = {rho:+.3f}")

    out = {"generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
           "source": a.nav, "split_date": SPLIT, "n_strategies": len(strat_cols),
           "pbo": pbo, "pbo_verdict": verdict_pbo,
           "deflated_sharpe": dsr,
           "best_full_sample": best_full,
           "best_by_is": pick_by_is,
           "is_best_equals_full_best": bool(same),
           "is_oos_rank_corr": rho,
           "is": {c: is_s[c] for c in strat_cols},
           "oos": {c: oos_s[c] for c in strat_cols},
           "full": full}
    (REP / "validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # ── 图表
    made = []
    # A. IS vs OOS 散点
    fig, ax = _fig(9.5, 6, "㊷ 样本内 vs 样本外 Sharpe（越靠右上越可信）")
    xs = [is_s[c].get("sharpe", 0) for c in strat_cols]
    ys = [oos_s[c].get("sharpe", 0) for c in strat_cols]
    ax.scatter(xs, ys, s=90, color=ACC2, alpha=.8, edgecolor=FG, linewidth=.4)
    for c, x, y in zip(strat_cols, xs, ys):
        ax.annotate(c, (x, y), fontsize=7, color=FG, xytext=(4, 4), textcoords="offset points")
    lim = [min(xs + ys) - .2, max(xs + ys) + .2]
    ax.plot(lim, lim, color=FG, ls="--", lw=1, label="IS=OOS 完全一致")
    ax.axhline(0, color=ACC, lw=.7); ax.axvline(0, color=ACC, lw=.7)
    ax.set_xlabel("样本内 Sharpe (2016-2022)"); ax.set_ylabel("样本外 Sharpe (2023-2026)")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    made.append(_save(fig, "42_is_vs_oos.png"))

    # B. PBO 可视化
    fig, ax = _fig(8, 5, "㊸ PBO 过拟合概率")
    v = pbo.get("pbo") or 0
    ax.barh(["PBO"], [v * 100], color=ACC if v >= .5 else ACC3)
    ax.text(v * 100, 0, f"  {v:.1%}", va="center", color=FG, fontsize=14, fontweight="bold")
    ax.axvline(50, color=ACC4, ls="--", lw=1.2, label="50% 警戒线")
    ax.set_xlim(0, 100); ax.set_xlabel("%")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    made.append(_save(fig, "43_pbo_final.png"))

    # C. 排名相关
    fig, ax = _fig(9, 5.4, f"㊹ IS/OOS Sharpe 排名一致性 (ρ={rho:+.3f})")
    order = is_rank.sort_values().index
    ax.plot([is_rank[c] for c in order], [oos_rank[c] for c in order],
            marker="o", color=ACC3, lw=1.6)
    for c in order:
        ax.annotate(c, (is_rank[c], oos_rank[c]), fontsize=6.8, color=FG,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("样本内排名（1=最好）"); ax.set_ylabel("样本外排名")
    ax.invert_xaxis(); ax.invert_yaxis()
    made.append(_save(fig, "44_rank_consistency.png"))

    print(f"\n图表: {[p.name for p in made]}")
    print(f"保存: reports/validation.json")
    print("\n" + "=" * 70)
    pv = pbo.get("pbo")
    print(f"结论: PBO={'—' if pv is None else f'{pv:.3f}'} -> {verdict_pbo}")
    print(f"      DSR={dsr['dsr']:.3f}（n_trials={n_trials}） -> "
          f"{'显著' if dsr['dsr'] > .95 else '不显著（多重检验后站不住）'}")
    if np.isfinite(rho):
        print(f"      IS/OOS 排名相关 ρ={rho:+.2f} -> "
              f"{'较稳定' if rho > .5 else '不稳定，存在过拟合迹象'}")
    else:
        print("      IS/OOS 排名相关 ρ=—（仅单一策略，无法计算）")
    print("=" * 70)
    return out


if __name__ == "__main__":
    main()
