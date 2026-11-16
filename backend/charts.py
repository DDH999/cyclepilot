# -*- coding: utf-8 -*-
"""
QuantPilot · 图表工厂
33 个图表生成器，统一视觉语言。全部输出到 reports/charts/*.png
每个图都回答一个具体问题，不为凑数。
"""
from __future__ import annotations
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "reports" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0e1117"
FG = "#e6e6e6"
GRID = "#2a2f3a"
ACC = "#ff4b4b"
ACC2 = "#2e86de"
ACC3 = "#26de81"
ACC4 = "#feca57"


def _fig(w=10, h=5.2, title=""):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8.5)
    ax.grid(alpha=.18, color=GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    if title:
        ax.set_title(title, color=FG, fontsize=11.5, fontweight="bold", pad=10)
    return fig, ax


def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


# ═══════════════════ 价格 / 收益类 (1-10) ═══════════════════
def c01_price_ma(df, sym):
    fig, ax = _fig(title=f"① {sym} 价格与均线")
    d = df.tail(500)
    ax.plot(d["date"], d["close"], color=FG, lw=1.4, label="收盘")
    for n, c in [(20, ACC2), (60, ACC4), (120, ACC)]:
        ax.plot(d["date"], d["close"].rolling(n).mean(), color=c, lw=1, label=f"MA{n}")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, f"01_price_ma_{sym}")


def c02_cum_return(df, sym):
    fig, ax = _fig(title=f"② {sym} 累计收益 (对数)")
    r = df["close"].pct_change(fill_method=None)
    ax.plot(df["date"], (1 + r).cumprod(), color=ACC2, lw=1.3)
    ax.set_yscale("log")
    return _save(fig, f"02_cumret_{sym}")


def c03_drawdown(df, sym):
    fig, ax = _fig(title=f"③ {sym} 回撤曲线")
    c = df["close"]
    dd = c / c.cummax() - 1
    ax.fill_between(df["date"], dd * 100, 0, color=ACC, alpha=.45)
    ax.plot(df["date"], dd * 100, color=ACC, lw=.9)
    ax.set_ylabel("回撤 %")
    return _save(fig, f"03_drawdown_{sym}")


def c04_rolling_vol(df, sym):
    fig, ax = _fig(title=f"④ {sym} 滚动波动率年化")
    r = df["close"].pct_change(fill_method=None)
    for n, c in [(20, ACC2), (60, ACC4), (250, ACC3)]:
        ax.plot(df["date"], r.rolling(n).std() * np.sqrt(252) * 100, color=c, lw=1, label=f"{n}日")
    ax.set_ylabel("年化波动率 %")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, f"04_rollvol_{sym}")


def c05_vol_term(df, sym):
    fig, ax = _fig(title=f"⑤ {sym} 波动率期限结构")
    r = df["close"].pct_change(fill_method=None)
    for n, c in [(5, ACC4), (20, ACC2), (60, ACC3)]:
        ax.plot(df["date"], r.rolling(n).std(), color=c, lw=.9, label=f"vol{n}")
    ax.set_yscale("log")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, f"05_volterm_{sym}")


def c06_return_dist(df, sym):
    fig, ax = _fig(title=f"⑥ {sym} 日收益分布 vs 正态")
    r = df["close"].pct_change(fill_method=None).dropna() * 100
    ax.hist(r, bins=90, color=ACC2, alpha=.65, density=True, label="实际")
    x = np.linspace(r.min(), r.max(), 300)
    from scipy import stats as st
    ax.plot(x, st.norm.pdf(x, r.mean(), r.std()), color=ACC4, lw=1.6, label="正态拟合")
    k = st.kurtosis(r)
    ax.text(.02, .95, f"峰度={k:.2f}  偏度={st.skew(r):.2f}", transform=ax.transAxes,
            color=FG, fontsize=9, va="top")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, f"06_retdist_{sym}")


def c07_qq(df, sym):
    fig, ax = _fig(title=f"⑦ {sym} 收益 Q-Q 图")
    from scipy import stats as st
    r = df["close"].pct_change(fill_method=None).dropna()
    st.probplot(r, dist="norm", plot=ax)
    ax.get_lines()[0].set(color=ACC2, ms=2.5)
    ax.get_lines()[1].set(color=ACC4, lw=1.5)
    ax.title.set_color(FG)
    return _save(fig, f"07_qq_{sym}")


def c08_acf(df, sym):
    fig, ax = _fig(title=f"⑧ {sym} 收益自相关 (检验可预测性)")
    r = df["close"].pct_change(fill_method=None).dropna()
    lags = range(1, 41)
    acf = [r.autocorr(l) for l in lags]
    ax.bar(list(lags), acf, color=[ACC if abs(a) > 2 / np.sqrt(len(r)) else ACC2 for a in acf])
    ax.axhline(2 / np.sqrt(len(r)), color=ACC4, ls="--", lw=.9)
    ax.axhline(-2 / np.sqrt(len(r)), color=ACC4, ls="--", lw=.9)
    ax.axhline(0, color=FG, lw=.6)
    ax.set_xlabel("滞后阶")
    return _save(fig, f"08_acf_{sym}")


def c09_monthly_heat(df, sym):
    d = df.copy()
    d["y"] = d["date"].dt.year
    d["m"] = d["date"].dt.month
    g = d.groupby(["y", "m"])["close"].apply(lambda s: s.iloc[-1] / s.iloc[0] - 1).unstack() * 100
    fig, ax = _fig(12, max(3.5, len(g) * .38), title=f"⑨ {sym} 月度收益热力图 %")
    vmax = np.nanmax(np.abs(g.values))
    im = ax.imshow(g.values, cmap="RdYlGn", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_yticks(range(len(g.index))); ax.set_yticklabels(g.index, fontsize=7)
    ax.set_xticks(range(12)); ax.set_xticklabels(["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"], fontsize=8)
    cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(colors=FG)
    return _save(fig, f"09_monthly_{sym}")


def c10_annual(df, sym):
    d = df.copy(); d["y"] = d["date"].dt.year
    g = d.groupby("y")["close"].apply(lambda s: s.iloc[-1] / s.iloc[0] - 1) * 100
    fig, ax = _fig(title=f"⑩ {sym} 年度收益 %")
    ax.bar(g.index.astype(str), g.values, color=[ACC3 if v > 0 else ACC for v in g.values])
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, f"10_annual_{sym}")


# ═══════════════════ 横截面 / 相关性 (11-16) ═══════════════════
def c11_corr(panel):
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).dropna()
    cm = px.pct_change(fill_method=None).dropna().corr()
    fig, ax = _fig(9, 7.5, "⑪ 标的收益相关性矩阵")
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index, fontsize=7.5)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.3, color="black" if abs(cm.values[i, j]) > .6 else FG)
    fig.colorbar(im, ax=ax)
    return _save(fig, "11_corr_matrix")


def c12_roll_corr(panel):
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).dropna()
    r = px.pct_change(fill_method=None).dropna()
    if r.shape[1] < 2:
        return None
    a, b = r.columns[0], r.columns[1]
    fig, ax = _fig(title=f"⑫ {a} vs {b} 滚动60日相关性")
    ax.plot(r.index, r[a].rolling(60).corr(r[b]), color=ACC2, lw=1.1)
    ax.axhline(0, color=FG, lw=.6)
    return _save(fig, "12_rolling_corr")


def c13_pca(panel):
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).dropna()
    r = px.pct_change(fill_method=None).dropna()
    X = (r - r.mean()) / r.std()
    C = np.cov(X.T.values)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    exp = ev / ev.sum()
    fig, ax = _fig(title="⑬ 主成分解释方差（市场共同因子占比）")
    ax.bar(range(1, len(exp) + 1), exp * 100, color=ACC2)
    ax.plot(range(1, len(exp) + 1), np.cumsum(exp) * 100, color=ACC4, marker="o", ms=3)
    ax.set_xlabel("主成分"); ax.set_ylabel("%")
    ax.text(.4, .8, f"PC1={exp[0]*100:.1f}%  → 系统性风险主导" if exp[0] > .35 else f"PC1={exp[0]*100:.1f}%",
            transform=ax.transAxes, color=FG, fontsize=9)
    return _save(fig, "13_pca_variance")


def c14_sector_rank(perf: dict):
    s = pd.Series(perf).sort_values()
    fig, ax = _fig(title="⑭ 标的区间收益排名 %")
    ax.barh(s.index, s.values, color=[ACC if v < 0 else ACC3 for v in s.values])
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:+.1f}%", va="center", fontsize=8, color=FG)
    return _save(fig, "14_sector_rank")


def c15_beta(panel, bench=None):
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).dropna()
    r = px.pct_change(fill_method=None).dropna()
    mkt = r.mean(axis=1)
    betas = {s: np.polyfit(mkt, r[s], 1)[0] for s in r.columns}
    fig, ax = _fig(title="⑮ 各标的对等权组合的 Beta")
    ks = list(betas); vs = [betas[k] for k in ks]
    ax.bar(ks, vs, color=[ACC if v > 1.15 else ACC3 if v < .85 else ACC2 for v in vs])
    ax.axhline(1, color=ACC4, ls="--", lw=1)
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "15_beta")


def c16_rolling_beta(panel):
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).dropna()
    r = px.pct_change(fill_method=None).dropna()
    mkt = r.mean(axis=1)
    fig, ax = _fig(title="⑯ 滚动120日 Beta（系统性暴露漂移）")
    for s in list(r.columns)[:4]:
        cov = r[s].rolling(120).cov(mkt); var = mkt.rolling(120).var()
        ax.plot(r.index, cov / var, lw=1, label=s)
    ax.axhline(1, color=ACC4, ls="--", lw=.9)
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3, ncol=2)
    return _save(fig, "16_rolling_beta")


# ═══════════════════ 模型 / 评估类 (17-28) ═══════════════════
def c17_auc_folds(res):
    fig, ax = _fig(title=f"⑰ AUC 逐折稳定性 — {res.get('label','')}")
    pf = res["per_fold"]
    ax.plot(range(1, len(pf) + 1), pf, marker="o", color=ACC2, lw=1.6)
    ax.axhline(.5, color=ACC, ls="--", lw=1.1, label="随机 0.5")
    ax.axhline(.7, color=ACC3, ls="--", lw=1.1, label="目标 0.7")
    ax.set_ylim(.3, 1.0); ax.set_xlabel("折"); ax.set_ylabel("AUC")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "17_auc_folds")


def c18_roc(y, p):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = _fig(title=f"⑱ ROC 曲线 (AUC={auc(fpr,tpr):.4f})")
    ax.plot(fpr, tpr, color=ACC2, lw=2)
    ax.plot([0, 1], [0, 1], color=ACC, ls="--", lw=1)
    ax.set_xlabel("假阳率"); ax.set_ylabel("真阳率")
    return _save(fig, "18_roc")


def c19_pr(y, p):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    pr, rc, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = _fig(title=f"⑲ Precision-Recall (AP={ap:.4f})")
    ax.plot(rc, pr, color=ACC3, lw=2)
    ax.axhline(np.mean(y), color=ACC, ls="--", lw=1, label=f"基准={np.mean(y):.2%}")
    ax.set_xlabel("召回"); ax.set_ylabel("精确")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "19_pr_curve")


def c20_calibration(y, p):
    from sklearn.calibration import calibration_curve
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = _fig(title="⑳ 概率校准（预测概率 vs 实际频率）")
    ax.plot([0, 1], [0, 1], color=ACC, ls="--", lw=1, label="完美校准")
    ax.plot(mp, fr, marker="o", color=ACC2, lw=1.6, label="模型")
    ax.set_xlabel("预测概率"); ax.set_ylabel("实际频率")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "20_calibration")


def c21_score_dist(y, p):
    fig, ax = _fig(title="㉑ 预测分数分布（分类别）")
    ax.hist(p[y == 0], bins=50, alpha=.62, color=ACC, label="负例", density=True)
    ax.hist(p[y == 1], bins=50, alpha=.62, color=ACC3, label="正例", density=True)
    ax.set_xlabel("预测概率")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "21_score_dist")


def c22_confusion(y, p, thr=.5):
    tp = int(((p >= thr) & (y == 1)).sum()); fp = int(((p >= thr) & (y == 0)).sum())
    tn = int(((p < thr) & (y == 0)).sum()); fn = int(((p < thr) & (y == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = _fig(6, 5, f"㉒ 混淆矩阵 (阈值={thr})")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["预测负", "预测正"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["实际负", "实际正"])
    return _save(fig, "22_confusion")


def c23_importance(imp: dict):
    s = pd.Series(imp).sort_values().tail(18)
    fig, ax = _fig(9, 6.5, "㉓ 特征重要性 Top18")
    ax.barh(s.index, s.values, color=ACC2)
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, "23_importance")


def c24_ic_decay(ic: list):
    fig, ax = _fig(title="㉔ 因子 IC 衰减（滞后阶数）")
    ax.bar([x["lag"] for x in ic], [x["ic"] for x in ic], color=ACC2)
    ax.axhline(0, color=FG, lw=.6)
    ax.axhline(.03, color=ACC3, ls="--", lw=.9, label="IC=0.03 可用线")
    ax.set_xlabel("滞后交易日"); ax.set_ylabel("Spearman IC")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "24_ic_decay")


def c25_regime_timeline(df, sym):
    r = df["close"].pct_change(fill_method=None)
    v = r.rolling(20).std()
    q = v.rolling(250).rank(pct=True)
    fig, ax = _fig(title=f"㉕ {sym} 波动率状态迁移（高波=红）")
    ax.plot(df["date"], df["close"], color=FG, lw=.9)
    hi = q > .7
    ax.fill_between(df["date"], df["close"].min(), df["close"].max(),
                    where=hi, color=ACC, alpha=.18, step="mid")
    ax.set_yscale("log")
    return _save(fig, f"25_regime_{sym}")


def c26_tail_risk(df, sym):
    r = df["close"].pct_change(fill_method=None).dropna()
    from scipy import stats as st
    var95 = -np.percentile(r, 5) * 100
    cvar95 = -r[r <= np.percentile(r, 5)].mean() * 100
    var99 = -np.percentile(r, 1) * 100
    fig, ax = _fig(7.5, 5, f"㉖ {sym} 尾部风险 VaR/CVaR")
    ax.bar(["VaR95", "CVaR95", "VaR99"], [var95, cvar95, var99], color=[ACC4, ACC, ACC])
    for i, v in enumerate([var95, cvar95, var99]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", color=FG, fontsize=10)
    ax.set_ylabel("单日损失 %")
    return _save(fig, f"26_tailrisk_{sym}")


def c27_sharpe_roll(df, sym):
    r = df["close"].pct_change(fill_method=None)
    fig, ax = _fig(title=f"㉗ {sym} 滚动 Sharpe（250日）")
    sh = r.rolling(250).mean() / r.rolling(250).std() * np.sqrt(252)
    ax.plot(df["date"], sh, color=ACC3, lw=1.1)
    ax.axhline(0, color=FG, lw=.6); ax.axhline(1, color=ACC4, ls="--", lw=.9)
    return _save(fig, f"27_sharpe_{sym}")


def c28_equity_vs_bench(panel, syms=None):
    px = pd.DataFrame({s: d.set_index("date")["close"] for s, d in panel.items()}).dropna()
    r = px.pct_change(fill_method=None).dropna()
    nav = (1 + r).cumprod()
    bench = (1 + r.mean(axis=1)).cumprod()
    fig, ax = _fig(title="㉘ 各标的累计净值 vs 等权基准")
    for s in (syms or list(nav.columns)):
        ax.plot(nav.index, nav[s], lw=.9, alpha=.75, label=s)
    ax.plot(bench.index, bench, color=FG, lw=2.2, ls="--", label="等权基准")
    ax.set_yscale("log")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=7.5, framealpha=.3, ncol=3)
    return _save(fig, "28_equity_vs_bench")


# ═══════════════════ 风险 / 归因类 (29-33) ═══════════════════
def c29_volume(df, sym):
    fig, ax = _fig(title=f"㉙ {sym} 成交量与价格（量价背离检测）")
    ax.bar(df["date"].tail(350), df["volume"].tail(350) / 1e6, color=ACC2, alpha=.55, width=1)
    ax.set_ylabel("成交量(百万)")
    ax2 = ax.twinx(); ax2.set_facecolor(BG)
    ax2.plot(df["date"].tail(350), df["close"].tail(350), color=ACC4, lw=1.2)
    ax2.tick_params(colors=FG)
    for s in ax2.spines.values():
        s.set_color(GRID)
    return _save(fig, f"29_volume_{sym}")


def c30_overfit_pbo(pbo_res):
    fig, ax = _fig(7.5, 5, "㉚ 回测过拟合概率 PBO")
    v = pbo_res.get("pbo")
    if v is None:
        ax.text(.5, .5, "样本不足，无法计算 PBO", ha="center", va="center", color=FG, fontsize=13)
    else:
        ax.barh(["PBO"], [v * 100], color=ACC if v > .5 else ACC3)
        ax.text(v * 100, 0, f"  {v:.1%}", va="center", color=FG, fontsize=13)
        ax.axvline(50, color=ACC4, ls="--", lw=1, label="50% 警戒线")
        ax.set_xlim(0, 100); ax.set_xlabel("%")
        ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "30_pbo")


def c31_deflated(dsr):
    fig, ax = _fig(7.5, 5, "㉛ 通缩夏普 (Deflated Sharpe)")
    d = dsr.get("dsr")
    if d is None:
        ax.text(.5, .5, "无法计算", ha="center", va="center", color=FG, fontsize=13)
    else:
        ax.bar(["DSR"], [d], color=ACC3 if d > .95 else ACC4 if d > .5 else ACC)
        ax.axhline(.95, color=ACC3, ls="--", lw=1.2, label="95% 可信线")
        ax.set_ylim(0, 1.05); ax.set_ylabel("概率")
        ax.text(0, d, f"  {d:.1%}", va="bottom", color=FG, fontsize=13)
        ax.text(.02, .05, f"原始SR={dsr['sr']:.3f} → 基准SR={dsr['sr_benchmark']:.3f}\n"
                          f"多重检验数={dsr['n_trials']}", transform=ax.transAxes,
                color=FG, fontsize=8.5)
        ax.legend(facecolor=BG, labelcolor=FG, fontsize=8, framealpha=.3)
    return _save(fig, "31_deflated_sharpe")


def c32_leak_audit(aud):
    labels = ["随机切分\n(错误做法)", "Purged WF\n(正确做法)", "洗牌标签\n(应≈0.50)", "注入近似标签\n(应显著升高)"]
    vals = [aud.get("random_split_auc") or 0, aud.get("purged_wf_auc") or 0,
            aud.get("shuffled_label_auc") or 0, aud.get("leak_injected_auc") or 0]
    cols = [ACC, ACC3, ACC4, ACC]
    fig, ax = _fig(10, 5.2, "㉜ 泄漏审计对照 —— 错误做法如何凭空造出 AUC")
    b = ax.bar(labels, vals, color=cols)
    for bb, v in zip(b, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                color=FG, fontsize=11, fontweight="bold")
    ax.axhline(.5, color=FG, ls=":", lw=1)
    if aud.get("random_split_auc") and aud.get("purged_wf_auc"):
        gap = aud["random_split_auc"] - aud["purged_wf_auc"]
        ax.annotate(f"虚假高估 +{gap*100:.1f}pp", xy=(0.5, max(vals[:2]) + .03),
                    ha="center", color=ACC4, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 1.08); ax.set_ylabel("AUC")
    ax.tick_params(axis="x", labelsize=9)
    return _save(fig, "32_leak_audit")


def c33_alpha_corr(panel):
    """多标的动量因子相关性 —— 检验因子分散度"""
    fac = {}
    for s, d in panel.items():
        c = d.set_index("date")["close"]
        fac[s] = c.pct_change(20)
    F = pd.DataFrame(fac).dropna()
    cm = F.corr()
    fig, ax = _fig(9, 7.5, "㉝ 20日动量因子横截面相关性")
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index, fontsize=7.5)
    fig.colorbar(im, ax=ax)
    return _save(fig, "33_factor_corr")


REGISTRY = {
    "01 price+MA": c01_price_ma, "02 累计收益": c02_cum_return, "03 回撤": c03_drawdown,
    "04 滚动波动率": c04_rolling_vol, "05 波动率期限": c05_vol_term, "06 收益分布": c06_return_dist,
    "07 Q-Q": c07_qq, "08 自相关": c08_acf, "09 月度热力": c09_monthly_heat,
    "10 年度收益": c10_annual, "11 相关矩阵": c11_corr, "12 滚动相关": c12_roll_corr,
    "13 PCA": c13_pca, "14 收益排名": c14_sector_rank, "15 Beta": c15_beta,
    "16 滚动Beta": c16_rolling_beta, "17 AUC分折": c17_auc_folds, "18 ROC": c18_roc,
    "19 PR": c19_pr, "20 校准": c20_calibration, "21 分数分布": c21_score_dist,
    "22 混淆矩阵": c22_confusion, "23 特征重要性": c23_importance, "24 IC衰减": c24_ic_decay,
    "25 状态迁移": c25_regime_timeline, "26 尾部风险": c26_tail_risk, "27 滚动Sharpe": c27_sharpe_roll,
    "28 净值对比": c28_equity_vs_bench, "29 量价": c29_volume, "30 PBO": c30_overfit_pbo,
    "31 通缩Sharpe": c31_deflated, "32 泄漏审计": c32_leak_audit, "33 因子相关": c33_alpha_corr,
}
