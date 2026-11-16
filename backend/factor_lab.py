# -*- coding: utf-8 -*-
"""
QuantPilot · 因子实验室（严格版）
=========================================================
对每个因子做完整诊断，并做**多重检验校正**：
  IC 均值 / ICIR / Newey-West t 值 / FDR 校正 p 值
  覆盖率 / 换手 / 十分组单调性 / 多空价差
  正交化后 IC（剔除动量偏好）/ 分年度稳定性
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, strategies as S, factors as FX

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)
CHARTS = REP / "charts"
CHARTS.mkdir(exist_ok=True)

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
    p = CHARTS / name
    fig.savefig(p, dpi=115, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


# ───────────────────────────── 统计工具
def newey_west_t(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """均值的新息调整 t 值（解决 IC 序列自相关）"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return np.nan, np.nan
    m = x.mean()
    e = x - m
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
        lags = max(1, min(lags, n // 4))
    v = (e @ e) / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        v += 2 * w * (e[l:] @ e[:-l]) / n
    if v <= 0:
        return np.nan, np.nan
    t = m / np.sqrt(v / n)
    return float(m), float(t)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正"""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if not len(idx):
        return out
    sub = p[idx]
    order = np.argsort(sub)
    m = len(sub)
    ranked = sub[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[idx] = adj
    return out


def load_panel():
    import hashlib
    cc = ROOT / ".cache"
    codes = sorted({m.group(1) for f in cc.glob("*.csv")
                    if (m := re.search(r"_cn_(\d{6})$", f.stem))})
    newest = int(max((f.stat().st_mtime for f in cc.glob("*.csv")), default=0))
    sig = f"{len(codes)}|2016-01-01|{newest}"
    cp = cc / f"panel_{hashlib.md5(sig.encode()).hexdigest()[:12]}.pkl"
    if cp.exists():
        return pd.read_pickle(cp)
    panel = datafeed.sanitize_panel(datafeed.get_panel({c: "cn" for c in codes}),
                                    start="2016-01-01")
    pd.to_pickle(panel, cp)
    return panel


# ───────────────────────────── 单因子诊断
def diagnose(name, mat, fwd, mom_z, desc, family):
    dates = [d for d in mat.index if d in fwd.index]
    ics, resid_ics = [], []
    grp_rets = [[] for _ in range(10)]
    for d in dates:
        f = mat.loc[d]
        y = fwd.loc[d]
        m = f.notna() & y.notna()
        if m.sum() < 40:
            continue
        ic = f[m].corr(y[m], method="spearman")
        if np.isfinite(ic):
            ics.append(ic)
        # 正交化：剔除动量偏好
        if d in mom_z.index:
            mm = mom_z.loc[d]
            mask = m & mm.notna()
            if mask.sum() >= 40:
                a = stats.rankdata(f[mask]); b = stats.rankdata(mm[mask])
                a = (a - a.mean()) / (a.std() + 1e-12)
                b = (b - b.mean()) / (b.std() + 1e-12)
                beta = (a * b).mean()
                res = a - beta * b
                yv = y[mask].values
                ric = stats.spearmanr(res, yv).statistic
                if np.isfinite(ric):
                    resid_ics.append(ric)
        # 十分组
        try:
            q = pd.qcut(f[m].rank(method="first"), 10, labels=False)
            for g in range(10):
                sel = (q == g).values
                if sel.sum():
                    grp_rets[g].append(y[m].values[sel].mean())
        except Exception:
            pass

    ics = np.array(ics)
    if len(ics) < 12:
        return None
    mean_ic, t = newey_west_t(ics)
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t))) if np.isfinite(t) else np.nan

    gm = np.array([np.nanmean(g) if g else np.nan for g in grp_rets])
    spread = float(gm[-1] - gm[0]) if np.isfinite(gm[-1]) and np.isfinite(gm[0]) else np.nan
    mono = float(stats.spearmanr(np.arange(10), gm).statistic) if np.isfinite(gm).sum() >= 8 else np.nan

    # 换手：相邻调仓日 top30 的重合度
    tops = []
    for d in dates:
        s = mat.loc[d].dropna()
        if len(s) >= 40:
            tops.append(set(s.nlargest(30).index))
    turn = np.nan
    if len(tops) > 2:
        turn = float(np.mean([1 - len(a & b) / max(len(a | b), 1)
                              for a, b in zip(tops[:-1], tops[1:])]))

    return {"name": name, "desc": desc, "family": family,
            "n_dates": int(len(ics)), "ic": float(mean_ic), "icir": icir,
            "t": float(t), "p": float(p),
            "ic_pos_ratio": float((ics > 0).mean()),
            "resid_ic": float(np.mean(resid_ics)) if resid_ics else np.nan,
            "spread": spread, "mono": mono, "turnover": turn,
            "coverage": float(mat.notna().mean().mean()),
            "ic_series": [float(x) for x in ics],
            "ic_dates": [str(d.date()) for d in dates[:len(ics)]],
            "decile": [float(x) if np.isfinite(x) else None for x in gm]}


def main():
    t0 = time.time()
    print("[1/4] 加载数据 ...")
    panel = load_panel()
    px = FX._px(panel)
    reb = S.rebalance_dates(px.index, "M")
    ind = {}
    try:
        import sector
        im = sector.load_industries()
        ind = {c: im.get(c, "未知") for c in panel}
    except Exception:
        ind = {c: "未知" for c in panel}
    print(f"      {len(panel)} 只, {len(reb)} 个调仓日")

    print("[2/4] 构建因子 ...")
    fx = FX.all_factors(panel, ind, reb)
    print(f"      {len(fx)} 个因子")

    fwd = (px.shift(-20) / px - 1).reindex(reb)
    mom = S.f_momentum(px, 120, 20).reindex(reb)
    mom_z = S.xs_zscore(S.winsorize_rows(mom))

    print("[3/4] 诊断（IC / NW-t / FDR / 正交化 / 单调性）...")
    rows = []
    for name, mat in fx.items():
        mat = mat.reindex(reb)
        desc, fam = FX.PRICE_FACTORS.get(name) or FX.FUND_META.get(name) or (name, "其他")
        r = diagnose(name, mat, fwd, mom_z, desc, fam)
        if r:
            rows.append(r)
    if not rows:
        print("无有效因子"); return
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("ic_series", "ic_dates", "decile")}
                       for r in rows])
    df["p_fdr"] = bh_fdr(df["p"].values)
    df["pass_fdr"] = df["p_fdr"] < 0.05
    df = df.sort_values("ic", key=lambda s: -s.abs())
    print(f"      {len(df)} 个因子完成诊断, FDR<0.05 通过 {int(df['pass_fdr'].sum())} 个\n")

    print(f"{'因子':<18}{'族':<10}{'IC':>7}{'ICIR':>7}{'NW-t':>7}{'FDR-p':>8}"
          f"{'正交IC':>8}{'多空':>8}{'单调':>6}{'换手':>6}")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['name']:<18}{r['family']:<10}{r['ic']:>7.4f}{r['icir']:>7.2f}{r['t']:>7.2f}"
              f"{r['p_fdr']:>8.3f}{(r['resid_ic'] if np.isfinite(r['resid_ic']) else 0):>8.4f}"
              f"{(r['spread'] or 0)*100:>7.2f}%{(r['mono'] or 0):>6.2f}{(r['turnover'] or 0):>6.2f}")

    print("\n[4/4] 图表 ...")
    # 1 IC 排序（FDR 通过高亮）
    d2 = df.sort_values("ic")
    fig, ax = _fig(11, max(5, len(d2) * .3), "㊼ 因子 IC 排序（绿色=FDR<0.05 通过）")
    ax.barh(d2["name"], d2["ic"], color=[ACC3 if p else ACC2 for p in d2["pass_fdr"]])
    ax.axvline(0, color=FG, lw=.8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("IC 均值（20日）")
    _save(fig, "50_factor_ic.png")

    # 2 正交化前后对比
    fig, ax = _fig(9.5, 6, "㊽ 原始 IC vs 剔除动量后 IC")
    ax.scatter(df["ic"], df["resid_ic"].fillna(0), s=70, color=ACC2, alpha=.8,
               edgecolor=FG, linewidth=.4)
    for _, r in df.iterrows():
        if abs(r["ic"]) > .02:
            ax.annotate(r["name"], (r["ic"], r["resid_ic"] if np.isfinite(r["resid_ic"]) else 0),
                        fontsize=6.5, color=FG, xytext=(3, 3), textcoords="offset points")
    lim = [df["ic"].min() - .01, df["ic"].max() + .01]
    ax.plot(lim, lim, color=FG, ls="--", lw=1, label="无变化线")
    ax.set_xlabel("原始 IC"); ax.set_ylabel("剔除动量后 IC")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    _save(fig, "51_factor_resid_ic.png")

    # 3 分年度 IC（取 |IC| 前 8）
    top = df.head(8)["name"].tolist()
    yrs = sorted({int(d[:4]) for r in rows for d in r["ic_dates"]})
    hm = np.full((len(top), len(yrs)), np.nan)
    for i, nm in enumerate(top):
        r = next(x for x in rows if x["name"] == nm)
        s = pd.Series(r["ic_series"], index=pd.to_datetime(r["ic_dates"]))
        for j, y in enumerate(yrs):
            v = s[s.index.year == y]
            if len(v):
                hm[i, j] = v.mean()
    fig, ax = _fig(11, max(4, len(top) * .45), "㊾ 分年度 IC 稳定性（前 8 因子）")
    vmax = np.nanmax(np.abs(hm))
    im = ax.imshow(hm, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs, fontsize=8)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
    for i in range(len(top)):
        for j in range(len(yrs)):
            if np.isfinite(hm[i, j]):
                ax.text(j, i, f"{hm[i,j]:.3f}", ha="center", va="center", fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax)
    _save(fig, "52_factor_ic_year.png")

    # 4 IC vs 换手（性价比）
    fig, ax = _fig(9.5, 6, "㊿ 因子性价比：IC 越高、换手越低越好")
    ax.scatter(df["turnover"], df["ic"].abs(), s=70, color=ACC4, alpha=.8,
               edgecolor=FG, linewidth=.4)
    for _, r in df.iterrows():
        if abs(r["ic"]) > .02:
            ax.annotate(r["name"], (r["turnover"], abs(r["ic"])), fontsize=6.5, color=FG,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("调仓换手率"); ax.set_ylabel("|IC|")
    _save(fig, "53_factor_cost_eff.png")

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_factors": int(len(df)), "n_pass_fdr": int(df["pass_fdr"].sum()),
           "n_dates": int(df["n_dates"].max()),
           "elapsed": round(time.time() - t0, 1),
           "factors": [{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                        for k, v in r.items() if k not in ("ic_series", "ic_dates")}
                       for r in rows]}
    (REP / "factor_lab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"保存 reports/factor_lab.json  ({len(df)} 因子, FDR 通过 {int(df['pass_fdr'].sum())})")


if __name__ == "__main__":
    main()
