# -*- coding: utf-8 -*-
"""
QuantPilot · 基本面因子测试
=========================================
1) 用 merge_asof 做严格时点对齐（announce_date <= 调仓日）
2) 每个因子单独回测 + IC 检验 + 与量价因子相关性
3) 样本内/样本外分段
"""
from __future__ import annotations
import sys, json, re, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, fundamentals as F, strategies as S
from backtest import Backtest, Costs, benchmark_ew, attribution

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)
SPLIT = "2023-01-01"


def cached_codes() -> list[str]:
    return sorted({m.group(1) for f in (ROOT / ".cache").glob("*.csv")
                   if (m := re.search(r"_cn_(\d{6})$", f.stem))})


def load_panel():
    import hashlib
    cc = ROOT / ".cache"
    newest = int(max((f.stat().st_mtime for f in cc.glob("*.csv")), default=0))
    sig = f"{len(cached_codes())}|2016-01-01|{newest}"
    cp = cc / f"panel_{hashlib.md5(sig.encode()).hexdigest()[:12]}.pkl"
    if cp.exists():
        return pd.read_pickle(cp)
    panel = datafeed.sanitize_panel(datafeed.get_panel({c: "cn" for c in cached_codes()}),
                                    start="2016-01-01")
    pd.to_pickle(panel, cp)
    return panel


def pit_factor_matrices(panel: dict, reb_dates: pd.DatetimeIndex,
                        fp: pd.DataFrame, px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """把报告期因子对齐到调仓日（严格只用 announce <= T 的报告）"""
    cols = [c for c in F.FACTORS if c in fp.columns]
    right = fp.dropna(subset=["announce"]).copy()
    right = right[["code", "announce", "bps", "eps_ttm"] + cols]
    right = right.sort_values("announce").reset_index(drop=True)

    codes = list(panel)
    grid = pd.DataFrame([(d, c) for d in reb_dates for c in codes], columns=["date", "code"])
    grid = grid.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(grid, right, left_on="date", right_on="announce",
                           by="code", direction="backward", tolerance=pd.Timedelta(days=400))
    # 价格对齐（向量化，只取调仓日）
    plong = px.loc[[d for d in reb_dates if d in px.index]].stack().rename("price").reset_index()
    plong.columns = ["date", "code", "price"]
    merged = merged.drop(columns=[c for c in ("price",) if c in merged.columns], errors="ignore")
    merged = merged.merge(plong, on=["date", "code"], how="left")
    merged["f_ey"] = merged["eps_ttm"] / merged["price"]          # 盈利收益率
    merged["f_by"] = merged["bps"] / merged["price"]              # 账面收益率
    merged.loc[merged["price"] <= 0, ["f_ey", "f_by"]] = np.nan

    use = cols + ["f_ey", "f_by"]
    out = {}
    for f in use:
        out[f] = merged.pivot(index="date", columns="code", values=f).reindex(
            index=reb_dates, columns=codes)
    return out


def ic_series(fac: pd.DataFrame, px: pd.DataFrame, horizon: int = 20) -> float:
    """截面 Spearman IC 均值"""
    fwd = (px.shift(-horizon) / px - 1)
    ics = []
    for d in fac.index:
        if d not in fwd.index:
            continue
        a = fac.loc[d]
        b = fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 30:
            continue
        ics.append(a[m].corr(b[m], method="spearman"))
    return float(np.nanmean(ics)) if ics else np.nan


def stats(nav: pd.Series) -> dict:
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    tot = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = float((nav / nav.cummax() - 1).min())
    return {"cagr": float(cagr), "sharpe": float(r.mean() * 252 / vol) if vol > 0 else np.nan,
            "mdd": dd, "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "total": float(tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    print("[1/5] 加载面板与基本面 ...")
    panel = load_panel()
    fp = F.build_factor_panel()
    if fp.empty:
        print("  基本面数据为空，请先运行 backend/fundamentals.py"); return
    print(f"     行情 {len(panel)} 只 | 基本面 {len(fp):,} 行 / {fp['code'].nunique()} 只")

    px = S.price_panel(panel)
    reb = S.rebalance_dates(px.index, "M")
    ind = {c: "未知" for c in panel}
    try:
        import sector
        im = sector.load_industries()
        ind = {c: im.get(c, "未知") for c in panel}
    except Exception:
        pass

    print("[2/5] 时点对齐（announce_date <= 调仓日）...")
    fm = pit_factor_matrices(panel, reb, fp, px)
    cov = {k: float(v.notna().mean().mean()) for k, v in fm.items()}
    print("     覆盖率: " + "  ".join(f"{k.replace('f_','')}={v:.0%}" for k, v in cov.items()))

    print("[3/5] IC 与相关性诊断 ...")
    mom = S.f_momentum(px, 120, 20)
    diag = {}
    for k, mat in fm.items():
        ic = ic_series(mat, px, 20)
        # 与动量因子的截面排名相关
        rr = []
        for d in mat.index:
            if d not in mom.index:
                continue
            fa, mb = mat.loc[d], mom.loc[d]
            m = fa.notna() & mb.notna()
            if m.sum() < 30:
                continue
            rr.append(fa[m].corr(mb[m], method="spearman"))
        diag[k] = {"ic": ic, "corr_mom": float(np.nanmean(rr)) if rr else np.nan}

    print("[4/5] 回测 ...")
    bt = Backtest(panel, capital=1_000_000, costs=Costs(), stop_loss=0.15, dd_brake=None)
    idx = datafeed.load_index("000300")
    s = idx.set_index("date")["close"]
    bnav = s / s.iloc[0]
    results = {}
    for k, mat in fm.items():
        z = S.xs_zscore(S.winsorize_rows(mat))
        w = S.to_weights_buffered(z, reb, top_n=a.top, buffer=0.5,
                                  industry=ind, max_per_ind=3)
        if not len(w):
            continue
        w = S.apply_vol_target(w, px, target_vol=0.15)
        w = S.regime_overlay(w, s)
        r = bt.run(w, max_positions=a.top)
        if "error" in r:
            continue
        nav = r["nav"]
        i_, o_ = stats(nav.loc[:SPLIT]), stats(nav.loc[SPLIT:])
        bt_ = attribution(nav, bnav.reindex(nav.index).ffill())
        results[k] = {"name": F.FACTORS.get(k, k), **{x: r[x] for x in
                                                     ("cagr", "sharpe", "max_drawdown", "calmar",
                                                      "ann_turnover", "total_return")},
                      "beta": bt_.get("beta"), "alpha_ann": bt_.get("alpha_ann"),
                      "ir": bt_.get("information_ratio"), "is": i_, "oos": o_,
                      **diag[k], "_nav": nav}

    print("[5/5] 结果\n")
    print(f"{'因子':<26}{'IC':>7}{'vs动量':>8}{'年化':>8}{'Sharpe':>8}{'回撤':>8}"
          f"{'IS年化':>8}{'OOS年化':>9}{'OOS Sharpe':>11}")
    print("-" * 105)
    for k, r in sorted(results.items(), key=lambda kv: -(kv[1]["sharpe"] or -9)):
        print(f"{r['name']:<26}{(r['ic'] or 0):>7.3f}{(r['corr_mom'] or 0):>8.2f}"
              f"{(r['cagr'] or 0)*100:>7.1f}%{(r['sharpe'] or 0):>8.2f}"
              f"{(r['max_drawdown'] or 0)*100:>7.1f}%"
              f"{(r['is'].get('cagr', 0))*100:>7.1f}%{(r['oos'].get('cagr', 0))*100:>8.1f}%"
              f"{(r['oos'].get('sharpe', 0)):>11.2f}")

    # 基本面 + 动量 组合
    print("\n组合检验（基本面 top2 + 动量）")
    best2 = sorted(results, key=lambda k: -(results[k]["sharpe"] or -9))[:2]
    zs = [S.xs_zscore(S.winsorize_rows(fm[k])) for k in best2] + [S.xs_zscore(S.winsorize_rows(mom))]
    comb = S.composite([fm[best2[0]], fm[best2[1]], mom], [0.4, 0.3, 0.3])
    w = S.to_weights_buffered(S.xs_zscore(S.winsorize_rows(comb)), reb, top_n=a.top,
                              buffer=0.5, industry=ind, max_per_ind=3)
    w = S.regime_overlay(S.apply_vol_target(w, px, target_vol=0.15), s)
    r = bt.run(w, max_positions=a.top)
    nav = r["nav"]
    i_, o_ = stats(nav.loc[:SPLIT]), stats(nav.loc[SPLIT:])
    print(f"  {'基本面+动量 合成':<24}年化{r['cagr']*100:6.1f}% Sharpe{r['sharpe']:5.2f} "
          f"回撤{r['max_drawdown']*100:6.1f}% | IS {i_.get('cagr',0)*100:5.1f}% "
          f"OOS {o_.get('cagr',0)*100:6.1f}% / Sharpe {o_.get('sharpe',0):.2f}")
    results["_combo"] = {"name": "基本面+动量 合成", "cagr": r["cagr"], "sharpe": r["sharpe"],
                         "max_drawdown": r["max_drawdown"], "is": i_, "oos": o_, "_nav": nav}

    if a.save:
        out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "split_date": SPLIT,
               "coverage": cov, "factors": {}}
        for k, r in results.items():
            out["factors"][k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                 for kk, vv in r.items() if not kk.startswith("_")}
        (REP / "fundamentals_result.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        navs = pd.DataFrame({k: v["_nav"] for k, v in results.items() if "_nav" in v})
        navs["基准"] = bnav.reindex(navs.index).ffill()
        navs.to_csv(REP / "fund_nav.csv", encoding="utf-8")
        print(f"\n[保存] fundamentals_result.json + fund_nav.csv")
    return results


if __name__ == "__main__":
    main()
