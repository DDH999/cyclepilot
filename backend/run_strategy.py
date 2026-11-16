# -*- coding: utf-8 -*-
"""
QuantPilot · 策略回测 runner（竞赛级 v2）
新增：个股止损 · 行业中性化 · 样本内/样本外分段 · 双重基准
"""
from __future__ import annotations
import sys, json, time, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, universe, strategies as S, sector
from backtest import Backtest, Costs, benchmark_ew, attribution

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)
SPLIT = "2023-01-01"          # 样本内 / 样本外 分界


def cached_codes(limit: int | None = None) -> list[str]:
    codes = []
    for f in (ROOT / ".cache").glob("*.csv"):
        m = re.search(r"_cn_(\d{6})$", f.stem)
        if m:
            codes.append(m.group(1))
    codes = sorted(set(codes))
    return codes[:limit] if limit else codes


def load_panel(codes: list[str], start: str = "2016-01-01") -> dict:
    """带面板缓存：首次构建慢，之后秒级"""
    import hashlib
    cc = ROOT / ".cache"
    cs = list(cc.glob("*.csv"))
    newest = int(max((f.stat().st_mtime for f in cs), default=0))
    sig = f"{len(codes)}|{start}|{newest}"
    cp = cc / f"panel_{hashlib.md5(sig.encode()).hexdigest()[:12]}.pkl"
    if cp.exists():
        try:
            print("      使用面板缓存")
            return pd.read_pickle(cp)
        except Exception:
            pass
    panel = datafeed.get_panel({c: "cn" for c in codes})
    panel = datafeed.sanitize_panel(panel, start=start)
    try:
        pd.to_pickle(panel, cp)
        for old in cc.glob("panel_*.pkl"):
            if old != cp:
                old.unlink(missing_ok=True)
    except Exception:
        pass
    return panel


def base_signals(panel: dict) -> dict[str, pd.DataFrame]:
    px = S.price_panel(panel)
    return {
        "动量 120-20": S.f_momentum(px, 120, 20),
        "动量 250-20": S.f_momentum(px, 250, 20),
        "短期反转 5":  S.f_reversal(px, 5),
        "低波动 60":   S.f_lowvol(px, 60),
        "趋势 20/60":  S.f_trend(px, 20, 60),
        "波动率压缩":  S.f_volratio(px, 5, 60),
    }


def build_weights(panel: dict, top_n: int, buffer: float, industry=None,
                  max_per_ind=None) -> dict[str, pd.DataFrame]:
    px = S.price_panel(panel)
    rb = S.rebalance_dates(px.index, "M")
    sigs = base_signals(panel)
    kw = dict(top_n=top_n, buffer=buffer, industry=industry, max_per_ind=max_per_ind)
    out = {}
    for name, sig in sigs.items():
        z = S.xs_zscore(S.winsorize_rows(sig))
        w = S.to_weights_buffered(z, rb, **kw)
        if len(w):
            out[name] = w
    comp = S.composite([sigs["动量 120-20"], sigs["低波动 60"], sigs["波动率压缩"]],
                       [0.5, 0.3, 0.2])
    w = S.to_weights_buffered(comp, rb, **kw)
    if len(w):
        out["多因子合成"] = w
    return out


def variants(weights: dict[str, pd.DataFrame], px: pd.DataFrame, vt: float | None,
             bench_close: pd.Series | None) -> dict[str, pd.DataFrame]:
    out = {}
    for name, w in weights.items():
        out[name] = w
        v = S.apply_vol_target(w, px, target_vol=vt) if vt else w.copy()
        if bench_close is not None:
            out[name + " +风控"] = S.regime_overlay(v, bench_close)
        elif vt:
            out[name + " +VT"] = v
    return out


def period_stats(nav: pd.Series) -> dict:
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    tot = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sh = (r.mean() * 252) / vol if vol > 0 else np.nan
    dd = float((nav / nav.cummax() - 1).min())
    return {"total": float(tot), "cagr": float(cagr), "sharpe": float(sh),
            "mdd": dd, "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan}


def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--buffer", type=float, default=0.5)
    ap.add_argument("--vt", type=float, default=0.15)
    ap.add_argument("--stop-loss", type=float, default=0.15, help="个股止损，0=关闭")
    ap.add_argument("--max-per-ind", type=int, default=3, help="单行业最大持仓数，0=不限制")
    ap.add_argument("--dd-brake", type=float, default=0.15, help="组合回撤制动阈值，0=关闭")
    ap.add_argument("--dd-scale", type=float, default=0.30, help="制动后仓位缩放")
    ap.add_argument("--impact-k", type=float, default=0.002, help="冲击成本系数")
    ap.add_argument("--delist-recovery", type=float, default=0.30, help="退市残值回收率")
    ap.add_argument("--no-delisted", action="store_true", help="排除已退市股票（会引入幸存者偏差）")
    ap.add_argument("--no-regime", action="store_true")
    ap.add_argument("--bench", default="000300")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    codes = cached_codes(a.limit or None)
    panel = load_panel(codes, a.start)
    # 区分"指数成分（存活）"与"退市股"
    uni_set = set()
    for f in (ROOT / ".cache").glob("universe_*.csv"):
        try:
            uni_set |= set(pd.read_csv(f, dtype={"code": str})["code"])
        except Exception:
            pass
    n_survivor = sum(1 for c in panel if c in uni_set)
    n_delist = len(panel) - n_survivor
    print(f"[1/6] 票池 {len(panel)} 只 = 指数成分 {n_survivor} + 退市 {n_delist} | "
          f"{min(d['date'].min() for d in panel.values()).date()} ~ "
          f"{max(d['date'].max() for d in panel.values()).date()}")

    # 行业
    ind = sector.load_industries()
    ind = {c: ind.get(c, "未知") for c in panel}
    n_ind = len({v for v in ind.values() if v != "未知"})
    print(f"[2/6] 行业分类 {sum(1 for v in ind.values() if v!='未知')}/{len(ind)} 只, {n_ind} 个行业"
          f" | 单行业上限={a.max_per_ind or 'off'}")

    # 基准
    idx_nav = None
    bench_close = None
    try:
        idx = datafeed.load_index(a.bench)
        s = idx.set_index("date")["close"]
        idx_nav = s / s.iloc[0]
        bench_close = s
        _s = s[s.index >= pd.Timestamp(a.start)]
        print(f"[3/6] 基准 {a.bench}: {len(idx)} 行, 回测区间收益 "
              f"{(_s.iloc[-1]/_s.iloc[0]-1)*100:.1f}%")
    except Exception as e:
        print(f"[3/6] 指数基准失败({e})，用等权替代")

    print(f"[4/6] 生成策略 (Top{a.top} 缓冲{a.buffer} VT{a.vt} 止损{a.stop_loss or 'off'} "
          f"回撤制动{a.dd_brake or 'off'})")
    px = S.price_panel(panel)
    w0 = build_weights(panel, a.top, a.buffer, ind, a.max_per_ind or None)
    wdict = variants(w0, px, a.vt or None, None if a.no_regime else bench_close)
    print(f"      {len(wdict)} 个策略/变体")

    print("[5/6] 回测中 ...")
    costs = Costs()
    ew_nav = benchmark_ew(panel); ew_nav = ew_nav / ew_nav.iloc[0]
    results = {}
    # 只构建一次回测矩阵，14 个策略复用（避免重复初始化）
    bt = Backtest(panel, capital=a.capital, costs=costs,
                  stop_loss=a.stop_loss or None,
                  dd_brake=a.dd_brake or None,
                  dd_brake_scale=a.dd_scale,
                  impact_k=a.impact_k,
                  delist_recovery=a.delist_recovery)
    for name, w in wdict.items():
        res = bt.run(w, max_positions=a.top)
        if "error" in res:
            print(f"  {name:22} 跳过 {res['error']}")
            continue
        b = (idx_nav.reindex(res["nav"].index).ffill() if idx_nav is not None
             else ew_nav.reindex(res["nav"].index).ffill())
        at = attribution(res["nav"], b)
        r = {k: v for k, v in res.items() if k not in ("nav", "returns", "drawdown", "turnover")}
        r.update(at)
        r["_nav"] = res["nav"]
        r["_is"] = period_stats(res["nav"].loc[:SPLIT])
        r["_oos"] = period_stats(res["nav"].loc[SPLIT:])
        results[name] = r
        print(f"  {name:22} 年化{fmt_pct(res['cagr']):>8} Sharpe{res['sharpe']:>6.2f} "
              f"回撤{fmt_pct(res['max_drawdown']):>8} Calmar{res['calmar']:>5.2f}")

    # ── 汇总
    bnav = (idx_nav if idx_nav is not None else ew_nav)
    bnav = bnav.reindex(next(iter(results.values()))["_nav"].index).ffill()
    bnav = bnav / bnav.iloc[0]          # 归一到回测起点
    bench_ret = float(bnav.iloc[-1] - 1)
    print(f"\n[6/6] 全样本（基准 {a.bench} {fmt_pct(bench_ret)}，"
          f"等权票池 {fmt_pct(float(ew_nav.iloc[-1]-1))}）")
    hdr = (f"{'策略':<24}{'年化':>8}{'Sharpe':>8}{'回撤':>8}{'Calmar':>8}{'Beta':>7}"
           f"{'超额':>9}{'IR':>7}")
    print(hdr); print("-" * len(hdr))
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("sharpe") or -9)):
        print(f"{name:<24}{fmt_pct(r['cagr']):>8}{r['sharpe']:>8.2f}{fmt_pct(r['max_drawdown']):>8}"
              f"{r['calmar']:>8.2f}{r.get('beta',0):>7.2f}{fmt_pct(r.get('alpha_ann')):>9}"
              f"{r.get('information_ratio',0):>7.2f}")

    # ── 样本内 / 样本外
    print(f"\n样本内({a.start[:4]}-2022) vs 样本外(2023-2026) —— 真实检验")
    h2 = f"{'策略':<24}{'IS年化':>9}{'IS Sharpe':>11}{'IS回撤':>9}{'OOS年化':>9}{'OOS Sharpe':>12}{'OOS回撤':>9}{'衰减':>8}"
    print(h2); print("-" * len(h2))
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("sharpe") or -9)):
        i, o = r.get("_is", {}), r.get("_oos", {})
        dec = (o.get("sharpe", 0) - i.get("sharpe", 0)) if i and o else None
        print(f"{name:<24}{fmt_pct(i.get('cagr')):>9}{i.get('sharpe',0):>11.2f}"
              f"{fmt_pct(i.get('mdd')):>9}{fmt_pct(o.get('cagr')):>9}{o.get('sharpe',0):>12.2f}"
              f"{fmt_pct(o.get('mdd')):>9}{(f'{dec:+.2f}' if dec is not None else '—'):>8}")

    if a.save and results:
        out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "split_date": SPLIT,
               "universe_n": len(panel), "n_survivor": n_survivor, "n_delisted": n_delist,
               "top_n": a.top, "buffer": a.buffer,
               "vol_target": a.vt, "stop_loss": a.stop_loss,
               "dd_brake": a.dd_brake, "dd_brake_scale": a.dd_scale,
               "impact_k": a.impact_k, "delist_recovery": a.delist_recovery,
               "max_per_industry": a.max_per_ind, "n_industries": n_ind,
               "regime_overlay": not a.no_regime, "benchmark": a.bench,
               "benchmark_total": bench_ret,
               "equalweight_total": float(ew_nav.iloc[-1] - 1),
               "costs": costs.describe(),
               "strategies": {n: {**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in r.items() if not k.startswith("_")},
                                  "is": r.get("_is", {}), "oos": r.get("_oos", {})}
                              for n, r in results.items()}}
        (REP / "backtest_results.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        navs = pd.DataFrame({n: r["_nav"] for n, r in results.items()})
        navs[f"基准_{a.bench}"] = bnav.reindex(navs.index).ffill()
        navs["等权票池"] = ew_nav.reindex(navs.index).ffill()
        navs.to_csv(REP / "nav_curves.csv", encoding="utf-8")
        print(f"\n[保存] backtest_results.json + nav_curves.csv ({len(navs)} 行)")
    return results


if __name__ == "__main__":
    main()
