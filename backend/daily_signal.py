# -*- coding: utf-8 -*-
"""
QuantPilot · 每日目标持仓生成器
====================================================================
把回测里的「目标权重」翻译成「今天该买的股票、股数、金额」。

用法:
    python backend/daily_signal.py --capital 100000
    python backend/daily_signal.py --capital 100000 --holdings holdings.csv
    python backend/daily_signal.py --capital 100000 --dry-run     # 只看不写文件

输出:
    reports/daily_signal_YYYY-MM-DD.json   完整信号
    reports/daily_signal_YYYY-MM-DD.csv    下单清单

重要约定:
  * 信号基于「最近一个交易日的收盘价」计算，应在 **下一个交易日** 执行
  * 每月最后一个交易日调仓（策略本身就是月频）
  * 权重之和 = 目标股票仓位，其余为现金
  * 股数按 100 股整数手向下取整；不足一手的标的不买
====================================================================
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, re, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed, strategies as S, factors as FX

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)
HOLD_DEFAULT = ROOT / "holdings.csv"

# 成交成本（用于估算，实际以券商为准）
COMMISSION, MIN_COMM, STAMP, TRANSFER = 0.00025, 5.0, 0.0005, 0.00001


def load_panel():
    import hashlib
    cc = ROOT / ".cache"
    codes = sorted({m.group(1) for f in cc.glob("*.csv")
                    if (m := re.search(r"_cn_(\d{6})$", f.stem))})
    if not codes:
        raise SystemExit("没有行情缓存，请先跑 python backend/universe.py csi800")
    newest = int(max((f.stat().st_mtime for f in cc.glob("*.csv")), default=0))
    sig = f"{len(codes)}|2016-01-01|{newest}"
    cp = cc / f"panel_{hashlib.md5(sig.encode()).hexdigest()[:12]}.pkl"
    if cp.exists():
        return pd.read_pickle(cp)
    print("  构建面板缓存（首次约 35 秒）...")
    panel = datafeed.sanitize_panel(datafeed.get_panel({c: "cn" for c in codes}),
                                    start="2016-01-01")
    pd.to_pickle(panel, cp)
    return panel


def build_target(panel, ind, reb, px, top_n=30, buffer=0.5, max_per_ind=3,
                 vt=0.15, regime=True):
    """返回 (目标权重 Series, 调仓日, 诊断信息)"""
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

    z = S.xs_zscore(S.winsorize_rows(S.composite(parts, [1.0] * len(parts))))
    w0 = S.to_weights_buffered(z, reb, top_n=top_n, buffer=buffer,
                               industry=ind, max_per_ind=max_per_ind)
    if not len(w0):
        raise SystemExit("无法生成权重：候选股不足")
    bench = datafeed.load_index("000300").set_index("date")["close"]
    w = S.apply_vol_target(w0, px, target_vol=vt) if vt else w0.copy()
    if regime:
        w = S.regime_overlay(w, bench)

    last_reb = w.index[-1]
    tgt = w.loc[last_reb]
    tgt = tgt[tgt > 0].sort_values(ascending=False)

    ma200 = bench.rolling(200).mean()
    on = bool(bench.iloc[-1] > ma200.iloc[-1]) if len(bench) > 200 else True
    return tgt, last_reb, {"gross": float(tgt.sum()), "regime_on": on,
                           "bench_last": float(bench.iloc[-1]),
                           "bench_ma200": float(ma200.iloc[-1]),
                           "n_candidates": int(z.loc[last_reb].notna().sum())}


def load_holdings(path: Path) -> dict[str, float]:
    """读现有持仓：code,shares（也接受 code,仓位金额）"""
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype={"code": str})
    except Exception as e:
        print(f"  ⚠ 持仓文件读取失败（{e}），按空仓处理")
        return {}
    if "code" not in df.columns:
        print("  ⚠ 持仓文件缺 code 列，按空仓处理")
        return {}
    col = "shares" if "shares" in df.columns else None
    if col is None:
        print("  ⚠ 持仓文件缺 shares 列，按空仓处理")
        return {}
    df["code"] = df["code"].astype(str).str.zfill(6)
    return {str(r["code"]): float(r[col]) for _, r in df.iterrows() if pd.notna(r[col])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100000, help="总资金（含现金）")
    ap.add_argument("--holdings", default=str(HOLD_DEFAULT), help="现有持仓 CSV")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--vt", type=float, default=0.15)
    ap.add_argument("--max-per-ind", type=int, default=3)
    ap.add_argument("--no-regime", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] 加载数据 (资金 {a.capital:,.0f} 元) ...")
    panel = load_panel()
    px = FX._px(panel)
    asof = px.index[-1]
    print(f"      {len(panel)} 只 | 最新交易日 {asof.date()}")

    ind = {}
    try:
        import sector
        im = sector.load_industries()
        ind = {c: im.get(c, "未知") for c in panel}
    except Exception:
        ind = {c: "未知" for c in panel}

    print("[2/4] 计算目标持仓 ...")
    reb = S.rebalance_dates(px.index, "M")
    tgt, last_reb, diag = build_target(panel, ind, reb, px, a.top, 0.5,
                                       a.max_per_ind, a.vt, not a.no_regime)
    nxt = [d for d in reb if d > asof]
    print(f"      调仓日 {last_reb.date()} | 下次调仓 {nxt[0].date() if nxt else '下月末'}")
    print(f"      候选 {diag['n_candidates']} 只 → 持有 {len(tgt)} 只 | "
          f"目标仓位 {diag['gross']*100:.1f}% | 现金 {(1-diag['gross'])*100:.1f}%")
    print(f"      市场状态 {'✅ 在 200 日线上（满仓档）' if diag['regime_on'] else '⚠ 跌破 200 日线（降仓档）'}")

    # 目标股数
    price = px.iloc[-1]
    gross = max(diag["gross"], 1e-9)
    # 最小可行资金：保证每只都能买满一手
    lot_cost = {c: 100 * float(price.get(c, np.nan)) for c in tgt.index}
    lot_cost = {c: v for c, v in lot_cost.items() if np.isfinite(v) and v > 0}
    min_cap = max((v * len(tgt) / gross for v in lot_cost.values()), default=0)

    rows, skipped = [], []
    for c, w in tgt.items():
        p = float(price.get(c, np.nan))
        if not np.isfinite(p) or p <= 0:
            continue
        lots = int(np.floor(w * a.capital / p / 100)) * 100
        if lots < 100:
            skipped.append({"code": c, "weight": float(w), "price": p,
                            "need": round(100 * p, 0),
                            "reason": "一手都不够"})
            continue
        rows.append({"code": c, "weight": float(w), "price": p,
                     "target_shares": lots, "target_value": round(lots * p, 2)})
    tg = pd.DataFrame(rows)

    print()
    if a.capital < min_cap:
        print(f"  ⚠ 资金不足：{a.capital:,.0f} 元 / 最小可行 {min_cap:,.0f} 元")
        print(f"    持仓将由 30 只降为 {len(tg)} 只，与回测口径不一致（回测是 30 只等权）")
        if skipped:
            print(f"    跳过 {len(skipped)} 只（每只一手需 "
                  f"{min(x['need'] for x in skipped):,.0f}~{max(x['need'] for x in skipped):,.0f} 元）")
    if tg.empty:
        raise SystemExit(f"\n目标持仓全为空：资金 {a.capital:,.0f} 元不足，"
                         f"建议至少 {min_cap:,.0f} 元（30 只 × 一手）或减少 --top")
    eff = float((tg["target_value"].sum()) / a.capital)
    print(f"      实际持仓 {len(tg)} 只 / 建仓金额 {tg['target_value'].sum():,.0f} 元 "
          f"/ 实际仓位 {eff*100:.1f}%（目标 {gross*100:.1f}%）")

    # 差额单
    hold = load_holdings(Path(a.holdings))
    print(f"[3/4] 对账（现有持仓 {len(hold)} 只）...")
    cur = {c: float(hold.get(c, 0)) for c in tg["code"]}
    for c in hold:
        if c not in cur:
            cur[c] = float(hold[c])
    orders = []
    allc = set(tg["code"]) | {c for c, q in cur.items() if q > 0}
    tgt_map = dict(zip(tg["code"], tg["target_shares"]))
    for c in sorted(allc):
        t = float(tgt_map.get(c, 0))
        q = float(cur.get(c, 0))
        d = t - q
        if abs(d) < 100 and not (t == 0 and q > 0):
            continue
        p = float(price.get(c, np.nan))
        if not np.isfinite(p) or p <= 0:
            continue
        side = "BUY" if d > 0 else "SELL"
        val = abs(d) * p
        fee = max(val * COMMISSION, MIN_COMM) + val * TRANSFER
        if side == "SELL":
            fee += val * STAMP
        orders.append({"code": c, "side": side, "shares": int(abs(d)),
                       "price": p, "value": round(val, 2), "fee_est": round(fee, 2),
                       "cur": int(q), "target": int(t),
                       "note": "清仓" if t == 0 else ("新建" if q == 0 else "调整")})
    od = pd.DataFrame(orders)
    buys = od[od["side"] == "BUY"] if len(od) else od
    sells = od[od["side"] == "SELL"] if len(od) else od

    print(f"[4/4] 下单清单")
    print(f"      买入 {len(buys)} 笔 / {buys['value'].sum() if len(buys) else 0:,.0f} 元")
    print(f"      卖出 {len(sells)} 笔 / {sells['value'].sum() if len(sells) else 0:,.0f} 元")
    if len(od):
        print(f"      预估费用 {od['fee_est'].sum():,.0f} 元")
    print()
    if len(od):
        print(f"  {'代码':<8}{'方向':<6}{'股数':>8}{'价格':>9}{'金额':>11}  备注")
        print("  " + "-" * 58)
        for _, r in od.sort_values(["side", "value"], ascending=[True, False]).iterrows():
            print(f"  {r['code']:<8}{r['side']:<6}{r['shares']:>8,}{r['price']:>9.2f}"
                  f"{r['value']:>11,.0f}  {r['note']}")

    # 落盘
    sig = {"as_of": str(asof.date()), "last_rebalance": str(last_reb.date()),
           "next_rebalance": str(nxt[0].date()) if nxt else None,
           "capital": a.capital, "gross_target": diag["gross"],
           "cash_target": 1 - diag["gross"], "regime_on": diag["regime_on"],
           "n_target": len(tg), "n_orders": len(od),
           "buy_value": float(buys["value"].sum()) if len(buys) else 0,
           "sell_value": float(sells["value"].sum()) if len(sells) else 0,
           "fee_est": float(od["fee_est"].sum()) if len(od) else 0,
           "targets": tg.to_dict("records"),
           "orders": od.to_dict("records") if len(od) else []}
    if a.dry_run:
        print("\n[dry-run] 未写文件")
    else:
        (REP / f"daily_signal_{asof.date()}.json").write_text(
            json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(od):
            od.to_csv(REP / f"daily_signal_{asof.date()}.csv", index=False, encoding="utf-8")
        if not HOLD_DEFAULT.exists():
            pd.DataFrame({"code": [], "shares": []}).to_csv(HOLD_DEFAULT, index=False,
                                                            encoding="utf-8")
            print(f"\n  ℹ 已生成持仓模板 {HOLD_DEFAULT.name}（填上你现在持有的 code/shares 再跑一次）")
        print(f"\n  保存 reports/daily_signal_{asof.date()}.json")
    print(f"\n  ⚠ 信号基于 {asof.date()} 收盘价，请在下一个交易日执行")
    print(f"  耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
