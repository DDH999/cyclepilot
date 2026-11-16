# -*- coding: utf-8 -*-
"""
退市股 + 时点票池 修复验收（2019-2021 子集）
====================================================
只做验证，不改策略参数（low_vol / top_n=10 / 20 日调仓 保持不变）。
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import backtest_strict as BS
import position_manager as PM
import strategy_lab as SL
import universe_pit as UP

START, END = "2019-01-02", "2021-12-31"
N_SURVIVOR = 300


def pick_subset(basic, index):
    """子集 = 窗口内曾入池的全部退市股 + 随机 300 只存续股。"""
    cal = [pd.Timestamp(d) for d in index]
    pool_any = set().union(*[index[str(d.date())] for d in cal])
    delisted = basic[basic["out_date"].notna()]
    dl_in = [c for c in delisted["code"] if c in pool_any
             and pd.Timestamp(basic.loc[basic["code"] == c, "out_date"].iloc[0])
             <= pd.Timestamp(END)]
    surv = [c for c in basic["code"] if c not in set(delisted["code"])]
    rng = np.random.default_rng(42)
    surv_s = list(rng.choice(sorted(surv), size=min(N_SURVIVOR, len(surv)),
                             replace=False))
    return delisted, sorted(set(dl_in)), sorted(surv_s)


def weights_with_universe(panel, index, factor="low_vol", lookback=20,
                          top_n=10, rebalance=20):
    """因子权重，但每天只在当日票池内排序。"""
    sc = SL.build_scores(panel, factor, lookback)
    rows, idx = [], []
    dates = list(sc.index)
    for i in range(0, len(dates), rebalance):
        d = dates[i]
        elig = index.get(str(pd.Timestamp(d).date()))
        if not elig:
            continue
        row = sc.loc[d].dropna()
        row = row[row.index.isin(elig)]
        if len(row) < top_n:
            continue
        pick = row.nlargest(top_n).index
        w = pd.Series(0.0, index=sc.columns)
        w[pick] = 1.0 / len(pick)
        rows.append(w); idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx)) if rows else \
        pd.DataFrame(columns=sc.columns)


def ann_of(series):
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)


def main():
    print("=== 1. 构建时点票池 ===")
    basic = UP.load_basic()
    cal = UP.trading_days(START, END)
    idx = UP.build_index(START, END, basic=basic, cal=cal)
    sm = UP.summarize(idx)
    print(f"  {sm['n_days']} 天，规模 {sm['min_size']}~{sm['max_size']}，"
          f"平均 {sm['avg_size']:.0f}")

    delisted, dl_codes, surv_codes = pick_subset(basic, idx["index"])
    print(f"  退市股（窗口内曾入池）{len(dl_codes)} 只；存续股抽样 {len(surv_codes)} 只")
    codes = sorted(set(dl_codes) | set(surv_codes))

    print(f"\n=== 2. 拉取 {len(codes)} 只行情（首次较慢，之后走缓存）===")
    got = BS.fetch_panel(codes, "2018-06-01", END, with_flags=True)
    panel = got["panel"]
    print(f"  成功 {got['n_ok']} 只，失败 {got['n_missing']} 只")
    if len(panel) < 30:
        print("样本太少，终止")
        return

    dmap = UP.delist_map(basic)
    dmap = {c: d for c, d in dmap.items() if c in panel}
    print(f"  面板内退市股 {len(dmap)} 只")

    print("\n=== 3. 修复后（时点票池 + 退市核销）===")
    w = weights_with_universe(panel, idx["index"], "low_vol", 20, 10, 20)
    bt = BS.StrictBacktest(panel, capital=1e6, delist_map=dmap)
    res = bt.run(w, max_positions=10)
    print(f"  年化 {res['annual']*100:.2f}%  夏普 {res['sharpe']}  "
          f"回撤 {res['max_drawdown']*100:.2f}%")
    print(f"  退市核销 {res['n_delist_liquidations']} 次")

    # 同池等权基准（同票池、同前复权）
    CL = pd.DataFrame({c: d.set_index("date")["close"]
                       for c, d in panel.items()}).sort_index()
    ew = CL.pct_change(fill_method=None).mean(axis=1).add(1).cumprod().dropna()
    ew = ew.loc[str(res["nav"].index[0].date()):]
    ew_ann = ann_of(ew)
    print(f"  同池等权年化 {ew_ann*100:.2f}%  →  超额 "
          f"{(res['annual']-ew_ann)*100:+.2f}pp")

    print("\n=== 4. 修复前（当前沪深300 成分，无退市股）===")
    old_codes = BS.default_universe(60)
    got_old = BS.fetch_panel(old_codes, START, END, with_flags=True)
    p_old = got_old["panel"]
    w_old = SL.factor_weights(SL.build_scores(p_old, "low_vol", 20),
                              "low_vol", 10, 20)
    res_old = BS.StrictBacktest(p_old, capital=1e6).run(w_old, max_positions=10)
    CLo = pd.DataFrame({c: d.set_index("date")["close"]
                        for c, d in p_old.items()}).sort_index()
    ewo = CLo.pct_change(fill_method=None).mean(axis=1).add(1).cumprod().dropna()
    ewo = ewo.loc[str(res_old["nav"].index[0].date()):]
    ewo_ann = ann_of(ewo)
    print(f"  年化 {res_old['annual']*100:.2f}%  同池等权 {ewo_ann*100:.2f}%  "
          f"超额 {(res_old['annual']-ewo_ann)*100:+.2f}pp")

    print("\n=== 5. 对比 ===")
    print(f"  {'指标':<12}{'修复前':>12}{'修复后':>12}")
    print(f"  {'策略年化':<12}{res_old['annual']*100:>11.2f}%{res['annual']*100:>11.2f}%")
    print(f"  {'同池等权':<12}{ewo_ann*100:>11.2f}%{ew_ann*100:>11.2f}%")
    print(f"  {'超额':<12}{(res_old['annual']-ewo_ann)*100:>+11.2f}pp"
          f"{(res['annual']-ew_ann)*100:>+11.2f}pp")
    print(f"  {'最大回撤':<12}{res_old['max_drawdown']*100:>11.2f}%"
          f"{res['max_drawdown']*100:>11.2f}%")
    print(f"  {'退市核销':<12}{res_old['n_delist_liquidations']:>12}"
          f"{res['n_delist_liquidations']:>12}")

    print("\n=== 6. 退市股入选清单 ===")
    cap0 = 1e6
    liq = {r["code"]: r for r in (res.get("delist_log") or [])}
    held_dl = [c for c in (res.get("contrib_by_code") or {}) if c in dmap]
    if not held_dl:
        print("  本次回测没有退市股进入持仓")
    else:
        print(f"  {'代码':<8}{'退市日':<12}{'入选日':<12}{'退出日':<12}"
              f"{'持仓天数':>8}{'盈亏(pp)':>10}  核销方式")
        tot = 0.0
        for cc in sorted(held_dl):
            pnl = float(res["contrib_by_code"].get(cc, 0.0)) / cap0 * 100
            tot += pnl
            r = liq.get(cc)
            way = (f"{r['reason']} rec={r['recovery']:.3f}" if r
                   else "退市前已清仓")
            print(f"  {cc:<8}{str(dmap[cc].date()):<12}"
                  f"{res['first_in_by_code'].get(cc, '-'):<12}"
                  f"{res['last_out_by_code'].get(cc, '-'):<12}"
                  f"{res['hold_days_by_code'].get(cc, 0):>8}"
                  f"{pnl:>+9.2f}%  {way}")
        print(f"  合计贡献 {tot:+.2f}pp")

    print("\n=== 7. 归因：存续股 vs 退市股 ===")
    dl_set = set(dmap)
    contrib = res["contrib_by_code"]
    tot_c = sum(contrib.values())
    dl_c = sum(v for k, v in contrib.items() if k in dl_set)
    sv_c = tot_c - dl_c
    print(f"  总贡献 {tot_c/cap0*100:+.2f}pp  "
          f"（存续股 {sv_c/cap0*100:+.2f}pp / 退市股 {dl_c/cap0*100:+.2f}pp）")
    print(f"  被选中的退市股 {len([k for k in contrib if k in dl_set])} 只 / "
          f"面板内退市股 {len(dl_set)} 只")


if __name__ == "__main__":
    main()
