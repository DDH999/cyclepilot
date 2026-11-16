# -*- coding: utf-8 -*-
"""
因子有效性最终验收（M53）
====================================================
只跑一次，出最终结论。包含：
1. 区间修正：热身数据（2018-06 起）只用于因子计算，**不进净值区间**
2. 三基准对照：策略 / 全市场等权 / 当前成分股等权
3. 三方归因：退市股 / 被调出指数股 / 仍在指数股，各解释多少
4. 失败样本分析：为什么不完整、是否非随机

不改因子定义、top_n、调仓频率。
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import backtest_strict as BS
import strategy_lab as SL
import universe_pit as UP

START, END = "2019-01-02", "2021-12-31"
WARMUP = "2018-06-01"
N_SAMPLE = 800


def ann_of(s: pd.Series):
    s = pd.Series(s).dropna()
    if len(s) < 2:
        return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)


def ew_nav(panel: dict, start=START, end=END) -> pd.Series:
    CL = pd.DataFrame({c: d.set_index("date")["close"]
                       for c, d in panel.items()}).sort_index()
    CL = CL.loc[start:end]
    return CL.pct_change(fill_method=None).mean(axis=1, skipna=True) \
        .add(1).cumprod().dropna()


def weights_pit(panel: dict, index: dict, factor="low_vol", lookback=20,
                top_n=10, rebalance=20, start=START):
    sc = SL.build_scores(panel, factor, lookback)
    rows, idx = [], []
    dates = [d for d in sc.index if d >= pd.Timestamp(start)]
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
    return (pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
            if rows else pd.DataFrame(columns=sc.columns))


def main():
    rep = []
    def W(s=""):
        print(s)
        rep.append(s)

    # ── 票池
    basic = UP.load_basic()
    cal = UP.trading_days(START, END)
    uidx = UP.build_index(START, END, basic=basic, cal=cal)
    index = uidx["index"]
    W(f"# 因子有效性最终验收\n")
    W(f"- 区间：{START} ~ {END}")
    W(f"- 时点票池：{uidx['n_codes']} 只，逐日规模 "
      f"{min(len(v) for v in index.values())}~{max(len(v) for v in index.values())}")

    # ── 抽样：全部窗口内退市股 + 随机 N
    delisted = basic[basic["out_date"].notna()].copy()
    pool_any = set().union(*index.values())
    dl_in = [c for c in delisted["code"] if c in pool_any
             and delisted.loc[delisted["code"] == c, "out_date"].iloc[0]
             <= pd.Timestamp(END)]
    surv = [c for c in basic["code"] if c not in set(delisted["code"])]
    rng = np.random.default_rng(42)
    surv_s = list(rng.choice(sorted(surv), size=N_SAMPLE, replace=False))
    codes = sorted(set(dl_in) | set(surv_s))
    W(f"- 抽样：退市股 {len(dl_in)} + 存续股 {len(surv_s)} = {len(codes)} 只")

    # ── 取数（含热身）
    got = BS.fetch_panel(codes, WARMUP, END, with_flags=True)
    panel = got["panel"]
    W(f"- 取数成功 {got['n_ok']} 只，失败 {got['n_missing']} 只，"
      f"短历史 {got.get('n_short', 0)} 只")

    # ── 失败样本分析
    fail = set(got["missing"])
    W("\n## 1. 失败样本分析\n")
    fb = basic[basic["code"].isin(fail)]
    n_fail = len(fb)
    if n_fail:
        late = int((fb["ipo_date"] > pd.Timestamp(END)).sum())
        dele = int(fb["out_date"].notna().sum())
        W(f"| 特征 | 只数 | 占比 |")
        W("|---|---|---|")
        W(f"| 失败总数 | {n_fail} | 100% |")
        W(f"| 上市晚于 {END}（窗口内无数据） | {late} | {late/n_fail*100:.0f}% |")
        W(f"| 已退市 | {dele} | {dele/n_fail*100:.0f}% |")
        W(f"| 其他（数据源缺失/停牌过长） | {n_fail-late-dele} | "
          f"{(n_fail-late-dele)/n_fail*100:.0f}% |")
        W(f"\n失败率 {n_fail/len(codes)*100:.1f}%；"
          f"其中 {late/n_fail*100:.0f}% 是上市晚于窗口的次新股——"
          f"属正常剔除，非数据缺失。")

    # ── 区间裁剪：热身只给因子，不进净值
    p_bt = {c: d[d["date"] >= pd.Timestamp(START)].reset_index(drop=True)
            for c, d in panel.items()}
    p_bt = {c: d for c, d in p_bt.items() if len(d) > 30}
    W(f"\n- 净值区间内可用 {len(p_bt)} 只（热身数据已剔除）")

    # ── 策略
    dmap = {c: d for c, d in UP.delist_map(basic).items() if c in p_bt}
    w = weights_pit(panel, index, "low_vol", 20, 10, 20)
    res = BS.StrictBacktest(p_bt, capital=1e6, delist_map=dmap).run(
        w, max_positions=10)
    nav = res["nav"]
    nav = nav[(nav.index >= pd.Timestamp(START))]
    strat_ann = ann_of(nav)

    # ── 基准1：全市场等权（PIT 抽样）
    ew_all = ew_nav(p_bt)
    ew_all_ann = ann_of(ew_all)

    # ── 基准2：当前成分股等权
    cur = BS.default_universe(300)
    got_cur = BS.fetch_panel(cur, START, END, with_flags=False)
    p_cur = got_cur["panel"]
    ew_cur = ew_nav(p_cur)
    ew_cur_ann = ann_of(ew_cur)
    W(f"- 当前沪深300成分取数 {got_cur['n_ok']} 只")

    # ── 三方归因
    W("\n## 2. 三基准对照（同一区间、同一口径）\n")
    W("| 组合 | 年化 | 说明 |")
    W("|---|---|---|")
    W(f"| **策略（low_vol top10）** | {strat_ann*100:.2f}% | 时点票池内选股 |")
    W(f"| **全市场等权（PIT 抽样）** | {ew_all_ann*100:.2f}% | 无选股能力基准 |")
    W(f"| **当前成分股等权** | {ew_cur_ann*100:.2f}% | 幸存者偏差基准 |")
    W(f"\n策略 vs 全市场等权：**{(strat_ann-ew_all_ann)*100:+.2f}pp**")
    W(f"全市场等权 vs 当前成分股等权：**{(ew_all_ann-ew_cur_ann)*100:+.2f}pp**")

    W("\n## 3. 三方归因：为什么当前成分股等权更高\n")
    cur_set = set(p_cur)
    dl_set = set(dl_in) & set(p_bt)
    g_del = [c for c in p_bt if c in dl_set]
    g_out = [c for c in p_bt if c not in dl_set and c not in cur_set]
    g_in = [c for c in p_bt if c in cur_set]
    W("| 分组 | 只数 | 等权年化 | 在抽样中的权重 |")
    W("|---|---|---|---|")
    for name, g in (("退市股", g_del), ("被调出指数（未退市）", g_out),
                    ("仍在指数内", g_in)):
        if len(g) < 3:
            W(f"| {name} | {len(g)} | 样本不足 | - |")
            continue
        a = ann_of(ew_nav({c: p_bt[c] for c in g}))
        W(f"| {name} | {len(g)} | {a*100:.2f}% | {len(g)/len(p_bt)*100:.1f}% |")

    # ── 退市股明细
    W("\n## 4. 退市股入选清单\n")
    contrib = res.get("contrib_by_code") or {}
    liq = {r["code"]: r for r in (res.get("delist_log") or [])}
    held = [c for c in contrib if c in dl_set]
    if held:
        W("| 代码 | 退市日 | 入选 | 退出 | 天数 | 盈亏(pp) | 核销 |")
        W("|---|---|---|---|---|---|---|")
        for c in sorted(held):
            r = liq.get(c)
            way = (f"{r['reason']} rec={r['recovery']:.3f}" if r else "退市前已清仓")
            W(f"| {c} | {dmap[c].date()} | "
              f"{res['first_in_by_code'].get(c,'-')} | "
              f"{res['last_out_by_code'].get(c,'-')} | "
              f"{res['hold_days_by_code'].get(c,0)} | "
              f"{contrib[c]/1e6*100:+.2f}% | {way} |")
        W(f"\n共 {len(held)} 只退市股进入持仓，"
          f"合计 {sum(contrib[c] for c in held)/1e6*100:+.2f}pp")
    else:
        W("无退市股进入持仓")

    # ── 结论
    W("\n## 5. 结论\n")
    alpha = strat_ann - ew_all_ann
    if alpha <= 0:
        W(f"**策略没有正 alpha。** 相对全市场等权，年化 "
          f"{alpha*100:+.2f}pp，扣成本后为负。")
    elif alpha < 0.03:
        W(f"**策略 alpha 微弱且不稳健**（{alpha*100:+.2f}pp/年），"
          f"三年样本无法与运气区分。")
    else:
        W(f"策略存在 {alpha*100:+.2f}pp/年 的超额，但需检查是否来自选参偏差。")
    W(f"\n- 策略最大回撤 {res['max_drawdown']*100:.2f}%")
    W(f"- 退市核销 {res['n_delist_liquidations']} 次")

    out = Path(__file__).resolve().parent.parent / "reports" / "final_conclusion.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep), encoding="utf-8")
    print(f"\n[报告] {out}")


if __name__ == "__main__":
    main()
