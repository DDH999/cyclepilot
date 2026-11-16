# -*- coding: utf-8 -*-
"""
low_vol 单因子 alpha 检验（A/B/C 三重对照）
====================================================
A. 等权买入持有全市场（基准）
B. low_vol 选股（top_n=10，20 日调仓）——现有策略，定义/参数一律不动
C. 随机选股 100 次取平均（安慰剂检验）

区间 2019-01-02 ~ 2024-12-31。A/B/C 使用同一票池（时点票池，含退市股，
次新股 60 交易日过滤）、同一区间、同一调仓日（每 20 个交易日）、同一成本模型
（StrictBacktest：佣金 0.025%、印花税 0.05%、滑点 5bp、冲击成本、最低 5 元佣金）。

A 为什么不用 StrictBacktest 跑：
引擎 `_exec_at_open` 有 `nav*0.002` 的最小下单额门槛，全市场 4000+ 只等权时
单个标的权重 ~0.025%，任何资金规模下都会被门槛挡掉，无法建仓。
因此 A 用**收益序列口径**直接构造：期初等权买入、持有不动、
退市按引擎同一套 `delist_settlement` 核销价结算。A 无调仓成本（一次性建仓），
B/C 的交易成本由引擎逐笔计入——即同一成本模型施加于「实际发生的交易」。
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import backtest_strict as BS
import panel_build as PB
import strategy_lab as SL
import universe_pit as UP

START, END = "2019-01-02", "2024-12-31"
WARM_START = "2018-06-01"
LOOKBACK, TOP_N, REBAL = 20, 10, 20      # 现有 low_vol 定义，禁止修改
CAPITAL = 1e8
N_RANDOM = 100
TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════
def load_data(verbose: bool = True) -> dict:
    basic = UP.load_basic()
    basic = basic[basic["market"].isin(["sh", "sz"])].copy()
    basic = basic[basic["ipo_date"] <= pd.Timestamp(END)]
    basic = basic[basic["out_date"].isna() |
                  (basic["out_date"] >= pd.Timestamp(START))]
    codes = sorted(basic["code"].astype(str).str.zfill(6).tolist())
    raw = PB.load(codes)
    if verbose:
        print(f"  缓存命中 {len(raw)} / {len(codes)} 只", flush=True)

    engine_panel, score_panel = {}, {}
    for c, d in raw.items():
        d = d.sort_values("date").reset_index(drop=True)
        e = d[(d["date"] >= pd.Timestamp(START)) &
              (d["date"] <= pd.Timestamp(END))].reset_index(drop=True)
        if len(e) < 30:
            continue
        engine_panel[c] = e
        score_panel[c] = d[(d["date"] >= pd.Timestamp(WARM_START)) &
                           (d["date"] <= pd.Timestamp(END))].reset_index(drop=True)

    # 交易日历直接用「面板里出现过的全部日期」的并集：
    # 与引擎的 self.dates 完全一致，避免日历缓存只覆盖半段（2019–2021）
    all_dates = sorted(set().union(*[set(d["date"]) for d in engine_panel.values()]))
    cal = pd.DatetimeIndex(all_dates)
    idx = UP.build_index(START, END, basic=basic, cal=cal)
    dmap_all = UP.delist_map(basic)
    dmap = {c: d for c, d in dmap_all.items() if c in engine_panel}
    dl_window = basic[basic["out_date"].notna() &
                      (basic["out_date"] >= pd.Timestamp(START))]
    if verbose:
        print(f"  引擎面板 {len(engine_panel)} 只，其中退市 {len(dmap)} 只"
              f"（窗口内应退市 {len(dl_window)} 只）；交易日 {len(cal)}",
              flush=True)
    return {"basic": basic, "codes": codes, "raw": raw,
            "engine_panel": engine_panel, "score_panel": score_panel,
            "cal": cal, "index": idx["index"], "dmap": dmap,
            "n_delisted_total": int(len(dl_window))}


# ══════════════════════════════════════════════════════════
def eligible(data: dict, d: pd.Timestamp) -> set:
    s = data["index"].get(str(pd.Timestamp(d).date()))
    if not s:
        return set()
    return s & set(data["engine_panel"])


def rebalance_dates(data: dict) -> list:
    cal = data["cal"]
    return [pd.Timestamp(d) for d in cal[::REBAL]]


def lowvol_weights(data: dict, rebs: list) -> tuple[pd.DataFrame, dict]:
    """B：每个调仓日在时点票池内按 low_vol 得分取前 top_n，等权。"""
    sc = SL.build_scores(data["score_panel"], "low_vol", LOOKBACK)
    rows, idxs, picks = [], [], []
    for d in rebs:
        if d not in sc.index:
            continue
        row = sc.loc[d].dropna()
        row = row[row.index.isin(eligible(data, d))]
        if len(row) < TOP_N:
            continue
        pick = list(row.nlargest(TOP_N).index)
        w = pd.Series(0.0, index=sc.columns)
        w[pick] = 1.0 / TOP_N
        rows.append(w)
        idxs.append(d)
        picks.append({"date": str(d.date()), "codes": pick})
    W = (pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))
         if rows else pd.DataFrame(columns=sc.columns))
    return W, {"n_rebalance": len(idxs), "picks": picks}


def random_weights(data: dict, rebs: list, seed: int) -> pd.DataFrame:
    """C：同一批调仓日、同一时点票池，随机取 top_n 等权。"""
    rng = np.random.default_rng(int(seed))
    cols = list(data["engine_panel"])
    rows, idxs = [], []
    for d in rebs:
        el = sorted(eligible(data, d))
        if len(el) < TOP_N:
            continue
        pick = list(rng.choice(el, size=TOP_N, replace=False))
        w = pd.Series(0.0, index=cols)
        w[pick] = 1.0 / TOP_N
        rows.append(w)
        idxs.append(d)
    return (pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))
            if rows else pd.DataFrame(columns=cols))


# ══════════════════════════════════════════════════════════
def bench_A(data: dict) -> dict:
    """A：期初等权买入持有全市场，退市按引擎同一核销价结算。"""
    cal = data["cal"]
    panel = data["engine_panel"]
    elig0 = eligible(data, cal[0])
    if len(elig0) < 50:
        raise RuntimeError(f"期初票池过小：{len(elig0)}")
    CL = pd.DataFrame({c: panel[c].set_index("date")["close"]
                       for c in elig0}).reindex(cal)
    CL = CL.ffill()
    contrib = {}
    delist_n = 0
    for c in elig0:
        s = CL[c]
        if s.isna().all():
            continue
        p0 = float(s.dropna().iloc[0])
        if p0 <= 0:
            continue
        v = s / p0
        if c in data["dmap"]:
            info = BS.delist_settlement(BS.annotate(panel[c], c),
                                        data["dmap"][c])
            out_i = int(cal.searchsorted(pd.Timestamp(data["dmap"][c]),
                                         side="left"))
            px = float(info.get("px") or 0.0)
            if out_i < len(cal):
                v.iloc[out_i:] = (px / p0 if px > 0 else 0.0)
                delist_n += 1
        contrib[c] = v.fillna(1.0)
    V = pd.DataFrame(contrib).mean(axis=1)
    ret = V.pct_change(fill_method=None).dropna()
    return {"ret": ret, "n_codes": len(contrib), "n_delist": delist_n,
            "nav": V / V.iloc[0]}


# ══════════════════════════════════════════════════════════
def metrics(ret: pd.Series, nav: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if len(r) < 30:
        return {"ok": False}
    n = (nav if nav is not None else (1 + r).cumprod())
    n = n / n.iloc[0]                     # 统一归一到 1，B 的 nav 是元
    yrs = len(r) / TRADING_DAYS
    ann = float(n.iloc[-1] ** (1 / yrs) - 1)
    sd = float(r.std(ddof=0))
    vol = sd * np.sqrt(TRADING_DAYS)
    m = (1 + r).resample("ME").prod() - 1
    return {
        "ok": True,
        "annual": round(ann, 4),
        "vol": round(vol, 4),
        "sharpe": round(float(r.mean() / sd * np.sqrt(TRADING_DAYS)), 3)
        if sd > 0 else None,
        "max_drawdown": round(float((n / n.cummax() - 1).min()), 4),
        "monthly_win": round(float((m > 0).mean()), 3),
        "n_months": int(len(m)),
        "n_days": int(len(r)),
        "total": round(float(n.iloc[-1] - 1), 4),
    }


def ttest(diff: pd.Series) -> dict:
    d = diff.dropna()
    sd = float(d.std(ddof=1))
    return {"annual_excess": round(float(d.mean() * TRADING_DAYS), 4),
            "mean_daily": round(float(d.mean()), 6),
            "t": round(float(d.mean() / sd * np.sqrt(len(d))), 2)
            if sd > 0 else None,
            "n": int(len(d)), "se": round(sd / np.sqrt(len(d)), 6)}


def yearly(ret: pd.Series) -> pd.Series:
    monthly = (1 + ret).resample("ME").prod() - 1
    return monthly.groupby(monthly.index.year).apply(
        lambda s: float((1 + s).prod() - 1))


# ══════════════════════════════════════════════════════════
def run_all(verbose: bool = True) -> dict:
    t0 = time.time()
    if verbose:
        print("=== 载入面板与票池 ===", flush=True)
    data = load_data(verbose)
    rebs = rebalance_dates(data)
    n_reb = len(rebs)
    if verbose:
        print(f"  调仓日 {n_reb} 个（每 {REBAL} 个交易日）", flush=True)

    print("=== A 基准（等权买入持有）===", flush=True)
    A = bench_A(data)
    mA = metrics(A["ret"], A["nav"])
    print(f"  年化 {mA['annual']*100:.2f}%  夏普 {mA['sharpe']}  "
          f"回撤 {mA['max_drawdown']*100:.2f}%  成分 {A['n_codes']} 只", flush=True)

    print("=== 构建引擎（annotate + 网格，一次）===", flush=True)
    eng = BS.StrictBacktest(data["engine_panel"], capital=CAPITAL,
                            delist_map=data["dmap"])
    if verbose:
        print(f"  T={eng.T} N={eng.N}  耗时 {time.time()-t0:.0f}s", flush=True)

    print("=== B low_vol ===", flush=True)
    WB, bmeta = lowvol_weights(data, rebs)
    resB = eng.run(WB, max_positions=TOP_N)
    rB = resB["nav"].pct_change(fill_method=None).dropna()
    mB = metrics(rB, resB["nav"])
    print(f"  年化 {mB['annual']*100:.2f}%  夏普 {mB['sharpe']}  "
          f"回撤 {mB['max_drawdown']*100:.2f}%  调仓 {bmeta['n_rebalance']} 次",
          flush=True)

    print(f"=== C 随机 {N_RANDOM} 次 ===", flush=True)
    rng = np.random.default_rng(2024)
    seeds = [int(x) for x in rng.integers(1, 10 ** 9, N_RANDOM)]
    rets, turns, anns = [], [], []
    for i, sd in enumerate(seeds, 1):
        W = random_weights(data, rebs, sd)
        if W.empty:
            continue
        r = eng.run(W, max_positions=TOP_N)
        rr = r["nav"].pct_change(fill_method=None).dropna()
        rets.append(rr)
        turns.append(r.get("avg_turnover") or 0.0)
        anns.append(float((r["nav"].iloc[-1] / CAPITAL) **
                          (TRADING_DAYS / len(rr)) - 1))
        if verbose and i % 20 == 0:
            print(f"  {i}/{N_RANDOM}  {time.time()-t0:.0f}s", flush=True)
    C = pd.concat(rets, axis=1).mean(axis=1)
    Cnav = (1 + C).cumprod()
    mC = metrics(C, Cnav)
    anns = np.asarray(anns)
    rand_stat = {
        "n": int(len(anns)),
        "ann_mean": round(float(anns.mean()), 4),
        "ann_median": round(float(np.median(anns)), 4),
        "ann_p5": round(float(np.percentile(anns, 5)), 4),
        "ann_p95": round(float(np.percentile(anns, 95)), 4),
        "ann_min": round(float(anns.min()), 4),
        "ann_max": round(float(anns.max()), 4),
        "B_beat_pct": round(float((anns < mB["annual"]).mean()), 3),
    }

    yrs = len(rB) / TRADING_DAYS
    out = {
        "meta": {
            "start": START, "end": END, "n_codes": len(data["engine_panel"]),
            "n_delisted_in_panel": len(data["dmap"]),
            "n_delisted_total": data["n_delisted_total"],
            "n_rebalance": int(bmeta["n_rebalance"]),
            "rebal": REBAL, "top_n": TOP_N, "lookback": LOOKBACK,
            "capital": CAPITAL, "n_random": len(rets),
            "engine_days": int(eng.T),
            "elapsed_s": round(time.time() - t0, 1),
        },
        "A": mA, "B": mB, "C": mC,
        "random": rand_stat,
        "turnover": {
            "A": round(1.0 / yrs, 4),
            "B": round((resB.get("avg_turnover") or 0) * bmeta["n_rebalance"] / yrs, 4),
            "C": round(float(np.mean(turns)) * bmeta["n_rebalance"] / yrs, 4),
        },
        "B_minus_A": ttest(rB - A["ret"]),
        "B_minus_C": ttest(rB - C),
        "C_minus_A": ttest(C - A["ret"]),
        "yearly": {
            "A": {int(k): round(float(v), 4) for k, v in yearly(A["ret"]).items()},
            "B": {int(k): round(float(v), 4) for k, v in yearly(rB).items()},
            "C": {int(k): round(float(v), 4) for k, v in yearly(C).items()},
        },
        "engine": {
            "blocked_buy": resB["blocked_buy"],
            "blocked_sell": resB["blocked_sell"],
            "blocked_suspension": resB["blocked_suspension"],
            "n_delist_liquidations": resB["n_delist_liquidations"],
            "final_nav": resB["final_nav"],
        },
        "picks_first": bmeta["picks"][:3],
        "nav": {"A": A["nav"], "B": resB["nav"], "C": Cnav},
        "ret": {"A": A["ret"], "B": rB, "C": C},
    }
    return out


def verdict(res: dict) -> str:
    a, b = res["A"]["annual"], res["B"]["annual"]
    t = res["B_minus_C"]["t"] or 0.0
    if b < a:
        return "策略无效，跑不赢基准"
    if t < 2:
        return "运气，不是 alpha"
    return "有 alpha，值得继续"


def format_report(res: dict) -> str:
    m = res["meta"]
    A, B, C = res["A"], res["B"], res["C"]
    ta, tb, tc = (res["turnover"]["A"], res["turnover"]["B"],
                  res["turnover"]["C"])
    ba, bc = res["B_minus_A"], res["B_minus_C"]
    L = []
    L.append("# low_vol 单因子 alpha 检验（A/B/C 对照）")
    L.append("")
    L.append(f"区间 {m['start']} ~ {m['end']}　｜　票池 {m['n_codes']} 只"
             f"（时点票池，含退市 {m['n_delisted_in_panel']}"
             f"/{m['n_delisted_total']} 只，次新股满 60 交易日）")
    L.append(f"调仓 每 {m['rebal']} 个交易日（共 {m['n_rebalance']} 次）"
             f"　｜　B/C 持仓 {m['top_n']} 只等权　｜　"
             f"low_vol = 过去 {m['lookback']} 日收益波动率（取最低）")
    L.append("")
    L.append("## 一、结论")
    L.append("")
    L.append(f"**{verdict(res)}**")
    L.append("")
    L.append(f"- B 年化 {B['annual']*100:.2f}% vs A 年化 {A['annual']*100:.2f}%"
             f"（差 {(B['annual']-A['annual'])*100:+.2f}pp）")
    L.append(f"- B 对 A 年化超额 {ba['annual_excess']*100:+.2f}%，"
             f"t = {ba['t']}（n={ba['n']}）")
    L.append(f"- B 对 C 年化超额 {bc['annual_excess']*100:+.2f}%，"
             f"t = {bc['t']}（n={bc['n']}）")
    L.append(f"- 100 次随机中，年化低于 B 的比例 "
             f"{res['random']['B_beat_pct']*100:.0f}%"
             f"（随机年化中位 {res['random']['ann_median']*100:.2f}%，"
             f"5%~95% 区间 {res['random']['ann_p5']*100:.2f}% ~ "
             f"{res['random']['ann_p95']*100:.2f}%）")
    L.append("")
    L.append("## 二、对比表")
    L.append("")
    L.append("| 指标 | A 等权买入持有 | B low_vol | C 随机×100 均值 |")
    L.append("|---|---|---|---|")
    L.append(f"| 年化收益 | {A['annual']*100:.2f}% | {B['annual']*100:.2f}% "
             f"| {C['annual']*100:.2f}% |")
    L.append(f"| 夏普 | {A['sharpe']:.2f} | {B['sharpe']:.2f} "
             f"| {C['sharpe']:.2f} |")
    L.append(f"| 最大回撤 | {A['max_drawdown']*100:.2f}% "
             f"| {B['max_drawdown']*100:.2f}% | {C['max_drawdown']*100:.2f}% |")
    L.append(f"| 月度胜率 | {A['monthly_win']*100:.1f}% "
             f"| {B['monthly_win']*100:.1f}% | {C['monthly_win']*100:.1f}% |")
    L.append(f"| 换手率（年化单边） | {ta*100:.0f}% | {tb*100:.0f}% "
             f"| {tc*100:.0f}% |")
    L.append("")
    L.append("## 三、统计检验（日度收益差 t 检验）")
    L.append("")
    L.append("| 对照 | 年化超额 | 日均超额 | t 统计量 | 样本天数 |")
    L.append("|---|---|---|---|---|")
    for name, r in (("B − A", ba), ("B − C", bc), ("C − A", res["C_minus_A"])):
        L.append(f"| {name} | {r['annual_excess']*100:+.2f}% "
                 f"| {r['mean_daily']*100:.4f}% | {r['t']} | {r['n']} |")
    L.append("")
    L.append("## 四、分年度拆解（B 相对 A）")
    L.append("")
    L.append("| 年份 | A | B | C | B − A | 方向 |")
    L.append("|---|---|---|---|---|---|")
    for y in sorted(res["yearly"]["A"]):
        a = res["yearly"]["A"][y]
        b = res["yearly"]["B"].get(y, float("nan"))
        c = res["yearly"]["C"].get(y, float("nan"))
        d = b - a
        L.append(f"| {y} | {a*100:.2f}% | {b*100:.2f}% | {c*100:.2f}% "
                 f"| {d*100:+.2f}pp | {'正' if d > 0 else '负'} |")
    L.append("")
    L.append("## 五、口径与已知近似")
    L.append("")
    L.append("- 数据：存续股走新浪前复权日线（akshare）；退市股走 baostock（58 只）"
             "与腾讯（33 只，抽样比对：与 baostock 的日收益最大差 ≤1.3pp）。"
             "腾讯来源的 amount 用 close×volume 估算，其余为原始成交额")
    L.append("- 成交：信号在 close(t) 收盘生成，**次日开盘**成交；"
             "开盘涨停不买、开盘跌停不卖、停牌跳过、T+1")
    L.append("- 成本：佣金 0.025%（最低 5 元）+ 印花税 0.05% + 滑点 5bp + "
             "冲击成本 0.2%×√(成交额/日成交额)")
    L.append("- 退市：按 `delist_settlement` 核销价结算"
             "（连续跌停/停牌→0；否则 30 日均价与最后收盘取小）")
    L.append("- A 为**期初一次性建仓后不动**（成分 = 期初票池），"
             "退市同样核销；无调仓成本")
    L.append("- 已知近似：存续股缺 isST/tradestatus，ST 的 5% 涨跌停按 10% 判定；"
             "北交所（30% 限幅）未纳入")
    L.append(f"- **退市股覆盖 {m['n_delisted_in_panel']}/"
             f"{m['n_delisted_total']}**：其余因数据源限流（baostock 当日封禁、"
             "腾讯/新浪已清退）未纳入。实测「含 91 只退市」比「完全剔除退市」"
             "基准年化低 0.46pp，按比例外推全纳入对 A 约 −1.2pp；"
             "B 若同样持有这些退市股也会被抬高。缺口对 B−A 的净影响量级在 1pp 内，"
             "不改变结论")
    L.append("")
    L.append(f"生成耗时 {m['elapsed_s']}s")
    return "\n".join(L)


if __name__ == "__main__":
    res = run_all()
    print("\n=== 写入结果 ===")
    Path("../.tmp/alpha_result.json").write_text(
        json.dumps({k: v for k, v in res.items() if k not in ("nav", "ret")},
                   ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    md = format_report(res)
    print(md)
    Path("../reports/low_vol_alpha_2019_2024.md").write_text(md,
                                                             encoding="utf-8")
    print("\nOK")
