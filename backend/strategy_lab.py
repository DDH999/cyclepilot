# -*- coding: utf-8 -*-
"""
多策略族样本外横评（M46）
====================================================
上一轮结论：**动量在 2019–2021 年化 42.68%，2022–2024 变成 -1.98%**。

但那只证明「动量失效」，不能推出「所有因子失效」。
本模块把四类因子放进**同一套严格引擎**、**同一组区间**，公平对比：

| 因子 | 逻辑 | 方向 |
|---|---|---|
| **动量** | 过去 N 日涨得多的继续涨 | 买高 |
| **短反转** | 过去 N 日跌得多的反弹 | 买低 |
| **长反转（价值代理）** | 过去 2 年跌得多的均值回归 | 买低 |
| **低波** | 波动小的风险调整后更优 | 买低波 |
| **等权基准** | 全池等权 | — |

两套评价口径
------------
1. **IC**（信息系数）：因子得分与未来收益的 Spearman 秩相关
   —— 衡量"因子本身有没有预测力"，与组合构建无关

   |IC| 经验标尺：<0.02 无效；0.02~0.04 弱；0.04~0.06 中；>0.06 强
2. **组合回测**：选因子前 N 名、等权、周期调仓，走严格引擎
   （次日开盘成交 + 涨跌停/停牌过滤 + 真实成本）

⚠ 结论的适用范围
-----------------
* 只覆盖 **A 股 60 只大盘股**（沪深300 成分）
* 只覆盖 **2019–2024**，不跨牛熊完整周期
* 「价值」用长反转代理，**不是真的 PB/PE 因子**（缺历史估值数据）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["FACTORS", "build_scores", "ic_series", "factor_weights",
           "run_matrix", "format_matrix", "format_ic"]

TRADING_DAYS = 252

# 因子定义：名称 → (类别, 说明, 是否买高)
FACTORS = {
    "momentum": ("动量", "过去 N 日涨幅排名，买最高", True),
    "reversal": ("短反转", "过去 N 日跌幅排名，买最低", False),
    "lt_reversal": ("长反转/价值代理", "过去 2 年跌幅，买最低", False),
    "low_vol": ("低波", "过去 N 日波动率，买最低", False),
}


def _close_panel(panel: dict) -> pd.DataFrame:
    d = {}
    for c, df in panel.items():
        d[c] = df.set_index("date")["close"].sort_index()
    return pd.DataFrame(d).sort_index()


def build_scores(panel: dict, factor: str, lookback: int) -> pd.DataFrame:
    """
    构造因子得分面板（date × code）。

    所有值只用**截至当日**的数据，不引入未来信息。
    """
    CL = _close_panel(panel)
    lb = max(int(lookback), 1)
    # 统一约定：得分 **越高越好**，组合层永远取 nlargest。
    # 早期版本对反转因子既取负又取最小，双重取反导致它选的股票
    # 与动量完全相同（自测时两者数字一模一样才暴露）。
    if factor == "momentum":
        s = CL.pct_change(lb, fill_method=None)          # 涨得多 = 高分
    elif factor == "reversal":
        s = -CL.pct_change(lb, fill_method=None)         # 跌得多 = 高分
    elif factor == "lt_reversal":
        s = -CL.pct_change(lb, fill_method=None)         # 长期跌得多 = 高分
    elif factor == "low_vol":
        r = CL.pct_change(fill_method=None)
        s = -r.rolling(lb).std(ddof=0)                   # 波动低 = 高分
    else:
        raise ValueError(f"未知因子 {factor!r}")
    return s


def ic_series(panel: dict, score: pd.DataFrame, horizon: int = 5,
              step: int = 5) -> dict:
    """
    因子 IC：每个调仓日算 Spearman(因子得分, 未来 horizon 日收益)。

    未来收益用 close(t) → close(t+h)，与因子得分在 t 的信息严格对齐。
    """
    from scipy.stats import spearmanr
    CL = _close_panel(panel)
    fwd = CL.shift(-int(horizon)) / CL - 1.0

    ics, dates = [], []
    idx = list(score.index)
    for i in range(0, len(idx), max(int(step), 1)):
        d = idx[i]
        if d not in fwd.index:
            continue
        a = score.loc[d].dropna()
        b = fwd.loc[d].reindex(a.index).dropna()
        a = a.reindex(b.index)
        if len(a) < 10:
            continue
        try:
            rho = spearmanr(a.to_numpy(), b.to_numpy()).statistic
        except Exception:
            continue
        if np.isfinite(rho):
            ics.append(float(rho))
            dates.append(d)
    if len(ics) < 5:
        return {"ok": False, "reason": f"有效 IC 点仅 {len(ics)}"}
    s = pd.Series(ics, index=pd.DatetimeIndex(dates))
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    return {"ok": True, "ic_series": s, "mean_ic": round(mean, 4),
            "ic_std": round(std, 4),
            "ic_ir": (round(mean / std, 3) if std > 0 else None),
            "positive_rate": round(float((s > 0).mean()), 3),
            "n": len(s),
            "verdict": _ic_verdict(mean)}


def _ic_verdict(mean_ic: float) -> str:
    a = abs(mean_ic)
    sign = "正" if mean_ic > 0 else "负"
    if a < 0.02:
        return f"|IC|={a:.3f} < 0.02 → 基本无预测力"
    if a < 0.04:
        return f"|IC|={a:.3f} → 弱预测力（{sign}向）"
    if a < 0.06:
        return f"|IC|={a:.3f} → 中等预测力（{sign}向）"
    return f"|IC|={a:.3f} → 强预测力（{sign}向）"


def factor_weights(score: pd.DataFrame, factor: str, top_n: int,
                   rebalance: int = 20) -> pd.DataFrame:
    """
    因子得分 → 持仓权重（每 rebalance 日取前 top_n 等权）。

    约定：得分已统一为「越高越好」，一律取 nlargest。
    """
    rows, idx = [], []
    dates = list(score.index)
    for i in range(0, len(dates), max(int(rebalance), 1)):
        d = dates[i]
        row = score.loc[d].dropna()
        if len(row) < top_n:
            continue
        pick = row.nlargest(top_n).index
        w = pd.Series(0.0, index=score.columns)
        w[pick] = 1.0 / len(pick)
        rows.append(w); idx.append(d)
    if not rows:
        return pd.DataFrame(columns=score.columns)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _slice(panel: dict, a: str, b: str) -> dict:
    out = {}
    for c, d in panel.items():
        s = d[(d["date"] >= pd.Timestamp(a)) & (d["date"] <= pd.Timestamp(b))]
        if len(s) > 80:
            out[c] = s.reset_index(drop=True)
    return out


def run_matrix(panel: dict, is_range: tuple[str, str],
               oos_range: tuple[str, str],
               factors: list[str] | None = None,
               grid: dict | None = None,
               capital: float = 1_000_000.0,
               verbose: bool = True) -> dict:
    """
    对每个因子：IS 上调参选最优，OOS 上检验，同时算 IS/OOS 的 IC。

    返回横评表。
    """
    import backtest_strict as BS

    factors = factors or list(FACTORS)
    grid = grid or {"lookback": [20, 60, 120], "top_n": [10, 20],
                    "rebalance": [10, 20]}
    p_is = _slice(panel, *is_range)
    p_oos = _slice(panel, *oos_range)
    if len(p_is) < 10 or len(p_oos) < 10:
        return {"ok": False,
                "reason": f"分区样本不足（IS {len(p_is)} / OOS {len(p_oos)}）"}

    results = []
    for f in factors:
        label, desc, _ = FACTORS[f]
        if verbose:
            print(f"\n  [{label}] {desc}")

        # ── IS 调参
        # 长反转是 2 年级别的均值回归，回看必须是 250+，否则不成立
        lbs = ([250, 500] if f == "lt_reversal"
               else list(grid["lookback"]))
        best = None
        for lb in lbs:
            for tn in grid["top_n"]:
                for rb in grid["rebalance"]:
                    try:
                        sc = build_scores(p_is, f, lb)
                        w = factor_weights(sc, f, tn, rb)
                        if w.empty:
                            continue
                        bt = BS.StrictBacktest(p_is, capital=capital).run(
                            w, max_positions=tn)
                    except Exception:
                        continue
                    if not bt.get("ok"):
                        continue
                    rec = {"lookback": lb, "top_n": tn, "rebalance": rb,
                           "is_annual": bt["annual"],
                           "is_sharpe": bt["sharpe"],
                           "is_dd": bt["max_drawdown"]}
                    if best is None or (bt["sharpe"] or -9) > (best[1]["is_sharpe"] or -9):
                        best = (rec, rec)
        if best is None:
            results.append({"factor": f, "label": label, "ok": False,
                            "reason": "IS 无可运行参数"})
            continue
        bp = best[1]

        # ── OOS 检验
        try:
            sc_oos = build_scores(p_oos, f, bp["lookback"])
            w_oos = factor_weights(sc_oos, f, bp["top_n"], bp["rebalance"])
            bt_oos = (BS.StrictBacktest(p_oos, capital=capital).run(
                w_oos, max_positions=bp["top_n"])
                if not w_oos.empty else None)
        except Exception:
            bt_oos = None
        if bt_oos is None:
            results.append({"factor": f, "label": label, "ok": False,
                            "reason": "OOS 无法运行"})
            continue

        # ── IC（IS / OOS）
        ic_is = ic_series(p_is, build_scores(p_is, f, bp["lookback"]),
                          horizon=5)
        ic_oos = ic_series(p_oos, sc_oos, horizon=5)

        decay = bt_oos["annual"] - bp["is_annual"]
        sdecay = (bt_oos["sharpe"] or 0) - (bp["is_sharpe"] or 0)
        if bt_oos["annual"] <= 0:
            verdict = "失效（OOS 负收益）"
        elif sdecay < -0.5:
            verdict = "明显衰减"
        elif decay < -0.05:
            verdict = "收益衰减"
        else:
            verdict = "稳定"

        rec = {
            "factor": f, "label": label, "ok": True,
            "params": {k: bp[k] for k in ("lookback", "top_n", "rebalance")},
            "is": {"annual": bp["is_annual"], "sharpe": bp["is_sharpe"],
                   "dd": bp["is_dd"]},
            "oos": {"annual": bt_oos["annual"], "sharpe": bt_oos["sharpe"],
                    "dd": bt_oos["max_drawdown"]},
            "decay_annual": round(decay, 4),
            "decay_sharpe": round(sdecay, 3),
            "ic_is": ic_is.get("mean_ic") if ic_is.get("ok") else None,
            "ic_oos": ic_oos.get("mean_ic") if ic_oos.get("ok") else None,
            "ic_oos_ir": ic_oos.get("ic_ir") if ic_oos.get("ok") else None,
            "ic_verdict": ic_oos.get("verdict") if ic_oos.get("ok") else None,
            "verdict": verdict,
        }
        results.append(rec)
        if verbose:
            print(f"    最优 lb={bp['lookback']} N={bp['top_n']} rb={bp['rebalance']}"
                  f"  IS {bp['is_annual']*100:>6.2f}% → "
                  f"OOS {bt_oos['annual']*100:>6.2f}%   {verdict}")
            if ic_oos.get("ok"):
                print(f"    OOS IC {ic_oos['mean_ic']:+.4f}  "
                      f"{ic_oos['verdict']}")

    # ── 等权基准（OOS 区间）
    try:
        p_o = p_oos
        first = min(d["date"].iloc[0] for d in p_o.values())
        bench_w = pd.DataFrame([{c: 1.0 / len(p_o) for c in p_o}],
                               index=[first])
        bench = BS.StrictBacktest(p_o, capital=capital).run(
            bench_w, max_positions=len(p_o))
        bench_res = {"annual": bench["annual"], "sharpe": bench["sharpe"],
                     "dd": bench["max_drawdown"]}
    except Exception:
        bench_res = None

    ok = [r for r in results if r.get("ok")]
    survivors = [r for r in ok if r["verdict"] == "稳定"]
    if survivors:
        best_f = max(survivors, key=lambda r: r["oos"]["sharpe"] or -9)
        headline = (f"2022–2024 仍稳定的因子：{best_f['label']}"
                    f"（OOS 年化 {best_f['oos']['annual']*100:.2f}%，"
                    f"夏普 {best_f['oos']['sharpe']}）")
    elif ok:
        best_f = max(ok, key=lambda r: r["oos"]["sharpe"] or -9)
        headline = (f"没有一个因子在 2022–2024 稳定；相对最好的是 "
                    f"{best_f['label']}（OOS 年化 "
                    f"{best_f['oos']['annual']*100:.2f}%）")
    else:
        headline = "所有因子都无法运行"

    return {"ok": True, "results": results, "benchmark": bench_res,
            "survivors": [r["factor"] for r in survivors],
            "headline": headline,
            "is_range": list(is_range), "oos_range": list(oos_range),
            "n_codes_is": len(p_is), "n_codes_oos": len(p_oos)}


def format_matrix(res: dict) -> str:
    L = ["=" * 78, "多策略族样本外横评", "=" * 78]
    if not res.get("ok"):
        L.append(f"  {res.get('reason')}")
        L.append("=" * 78)
        return "\n".join(L)

    L.append(f"  IS {res['is_range'][0]} ~ {res['is_range'][1]}   "
             f"OOS {res['oos_range'][0]} ~ {res['oos_range'][1]}")
    L.append(f"  股票池 {res['n_codes_oos']} 只（OOS）")
    L.append("-" * 78)
    L.append("  " + f"{'因子':<14}{'IS年化':>9}{'OOS年化':>9}{'衰减':>9}"
             f"{'OOS夏普':>9}{'OOS IC':>9}{'结论':>14}")
    L.append("  " + "-" * 74)
    for r in res["results"]:
        if not r.get("ok"):
            L.append("  " + f"{r['label']:<14}  无法运行：{r.get('reason')}")
            continue
        L.append("  " + f"{r['label']:<14}"
                 f"{r['is']['annual']*100:>8.2f}%"
                 f"{r['oos']['annual']*100:>8.2f}%"
                 f"{r['decay_annual']*100:>8.2f}pp"
                 f"{(r['oos']['sharpe'] or 0):>9.2f}"
                 f"{(r['ic_oos'] if r['ic_oos'] is not None else 0):>9.4f}"
                 f"{r['verdict']:>14}")
    b = res.get("benchmark")
    if b:
        L.append("  " + "-" * 74)
        L.append("  " + f"{'等权基准':<14}{'—':>9}"
                 f"{b['annual']*100:>8.2f}%{'—':>9}"
                 f"{(b['sharpe'] or 0):>9.2f}{'—':>9}{'基准':>14}")
    L.append("-" * 78)
    L.append(f"  {res['headline']}")
    L.append("=" * 78)
    L.append("  ⚠ 只覆盖 A 股 60 只大盘股与 2019–2024；")
    L.append("     「价值」用长反转代理，不是真的 PB/PE 因子。")
    L.append("=" * 78)
    return "\n".join(L)


def format_ic(res: dict) -> str:
    L = ["=" * 72, "因子 IC 一览", "=" * 72]
    for r in res.get("results") or []:
        if not r.get("ok"):
            continue
        L.append(f"  {r['label']:<14} IS IC {r['ic_is']}   "
                 f"OOS IC {r['ic_oos']}   IR {r['ic_oos_ir']}")
        if r.get("ic_verdict"):
            L.append(f"       OOS：{r['ic_verdict']}")
    L.append("=" * 72)
    L.append("  |IC|<0.02 无效；0.02~0.04 弱；0.04~0.06 中；>0.06 强")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-02", periods=800)
    n = 12
    panel = {}
    for i in range(n):
        c = 10 * np.cumprod(1 + rng.normal(0.0003, 0.015, 800))
        panel[f"6000{i:02d}"] = pd.DataFrame({
            "date": idx, "open": c, "high": c * 1.01, "low": c * 0.99,
            "close": c, "volume": 1e6, "amount": 1e7})

    sc = build_scores(panel, "momentum", 20)
    print(f"动量得分面板 {sc.shape}")
    assert sc.shape[1] == n
    assert sc.iloc[:20].isna().all().all(), "回看期内应为 NaN"

    w = factor_weights(sc, "momentum", 3, 20)
    print(f"权重表 {w.shape}，每行非零 {int((w > 0).sum(axis=1).iloc[0])} 只")
    assert w.shape[1] == n

    ic = ic_series(panel, sc, horizon=5)
    print(f"IC: {ic.get('mean_ic')}  {ic.get('verdict')}")
    assert ic["ok"]

    res = run_matrix(panel,
                     is_range=("2020-01-02", "2021-06-30"),
                     oos_range=("2021-07-01", "2022-12-30"),
                     grid={"lookback": [20, 60], "top_n": [3],
                           "rebalance": [20]},
                     verbose=False)
    print()
    print(format_matrix(res))
    assert res["ok"]
    print("\nOK v" + __version__)
