# -*- coding: utf-8 -*-
"""
收益归因与成本核算（M6）
====================================================
回答三个最实际的问题：

1. **我赚的钱是哪来的？** —— 市场给的（beta）还是自己选的（alpha）？
2. **我一年被成本吃掉多少？** —— 佣金/印花税/过户费/滑点明细
3. **降换手能省多少？** —— 不同调仓频率下的成本对照

这三项都是**确定性收益**，不依赖任何 alpha。
对 10 万本金来说，省下 1.6% 比找到 1.6% 的超额收益容易得多。

依赖：numpy / pandas
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = [
    "TRADING_DAYS", "recovery_needed", "cost_analysis", "turnover_sensitivity",
    "performance_attribution", "load_bench", "format_report", "format_cost_report",
]

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
BENCH_CACHE = _ROOT / ".cache" / "bench_000300.csv"

TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════
# 基础
# ══════════════════════════════════════════════════════════
def _series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] == 1 else x.mean(axis=1)
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    return s


def recovery_needed(drawdown: float) -> float:
    """
    回本所需涨幅。

    drawdown 传负数（如 -0.30）或正数（0.30）均可。
    返回 0.4286 表示需涨 42.86% 才回本。
    """
    dd = abs(float(drawdown))
    if dd <= 0:
        return 0.0
    if dd >= 1.0:
        return float("inf")
    return dd / (1.0 - dd)


def max_drawdown(nav: np.ndarray) -> float:
    a = np.asarray(nav, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    peak = np.maximum.accumulate(a)
    dd = a / np.where(peak <= 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd)) if np.isfinite(dd).any() else 0.0


# ══════════════════════════════════════════════════════════
# 成本核算
# ══════════════════════════════════════════════════════════
DEFAULT_FEES = {
    "commission_bps": 2.5,    # 佣金 万2.5（双向）
    "stamp_bps": 5.0,         # 印花税 万5（仅卖出）
    "transfer_bps": 0.1,      # 过户费 万0.1（双向）
    "slippage_bps": 5.0,      # 滑点 5bps（双向）
}


def cost_analysis(capital: float, turnover_annual: float,
                  commission_bps: float = DEFAULT_FEES["commission_bps"],
                  stamp_bps: float = DEFAULT_FEES["stamp_bps"],
                  transfer_bps: float = DEFAULT_FEES["transfer_bps"],
                  slippage_bps: float = DEFAULT_FEES["slippage_bps"]) -> dict:
    """
    年交易成本核算。

    参数
    ----
    capital         : 本金
    turnover_annual : **年换手率**。1.0 = 一年把全部仓位换一遍；
                      12.0 = 月频全仓调仓；0 = 买入持有

    成交额推算
    ----------
    换手率 T 意味着一年"卖出 T 倍本金、买入 T 倍本金"，
    因此双边成交额 = 2 × T × capital。

    返回各大项金额与占比（bps 均按成交额计）。
    """
    cap = float(capital)
    T = max(float(turnover_annual), 0.0)
    amount_one_side = T * cap
    amount_both = 2.0 * amount_one_side

    commission = amount_both * commission_bps / 1e4
    stamp = amount_one_side * stamp_bps / 1e4          # 只有卖出交印花税
    transfer = amount_both * transfer_bps / 1e4
    slippage = amount_both * slippage_bps / 1e4
    total = commission + stamp + transfer + slippage

    return {
        "capital": round(cap, 2),
        "turnover_annual": round(T, 4),
        "amount_both": round(amount_both, 2),
        "commission": round(commission, 2),
        "stamp_duty": round(stamp, 2),
        "transfer_fee": round(transfer, 2),
        "slippage": round(slippage, 2),
        "total": round(total, 2),
        "total_pct": round(total / cap, 6) if cap > 0 else float("nan"),
        "bps_of_capital": round(total / cap * 1e4, 2) if cap > 0 else float("nan"),
    }


def turnover_sensitivity(capital: float, turnovers: Iterable[float] | None = None,
                         **fee_kw) -> pd.DataFrame:
    """
    不同换手率下的成本对照表。

    默认对比：买入持有 / 季频 / 月频 / 双周 / 周频。
    """
    if turnovers is None:
        turnovers = [0.0, 4.0, 12.0, 24.0, 48.0]
    rows = []
    for t in turnovers:
        r = cost_analysis(capital, t, **fee_kw)
        rows.append({"年换手": t, "成交额": r["amount_both"],
                     "总成本": r["total"], "占本金%": r["total_pct"] * 100})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 收益归因
# ══════════════════════════════════════════════════════════
def performance_attribution(nav, bench, rf_annual: float = 0.0) -> dict:
    """
    把总收益拆成「市场给的」与「自己选的」。

    方法：以基准日收益做单变量回归（CAPM/Jensen alpha）

        r_p = alpha + beta * r_b + e

    * **市场贡献** = beta × 基准区间收益 —— 跟着大盘赚的
    * **选股贡献** = 总收益 − 市场贡献 —— 剔除 beta 后剩下的
                    （含 alpha 与残差，样本短时不要过度解读）

    参数
    ----
    nav   : 组合净值序列（正数），索引为日期
    bench : 基准净值序列，索引为日期（会自动对齐交集）

    返回 dict，数值为 float64 全精度。
    """
    s_nav = _series(nav)
    s_bench = _series(bench)
    idx = s_nav.index.intersection(s_bench.index)
    if len(idx) < 30:
        raise ValueError(f"重叠交易日仅 {len(idx)} 天，至少需要 30 天")

    s_nav, s_bench = s_nav.loc[idx], s_bench.loc[idx]
    rp = s_nav.pct_change(fill_method=None).dropna().to_numpy()
    rb = s_bench.pct_change(fill_method=None).dropna().to_numpy()
    m = min(len(rp), len(rb))
    rp, rb = rp[-m:], rb[-m:]

    n = len(rp)
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    ep, eb = rp - rf_d, rb - rf_d

    var_b = float(np.var(eb, ddof=1)) if n > 1 else 0.0
    beta = (float(np.cov(eb, ep, ddof=1)[0, 1]) / var_b) if var_b > 0 else float("nan")

    # 区间总收益（净值口径）
    total_ret = float(s_nav.iloc[-1] / s_nav.iloc[0] - 1)
    bench_ret = float(s_bench.iloc[-1] / s_bench.iloc[0] - 1)
    market_contrib = float(beta * bench_ret) if np.isfinite(beta) else float("nan")
    selection = total_ret - market_contrib if np.isfinite(market_contrib) else float("nan")

    # 年化 alpha（Jensen）
    alpha_d = float(np.mean(ep) - beta * np.mean(eb)) if np.isfinite(beta) else float("nan")
    alpha_annual = alpha_d * TRADING_DAYS

    # 解释度
    r2 = float("nan")
    if var_b > 0:
        resid = ep - alpha_d - beta * eb
        ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((ep - ep.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    years = n / TRADING_DAYS
    ann_p = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else float("nan")
    ann_b = (1 + bench_ret) ** (1 / years) - 1 if years > 0 and bench_ret > -1 else float("nan")
    vol = float(np.std(rp, ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = ((np.mean(rp) - rf_d) / np.std(rp, ddof=0) * np.sqrt(TRADING_DAYS)
              if np.std(rp, ddof=0) > 0 else float("nan"))
    mdd = max_drawdown(np.cumprod(1 + rp))

    # 胜率
    win_rate = float(np.mean(rp > 0))
    up_days = rb > 0
    win_when_up = float(np.mean(rp[up_days] > 0)) if up_days.any() else float("nan")

    if not np.isfinite(selection):
        verdict = "无法拆分（基准缺失或样本过短）"
    elif total_ret <= 0:
        verdict = "本期亏损；归因仅用于判断亏损来源"
    elif abs(selection) < 0.02:
        verdict = "收益主要来自市场，选股几乎没有贡献（接近被动持有）"
    elif selection > 0.02:
        verdict = "除市场外还有正贡献，需进一步检验是否为可复现的 alpha"
    else:
        verdict = "跑输市场：市场在涨，但选股拖了后腿"

    return {
        "n_days": n,
        "start": str(idx[0].date()) if hasattr(idx[0], "date") else str(idx[0]),
        "end": str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1]),
        "total_return": total_ret,
        "bench_return": bench_ret,
        "market_contribution": market_contrib,
        "selection_contribution": selection,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "r2": r2,
        "annual_return": ann_p,
        "bench_annual": ann_b,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "recovery_needed": recovery_needed(mdd),
        "win_rate": win_rate,
        "win_rate_on_up_days": win_when_up,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════
# 基准加载（多源兜底 + 缓存）
# ══════════════════════════════════════════════════════════
def load_bench(start: str | None = None, end: str | None = None,
               cache: _Path | str = BENCH_CACHE,
               refresh: bool = False) -> pd.Series:
    """
    加载沪深300 收盘价，返回以日期为索引的 Series。

    顺序：本地缓存 → akshare 多个接口（任一成功即写入缓存）。
    全部失败抛 RuntimeError，由调用方决定是否降级。
    """
    cache = _Path(cache)
    if cache.exists() and not refresh:
        try:
            df = pd.read_csv(cache, parse_dates=["date"])
            s = df.set_index("date")["close"].astype("float64").sort_index()
            if start:
                s = s.loc[str(start):]
            if end:
                s = s.loc[:str(end)]
            if len(s) > 30:
                return s
        except Exception:
            pass

    try:
        import akshare as ak
    except Exception as ex:
        raise RuntimeError(f"akshare 不可用：{ex}") from None

    s_start = start or "2016-01-01"
    s_end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    sources = [
        ("index_zh_a_hist", lambda: ak.index_zh_a_hist(
            symbol="000300", period="daily",
            start_date=pd.Timestamp(s_start).strftime("%Y%m%d"),
            end_date=pd.Timestamp(s_end).strftime("%Y%m%d"))),
        ("stock_zh_index_daily", lambda: ak.stock_zh_index_daily(symbol="sh000300")),
    ]
    last_err = ""
    for name, fn in sources:
        try:
            df = fn()
            if df is None or len(df) == 0:
                continue
            dcol = "日期" if "日期" in df.columns else (
                "date" if "date" in df.columns else df.columns[0])
            ccol = "收盘" if "收盘" in df.columns else (
                "close" if "close" in df.columns else df.columns[-1])
            s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").to_numpy(),
                          index=pd.to_datetime(df[dcol])).dropna().sort_index()
            if len(s) <= 30:
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            s.rename("close").rename_axis("date").to_csv(cache, float_format="%.4f")
            return s
        except Exception as ex:
            last_err = f"{name}: {ex}"
    raise RuntimeError(f"基准数据获取失败（{last_err}）")


# ══════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════
def format_report(attr: dict, cost: dict | None = None) -> str:
    L = ["=" * 66, "收益归因与成本核算", "=" * 66]
    L.append(f"  区间 {attr.get('start')} ~ {attr.get('end')}"
             f"（{attr.get('n_days')} 个交易日）")
    L.append("-" * 66)
    L.append(f"  组合收益      {attr['total_return'] * 100:>8.2f}%"
             f"    基准 {attr['bench_return'] * 100:>8.2f}%")
    L.append(f"  年化          {attr['annual_return'] * 100:>8.2f}%"
             f"    基准 {attr['bench_annual'] * 100:>8.2f}%")
    L.append(f"  波动率        {attr['volatility'] * 100:>8.2f}%")
    L.append(f"  夏普          {attr['sharpe']:>8.2f}")
    L.append(f"  最大回撤      {attr['max_drawdown'] * 100:>8.2f}%"
             f"   回本需涨 {attr['recovery_needed'] * 100:>6.1f}%")
    L.append("-" * 66)
    L.append("  【收益拆分】")
    L.append(f"  市场贡献      {attr['market_contribution'] * 100:>8.2f}%"
             f"   （Beta {attr['beta']:.2f} 跟来的）")
    L.append(f"  选股贡献      {attr['selection_contribution'] * 100:>8.2f}%"
             f"   （剔除市场后剩下的）")
    L.append(f"  年化 Alpha    {attr['alpha_annual'] * 100:>8.2f}%"
             f"   基准解释度 R² {attr['r2']:.2f}")
    L.append("-" * 66)
    L.append(f"  上涨日胜率    {attr['win_rate_on_up_days'] * 100:>8.1f}%"
             f"   整体胜率 {attr['win_rate'] * 100:>5.1f}%")
    L.append(f"  结论: {attr['verdict']}")
    if cost:
        L.append("-" * 66)
        L.append("  【成本核算】")
        L.append(f"  本金 {cost['capital']:,.0f}  年换手 {cost['turnover_annual']:.1f}x"
                 f"  双边成交额 {cost['amount_both']:,.0f}")
        L.append(f"  佣金 {cost['commission']:>8,.0f}   印花税 {cost['stamp_duty']:>8,.0f}"
                 f"   过户费 {cost['transfer_fee']:>7,.0f}   滑点 {cost['slippage']:>8,.0f}")
        L.append(f"  合计 {cost['total']:>8,.0f} 元   占本金 {cost['total_pct'] * 100:.2f}%")
    L.append("=" * 66)
    return "\n".join(L)


def format_cost_report(df: pd.DataFrame, capital: float) -> str:
    L = ["=" * 66, f"交易成本敏感性（本金 {capital:,.0f}）", "=" * 66]
    L.append(f"  {'年换手':>8}{'双边成交额':>14}{'年成本':>12}{'占本金':>10}")
    L.append("  " + "-" * 46)
    for _, r in df.iterrows():
        L.append(f"  {r['年换手']:>8.0f}{r['成交额']:>14,.0f}"
                 f"{r['总成本']:>12,.0f}{r['占本金%']:>9.2f}%")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    # 自测：成本 + 归因
    cap = 100000
    df = turnover_sensitivity(cap)
    print(format_cost_report(df, cap))

    rng = np.random.default_rng(7)
    days = pd.bdate_range("2023-01-02", periods=500)
    rb = rng.normal(0.0004, 0.010, len(days))
    rp = 0.6 * rb + rng.normal(0.0002, 0.008, len(days))
    nav = pd.Series(100000 * np.cumprod(1 + rp), index=days)
    bench = pd.Series(3000 * np.cumprod(1 + rb), index=days)
    attr = performance_attribution(nav, bench)
    print()
    print(format_report(attr, cost_analysis(cap, 12.0)))

    assert abs(recovery_needed(-0.30) - 0.4286) < 1e-3
    assert attr["beta"] > 0.3
    print(f"\nOK v{__version__}")
# ══════════════════════════════════════════════════════════
# M65 追加：截面归因（单票 / 分组 / Brinson / 对平检查）
# 原有绩效归因保持不变
# ══════════════════════════════════════════════════════════
TRADING_DAYS_ = 252


def by_stock(contrib: dict, capital: float = 1e6, min_pp: float = 0.0,
             top_n: int | None = None) -> pd.DataFrame:
    """按单票贡献（pp）。contrib 单位是元。"""
    if not contrib:
        return pd.DataFrame(columns=["code", "contrib_pp"])
    d = pd.DataFrame([{"code": k, "contrib_pp": v / float(capital) * 100}
                      for k, v in contrib.items()])
    d = d[d["contrib_pp"].abs() >= float(min_pp)]
    d = d.sort_values("contrib_pp", ascending=False)
    if top_n:
        d = d.head(int(top_n))
    return d.reset_index(drop=True)


def by_group(contrib: dict, mapping, capital: float = 1e6) -> pd.DataFrame:
    """按分组（行业/板块）聚合贡献。"""
    if not contrib:
        return pd.DataFrame(columns=["group", "n", "contrib_pp"])
    m = pd.Series(mapping).astype(str)
    rows = {}
    for code, v in contrib.items():
        g = m.get(str(code), "UNKNOWN")
        r = rows.setdefault(g, {"group": g, "n": 0, "contrib_pp": 0.0})
        r["n"] += 1
        r["contrib_pp"] += float(v) / float(capital) * 100
    d = pd.DataFrame(rows.values())
    return d.sort_values("contrib_pp", ascending=False).reset_index(drop=True)


def by_time(contrib_daily, freq: str = "ME") -> pd.DataFrame:
    """按时间聚合贡献。"""
    s = pd.Series(contrib_daily).dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["period", "contrib"])
    g = s.resample(freq).sum()
    return pd.DataFrame({"period": g.index.astype(str),
                         "contrib": g.to_numpy()})


def vs_benchmark(port_nav, bench_nav) -> dict:
    """相对基准：年化超额 / 跟踪误差 / 信息比率 / beta / alpha。"""
    p = pd.Series(port_nav).dropna()
    b = pd.Series(bench_nav).dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 20:
        return {"ok": False, "reason": "重叠区间不足"}
    p, b = p.loc[idx], b.loc[idx]
    rp = p.pct_change(fill_method=None).dropna()
    rb = b.pct_change(fill_method=None).dropna()
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    ann_p = float((p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1)
    ann_b = float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1)
    diff = rp - rb
    sd = float(diff.std(ddof=0))
    te = sd * np.sqrt(TRADING_DAYS_)
    ir = float(diff.mean() / sd * np.sqrt(TRADING_DAYS_)) if sd > 0 else 0.0
    var_b = float(rb.var(ddof=0))
    beta = float(rp.cov(rb) / var_b) if var_b > 0 else 0.0
    alpha_d = float(rp.mean() - beta * rb.mean())
    return {"ok": True, "n_days": int(len(idx)),
            "port_annual": ann_p, "bench_annual": ann_b,
            "excess_annual": ann_p - ann_b,
            "tracking_error": te, "info_ratio": ir, "beta": beta,
            "alpha_annual": alpha_d * TRADING_DAYS_,
            "win_rate": float((diff > 0).mean())}


def brinson(port_w, bench_w, port_r, bench_r) -> pd.DataFrame:
    """Brinson：配置 + 选择 + 交互。入参均按同一分组索引。"""
    idx = (pd.Series(port_w).index.union(pd.Series(bench_w).index)
           .union(pd.Series(port_r).index).union(pd.Series(bench_r).index))
    pw = pd.Series(port_w).reindex(idx).fillna(0.0)
    bw = pd.Series(bench_w).reindex(idx).fillna(0.0)
    pr = pd.Series(port_r).reindex(idx).fillna(0.0)
    br = pd.Series(bench_r).reindex(idx).fillna(0.0)
    rb_total = float((bw * br).sum())
    alloc = (pw - bw) * (br - rb_total)
    select = bw * (pr - br)
    inter = (pw - bw) * (pr - br)
    return pd.DataFrame({
        "group": idx,
        "port_w": pw.to_numpy(), "bench_w": bw.to_numpy(),
        "port_r": pr.to_numpy(), "bench_r": br.to_numpy(),
        "allocation": alloc.to_numpy(), "selection": select.to_numpy(),
        "interaction": inter.to_numpy(),
        "total": (alloc + select + inter).to_numpy(),
    }).sort_values("total", ascending=False).reset_index(drop=True)


def recombination_check(contrib: dict, port_total_return: float,
                        capital: float = 1e6) -> dict:
    """归因对平：逐票贡献之和 vs 策略实际总收益。"""
    s = float(sum(contrib.values())) / float(capital) if contrib else 0.0
    diff = s - float(port_total_return)
    return {"attributed": round(s, 6), "actual": round(float(port_total_return), 6),
            "gap": round(diff, 6), "gap_pp": round(diff * 100, 4),
            "ok": abs(diff) < 0.01}


def format_cross_report(stock: pd.DataFrame,
                        group: pd.DataFrame | None = None,
                        bench: dict | None = None,
                        recon: dict | None = None) -> str:
    """截面归因报告（与原有的 format_report 并存）。"""
    L = ["=" * 76, "截面归因", "=" * 76]
    if stock is not None and len(stock):
        top = stock[stock["contrib_pp"] > 0].head(5)
        bot = stock[stock["contrib_pp"] < 0].tail(5)
        if len(top):
            L.append("  贡献最大（pp）：")
            for _, r in top.iterrows():
                L.append(f"    {r['code']:<10}{r['contrib_pp']:>+8.2f}")
        if len(bot):
            L.append("  拖累最大（pp）：")
            for _, r in bot.iterrows():
                L.append(f"    {r['code']:<10}{r['contrib_pp']:>+8.2f}")
    if group is not None and len(group):
        L.append("-" * 76)
        L.append("  按分组：")
        for _, r in group.iterrows():
            L.append(f"    {str(r['group'])[:12]:<14}n={r['n']:<4}"
                     f"{r['contrib_pp']:>+8.2f}pp")
    if bench and bench.get("ok"):
        L.append("-" * 76)
        L.append(f"  相对基准：超额 {bench['excess_annual']*100:+.2f}pp/年   "
                 f"TE {bench['tracking_error']*100:.2f}%   "
                 f"IR {bench['info_ratio']:.2f}")
        L.append(f"  beta {bench['beta']:.2f}   "
                 f"alpha {bench['alpha_annual']*100:+.2f}pp/年   "
                 f"胜率 {bench['win_rate']*100:.1f}%")
    if recon:
        L.append("-" * 76)
        L.append(f"  归因对平：逐票合计 {recon['attributed']*100:+.2f}%  "
                 f"实际 {recon['actual']*100:+.2f}%  "
                 f"差 {recon['gap_pp']:+.2f}pp "
                 f"{'✓' if recon['ok'] else '✗ 口径需查'}")
    L.append("=" * 76)
    return "\n".join(L)
