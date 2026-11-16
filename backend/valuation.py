# -*- coding: utf-8 -*-
"""
估值分位（M8）
====================================================
价值投资里唯一可量化的东西：**现在贵不贵**。

做法：拿到历史 PE/PB 序列，算「当前值在过去 N 年中的百分位」。

    分位 10% → 比过去 90% 的时间都便宜
    分位 90% → 比过去 90% 的时间都贵

数据源（akshare，均带本地缓存）
-------------------------------
* 指数 PE：`stock_index_pe_lg`（沪深300 等，2005 年至今，月度）
* 指数 PB：`stock_index_pb_lg`
* 个股估值：`stock_zh_valuation_baidu`（市盈率TTM / 市净率 等，日频）

注意：估值分位回答的是"贵不贵"，**不回答"会不会涨"**。
低分位可以更便宜，也可以继续更便宜（价值陷阱）。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = [
    "Zones", "DEFAULTS_ZONES", "percentile_rank", "valuation_stats",
    "zone_from_pct", "fetch_index_pe", "fetch_index_pb", "fetch_stock_pe",
    "valuation_report", "format_report",
]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "valuation"
CACHE_DAYS = 3          # 缓存有效期（天）

INDEX_PE_COL = "滚动市盈率"
INDEX_PB_COL = "市净率"
STOCK_INDICATORS = ("市盈率(TTM)", "市净率", "市销率", "股息率")


class Zones:
    """分位 → 估值区间。阈值为经验值，可整体传入覆盖。"""

    def __init__(self, cheap: float = 20.0, low: float = 40.0,
                 high: float = 60.0, expensive: float = 80.0):
        self.cheap, self.low, self.high, self.expensive = cheap, low, high, expensive


DEFAULTS_ZONES = Zones()


# ══════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════
def _clean(series) -> pd.Series:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    return s.sort_index()


def percentile_rank(series, value: float | None = None,
                    lookback_years: float | None = None) -> float:
    """
    当前值在历史序列中的百分位（0~100）。

    >50 表示比历史多数时间更贵。
    value 缺省取序列最后一个点；lookback_years 缺省用全部历史。
    """
    s = _clean(series)
    if len(s) < 5:            # 纯统计只需少量点；20 点的门槛留给 valuation_stats
        return float("nan")
    if lookback_years:
        cutoff = s.index[-1] - pd.Timedelta(days=int(365.25 * lookback_years))
        s = s.loc[cutoff:]
        if len(s) < 20:
            return float("nan")
    v = float(s.iloc[-1]) if value is None else float(value)
    if not np.isfinite(v):
        return float("nan")
    return float((s < v).sum() / len(s) * 100.0)


def valuation_stats(series, lookback_years: float | None = None,
                    zones: Zones = DEFAULTS_ZONES) -> dict:
    """
    估值统计：当前值、分位、历史区间、区间定性。

    返回 dict（不可用时 ok=False）。
    """
    s = _clean(series)
    if lookback_years:
        cutoff = s.index[-1] - pd.Timedelta(days=int(365.25 * lookback_years))
        s = s.loc[cutoff:]
    if len(s) < 20:
        return {"ok": False, "reason": f"样本仅 {len(s)} 个点，至少需要 20 个"}

    cur = float(s.iloc[-1])
    pct = percentile_rank(s, cur)
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "ok": True,
        "n": int(len(s)),
        "start": str(s.index[0].date()) if hasattr(s.index[0], "date") else str(s.index[0]),
        "end": str(s.index[-1].date()) if hasattr(s.index[-1], "date") else str(s.index[-1]),
        "current": round(cur, 4),
        "percentile": round(pct, 2),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "p10": round(float(q.loc[0.1]), 4),
        "p25": round(float(q.loc[0.25]), 4),
        "median": round(float(q.loc[0.5]), 4),
        "p75": round(float(q.loc[0.75]), 4),
        "p90": round(float(q.loc[0.9]), 4),
        "zone": zone_from_pct(pct, zones),
    }


def zone_from_pct(pct: float, zones: Zones = DEFAULTS_ZONES) -> str:
    """分位 → 中文定性。"""
    if pct != pct:
        return "无法判断"
    if pct < zones.cheap:
        return "低估区（历史便宜位）"
    if pct < zones.low:
        return "偏低（低于历史中位）"
    if pct <= zones.high:
        return "中性（历史中位附近）"
    if pct <= zones.expensive:
        return "偏高（高于历史中位）"
    return "高估区（历史昂贵位）"


# ══════════════════════════════════════════════════════════
# 取数（带缓存）
# ══════════════════════════════════════════════════════════
def _cache_path(kind: str, symbol: str) -> Path:
    safe = "".join(ch for ch in str(symbol) if ch.isalnum() or ch in "-_")
    return CACHE_DIR / f"{kind}_{safe}.csv"


def _read_cache(fp: Path, max_age_days: int = CACHE_DAYS) -> pd.Series | None:
    if not fp.exists():
        return None
    age_days = (time.time() - fp.stat().st_mtime) / 86400
    if age_days > max_age_days:
        return None
    try:
        df = pd.read_csv(fp, parse_dates=["date"])
        s = df.set_index("date")["value"].astype("float64")
        return s if len(s) >= 20 else None
    except Exception:
        return None


def _write_cache(fp: Path, s: pd.Series) -> None:
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        s.rename("value").rename_axis("date").to_csv(fp, float_format="%.6f")
    except Exception:
        pass


def _pick_col(df: pd.DataFrame, prefer: list[str]) -> str | None:
    cols = [str(c) for c in df.columns]
    for p in prefer:
        if p in cols:
            return p
    # 退化：找第一个能转成数值且不是日期的列
    for c in cols:
        if c in ("日期", "date", "时间", "指数"):
            continue
        try:
            pd.to_numeric(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None


def _series_from_df(df: pd.DataFrame, prefer_cols: list[str]) -> pd.Series:
    dcol = next((c for c in ("日期", "date", "时间") if c in df.columns), df.columns[0])
    ccol = _pick_col(df, prefer_cols)
    if ccol is None:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(df[dcol], errors="coerce")
    val = pd.to_numeric(df[ccol], errors="coerce")
    return pd.Series(val.to_numpy(), index=idx).dropna().sort_index()


def _fetch(kind: str, symbol: str, builder, prefer_cols: list[str],
           refresh: bool = False) -> pd.Series:
    fp = _cache_path(kind, symbol)
    if not refresh:
        s = _read_cache(fp)
        if s is not None:
            return s
    try:
        df = builder()
    except Exception as ex:
        raise RuntimeError(f"取数失败（{kind}/{symbol}）：{ex}") from None
    if df is None or len(df) == 0:
        raise RuntimeError(f"取数为空（{kind}/{symbol}）")
    s = _series_from_df(df, prefer_cols)
    if len(s) < 20:
        raise RuntimeError(f"数据点太少（{kind}/{symbol}，{len(s)} 个）")
    _write_cache(fp, s)
    return s


def fetch_index_pe(symbol: str = "沪深300", refresh: bool = False) -> pd.Series:
    """指数滚动市盈率（月度，2005 年至今）。"""
    import akshare as ak

    return _fetch("index_pe", symbol,
                  lambda: ak.stock_index_pe_lg(symbol=symbol),
                  [INDEX_PE_COL, "静态市盈率"], refresh)


def fetch_index_pb(symbol: str = "沪深300", refresh: bool = False) -> pd.Series:
    """指数市净率（月度）。"""
    import akshare as ak

    return _fetch("index_pb", symbol,
                  lambda: ak.stock_index_pb_lg(symbol=symbol),
                  [INDEX_PB_COL, "市净率中位数"], refresh)


def fetch_stock_pe(code: str, indicator: str = "市盈率(TTM)",
                   refresh: bool = False) -> pd.Series:
    """个股估值序列（日频，最长约 3 年）。"""
    import akshare as ak

    if indicator not in STOCK_INDICATORS:
        raise ValueError(f"indicator 必须是 {STOCK_INDICATORS} 之一")
    return _fetch(f"stock_{indicator}", code,
                  lambda: ak.stock_zh_valuation_baidu(
                      symbol=code, indicator=indicator, period="全部"),
                  ["value", "数值"], refresh)


# ══════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════
def valuation_report(kind: str = "index", symbol: str = "沪深300",
                     metric: str = "pe", refresh: bool = False) -> dict:
    """
    一站式估值报告。

    kind   : "index" 或 "stock"
    metric : "pe" / "pb"（index）；个股固定按 indicator 取
    """
    try:
        if kind == "index":
            s = (fetch_index_pe(symbol, refresh) if metric == "pe"
                 else fetch_index_pb(symbol, refresh))
            label = f"{symbol} {'PE(TTM)' if metric == 'pe' else 'PB'}"
        else:
            s = fetch_stock_pe(symbol, metric if metric in STOCK_INDICATORS
                               else "市盈率(TTM)", refresh)
            label = f"{symbol} {metric if metric in STOCK_INDICATORS else '市盈率(TTM)'}"
    except (RuntimeError, ValueError) as ex:
        return {"ok": False, "reason": str(ex), "label": f"{symbol}"}

    full = valuation_stats(s)
    windows = {}
    for yrs in (3, 5, 10):
        st = valuation_stats(s, lookback_years=yrs)
        windows[f"{yrs}年"] = {"percentile": st.get("percentile"),
                              "zone": st.get("zone")} if st.get("ok") else None

    if not full.get("ok"):
        return {"ok": False, "reason": full.get("reason"), "label": label}

    pct = full["percentile"]
    if pct < 20:
        verdict = "历史上很便宜：适合分批建仓，但要先确认不是价值陷阱（基本面恶化）"
    elif pct < 40:
        verdict = "偏便宜：可逐步买入，留出现金应对更低位置"
    elif pct <= 60:
        verdict = "估值合理：按计划持有，不必追高也不必恐慌"
    elif pct <= 80:
        verdict = "偏贵：慎追，可考虑降低新增仓位速度"
    else:
        verdict = "历史上很贵：新增仓位应非常谨慎，注意回撤风险"

    return {"ok": True, "label": label, "stats": full, "windows": windows,
            "verdict": verdict, "series_start": full["start"],
            "series_end": full["end"]}


def format_report(rep: dict) -> str:
    L = ["=" * 66, "估值分位报告", "=" * 66]
    if not rep.get("ok"):
        L.append(f"  {rep.get('label', '')}  无法评估：{rep.get('reason', '')}")
        L.append("=" * 66)
        return "\n".join(L)

    s = rep["stats"]
    L.append(f"  {rep['label']}   区间 {s['start']} ~ {s['end']}（{s['n']} 个点）")
    L.append("-" * 66)
    L.append(f"  当前值        {s['current']:>10.2f}")
    L.append(f"  历史分位      {s['percentile']:>10.1f}%   ← 比过去 {s['percentile']:.0f}% 的时间贵")
    L.append(f"  区间定性      {s['zone']}")
    L.append("-" * 66)
    L.append(f"  历史分布      P10 {s['p10']:.2f} | P25 {s['p25']:.2f} | "
             f"中位 {s['median']:.2f} | P75 {s['p75']:.2f} | P90 {s['p90']:.2f}")
    L.append(f"  极值          最低 {s['min']:.2f}   最高 {s['max']:.2f}")
    L.append("-" * 66)
    L.append("  分窗口：")
    for k, v in (rep.get("windows") or {}).items():
        if v:
            L.append(f"    {k:<4} 分位 {v['percentile']:>5.1f}%   {v['zone']}")
        else:
            L.append(f"    {k:<4} 数据不足")
    L.append("-" * 66)
    L.append(f"  结论: {rep['verdict']}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 离线自测：造一段序列验证分位逻辑
    idx = pd.bdate_range("2015-01-01", periods=2000)
    rng = np.random.default_rng(7)
    fake = pd.Series(rng.normal(15, 3, len(idx)).cumsum() * 0.05 + 15, index=idx)
    st = valuation_stats(fake)
    assert st["ok"] and 0 <= st["percentile"] <= 100
    assert abs(percentile_rank([1, 2, 3, 4, 5], 5.0) - 80.0) < 1e-9
    assert abs(percentile_rank([1, 2, 3, 4, 5], 0.0) - 0.0) < 1e-9
    assert zone_from_pct(5) == "低估区（历史便宜位）"
    assert zone_from_pct(90) == "高估区（历史昂贵位）"
    print(f"OK v{__version__}  离线自测通过")
