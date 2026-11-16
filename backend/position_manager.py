# -*- coding: utf-8 -*-
"""
仓位管理器（M47）：按大盘趋势动态限制最大仓位
====================================================
规则（按需求）
--------------
* 沪深300 在 **20 日均线下方** → 最大仓位 **30%**
* 沪深300 在 **20 日均线上方** → 最大仓位 **80%**
* 其余资金留在现金

为什么需要它
------------
策略给的是「相对权重」，但**市场环境决定该不该满仓**。
2022-2024 的实测已经证明：同样的因子在熊市里会被系统性拖累。
趋势过滤是最朴素、最不依赖预测的降风险手段。

⚠ 三个必须说清的点
------------------
1. **会有频繁翻转**。20 日均线穿越很常见，每次翻转都带来换手成本。
   本模块提供两个抑制手段（默认关闭，保持与需求一致）：
   * `band`：均线上下 ±N% 的缓冲带，价格在带内维持原状态
   * `confirm`：连续 N 日站在同一侧才切换
2. **趋势过滤不是万能的**。震荡市里它会被反复打脸（假突破），
   历史上胜率并不高，它的价值在于**降低尾部风险**而不是提高收益。
3. **无未来函数**：第 t 日收盘算出均线状态，影响的是第 t+1 日开盘的仓位。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["DEFAULT_CAP_LOW", "DEFAULT_CAP_HIGH", "DEFAULT_WINDOW",
           "load_index", "ma_state", "position_cap", "apply_cap",
           "cap_summary", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "index"
CACHE_MAX_AGE_HOURS = 12

DEFAULT_WINDOW = 20
DEFAULT_CAP_LOW = 0.30
DEFAULT_CAP_HIGH = 0.80
DEFAULT_BAND = 0.0        # 0 = 严格按需求（穿线即切换）
DEFAULT_CONFIRM = 1       # 1 = 当日确认


def load_index(code: str = "sh000300", refresh: bool = False) -> pd.Series:
    """
    取指数收盘价序列（12 小时缓存）。

    数据源：akshare `stock_zh_index_daily`（新浪）。
    """
    import akshare as ak
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = str(code).replace("/", "_")
    fp = CACHE_DIR / f"{safe}.csv"
    if not refresh and fp.exists():
        try:
            if (time.time() - fp.stat().st_mtime) / 3600 <= CACHE_MAX_AGE_HOURS:
                d = pd.read_csv(fp, parse_dates=["date"])
                if len(d) > 100:
                    return d.set_index("date")["close"].sort_index()
        except Exception:
            pass
    raw = ak.stock_zh_index_daily(symbol=str(code))
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"指数 {code} 取数失败")
    d = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
    }).dropna()
    d = d[d["close"] > 0].drop_duplicates("date").sort_values("date")
    d.to_csv(fp, index=False)
    return d.set_index("date")["close"]


def ma_state(close: pd.Series, window: int = DEFAULT_WINDOW,
             band: float = DEFAULT_BAND,
             confirm: int = DEFAULT_CONFIRM) -> pd.DataFrame:
    """
    计算每日的均线状态。

    返回 DataFrame：ma、above（是否在均线上方）、state（1/0）。
    """
    s = pd.Series(close).astype(float).dropna()
    ma = s.rolling(int(window), min_periods=int(window)).mean()
    diff = s / ma - 1.0

    raw_above = (diff > float(band)).astype(int)
    raw_below = (diff < -float(band)).astype(int)

    state = np.zeros(len(s), dtype=int)
    cur = 0
    run_above = run_below = 0
    for i in range(len(s)):
        if not np.isfinite(ma.iloc[i]):
            state[i] = 0
            continue
        if raw_above.iloc[i]:
            run_above += 1
            run_below = 0
        elif raw_below.iloc[i]:
            run_below += 1
            run_above = 0
        else:
            run_above = run_below = 0
            state[i] = cur
            continue
        if run_above >= int(confirm):
            cur = 1
        elif run_below >= int(confirm):
            cur = 0
        state[i] = cur

    out = pd.DataFrame({"close": s, "ma": ma, "diff": diff,
                        "state": state}, index=s.index)
    out["above"] = out["state"].astype(bool)
    return out


def position_cap(close: pd.Series, window: int = DEFAULT_WINDOW,
                 cap_low: float = DEFAULT_CAP_LOW,
                 cap_high: float = DEFAULT_CAP_HIGH,
                 band: float = DEFAULT_BAND,
                 confirm: int = DEFAULT_CONFIRM) -> pd.DataFrame:
    """每日最大仓位上限（0~1）。均线下方 cap_low，上方 cap_high。"""
    if not (0 <= cap_low <= cap_high <= 1.0):
        raise ValueError(f"仓位参数不合法：{cap_low} ~ {cap_high}")
    st = ma_state(close, window, band, confirm)
    st["cap"] = np.where(st["state"] > 0, float(cap_high), float(cap_low))
    # 均线尚未形成的前 window 天：保守用 cap_low
    st.loc[st["ma"].isna(), "cap"] = float(cap_low)
    return st


def apply_cap(weights: pd.DataFrame,
              caps: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    按日把权重总和压到上限以内（等比例缩放，不改变相对排序）。
    """
    w = weights.copy()
    if isinstance(caps, pd.DataFrame):
        cap_series = caps["cap"]
    else:
        cap_series = pd.Series(caps)

    out = w.copy()
    for d in out.index:
        row = out.loc[d]
        tot = float(row.sum())
        if tot <= 0:
            continue
        # 取「不晚于该日」的上限（信号日与上限日对齐）
        cap = np.nan
        try:
            avail = cap_series.loc[:d]
            if len(avail):
                cap = float(avail.iloc[-1])
        except Exception:
            cap = np.nan
        if not np.isfinite(cap):
            continue
        cap = float(np.clip(cap, 0.0, 1.0))
        if tot > cap + 1e-12:
            out.loc[d] = row * (cap / tot)
    return out


def cap_summary(caps: pd.DataFrame) -> dict:
    """仓位上限的统计：高低占比、切换次数、平均上限。"""
    st = caps["state"].to_numpy()
    switches = int(np.sum(np.abs(np.diff(st)) > 0))
    n_valid = int(np.isfinite(caps["ma"]).sum())
    return {"n_days": int(len(caps)),
            "n_valid_ma": n_valid,
            "above_days": int((st > 0).sum()),
            "below_days": int((st <= 0).sum()),
            "above_pct": round(float((st > 0).mean()) * 100, 2),
            "switches": switches,
            "switches_per_year": (round(switches / max(len(caps) / 252.0, 1e-9), 1)),
            "avg_cap": round(float(caps["cap"].mean()), 4),
            "current_state": ("均线上方" if st[-1] > 0 else "均线下方"),
            "current_cap": round(float(caps["cap"].iloc[-1]), 4),
            "as_of": str(caps.index[-1].date())}


def format_report(caps: pd.DataFrame, sm: dict,
                  window: int = DEFAULT_WINDOW,
                  cap_low: float = DEFAULT_CAP_LOW,
                  cap_high: float = DEFAULT_CAP_HIGH) -> str:
    L = ["=" * 70, "仓位管理器（大盘趋势过滤）", "=" * 70]
    L.append(f"  标的：沪深300   均线：{window} 日")
    L.append(f"  规则：均线上方 → 最大 {cap_high*100:.0f}%    "
             f"下方 → 最大 {cap_low*100:.0f}%")
    L.append("-" * 70)
    last = caps.iloc[-1]
    L.append(f"  截至 {sm['as_of']}")
    L.append(f"    收盘 {float(last['close']):>10,.2f}   "
             f"均线 {float(last['ma']):>10,.2f}   "
             f"偏离 {float(last['diff'])*100:+.2f}%")
    L.append(f"    当前状态：{sm['current_state']}   →  "
             f"最大仓位 {sm['current_cap']*100:.0f}%")
    L.append("-" * 70)
    L.append(f"  历史统计（共 {sm['n_days']} 天，均线有效 {sm['n_valid_ma']} 天）")
    L.append(f"    上方天数 {sm['above_days']:>5}（{sm['above_pct']:.1f}%）"
             f"   下方天数 {sm['below_days']:>5}")
    L.append(f"    状态切换 {sm['switches']} 次  "
             f"≈ 每年 {sm['switches_per_year']} 次")
    L.append(f"    平均仓位上限 {sm['avg_cap']*100:.1f}%")
    L.append("=" * 70)
    L.append("  ⚠ 趋势过滤降低尾部风险，但震荡市会被反复打脸；")
    L.append("     切换越频繁，换手成本越高。可用 band / confirm 抑制。")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # 离线自测：构造确定性序列
    idx = pd.bdate_range("2020-01-02", periods=120)
    up = pd.Series(np.linspace(100, 140, 120), index=idx)      # 一路向上
    down = pd.Series(np.linspace(140, 100, 120), index=idx)    # 一路向下

    c_up = position_cap(up)
    c_dn = position_cap(down)
    print(f"单边上涨：均线上方占比 {cap_summary(c_up)['above_pct']:.1f}%  "
          f"平均上限 {cap_summary(c_up)['avg_cap']*100:.1f}%")
    print(f"单边下跌：均线上方占比 {cap_summary(c_dn)['above_pct']:.1f}%  "
          f"平均上限 {cap_summary(c_dn)['avg_cap']*100:.1f}%")
    assert cap_summary(c_up)["avg_cap"] > cap_summary(c_dn)["avg_cap"], \
        "上涨行情应给出更高仓位上限"
    assert abs(cap_summary(c_dn)["avg_cap"] - 0.30) < 1e-6

    # 参数校验
    try:
        position_cap(up, cap_low=0.9, cap_high=0.3)
        raise AssertionError("非法参数应报错")
    except ValueError:
        print("非法仓位参数正确报错 ✓")

    # apply_cap
    w = pd.DataFrame([{"A": 0.5, "B": 0.5}], index=[idx[30]])
    capped = apply_cap(w, c_up)
    tot = float(capped.iloc[0].sum())
    print(f"权重 1.0 → 上限内 {tot:.2f}（上限 {c_up.loc[idx[30], 'cap']:.2f}）")
    assert tot <= c_up.loc[idx[30], "cap"] + 1e-9
    assert abs(capped.iloc[0, 0] - capped.iloc[0, 1]) < 1e-12, "应等比例缩放"

    # band / confirm 应减少切换
    zig = pd.Series(np.tile([100, 130], 60), index=idx[:120])
    a = cap_summary(position_cap(zig))
    b = cap_summary(position_cap(zig, band=0.05, confirm=3))
    print(f"锯齿行情：切换 {a['switches']} 次 → 加缓冲后 {b['switches']} 次")
    assert b["switches"] <= a["switches"]

    print()
    print(format_report(c_up, cap_summary(c_up)))
    print("\nOK v" + __version__)
