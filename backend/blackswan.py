# -*- coding: utf-8 -*-
"""
黑天鹅与事件风险（M9）
====================================================
先说清楚一件事：**没有人能预测黑天鹅。**

所以这个模块不做"预测"，只做三件能做对的事：

1. **事件日历**——把「日期确定、影响不确定」的事件先标出来
   （美国大选/中期选举按规则可算，A股财报披露截止日固定）
2. **历史情景回放**——用过去真实的冲击窗口，量出你的组合会疼多少
3. **尾部风险度量**——VaR / CVaR / 偏度 / 峰度 / 最差单日 / 恢复天数

核心区别：
    ❌ 不预测"选举结果会怎样影响哪个板块"（这类因果链极不稳定）
    ✅ 只回答"如果再来一次 2024 年 1 月那种崩盘，我会亏多少、要几年恢复"
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = [
    "us_election_days", "a_share_report_deadlines", "event_calendar",
    "HISTORICAL_SHOCKS", "tail_statistics", "stress_windows",
    "fragility_score", "format_report",
]


# ══════════════════════════════════════════════════════════
# 一、确定性事件日历（规则可算，不依赖数据源）
# ══════════════════════════════════════════════════════════
def _first_monday(year: int, month: int = 11) -> date:
    d = date(year, month, 1)
    while d.weekday() != 0:            # 0 = Monday
        d += timedelta(days=1)
    return d


def us_election_days(start_year: int = 2000, end_year: int = 2040) -> list[dict]:
    """
    美国大选与中期选举日期（规则：11 月第一个星期一之后的星期二）。

    * 年份 % 4 == 0 → 总统大选
    * 年份 % 4 == 2 → 中期选举
    两者都在同一个规则日。日期是确定的，**结果与市场反应不可预测**。
    """
    out = []
    for y in range(int(start_year), int(end_year) + 1):
        if y % 2:
            continue
        d = _first_monday(y) + timedelta(days=1)
        out.append({
            "date": d.isoformat(),
            "kind": "美国大选" if y % 4 == 0 else "美国中期选举",
            "year": y,
            "note": "日期确定、结果不确定；市场反应方向不可预测",
        })
    return out


def a_share_report_deadlines(start_year: int = 2020, end_year: int = 2030) -> list[dict]:
    """
    A 股定期报告披露截止日（法定，规则可算）。

    * 年报：4 月 30 日
    * 半年报：8 月 31 日
    * 三季报：10 月 31 日
    """
    out = []
    for y in range(int(start_year), int(end_year) + 1):
        out += [
            {"date": f"{y}-04-30", "kind": "年报披露截止", "year": y,
             "note": "业绩雷高发期，注意预告与商誉减值"},
            {"date": f"{y}-08-31", "kind": "半年报披露截止", "year": y,
             "note": "中报集中披露，易出现预期修正"},
            {"date": f"{y}-10-31", "kind": "三季报披露截止", "year": y,
             "note": "全年指引的第一次强信号"},
        ]
    return out


def event_calendar(as_of: str | date | None = None,
                   months_ahead: int = 12,
                   include_a_share: bool = True) -> list[dict]:
    """
    未来 N 个月内的事件（按日期升序）。

    只收录**日期确定**的事件；不含任何"可能发生"的猜测。
    """
    today = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    horizon = today + pd.DateOffset(months=int(months_ahead))

    items: list[dict] = []
    items += us_election_days(today.year - 1, horizon.year + 1)
    if include_a_share:
        items += a_share_report_deadlines(today.year, horizon.year + 1)

    out = []
    for it in items:
        d = pd.Timestamp(it["date"])
        if today <= d <= horizon:
            days = (d - today).days
            out.append({**it, "days_until": days})
    out.sort(key=lambda x: x["date"])
    return out


# ══════════════════════════════════════════════════════════
# 二、历史冲击窗口库
# ══════════════════════════════════════════════════════════
HISTORICAL_SHOCKS = {
    "2015 股灾（去杠杆）": ("2015-06-12", "2015-08-26"),
    "2016 熔断": ("2016-01-04", "2016-01-28"),
    "2018 贸易摩擦": ("2018-01-24", "2018-12-28"),
    "2020 新冠冲击": ("2020-01-20", "2020-03-23"),
    "2022 俄乌+加息": ("2022-01-05", "2022-04-26"),
    "2024 微盘股流动性危机": ("2024-01-01", "2024-02-08"),
    "2024 关税冲击": ("2024-10-08", "2024-10-18"),
}


# ══════════════════════════════════════════════════════════
# 三、尾部风险
# ══════════════════════════════════════════════════════════
def _returns(series, is_nav: bool | None = None) -> pd.Series:
    """
    统一成日收益。

    判别净值 vs 收益：用**变异系数** std/|mean|。
      * 净值：平滑，变异系数小（通常 < 0.5）
      * 收益：均值接近 0，变异系数极大（几十以上）

    注意：不能用「中位数 > 2」这类阈值——
    归一化净值从 1.0 起步，永远到不了 2，会被误判成收益，
    导致"最差单日 +50%"这种荒谬结果。
    """
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if len(s) < 3:
        return s

    level = is_nav
    if level is None:
        mean_abs = abs(float(s.mean()))
        cv = float(s.std(ddof=0)) / mean_abs if mean_abs > 1e-12 else float("inf")
        pos_ratio = float((s > 0).mean())
        level = bool(pos_ratio > 0.95 and cv < 0.5)

    return s.pct_change(fill_method=None).dropna() if level else s


def tail_statistics(returns, is_nav: bool | None = None) -> dict:
    """
    尾部风险指标。

    * VaR95 / VaR99  : 单日损失的 95% / 99% 分位（负数）
    * CVaR95         : 超过 VaR 的平均损失（更保守）
    * 偏度 / 峰度     : 负偏 + 高峰度 = 尾部风险大
    * 最差单日/单周   : 历史极端值
    * 恢复交易日数    : 从最大回撤底部回到前高所需交易日（未恢复则为 None）
    """
    r = _returns(returns, is_nav)
    if len(r) < 30:
        return {"ok": False, "reason": f"样本仅 {len(r)} 天，至少需要 30 天"}

    a = r.to_numpy()
    var95 = float(np.percentile(a, 5))
    var99 = float(np.percentile(a, 1))
    tail = a[a <= var95]
    cvar95 = float(tail.mean()) if tail.size else var95
    skew = float(pd.Series(a).skew())
    kurt = float(pd.Series(a).kurt())

    nav = np.cumprod(1 + a)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.where(peak <= 0, np.nan, peak) - 1
    mdd = float(np.nanmin(dd))
    trough_i = int(np.nanargmin(dd))
    recover = None
    for i in range(trough_i, len(nav)):
        if nav[i] >= peak[trough_i]:
            recover = i - trough_i
            break

    week = r.resample("W").apply(lambda x: (1 + x).prod() - 1) if isinstance(
        r.index, pd.DatetimeIndex) else pd.Series(dtype="float64")

    return {
        "ok": True, "n": int(len(r)),
        "var95": round(var95, 6), "var99": round(var99, 6),
        "cvar95": round(cvar95, 6),
        "skew": round(skew, 4), "kurtosis": round(kurt, 4),
        "worst_day": round(float(a.min()), 6),
        "best_day": round(float(a.max()), 6),
        "worst_week": round(float(week.min()), 6) if len(week) else None,
        "max_drawdown": round(mdd, 6),
        "recovery_days": recover,
        "excess_kurtosis": round(kurt, 4),
    }


def stress_windows(nav, shocks: dict | None = None) -> list[dict]:
    """
    历史情景回放：每个冲击窗口内，组合与沪深300 的表现。

    只用到净值序列自身的日期；不覆盖的窗口标为未覆盖（不扣分）。
    """
    s = pd.Series(nav).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("净值索引必须是日期（DatetimeIndex）")

    shocks = shocks or HISTORICAL_SHOCKS
    out = []
    for name, (ds, de) in shocks.items():
        win = s.loc[ds:de]
        if len(win) < 2:
            out.append({"name": name, "window": f"{ds} ~ {de}", "ok": False,
                        "reason": "净值未覆盖该区间"})
            continue
        r = win.pct_change(fill_method=None).dropna().to_numpy()
        nav_w = np.cumprod(1 + r)
        peak = np.maximum.accumulate(nav_w)
        mdd = float(np.nanmin(nav_w / np.where(peak <= 0, np.nan, peak) - 1))
        out.append({
            "name": name, "window": f"{ds} ~ {de}", "ok": True,
            "days": int(len(win)),
            "return": round(float(nav_w[-1] - 1), 6),
            "max_drawdown": round(mdd, 6),
            "worst_day": round(float(r.min()), 6),
        })
    out.sort(key=lambda x: (not x.get("ok"), x.get("return", 0)))
    return out


def fragility_score(tail: dict, stress: list[dict] | None = None) -> dict:
    """
    脆弱度评分（0~100，越高越脆弱）。

    评分构成（各占权重）：
      * 回撤深度 25     * 单日最差 20
      * VaR99    20     * 负偏/高峰度 20
      * 历史情景平均跌幅 15

    **这是"你会疼多少"的度量，不是"会不会发生"的预测。**
    """
    if not tail.get("ok"):
        return {"ok": False, "reason": tail.get("reason")}

    def ramp(x, lo, hi):
        if not np.isfinite(x):
            return 0.0
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return (x - lo) / (hi - lo)

    mdd = ramp(abs(tail["max_drawdown"]), 0.05, 0.50) * 25
    wd = ramp(abs(tail["worst_day"]), 0.01, 0.10) * 20
    v99 = ramp(abs(tail["var99"]), 0.01, 0.08) * 20
    tail_shape = (ramp(-tail["skew"], 0.0, 1.5) * 0.5
                  + ramp(tail["kurtosis"], 0.0, 6.0) * 0.5) * 20

    scen = 0.0
    scored = [x for x in (stress or []) if x.get("ok")]
    if scored:
        avg = float(np.mean([x["return"] for x in scored]))
        scen = ramp(abs(min(avg, 0)), 0.02, 0.30) * 15

    score = float(np.clip(mdd + wd + v99 + tail_shape + scen, 0, 100))
    if score < 30:
        level, advice = "低", "尾部风险较小，维持现有纪律即可"
    elif score < 55:
        level, advice = "中", "存在一定尾部暴露，建议保留现金缓冲"
    elif score < 75:
        level, advice = "较高", "尾部风险偏高，建议降低集中度、设定组合级止损"
    else:
        level, advice = "高", "对极端行情极不抗打，必须降低仓位或做对冲"

    return {"ok": True, "score": round(score, 2), "level": level, "advice": advice,
            "breakdown": {"回撤深度": round(mdd, 2), "单日最差": round(wd, 2),
                          "VaR99": round(v99, 2), "尾部形状": round(tail_shape, 2),
                          "历史情景": round(scen, 2)},
            "recovery_days": tail.get("recovery_days")}


# ══════════════════════════════════════════════════════════
# 四、报告
# ══════════════════════════════════════════════════════════
def format_report(tail: dict, events: list[dict] | None = None,
                  stress: list[dict] | None = None,
                  frag: dict | None = None) -> str:
    L = ["=" * 66, "黑天鹅与事件风险", "=" * 66]

    if events:
        L.append("  【未来事件（日期确定、影响不确定）】")
        for e in events[:8]:
            L.append(f"    {e['date']}  {e['days_until']:>3} 天后  "
                     f"{e['kind']}")
        if not any("选举" in e["kind"] for e in events):
            L.append("    （本次窗口内无美国大选/中期选举）")
        L.append("-" * 66)

    if tail and tail.get("ok"):
        L.append("  【尾部风险】")
        L.append(f"    VaR95 {tail['var95']*100:>7.2f}%   VaR99 {tail['var99']*100:>7.2f}%"
                 f"   CVaR95 {tail['cvar95']*100:>7.2f}%")
        L.append(f"    最差单日 {tail['worst_day']*100:>7.2f}%   最差单周 "
                 + (f"{tail['worst_week']*100:>7.2f}%" if tail.get("worst_week") is not None else "   N/A"))
        L.append(f"    偏度 {tail['skew']:>6.2f}   峰度 {tail['kurtosis']:>6.2f}"
                 f"   最大回撤 {tail['max_drawdown']*100:>7.2f}%")
        rd = tail.get("recovery_days")
        L.append(f"    回撤恢复    {str(rd) + ' 个交易日' if rd is not None else '尚未恢复'}")
        L.append("-" * 66)

    if stress:
        L.append("  【历史情景回放】")
        for x in stress:
            if x.get("ok"):
                L.append(f"    {x['name']:<22} {x['window']}  收益 {x['return']*100:>7.2f}%"
                         f"  回撤 {x['max_drawdown']*100:>7.2f}%")
            else:
                L.append(f"    {x['name']:<22} 未覆盖")
        L.append("-" * 66)

    if frag and frag.get("ok"):
        L.append(f"  【脆弱度】 {frag['score']:.0f}/100  等级 {frag['level']}")
        L.append(f"    构成: {frag['breakdown']}")
        L.append(f"    建议: {frag['advice']}")

    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    # 选举日期校验
    days = {d["year"]: d["date"] for d in us_election_days(2024, 2028)}
    assert days[2024] == "2024-11-05", days[2024]
    assert days[2026] == "2026-11-03", days[2026]
    assert days[2028] == "2028-11-07", days[2028]
    print("选举日期规则校验通过:", days)

    cal = event_calendar("2026-11-08", months_ahead=12)
    print(f"\n未来 12 个月确定性事件 {len(cal)} 条")
    for e in cal[:5]:
        print(f"  {e['date']}  {e['days_until']:>3} 天后  {e['kind']}")

    # 尾部风险 + 脆弱度
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=600)
    normal = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.011, 600)), index=idx)
    # 真正的"肥尾"是少数极端单日，而不是慢性下跌
    r_fat = rng.normal(0.0005, 0.011, 600)
    r_fat[[150, 260, 380, 500]] = -0.11               # 四天各跌 11%
    fat = pd.Series(np.cumprod(1 + r_fat), index=idx)
    t_n, t_f = tail_statistics(normal), tail_statistics(fat)
    f_n, f_f = fragility_score(t_n), fragility_score(t_f)
    assert f_f["score"] > f_n["score"] + 5, (f_n["score"], f_f["score"])
    print(f"\n脆弱度：正常序列 {f_n['score']:.0f} < 肥尾序列 {f_f['score']:.0f} ✓")
    print(f"OK v{__version__}")
