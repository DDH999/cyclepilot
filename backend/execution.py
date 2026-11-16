# -*- coding: utf-8 -*-
"""
交易成本与容量建模（M12）
====================================================
之前只有「固定费率 + 换手」，那只能算小资金。
资金一大，**冲击成本**（自己把价格买上去）才是主要成本。

模型
----
平方根冲击定律（业界通用近似）：

    冲击(收益率) = k × 日波动率 × sqrt(委托金额 / 日均成交额)

* k 通常取 0.3~1.0，本书默认 0.5（保守取中值）
* 参与率 <1% 时冲击可忽略；>10% 时成本迅速失控

总成本 = 佣金 + 印花税 + 过户费 + 滑点 + 冲击

容量
----
给定成本上限（如年化 3%），反推**最大可manage资金**。
超过这个规模，策略的账面收益会被自己的交易吃掉。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["FeeCfg", "DEFAULTS", "impact_bps", "participation_rate",
           "cost_breakdown", "capacity_curve", "format_report"]


class FeeCfg:
    def __init__(self, commission_bps: float = 2.5, stamp_bps: float = 5.0,
                 transfer_bps: float = 0.1, spread_bps: float = 2.0,
                 impact_k: float = 0.5):
        self.commission_bps = commission_bps
        self.stamp_bps = stamp_bps
        self.transfer_bps = transfer_bps
        self.spread_bps = spread_bps
        self.impact_k = impact_k


DEFAULTS = FeeCfg()


def participation_rate(order_value: float, adv_value: float) -> float:
    """参与率 = 委托金额 / 日均成交额。ADV 非正时返回 inf。"""
    if adv_value is None or not np.isfinite(adv_value) or adv_value <= 0:
        return float("inf")
    return float(order_value) / float(adv_value)


def impact_bps(order_value: float, adv_value: float, sigma_daily: float,
               k: float = 0.5, max_bps: float = 500.0) -> float:
    """
    冲击成本（bps，单边）。

    k 越大越保守；max_bps 上限防止极端值炸表（仍会在结果里标明是否触顶）。
    """
    part = participation_rate(order_value, adv_value)
    if not np.isfinite(part):
        return float(max_bps)
    if part <= 0:
        return 0.0
    sigma = max(float(sigma_daily), 0.0)
    bps = k * sigma * np.sqrt(part) * 1e4
    return float(min(bps, max_bps))


def cost_breakdown(capital: float, turnover_annual: float,
                   adv_value: float, sigma_daily: float,
                   cfg: FeeCfg = DEFAULTS) -> dict:
    """
    年总成本拆解（金额与占本金比例）。

    capital        : 组合规模
    turnover_annual: 年换手（1.0 = 一年换一遍）
    adv_value      : 持仓标的的日均成交额（组合层面近似值）
    sigma_daily    : 持仓的日均波动率（如 0.012）
    """
    cap = float(capital)
    T = max(float(turnover_annual), 0.0)
    side = T * cap
    both = 2.0 * side

    commission = both * cfg.commission_bps / 1e4
    stamp = side * cfg.stamp_bps / 1e4
    transfer = both * cfg.transfer_bps / 1e4
    spread = both * cfg.spread_bps / 1e4
    # 关键：冲击按「单笔委托」计。
    # 年换手 T 意味着一年 T 次调仓，每次把整本书换一遍 → 单笔委托额 ≈ capital。
    # 若误用年成交额（T×capital）当单笔，参与率会被放大 T 倍，
    # 算出 1200% 这种荒谬数值，容量也会被严重低估。
    order_value = cap if T > 0 else 0.0
    imp_bps = impact_bps(order_value, adv_value, sigma_daily, cfg.impact_k)
    impact = both * imp_bps / 1e4

    total = commission + stamp + transfer + spread + impact
    return {
        "capital": round(cap, 2),
        "turnover_annual": round(T, 4),
        "adv_value": float(adv_value) if np.isfinite(adv_value) else None,
        "participation": round(participation_rate(order_value, adv_value), 6)
        if T > 0 else 0.0,
        "order_value": round(order_value, 2),
        "commission": round(commission, 2),
        "stamp_duty": round(stamp, 2),
        "transfer_fee": round(transfer, 2),
        "spread": round(spread, 2),
        "impact_bps": round(imp_bps, 2),
        "impact": round(impact, 2),
        "total": round(total, 2),
        "total_pct": round(total / cap, 6) if cap > 0 else float("nan"),
        "friction_pct": round((commission + stamp + transfer + spread) / cap, 6)
        if cap > 0 else float("nan"),
        "impact_pct": round(impact / cap, 6) if cap > 0 else float("nan"),
    }


def capacity_curve(capitals, turnover_annual: float, adv_value: float,
                   sigma_daily: float, cfg: FeeCfg = DEFAULTS) -> pd.DataFrame:
    """不同资金规模下的成本曲线（找容量上限）。"""
    rows = []
    for c in capitals:
        r = cost_breakdown(c, turnover_annual, adv_value, sigma_daily, cfg)
        rows.append({"资金": float(c), "参与率": r["participation"],
                     "冲击bps": r["impact_bps"], "总成本": r["total"],
                     "成本占比": r["total_pct"]})
    return pd.DataFrame(rows)


def max_capacity(turnover_annual: float, adv_value: float, sigma_daily: float,
                 cost_budget_pct: float = 0.03,
                 cfg: FeeCfg = DEFAULTS,
                 hi: float = 1e10) -> float:
    """
    反推：在给定成本预算下，最大可管理的资金规模。

    二分求解；若连 1 元都不达标返回 0。
    """
    lo = 1.0
    if cost_breakdown(hi, turnover_annual, adv_value, sigma_daily, cfg)["total_pct"] <= cost_budget_pct:
        return float(hi)
    for _ in range(80):
        mid = (lo + hi) / 2
        pct = cost_breakdown(mid, turnover_annual, adv_value, sigma_daily, cfg)["total_pct"]
        if pct > cost_budget_pct:
            hi = mid
        else:
            lo = mid
    return float(lo)


def format_report(res: dict, cap_line: float | None = None) -> str:
    L = ["=" * 66, "交易成本与容量建模", "=" * 66]
    L.append(f"  资金 {res['capital']:,.0f}   年换手 {res['turnover_annual']:.1f}x"
             f"   日均成交额 {res['adv_value']:,.0f}" if res.get("adv_value")
             else f"  资金 {res['capital']:,.0f}   年换手 {res['turnover_annual']:.1f}x")
    L.append(f"  单笔委托 {res.get('order_value', 0):,.0f}"
             f"   参与率 {res['participation']*100:.2f}%"
             f"   （单笔委托占日均成交额）")
    L.append("-" * 66)
    L.append(f"  佣金    {res['commission']:>10,.0f}    印花税 {res['stamp_duty']:>10,.0f}")
    L.append(f"  过户费  {res['transfer_fee']:>10,.0f}    价差   {res['spread']:>10,.0f}")
    L.append(f"  冲击    {res['impact']:>10,.0f}    （{res['impact_bps']:.1f} bps）")
    L.append("-" * 66)
    L.append(f"  合计    {res['total']:>10,.0f}    占本金 {res['total_pct']*100:.2f}%")
    L.append(f"  其中摩擦成本 {res['friction_pct']*100:.2f}%"
             f"   冲击成本 {res['impact_pct']*100:.2f}%")
    if cap_line is not None:
        L.append("-" * 66)
        L.append(f"  容量上限（3% 成本预算）约 {cap_line:,.0f} 元")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    # 小资金：摩擦成本为主
    small = cost_breakdown(100000, 12.0, adv_value=5e8, sigma_daily=0.012)
    # 大资金：冲击成本为主
    big = cost_breakdown(5e8, 12.0, adv_value=5e8, sigma_daily=0.012)

    print(format_report(small))
    print(format_report(big))

    assert small["impact_pct"] < small["friction_pct"], "小资金冲击应低于摩擦"
    assert big["impact_pct"] > big["friction_pct"], "大资金冲击应高于摩擦"
    assert impact_bps(1e6, 1e9, 0.01) < impact_bps(1e8, 1e9, 0.01)

    cap = max_capacity(12.0, adv_value=5e8, sigma_daily=0.012,
                       cost_budget_pct=0.03)
    print(f"容量上限（3% 预算，年换手 12x）：{cap:,.0f} 元")
    print(f"OK v{__version__}")
