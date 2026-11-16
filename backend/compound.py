# -*- coding: utf-8 -*-
"""
复利测算器（M7）
====================================================
长线价值投资的核心是复利。这个模块把复利害在哪里、成本贵在哪里，
全部算成具体数字：

1. **增长测算**：本金 + 年化 + 年限 → 终值（含定投）
2. **目标反推**：想要多少钱，需要多少年化 / 多少年
3. **成本代价**：每年被吃掉 1.5%，20 年后差多少
4. **回撤代价**：亏 30% 后，按 8% 复利要几年才补回来
5. **通胀折算**：名义收益背后的真实购买力

依赖：仅标准库
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "compound_growth", "required_rate", "years_to_target",
    "inflation_adjust", "fee_drag_effect", "drawdown_recovery_years",
    "format_growth", "format_compare",
]


# ══════════════════════════════════════════════════════════
def compound_growth(principal: float, annual_rate: float, years: int,
                    monthly_add: float = 0.0) -> dict:
    """
    逐年复利增长表。

    参数
    ----
    principal    : 初始本金
    annual_rate  : 年化收益率（0.08 表示 8%）
    years        : 年数
    monthly_add  : 每月追加投入（定投），默认 0

    返回
    ----
    table       : [{year, end_balance, principal_total, profit, gain_pct}]
    final       : 终值
    total_in    : 累计投入（本金 + 定投）
    profit      : 总收益
    multiple    : 终值 / 累计投入
    cagr        : 实际年化（考虑定投后）
    """
    p0 = float(principal)
    r = float(annual_rate)
    y = max(int(years), 0)
    add = float(monthly_add)
    monthly_rate = (1 + r) ** (1 / 12) - 1

    bal = p0
    invested = p0
    table = [{"year": 0, "end_balance": round(bal, 2),
              "principal_total": round(invested, 2), "profit": 0.0, "gain_pct": 0.0}]
    for yr in range(1, y + 1):
        for _ in range(12):
            bal = bal * (1 + monthly_rate) + add
            invested += add
        table.append({
            "year": yr,
            "end_balance": round(bal, 2),
            "principal_total": round(invested, 2),
            "profit": round(bal - invested, 2),
            "gain_pct": round(bal / invested - 1, 6) if invested > 0 else 0.0,
        })

    final = bal
    total_in = invested
    profit = final - total_in
    multiple = final / total_in if total_in > 0 else float("nan")
    cagr = (final / total_in) ** (1 / y) - 1 if y > 0 and total_in > 0 and final > 0 else float("nan")
    return {"table": table, "final": round(final, 2),
            "total_in": round(total_in, 2), "profit": round(profit, 2),
            "multiple": round(multiple, 4) if multiple == multiple else None,
            "cagr": cagr, "years": y, "annual_rate": r,
            "monthly_add": add, "principal": p0}


def required_rate(target: float, principal: float, years: int,
                  monthly_add: float = 0.0) -> float:
    """
    反推：想在第 years 年达到 target，需要多少年化收益率。

    用二分法求解，保证稳定。无法达成（需无限收益）时返回 nan。
    """
    if years <= 0 or principal <= 0:
        return float("nan")
    lo, hi = -0.99, 5.0
    if compound_growth(principal, hi, years, monthly_add)["final"] < target:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if compound_growth(principal, mid, years, monthly_add)["final"] < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def years_to_target(target: float, principal: float, annual_rate: float,
                    monthly_add: float = 0.0, max_years: int = 100) -> int | None:
    """反推：按给定年化，需要多少年达到 target。达不到返回 None。"""
    if principal <= 0:
        return None
    for y in range(1, max_years + 1):
        if compound_growth(principal, annual_rate, y, monthly_add)["final"] >= target:
            return y
    return None


def inflation_adjust(nominal: float, years: int, inflation: float = 0.03) -> dict:
    """名义金额折算为实际购买力。"""
    real = nominal / ((1 + inflation) ** years) if years >= 0 else float("nan")
    return {"nominal": round(nominal, 2), "real": round(real, 2),
            "years": years, "inflation": inflation,
            "loss_pct": round(1 - real / nominal, 6) if nominal > 0 else float("nan")}


def fee_drag_effect(principal: float, annual_rate: float, years: int,
                    fee_annual: float) -> dict:
    """
    每年被成本吃掉 fee_annual，长期差多少。

    典型场景：年换手 12 次的策略年成本约 2.4%，买入持有约 0.2%。
    """
    gross = compound_growth(principal, annual_rate, years)
    net = compound_growth(principal, annual_rate - fee_annual, years)
    return {
        "gross_final": gross["final"], "net_final": net["final"],
        "difference": round(gross["final"] - net["final"], 2),
        "difference_pct": round((gross["final"] - net["final"]) / gross["final"], 6)
        if gross["final"] > 0 else float("nan"),
        "fee_annual": fee_annual, "years": years,
    }


def drawdown_recovery_years(drawdown: float, annual_rate: float,
                            max_years: int = 50) -> int | None:
    """
    亏了 drawdown（传 -0.30 或 0.30）之后，按给定年化需要几年补回原净值。
    """
    dd = abs(float(drawdown))
    if dd <= 0:
        return 0
    if dd >= 1:
        return None
    need = 1.0 / (1.0 - dd)          # 需要涨到的倍数
    if annual_rate <= 0:
        return None
    years = 0
    v = 1.0
    while v < need and years < max_years:
        v *= (1 + annual_rate)
        years += 1
    return years if v >= need else None


# ══════════════════════════════════════════════════════════
# 渲染
# ══════════════════════════════════════════════════════════
def format_growth(res: dict) -> str:
    L = ["=" * 66,
         f"复利测算：本金 {res['principal']:,.0f}  年化 {res['annual_rate']*100:.1f}%"
         f"  年限 {res['years']} 年"
         + (f"  月投 {res['monthly_add']:,.0f}" if res["monthly_add"] else ""),
         "=" * 66]
    L.append(f"  {'年':>4}{'期末余额':>14}{'累计投入':>14}{'收益':>14}{'收益率':>10}")
    L.append("  " + "-" * 56)
    for row in res["table"]:
        if row["year"] % max(1, res["years"] // 8) and row["year"] != res["years"]:
            continue
        L.append(f"  {row['year']:>4}{row['end_balance']:>14,.0f}"
                 f"{row['principal_total']:>14,.0f}{row['profit']:>14,.0f}"
                 f"{row['gain_pct']*100:>9.1f}%")
    L.append("  " + "-" * 56)
    L.append(f"  终值 {res['final']:,.0f}   累计投入 {res['total_in']:,.0f}"
             f"   净赚 {res['profit']:,.0f}   倍数 {res['multiple']:.2f}x")
    L.append("=" * 66)
    return "\n".join(L)


def format_compare(principal: float, years: int, rates=(0.04, 0.06, 0.08, 0.10),
                   monthly_add: float = 0.0) -> str:
    """多档年化对比（含通胀折算），一眼看出"多 2% 有多重要"。"""
    L = ["=" * 66,
         f"年化差异的长期影响（本金 {principal:,.0f}，{years} 年"
         + (f"，月投 {monthly_add:,.0f}" if monthly_add else "") + "）",
         "=" * 66]
    L.append(f"  {'年化':>6}{'终值':>16}{'净赚':>16}{'扣通胀后':>16}")
    L.append("  " + "-" * 54)
    for r in rates:
        g = compound_growth(principal, r, years, monthly_add)
        real = inflation_adjust(g["final"], years)["real"]
        L.append(f"  {r*100:>5.0f}%{g['final']:>16,.0f}{g['profit']:>16,.0f}"
                 f"{real:>16,.0f}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    print(format_compare(100000, 20))
    print()
    print(format_growth(compound_growth(100000, 0.08, 20)))
    print()

    fee = fee_drag_effect(100000, 0.08, 20, 0.021)   # 月频换手 vs 买入持有
    print(f"每年多花 2.1% 成本，20 年后少 {fee['difference']:,.0f} 元"
          f"（占 {fee['difference_pct']*100:.1f}%）")
    for dd in (0.10, 0.20, 0.30, 0.50):
        print(f"  亏 {dd*100:.0f}% → 按 8% 复利需 {drawdown_recovery_years(dd, 0.08)} 年补回")

    r = required_rate(1000000, 100000, 20)
    print(f"\n10 万想 20 年变成 100 万，需要年化 {r*100:.2f}%")
    print(f"10 万按 8% 到 100 万需要 {years_to_target(1000000, 100000, 0.08)} 年")

    assert abs(compound_growth(100000, 0.08, 20)["final"] - 466095.71) < 1
    print(f"\nOK v{__version__}")
