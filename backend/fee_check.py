# -*- coding: utf-8 -*-
"""
费率核对器（M41）
====================================================
A 类份额 vs C 类份额，长期持有到底差多少？

这是**确定性最高**的"提高收益"手段之一：
同一只指数、同样的底层，只是收费结构不同，选错就白送钱。

两类份额的成本结构
------------------
**A 类**（前端收费）
* 申购费：一次性，标准 1.0%~1.5%，很多平台打 1 折 → 0.10%~0.15%
* 赎回费：持有 <7 天 1.5%；7 天~1 年 0.5%；1~2 年 0.25%；**≥2 年通常 0**
* 管理费 + 托管费：长期持有期间按年收取（两类相同）

**C 类**（销售服务费）
* 申购费：**0**
* 销售服务费：**按年**收取，0.20%~0.60%
* 赎回费：持有 ≥7~30 天通常 0
* 管理费 + 托管费：与 A 类相同

判断规则（一句话）
------------------
> **持有时间 > 临界点，选 A 类；否则选 C 类。**
> 临界点 = 申购费 ÷ 销售服务费年率

例：申购费 0.15%（1 折）、销售服务费 0.40%/年
→ 临界点 = 0.375 年 ≈ **4.5 个月**。持有超过 4.5 个月，A 类更省。

⚠ 诚实的边界
-------------
* 本工具**不保证**实时拉取到真实费率——公募费率变动、平台折扣各异
* `probe_fees()` 会尝试联网，取不到就如实说"请手工输入"，**绝不编数字**
* 计算假设：两类份额跟踪同一指数、收益率相同（实际会有微小跟踪差异）
"""
from __future__ import annotations

import numpy as np

__version__ = "0.1.0"
__all__ = ["DEFAULT_FEES", "breakeven_months", "fee_impact",
           "compare_ac", "probe_fees", "format_report"]

# 常见默认值（保守取整，不代表具体基金）
DEFAULT_FEES = {
    "a_purchase": 0.0015,      # 申购费 0.15%（1 折后）
    "a_purchase_std": 0.015,   # 申购费标准 1.5%
    "c_service": 0.0040,       # 销售服务费 0.40%/年
    "mgmt": 0.0050,            # 管理费 0.50%/年（两类相同）
    "custody": 0.0010,         # 托管费 0.10%/年（两类相同）
}


def breakeven_months(purchase_fee: float, service_fee_annual: float) -> dict:
    """
    A 类与 C 类的成本临界点（月）。

    A 类一次性付 purchase_fee；C 类每年付 service_fee_annual。
    持有 T 年时：A 成本 = purchase_fee；C 成本 = service_fee_annual × T
    → 临界 T = purchase_fee / service_fee_annual
    """
    try:
        pf = float(purchase_fee)
        sf = float(service_fee_annual)
    except Exception:
        return {"ok": False, "reason": "费率不是数字"}
    if sf <= 0:
        return {"ok": False, "reason": "销售服务费为 0，C 类永远更省（异常）"}
    if pf < 0:
        return {"ok": False, "reason": "申购费为负（异常）"}
    years = pf / sf
    months = years * 12
    return {"ok": True, "years": round(years, 3), "months": round(months, 1),
            "purchase_fee": round(pf, 6),
            "service_fee": round(sf, 6),
            "verdict": (f"持有超过 {months:.1f} 个月 → 选 A 类；"
                        f"否则选 C 类")}


def _fv(monthly: float, years: float, annual_return: float,
        annual_cost: float) -> float:
    """月定投的年金终值，扣除年化成本。"""
    r = (1.0 + annual_return - annual_cost) ** (1.0 / 12.0) - 1.0
    n = int(round(years * 12))
    if n <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return monthly * n
    return monthly * ((1.0 + r) ** n - 1.0) / r


def fee_impact(monthly: float = 4000.0, years: float = 20.0,
               annual_return: float = 0.07,
               fee_low: float = 0.0, fee_high: float = 0.0) -> dict:
    """
    两个年费率之差的长期影响（年化成本口径）。

    fee_low/fee_high 都是「年化总费用率」。
    """
    try:
        lo = float(fee_low)
        hi = float(fee_high)
    except Exception:
        return {"ok": False, "reason": "费率不是数字"}
    if lo > hi:
        lo, hi = hi, lo
    fv_lo = _fv(monthly, years, annual_return, lo)
    fv_hi = _fv(monthly, years, annual_return, hi)
    diff = fv_lo - fv_hi
    base = _fv(monthly, years, annual_return, 0.0)
    return {"ok": True, "monthly": monthly, "years": years,
            "annual_return": annual_return,
            "fee_low": round(lo, 6), "fee_high": round(hi, 6),
            "fee_gap_pp": round((hi - lo) * 100, 3),
            "fv_low_fee": round(fv_lo, 2), "fv_high_fee": round(fv_hi, 2),
            "fv_no_fee": round(base, 2),
            "diff": round(diff, 2),
            "diff_pct": round(diff / fv_hi, 4) if fv_hi > 0 else None,
            "verdict": (f"年费率差 {round((hi-lo)*100, 2)} 个百分点，"
                        f"{years:.0f} 年后终值差 {diff:,.0f} 元")}


def compare_ac(monthly: float = 4000.0, years: float = 20.0,
               annual_return: float = 0.07,
               a_purchase: float | None = None,
               c_service: float | None = None,
               mgmt: float | None = None,
               custody: float | None = None) -> dict:
    """
    A 类 vs C 类，在同一持有期下的终值对比。

    A 类：期初一次性申购费（按比例摊到第一笔，简化处理为年化成本）
    C 类：按年收销售服务费
    """
    a_purchase = DEFAULT_FEES["a_purchase"] if a_purchase is None else float(a_purchase)
    c_service = DEFAULT_FEES["c_service"] if c_service is None else float(c_service)
    mgmt = DEFAULT_FEES["mgmt"] if mgmt is None else float(mgmt)
    custody = DEFAULT_FEES["custody"] if custody is None else float(custody)

    be = breakeven_months(a_purchase, c_service,
                          ) if c_service > 0 else {"ok": False}
    # A 类一次性费用摊到持有期 → 相当于年化 a_purchase/years
    a_annual = mgmt + custody + (a_purchase / max(years, 1e-9))
    c_annual = mgmt + custody + c_service

    fv_a = _fv(monthly, years, annual_return, a_annual)
    fv_c = _fv(monthly, years, annual_return, c_annual)
    better = "A类" if fv_a >= fv_c else "C类"

    return {"ok": True, "monthly": monthly, "years": years,
            "a_purchase": a_purchase, "c_service": c_service,
            "mgmt": mgmt, "custody": custody,
            "a_annual_cost": round(a_annual, 6),
            "c_annual_cost": round(c_annual, 6),
            "fv_a": round(fv_a, 2), "fv_c": round(fv_c, 2),
            "diff": round(fv_a - fv_c, 2),
            "better": better,
            "breakeven": be,
            "verdict": (f"持有 {years:.0f} 年 → 选 {better}，"
                        f"终值差 {abs(fv_a - fv_c):,.0f} 元")}


def probe_fees(code: str) -> dict:
    """
    尝试联网取真实费率。取不到就如实说，不编数字。
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
    except Exception as ex:
        return {"ok": False, "code": code,
                "reason": f"akshare 不可用：{ex}"}

    attempts = []
    for label, fn in (
        ("fund_fee_em", lambda: ak.fund_fee_em(symbol=code,
                                               indicator="申购费率")),
        ("fund_individual_detail_hold_xq",
         lambda: ak.fund_individual_detail_hold_xq(symbol=code)),
    ):
        try:
            df = fn()
            if df is None or len(df) == 0:
                attempts.append(f"{label}: 返回空")
                continue
            return {"ok": True, "code": code, "source": label,
                    "rows": int(len(df)),
                    "columns": [str(c) for c in df.columns][:8],
                    "preview": df.head(5).to_dict("records")}
        except Exception as ex:
            attempts.append(f"{label}: {type(ex).__name__} {str(ex)[:50]}")

    return {"ok": False, "code": code,
            "reason": "公开接口没有稳定返回费率数据",
            "attempts": attempts,
            "next_action": ("请查基金合同/招募说明书里的：申购费率、"
                            "销售服务费率、管理费率、托管费率，用参数传入")}


def format_report(res: dict, impact: dict | None = None) -> str:
    L = ["=" * 70, "费率核对：A 类 vs C 类", "=" * 70]
    if not res.get("ok"):
        L.append(f"  失败：{res.get('reason')}")
        L.append("=" * 70)
        return "\n".join(L)

    L.append(f"  月定投 {res['monthly']:,.0f} 元   持有 {res['years']:.0f} 年")
    L.append(f"  A 类：申购费 {res['a_purchase']*100:.2f}%（一次性）"
             f"  年化成本 {res['a_annual_cost']*100:.3f}%")
    L.append(f"  C 类：销售服务费 {res['c_service']*100:.2f}%（按年）"
             f"  年化成本 {res['c_annual_cost']*100:.3f}%")
    L.append("-" * 70)
    L.append(f"  A 类终值 {res['fv_a']:>14,.0f}")
    L.append(f"  C 类终值 {res['fv_c']:>14,.0f}")
    L.append(f"  差额     {res['diff']:>14,.0f}")
    L.append("-" * 70)
    be = res.get("breakeven") or {}
    if be.get("ok"):
        L.append(f"  临界点：{be['verdict']}")
    L.append(f"  结论：{res['verdict']}")

    if impact and impact.get("ok"):
        L.append("-" * 70)
        L.append("  费率差的影响（年化成本口径）：")
        L.append(f"    {impact['fee_low']*100:.2f}% vs "
                 f"{impact['fee_high']*100:.2f}%  "
                 f"→ {impact['verdict']}")

    L.append("=" * 70)
    L.append("  ⚠ 费率以基金合同为准。本工具用的是常见默认值，")
    L.append("     请把你那只基金的「申购费 / 销售服务费」填进参数再算一次。")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    be = breakeven_months(0.0015, 0.0040)
    print(f"临界点（申购 0.15% vs 销售服务 0.40%/年）：{be['verdict']}")
    assert be["ok"] and abs(be["months"] - 4.5) < 0.1, be

    be2 = breakeven_months(0.015, 0.0040)
    print(f"标准申购费 1.5% 不打折：{be2['months']:.1f} 个月")

    r = compare_ac(monthly=4000, years=20)
    print()
    print(format_report(r))
    assert r["ok"]
    assert r["better"] == "A类", r["better"]

    # 短期持有应该选 C 类
    r_short = compare_ac(monthly=4000, years=0.25)
    print(f"\n只持有 3 个月：{r_short['better']}（{r_short['verdict']}）")
    assert r_short["better"] == "C类", r_short["better"]

    imp = fee_impact(4000, 20, 0.07, 0.0060, 0.0110)
    print(f"\n费率差 0.5pp：{imp['verdict']}")
    assert imp["ok"] and imp["diff"] > 0

    assert breakeven_months(0.001, 0)["ok"] is False
    print("\nOK v" + __version__)
