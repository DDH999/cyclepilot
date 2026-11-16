# -*- coding: utf-8 -*-
"""
确定性收益清单（M71）
====================================================
与"找 alpha"不同，这里四项收益**不依赖选股能力**：

| 项目 | 来源 | 确定性 |
|---|---|---|
| A/C 类费率差 | 同一只基金，A 类一次性 vs C 类按日计提 | **高**（合同约定） |
| 可转债打新 | 中签后上市首日卖出 | 中（有破发，靠数量摊薄） |
| 国债逆回购 | 闲钱放 GC001/GC007 | **高**（利率公开） |
| 定投 vs 一次性 | 降低择时风险，不提高期望收益 | 中（看波动环境） |

⚠ 口径纪律
----------
* 费率差是**合同约定**，可直接算
* 打新收益必须标注"估算"，因为它取决于中签率与首日涨幅，不是承诺
* 逆回购必须用**实时/最新**利率，拉不到就如实标注"未取到"，不能编
* 定投不产生额外期望收益，只改变路径，不能算成"收益提升"

用法
----
```
python backend\\certain.py --capital 100000
python backend\\certain.py --capital 100000 --json
```
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

__version__ = "0.1.0"
__all__ = ["repo_rate", "fee_saving", "cb_opportunity", "summary",
           "format_report"]

# 打新估算的保守假设（明确写出来，便于你改成自己的）
# ⚠ 用「单账户年收益区间」而不是「期望值」：
# 期望值模型算出来只有几元/年（因为中签率极低），
# 但实际单账户一年能中 5~15 签，这是"参与次数×中签概率"的经验结果。
CB_ASSUMPTIONS = {
    "年可申购只数": 80,        # 近年一年可申购转债数量级
    "单账户年签到数": (5, 15),  # 经验区间
    "单签盈利": (100, 250),    # 元/签（首日卖出，近年偏保守）
}


def repo_rate(symbol: str = "GC007") -> dict:
    """
    国债逆回购最新年化。

    ⚠ akshare 的 `repo_rate_hist(start_date, end_date)` **没有 symbol 参数**，
    返回的是各品种（GC001/GC002/GC003/GC004/GC007/GC014/GC028...）的整表，
    需要自己按品种名筛行。
    """
    try:
        import akshare as ak
        from datetime import date, timedelta
        end = date.today()
        start = end - timedelta(days=30)
        d = ak.repo_rate_hist(start_date=start.strftime("%Y%m%d"),
                              end_date=end.strftime("%Y%m%d"))
        if d is None or len(d) == 0:
            return {"ok": False, "reason": "接口返回空"}
        cols = [str(c) for c in d.columns]
        name_col = next((c for c in cols if "品种" in c or "名称" in c
                         or "code" in c.lower()), cols[0])
        rate_col = next((c for c in cols if "利率" in c or "收盘" in c
                         or "rate" in c.lower()), cols[-1])
        sub = d[d[name_col].astype(str).str.contains(symbol, na=False)]
        if len(sub) == 0:
            sub = d
        v = float(pd.to_numeric(sub[rate_col], errors="coerce").dropna().iloc[-1])
        pct = v if v > 0.5 else v * 100
        return {"ok": True, "symbol": symbol, "rate_pct": round(pct, 3),
                "source": "akshare repo_rate_hist"}
    except Exception as ex:
        return {"ok": False, "symbol": symbol,
                "reason": f"{type(ex).__name__}: {str(ex)[:60]}"}


def fee_saving(capital: float = 100_000.0, years: float = 10.0,
               monthly: float | None = None) -> dict:
    """A/C 类费率差（用已有 fee_check.compare_ac）。"""
    try:
        import fee_check as FC
        m = monthly if monthly is not None else capital / max(years * 12, 1)
        r = FC.compare_ac(monthly=float(m), years=float(years))
        if isinstance(r, dict):
            diff = r.get("diff")
            a, c = r.get("fv_a"), r.get("fv_c")
            better = "A 类" if (diff is not None and diff > 0) else "C 类"
            return {"ok": True, "better": better,
                    "fv_a": round(float(a), 0) if a is not None else None,
                    "fv_c": round(float(c), 0) if c is not None else None,
                    "diff": round(float(diff), 0) if diff is not None else None,
                    "a_annual_cost": r.get("a_annual_cost"),
                    "c_annual_cost": r.get("c_annual_cost")}
        return {"ok": True, "raw": str(r)[:200]}
    except Exception as ex:
        return {"ok": False, "reason": f"{type(ex).__name__}: {str(ex)[:60]}"}


def cb_opportunity(days_ahead: int = 7) -> dict:
    """
    可转债打新：可申购数量 + 估算年化（标注为估算）。
    """
    out = {"ok": False, "assumptions": CB_ASSUMPTIONS}
    try:
        import cb_new as CN
        r = CN.scan(days_ahead=int(days_ahead))
        if isinstance(r, dict):
            items = r.get("items") or r.get("bonds") or r.get("rows") or []
        else:
            items = r
        n = len(items) if items is not None else 0
        out.update({"ok": True, "n_upcoming": int(n),
                    "window_days": int(days_ahead)})
    except Exception as ex:
        out["reason"] = f"{type(ex).__name__}: {str(ex)[:60]}"
        return out

    a = CB_ASSUMPTIONS
    lo = a["单账户年签到数"][0] * a["单签盈利"][0]
    hi = a["单账户年签到数"][1] * a["单签盈利"][1]
    out["range_low"] = lo
    out["range_high"] = hi
    out["expected_annual"] = round((lo + hi) / 2, 0)     # 中位估计
    out["note"] = (f"单账户经验区间 {lo:,.0f}~{hi:,.0f} 元/年，"
                   f"取决于中签率与上市涨幅，不是承诺")
    return out


def dca_vs_lump(monthly: float = 3000.0, years: float = 5.0,
                annual_return: float = 0.06, annual_vol: float = 0.20) -> dict:
    """
    定投 vs 一次性：**不提高期望收益**，只降低"一次性买在高点"的风险。
    用蒙特卡洛给出路径差异，明确不是收益增强。
    """
    import numpy as np
    rng = np.random.default_rng(7)
    n_sim = 2000
    months = int(years * 12)
    m_mu = annual_return / 12
    m_sd = annual_vol / np.sqrt(12)
    lump_final, dca_final = [], []
    for _ in range(n_sim):
        r = rng.normal(m_mu, m_sd, months)
        path = np.cumprod(1 + r)
        lump_final.append(monthly * months * path[-1])
        # 定投：每期投入，按当期价格累积
        units = float(np.sum((monthly / path)))
        dca_final.append(units * path[-1])
    lump, dca = np.array(lump_final), np.array(dca_final)
    return {
        "ok": True, "months": months, "n_sim": n_sim,
        "lump_median": round(float(np.median(lump)), 0),
        "dca_median": round(float(np.median(dca)), 0),
        "lump_p5": round(float(np.percentile(lump, 5)), 0),
        "dca_p5": round(float(np.percentile(dca, 5)), 0),
        "note": ("定投期望与一次性接近，但 p5（最差 5%）更高，"
                 "即路径更稳，不是收益更高"),
    }


def summary(capital: float = 100_000.0, years: float = 10.0) -> dict:
    """汇总四项，给出年化 pp 与年金额。"""
    fee = fee_saving(capital, years)
    cb = cb_opportunity()
    repo = repo_rate()
    dca = dca_vs_lump(monthly=capital / max(years * 12, 1), years=min(years, 5))

    items = []
    # 费率
    fd = fee.get("diff") if fee.get("ok") else None
    f_amt = (fd / years) if (fd and years) else None
    f_pp = (f_amt / capital * 100) if (f_amt and capital) else None
    items.append({
        "项目": "A/C 类费率差", "确定性": "高",
        "年化pp": round(f_pp, 3) if f_pp is not None else None,
        "年金额": round(f_amt, 0) if f_amt is not None else None,
        "依据": f"选 {fee.get('better', '?')}（10年终值差 {fd:,.0f} 元）"
                if fd is not None else "fee_check.compare_ac",
        "状态": "OK" if fee.get("ok") else "未取到",
        "说明": (f"年成本 A {fee.get('a_annual_cost', 0)*100:.2f}% / "
                 f"C {fee.get('c_annual_cost', 0)*100:.2f}%"
                 if fee.get("ok") and fee.get("a_annual_cost") else ""),
    })
    # 打新
    exp = cb.get("expected_annual")
    pp = (exp / capital * 100) if (exp and capital) else None
    items.append({
        "项目": "可转债打新（估算）", "确定性": "中",
        "年化pp": round(pp, 2) if pp else None,
        "年金额": exp,
        "依据": "单账户 5~15 签 × 100~250 元/签（经验区间）",
        "状态": "OK" if cb.get("ok") else "未取到",
        "说明": cb.get("note", ""),
    })
    # 逆回购
    r = repo.get("rate_pct")
    items.append({
        "项目": "国债逆回购（闲钱）", "确定性": "高",
        "年化pp": round(r, 2) if r else None,
        "年金额": round(capital * r / 100, 0) if r else None,
        "依据": f"最新 {repo.get('symbol')} 年化",
        "状态": "OK" if repo.get("ok") else "未取到",
        "说明": "" if repo.get("ok") else str(repo.get("reason"))[:50],
    })
    # 定投
    items.append({
        "项目": "定投 vs 一次性", "确定性": "中",
        "年化pp": None,
        "依据": "蒙特卡洛 2000 次",
        "状态": "OK" if dca.get("ok") else "未取到",
        "说明": dca.get("note", ""),
    })
    return {"ok": True, "capital": capital, "years": years,
            "items": items, "detail": {"fee": fee, "cb": cb,
                                       "repo": repo, "dca": dca}}


def format_report(res: dict) -> str:
    L = ["=" * 80, "确定性收益清单", "=" * 80]
    L.append(f"  本金 {res['capital']:,.0f} 元   周期 {res['years']:.0f} 年")
    L.append("-" * 80)
    L.append(f"  {'项目':<20}{'确定性':<6}{'年化pp':>8}{'年金额':>12}  状态")
    for it in res["items"]:
        pp = it.get("年化pp")
        amt = it.get("年金额")
        L.append(f"  {it['项目']:<20}{it['确定性']:<6}"
                 f"{(f'{pp:.2f}' if pp is not None else '-'):>8}"
                 f"{(f'{amt:,.0f}' if amt else '-'):>12}  {it['状态']}")
        if it.get("说明"):
            L.append(f"    {it['说明'][:70]}")
    L.append("-" * 80)
    L.append("  说明：前两项的『年化pp』为估算，不构成收益承诺；")
    L.append("        定投不提高期望收益，只改善最差情形（p5）。")
    L.append("=" * 80)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    r = summary(a.capital, a.years)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        print(format_report(r))
