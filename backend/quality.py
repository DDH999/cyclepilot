# -*- coding: utf-8 -*-
"""
基本面质量维度（M34）
====================================================
给六维体检加**第七维：基本面质量**（美股，数据来自 ROIC.ai）。

⚠ 注意与 `fundamentals.py` 的区别：
* `fundamentals.py` 是**老系统**的 A 股基本面层（akshare 业绩报表 + 时点对齐）
* 本文件是**新平台**的第七维，只覆盖**美股**（ROIC.ai）

为什么需要
----------
原六维回答"有没有作弊、会不会崩"，但**不问"买的东西好不好"**。
校准时发现的盲区：纯 beta 策略六维看着正常，只是没有 alpha。

四个子维度
----------
| 子项 | 权重 | 看什么 |
|---|---|---|
| 盈利能力 | 30 | ROE、净利率、毛利率 |
| 成长性 | 25 | 营收增速、净利增速 |
| 财务健康 | 25 | 资产负债率、流动比率 |
| 现金流质量 | 20 | 经营现金流 / 净利润 |

诚实边界
--------
* **只覆盖美股**，A 股基本面还没接
* 字段名按 ROIC 实际返回做**模糊匹配**，找不到就如实标缺，**不猜**
* ROIC 免费档 5 次/分钟，拉三张表约 36 秒（已缓存 24 小时）
* 基本面好 ≠ 股价会涨
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

__version__ = "0.1.0"
__all__ = ["SUB_WEIGHTS", "fetch_statements", "extract_metrics",
           "score_quality", "quality_report", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "us" / "fundamentals"
CACHE_MAX_AGE_HOURS = 24

SUB_WEIGHTS = {"盈利能力": 30.0, "成长性": 25.0, "财务健康": 25.0,
               "现金流质量": 20.0}

# 字段模糊匹配规则：找第一个「所有关键词都出现在键名里」的字段
_FIELD_RULES = {
    "revenue": [("sales_revenue",), ("revenue",), ("net_sales",)],
    "cogs": [("cogs",), ("cost_of_goods",), ("cost_of_revenue",)],
    "gross_profit": [("gross_profit",)],
    "operating_income": [("oper_income",), ("operating_income",)],
    "net_income": [("net_income",), ("net_profit",)],
    "total_assets": [("total_assets",)],
    "total_liabilities": [("total_liabilities",), ("total_liab",)],
    "total_equity": [("total_equity",), ("stockholders_equity",),
                     ("shareholders_equity",)],
    "current_assets": [("current_assets",)],
    "current_liabilities": [("current_liabilities",)],
    "ocf": [("net_cash", "operating"), ("cash", "operating"),
            ("operating_cash",)],
}


def _pick(d: dict, key: str) -> float | None:
    """按规则找字段；找不到返回 None（不猜 0，不拿别的字段凑）。"""
    for rule in _FIELD_RULES.get(key, []):
        for k, v in d.items():
            lk = str(k).lower()
            if all(tok in lk for tok in rule):
                try:
                    f = float(v)
                    if np.isfinite(f):
                        return f
                except Exception:
                    continue
    return None


def _cache_path(ticker: str, kind: str) -> Path:
    safe = str(ticker).replace(":", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{kind}.json"


def _read_cache(ticker: str, kind: str) -> list | None:
    import json
    fp = _cache_path(ticker, kind)
    if not fp.exists():
        return None
    try:
        if (time.time() - fp.stat().st_mtime) / 3600 > CACHE_MAX_AGE_HOURS:
            return None
        data = json.loads(fp.read_text(encoding="utf-8"))
        return data if isinstance(data, list) and data else None
    except Exception:
        return None


def _write_cache(ticker: str, kind: str, items: list) -> None:
    import json
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(ticker, kind).write_text(
            json.dumps(items, ensure_ascii=False, default=str),
            encoding="utf-8")
    except Exception:
        pass


def _to_dicts(pager, limit: int) -> list[dict]:
    """ROIC 的 list() 返回 SyncPager，不是 list —— 必须显式取 data/迭代。"""
    items = getattr(pager, "data", None)
    if items is None:
        try:
            items = list(pager)
        except Exception:
            items = []
    out = []
    for it in items or []:
        out.append(it.model_dump() if hasattr(it, "model_dump") else dict(it))
        if len(out) >= limit:
            break
    return out


def fetch_statements(ticker: str, kind: str = "income",
                     period_type: str = "annual", limit: int = 2,
                     refresh: bool = False) -> dict:
    """
    拉财报表（income / balance / cashflow），24 小时缓存。

    每次真实调用消耗 1 次 ROIC 配额。
    """
    import us_source as US

    cached = None if refresh else _read_cache(ticker, kind)
    if cached:
        return {"ok": True, "items": cached, "source": "cache", "n": len(cached)}

    cli = US.roicai_client()
    US._throttle("roicai")
    sym = US._roic_symbol(ticker)
    try:
        if kind == "income":
            pager = cli.fundamentals.income_statements.list(
                sym, period_type=period_type, limit=limit)
        elif kind == "balance":
            pager = cli.fundamentals.balance_sheets.list(
                sym, period_type=period_type, limit=limit)
        elif kind == "cashflow":
            pager = cli.fundamentals.cash_flow_statements.list(
                sym, period_type=period_type, limit=limit)
        else:
            raise ValueError(f"未知报表类型 {kind!r}")
    except Exception as ex:
        raise RuntimeError(f"拉取 {kind} 失败：{str(ex)[:150]}") from None

    items = _to_dicts(pager, limit)
    if not items:
        raise RuntimeError(f"{kind} 返回空")
    _write_cache(ticker, kind, items)
    return {"ok": True, "items": items, "source": "live", "n": len(items)}


def extract_metrics(income: list[dict], balance: list[dict] | None = None,
                    cashflow: list[dict] | None = None) -> dict:
    """从报表抽指标。缺哪张表就少算哪几项，不补 0。"""
    missing: list[str] = []
    if not income:
        return {"ok": False, "reason": "没有利润表数据",
                "metrics": {}, "raw": {}, "missing": ["income"]}

    cur = income[0]
    prev = income[1] if len(income) > 1 else None

    rev = _pick(cur, "revenue")
    net = _pick(cur, "net_income")
    gp = _pick(cur, "gross_profit")
    oi = _pick(cur, "operating_income")
    if rev is None:
        missing.append("revenue")
    if net is None:
        missing.append("net_income")

    bal = (balance or [{}])[0] if balance else {}
    cf = (cashflow or [{}])[0] if cashflow else {}
    assets = _pick(bal, "total_assets")
    liab = _pick(bal, "total_liabilities")
    equity = _pick(bal, "total_equity")
    ca = _pick(bal, "current_assets")
    cl_ = _pick(bal, "current_liabilities")
    ocf = _pick(cf, "ocf")

    m: dict = {}
    if rev and rev > 0:
        if gp is not None:
            m["gross_margin"] = gp / rev
        if oi is not None:
            m["operating_margin"] = oi / rev
        if net is not None:
            m["net_margin"] = net / rev
    if net is not None and equity and equity > 0:
        m["roe"] = net / equity
    if net is not None and assets and assets > 0:
        m["roa"] = net / assets
    if liab is not None and assets and assets > 0:
        m["debt_ratio"] = liab / assets
    if ca is not None and cl_ and cl_ > 0:
        m["current_ratio"] = ca / cl_
    if ocf is not None and net and net > 0:
        m["ocf_to_net"] = ocf / net

    if prev:
        prev_rev = _pick(prev, "revenue")
        prev_net = _pick(prev, "net_income")
        if rev and prev_rev and prev_rev > 0:
            m["revenue_growth"] = rev / prev_rev - 1
        if net and prev_net and prev_net > 0:
            m["net_growth"] = net / prev_net - 1

    return {
        "ok": True,
        "period": cur.get("period_end_date"),
        "fiscal_year": cur.get("fiscal_year"),
        "currency": cur.get("currency"),
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "raw": {"revenue": rev, "net_income": net, "gross_profit": gp,
                "operating_income": oi, "total_assets": assets,
                "total_liabilities": liab, "total_equity": equity,
                "current_assets": ca, "current_liabilities": cl_,
                "ocf": ocf},
        "missing": missing,
    }


def _ramp(x, lo, hi) -> float:
    """线性打分：x>=hi → 1；x<=lo → 0。nan 表示不可算。"""
    if x is None:
        return float("nan")
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    if x >= hi:
        return 1.0
    if x <= lo:
        return 0.0
    return (x - lo) / (hi - lo)


def score_quality(metrics: dict) -> dict:
    """
    四子项打分。缺项按「已给子项」等比例折算，并报告覆盖率。
    """
    parts: dict[str, float | None] = {}

    sub = [v for v in (_ramp(metrics.get("roe"), 0.05, 0.25),
                       _ramp(metrics.get("net_margin"), 0.02, 0.25),
                       _ramp(metrics.get("gross_margin"), 0.15, 0.60))
           if np.isfinite(v)]
    parts["盈利能力"] = (float(np.mean(sub)) * SUB_WEIGHTS["盈利能力"]
                        if sub else None)

    sub = [v for v in (_ramp(metrics.get("revenue_growth"), 0.0, 0.20),
                       _ramp(metrics.get("net_growth"), 0.0, 0.25))
           if np.isfinite(v)]
    parts["成长性"] = (float(np.mean(sub)) * SUB_WEIGHTS["成长性"]
                      if sub else None)

    sub = []
    dr = metrics.get("debt_ratio")
    if dr is not None and np.isfinite(float(dr)):
        sub.append(1.0 - _ramp(dr, 0.30, 0.85))
    cr = _ramp(metrics.get("current_ratio"), 1.0, 2.0)
    if np.isfinite(cr):
        sub.append(cr)
    parts["财务健康"] = (float(np.mean(sub)) * SUB_WEIGHTS["财务健康"]
                        if sub else None)

    oc = _ramp(metrics.get("ocf_to_net"), 0.5, 1.3)
    parts["现金流质量"] = (float(oc) * SUB_WEIGHTS["现金流质量"]
                          if np.isfinite(oc) else None)

    valid = {k: v for k, v in parts.items() if v is not None}
    if not valid:
        return {"ok": False, "reason": "没有任何可用指标", "score": None,
                "breakdown": parts}

    raw = float(sum(valid.values()))
    max_possible = float(sum(SUB_WEIGHTS[k] for k in valid))
    score = float(np.clip(raw / max_possible * 100.0, 0, 100))
    coverage = len(valid) / len(SUB_WEIGHTS)

    # ── 防止「看得越少分越高」 ──────────────────────────────
    # 实测：只有利润表时得 86 分，三张表齐全反而 63 分 ——
    # 因为按可用子项归一化，缺掉的那项（往往最差）不参与拉低分数。
    # 这是典型的「缺数据反而好看」，必须抑制：
    #   * 覆盖 < 50%  → 不给分（交给上层记 N/A）
    #   * 覆盖 < 75%  → 分数封顶 70，并标记低置信
    if coverage < 0.5:
        return {"ok": False,
                "reason": f"可算子项仅 {len(valid)}/4（{coverage:.0%}），证据不足",
                "score": None, "breakdown": parts,
                "coverage": round(coverage, 2)}
    capped = False
    # 封顶 60（C 级）：完整数据都只有 63 分，缺数据不该"看起来更好"。
    if coverage < 0.75 and score > 60.0:
        score = 60.0
        capped = True

    if score >= 80:
        level, note = "优秀", "盈利、成长、负债、现金流四项整体健康"
    elif score >= 60:
        level, note = "良好", "基本面扎实，个别项偏弱"
    elif score >= 40:
        level, note = "一般", "有明显短板，需看具体哪一项"
    else:
        level, note = "偏弱", "盈利或偿债能力存在实质问题"

    return {"ok": True, "score": round(score, 2),
            "max_possible": round(max_possible, 1), "raw": round(raw, 2),
            "capped_by_coverage": capped,
            "breakdown": {k: (None if v is None else round(v, 2))
                          for k, v in parts.items()},
            "coverage": round(coverage, 2),
            "low_confidence": coverage <= 0.5,
            "level": level, "note": note}


def quality_report(ticker: str, refresh: bool = False) -> dict:
    """
    拉三张表 → 抽指标 → 打分。

    ⚠ 最多消耗 3 次 ROIC 配额（缓存命中不消耗）。
    """
    warns: list[str] = []
    try:
        inc = fetch_statements(ticker, "income", refresh=refresh)
    except Exception as ex:
        return {"ok": False, "ticker": ticker,
                "reason": f"利润表失败：{str(ex)[:150]}"}

    bal = cf = None
    try:
        bal = fetch_statements(ticker, "balance", refresh=refresh)["items"]
    except Exception as ex:
        warns.append(f"资产负债表取不到：{str(ex)[:90]}")
    try:
        cf = fetch_statements(ticker, "cashflow", refresh=refresh)["items"]
    except Exception as ex:
        warns.append(f"现金流量表取不到：{str(ex)[:90]}")

    ext = extract_metrics(inc["items"], bal, cf)
    if not ext.get("ok"):
        return {"ok": False, "ticker": ticker, "reason": ext.get("reason")}

    sc = score_quality(ext["metrics"])
    return {
        "ok": bool(sc.get("ok")), "ticker": ticker,
        "period": ext.get("period"), "fiscal_year": ext.get("fiscal_year"),
        "currency": ext.get("currency"),
        "metrics": ext["metrics"], "score": sc.get("score"),
        "level": sc.get("level"), "note": sc.get("note"),
        "breakdown": sc.get("breakdown"), "coverage": sc.get("coverage"),
        "low_confidence": sc.get("low_confidence"),
        "missing": ext.get("missing"), "warnings": warns,
        "reason": sc.get("reason"),
    }


def format_report(r: dict) -> str:
    L = ["=" * 68, f"基本面质量：{r.get('ticker', '')}", "=" * 68]
    if not r.get("ok"):
        L.append(f"  无法评估：{r.get('reason')}")
        for w in r.get("warnings") or []:
            L.append(f"    - {w}")
        L.append("=" * 68)
        return "\n".join(L)

    L.append(f"  期间 {r.get('period')}（FY{r.get('fiscal_year')}，"
             f"{r.get('currency')}）")
    L.append(f"  评分 {r['score']}/100   等级 {r['level']}   {r['note']}")
    L.append("-" * 68)
    for k, v in (r.get("breakdown") or {}).items():
        full = SUB_WEIGHTS.get(k, 0)
        shown = "-" if v is None else format(v, ".1f")
        L.append("  " + f"{k:<8}{shown:>7} / {full:.0f}")
    L.append("-" * 68)
    m = r.get("metrics") or {}
    for k, lab in (("gross_margin", "毛利率"), ("net_margin", "净利率"),
                   ("roe", "ROE"), ("roa", "ROA"),
                   ("debt_ratio", "资产负债率"), ("current_ratio", "流动比率"),
                   ("ocf_to_net", "现金流/净利"),
                   ("revenue_growth", "营收增速"), ("net_growth", "净利增速")):
        if k in m:
            L.append(f"  {lab:<10}{m[k]*100:>9.2f}%")
    if r.get("missing"):
        L.append(f"  缺字段：{', '.join(r['missing'])}")
    if r.get("low_confidence"):
        L.append("  ⚠ 可算子项不足一半，评分仅供参考")
    if r.get("capped_by_coverage"):
        L.append("  ⚠ 可算子项不足 3/4，评分已封顶 60")
    for w in r.get("warnings") or []:
        L.append(f"  ⚠ {w}")
    L.append("=" * 68)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    INC = [
        {"period_end_date": "2026-09-30", "fiscal_year": 2026, "currency": "USD",
         "is_sales_revenue_turnover": 478834e6, "is_gross_profit": 233602e6,
         "is_oper_income": 158063e6, "net_income": 112000e6},
        {"period_end_date": "2025-09-30", "fiscal_year": 2025, "currency": "USD",
         "is_sales_revenue_turnover": 391035e6, "is_gross_profit": 180683e6,
         "is_oper_income": 123216e6, "net_income": 93736e6},
    ]
    BAL = [{"total_assets": 364980e6, "total_liabilities": 308030e6,
            "total_equity": 56950e6, "current_assets": 152987e6,
            "current_liabilities": 145308e6}]
    CF = [{"net_cash_from_operating_activities": 118254e6}]

    ext = extract_metrics(INC, BAL, CF)
    assert ext["ok"]
    m = ext["metrics"]
    print("抽取指标:", {k: round(v, 4) for k, v in m.items()})
    assert abs(m["net_margin"] - 112000 / 478834) < 1e-6
    assert abs(m["roe"] - 112000 / 56950) < 1e-6

    sc = score_quality(m)
    print(f"\n评分 {sc['score']}/100  {sc['level']}  覆盖 {sc['coverage']}")
    print("分解:", sc["breakdown"])
    assert sc["ok"] and 0 <= sc["score"] <= 100

    sc2 = score_quality(extract_metrics(INC)["metrics"])
    print(f"\n只有利润表：{sc2['score']}/100 覆盖 {sc2['coverage']}")
    assert sc2["coverage"] < sc["coverage"]

    assert score_quality({})["ok"] is False
    print("\nOK v" + __version__)
