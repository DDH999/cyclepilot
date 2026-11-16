# -*- coding: utf-8 -*-
"""
实盘决策引擎（M36）
====================================================
前面的模块都在"体检"和"测量"，这一层回答唯一重要的问题：

    **今天该做什么？**

针对你的情况（4 个海外指数 QDII 定投、长线、不动杠杆），四个决策源：

1. **估值/价格分位 → 定投倍数**
   分位低多投、分位高少投。规则固定、可复现、不靠感觉。
2. **组合重叠度 → 是否假分散**
   标普500 与纳指100 相关性长期 0.85+，同时定投 ≈ 加杠杆押美国科技。
3. **再平衡 → 偏离多少该调回去**
   涨多的减、跌多的加，把权重拉回目标。
4. **回撤纪律 → 跌多少该加多少**
   预先写死规则，避免跌的时候靠情绪决定。

输出原则
--------
* **最多 3 条动作**，每条带数字和理由
* 没有值得做的事就明确说「**今天不用动**」——
  对长期投资者，避免过度操作本身就是价值
* 所有建议基于**当前可得的公开数据**，不做方向预测

⚠ 诚实的预期
-------------
按估值分位调整定投额（value-averaging）在历史上能把年化 IRR 改善
**约 0.5~1.5 个百分点**，并降低回撤；但它**不是稳赚**，
在长期单边上行时反而跑输"无脑定投"。这是权衡，不是免费午餐。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["DCA_LADDER", "dca_multiplier", "price_percentile",
           "portfolio_correlation", "rebalance_plan", "drawdown_playbook",
           "advise", "format_report"]

# 分位 → 定投倍数（规则写死，可复现）
# 注意：分位统一用 **0~100**（与 valuation / riskcontrol 的口径一致）。
# 早期版本这里写的是 0~1，导致 98% 分位被当成"未知"给了 1.0 倍（踩过）。
DCA_LADDER = [
    (20.0, 1.50, "历史低位：加倍投"),
    (40.0, 1.20, "偏低：小幅多投"),
    (60.0, 1.00, "中性：按计划"),
    (80.0, 0.80, "偏高：减少投入"),
    (100.01, 0.50, "历史高位：半仓观望"),
]

# 回撤 → 额外加仓（占年定投额的比例）
DRAWDOWN_PLAYBOOK = [
    (0.10, 0.00, "回撤 10% 以内：不动"),
    (0.20, 0.25, "回撤 10~20%：额外加 25% 年定投额"),
    (0.35, 0.50, "回撤 20~35%：额外加 50%"),
    (1.00, 1.00, "回撤超 35%：动用全部备用金"),
]


def dca_multiplier(pct: float | None) -> dict:
    """
    价格分位（**0~100**）→ 定投倍数。分位未知返回 1.0（不猜）。

    传 0~1 的值会被识别并自动 ×100（容错，不静默出错）。
    """
    if pct is None or not np.isfinite(pct):
        return {"multiplier": 1.0, "tier": "未知", "note": "分位不可用，按计划投"}
    pct = float(pct)
    if 0.0 < pct <= 1.0:      # 容错：有人会传 0.98 表示 98%
        pct *= 100.0
    for bound, mult, note in DCA_LADDER:
        if pct < bound:
            return {"multiplier": mult, "tier": note,
                    "percentile": round(pct, 1)}
    return {"multiplier": 1.0, "tier": "未知", "note": "分位异常"}


def price_percentile(index_key: str, years: float = 3.0,
                     days: int = 1000) -> dict:
    """
    指数当前价在近 N 年中的分位（0~100）。

    注意：这是**价格分位**，不是 PE 分位。
    对宽基指数长期上行，价格分位天然偏高，所以阈值用宽一些（20/40/60/80）。
    """
    import overseas as OS
    try:
        got = OS.index_history(index_key, days=days)
    except Exception as ex:
        return {"ok": False, "key": index_key, "reason": str(ex)[:120]}
    if not got.get("ok"):
        return {"ok": False, "key": index_key,
                "reason": got.get("reason") or "取数失败"}

    s = got["series"]
    days_needed = int(252 * years)
    s = s.tail(min(days_needed, len(s)))
    if len(s) < 60:
        return {"ok": False, "key": index_key,
                "reason": f"历史仅 {len(s)} 个点，不足以算分位"}

    cur = float(s.iloc[-1])
    pct = float((s < cur).sum() / len(s) * 100.0)
    return {"ok": True, "key": index_key, "name": got["name"],
            "current": round(cur, 2), "percentile": round(pct, 1),
            "source": got["source"], "n": len(s),
            "low": round(float(s.min()), 2), "high": round(float(s.max()), 2),
            "years": years}


def portfolio_correlation(navs: dict[str, pd.Series]) -> dict:
    """
    组合内相关性矩阵。用来发现「假分散」。

    净值序列传入，返回日收益相关性。
    """
    series = {}
    for code, s in navs.items():
        r = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 40:
            continue
        mean_abs = abs(float(r.mean()))
        cv = (float(r.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
        if float((r > 0).mean()) > 0.95 and cv < 0.5:
            r = r.pct_change(fill_method=None).dropna()
        series[code] = r

    if len(series) < 2:
        return {"ok": False, "reason": "可用净值序列不足 2 条"}

    df = pd.DataFrame(series).dropna()
    if len(df) < 60:
        return {"ok": False, "reason": f"重叠期仅 {len(df)} 天"}

    corr = df.corr()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iloc[i, j])
            pairs.append({"a": cols[i], "b": cols[j], "corr": round(v, 3)})
    pairs.sort(key=lambda x: -x["corr"])

    high = [p for p in pairs if p["corr"] >= 0.85]
    return {"ok": True, "n_days": len(df), "pairs": pairs,
            "high_pairs": high,
            "avg_corr": round(float(np.mean([p["corr"] for p in pairs])), 3),
            "verdict": ("高度同质：这几只基金涨跌几乎同步，同时持有≈加杠杆"
                        if len(high) >= 1 else
                        "相关性可接受，分散有一定效果")}


def rebalance_plan(weights: dict[str, float],
                   targets: dict[str, float] | None = None,
                   band: float = 0.05) -> dict:
    """
    再平衡：当前权重 vs 目标权重的偏离。超过 band 才建议调整。
    """
    if not weights:
        return {"ok": False, "reason": "无持仓权重"}
    weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
    # 权重合计不足 1 时，差额就是现金。
    # 不补这一项的话，单只基金看起来永远"没配满"，会一直建议加仓。
    total_w = sum(weights.values())
    if total_w < 0.995 and "CASH" not in weights:
        weights["CASH"] = round(1.0 - total_w, 6)
    codes = [c for c in weights if weights[c] > 0]
    if targets is None:
        t = 1.0 / len(codes)
        targets = {c: t for c in codes}

    rows = []
    for c in codes:
        cur = float(weights.get(c, 0.0))
        tgt = float(targets.get(c, 0.0))
        gap = cur - tgt
        # 现金不是"持仓"：现金多了要说「投出去」，不是「减仓」
        if gap > band:
            act = "投出现金" if str(c).upper() == "CASH" else "减仓"
        elif gap < -band:
            act = "补充现金" if str(c).upper() == "CASH" else "加仓"
        else:
            act = "不动"
        rows.append({"code": c, "current": round(cur, 4),
                     "target": round(tgt, 4), "gap": round(gap, 4),
                     "action": act})
    need = [r for r in rows if r["action"] != "不动"]
    return {"ok": True, "band": band, "rows": rows, "need_action": need,
            "max_gap": round(max(abs(r["gap"]) for r in rows), 4),
            "verdict": ("偏离在 %.0f%% 以内，不用动" % (band * 100)
                        if not need else
                        f"{len(need)} 只偏离超过 {band*100:.0f}%，建议调回")}


def drawdown_playbook(drawdown: float | None) -> dict:
    """回撤 → 该额外加多少（占年定投额）。"""
    if drawdown is None or not np.isfinite(drawdown):
        return {"ok": False, "reason": "缺回撤数据", "extra_pct": 0.0}
    dd = abs(float(drawdown))
    for bound, extra, note in DRAWDOWN_PLAYBOOK:
        if dd < bound:
            return {"ok": True, "drawdown": round(-dd, 4),
                    "extra_pct": extra, "note": note}
    return {"ok": True, "drawdown": round(-dd, 4), "extra_pct": 1.0,
            "note": "回撤超 35%：动用全部备用金"}


# ══════════════════════════════════════════════════════════
def advise(holdings_codes: list[str],
           fund_names: dict[str, str] | None = None,
           weights: dict[str, float] | None = None,
           monthly_amount: float = 1000.0,
           portfolio_drawdown: float | None = None,
           index_percentiles: dict[str, dict] | None = None,
           use_ml: bool = False) -> dict:
    """
    汇总成「今天该做什么」。

    index_percentiles 可外部传入（避免重复联网），形如 {"sp500": {...}}。
    use_ml: 是否叠加机器学习回撤风险信号（慢一些，且只有通过验证的市场才用）。
    """
    fund_names = fund_names or {}
    actions: list[dict] = []
    details: dict = {}

    # ── 1) 每只基金的基准指数分位 → 定投倍数
    import overseas as OS
    per_index: dict[str, dict] = {}
    mapping: dict[str, str] = {}
    for code in holdings_codes:
        name = fund_names.get(code, "")
        m = OS.match_fund_index(name) if name else {"ok": False}
        if not m.get("ok"):
            continue
        key = m["key"]
        mapping[code] = key
        if index_percentiles and key in index_percentiles:
            per_index[key] = index_percentiles[key]
        elif key not in per_index:
            per_index[key] = price_percentile(key)

    if per_index:
        details["index_percentiles"] = per_index
        lows = [(k, v) for k, v in per_index.items()
                if v.get("ok") and v["percentile"] < 40]
        # 只对最极端的那一个动，避免一次给出过多建议
        highs = [(k, v) for k, v in per_index.items()
                 if v.get("ok") and v["percentile"] >= 70]
        if lows:
            k, v = min(lows, key=lambda x: x[1]["percentile"])
            mul = dca_multiplier(v["percentile"])
            actions.append({
                "action": "加投",
                "target": v["name"],
                "detail": f"价格处于近 {v['years']:.0f} 年 {v['percentile']:.0f}% 分位"
                          f"（当前 {v['current']}），按规则本期投 {mul['multiplier']:.1f} 倍",
                "amount": round(monthly_amount * mul["multiplier"], 2),
                "reason": mul["tier"], "priority": 1,
            })
        if highs:
            k, v = max(highs, key=lambda x: x[1]["percentile"])
            mul = dca_multiplier(v["percentile"])
            actions.append({
                "action": "减投",
                "target": v["name"],
                "detail": f"价格处于近 {v['years']:.0f} 年 {v['percentile']:.0f}% 分位"
                          f"（当前 {v['current']}），按规则本期投 {mul['multiplier']:.1f} 倍",
                "amount": round(monthly_amount * mul["multiplier"], 2),
                "reason": mul["tier"], "priority": 2,
            })

    # ── 2) 再平衡
    if weights:
        rb = rebalance_plan(weights)
        details["rebalance"] = rb
        if rb.get("ok") and rb["need_action"]:
            worst = max(rb["need_action"], key=lambda r: abs(r["gap"]))
            actions.append({
                "action": worst["action"],
                "target": worst["code"],
                "detail": f"当前 {worst['current']*100:.1f}% vs 目标 "
                          f"{worst['target']*100:.1f}%，偏离 "
                          f"{worst['gap']*100:+.1f}%",
                "amount": None, "reason": rb["verdict"], "priority": 3,
            })

    # ── 3) 回撤加仓纪律
    dp = drawdown_playbook(portfolio_drawdown)
    details["drawdown"] = dp
    if dp.get("ok") and dp["extra_pct"] > 0:
        actions.append({
            "action": "额外加仓",
            "target": "组合",
            "detail": f"组合回撤 {abs(dp['drawdown'])*100:.1f}%，"
                      f"按纪律可额外投入年定投额的 {dp['extra_pct']*100:.0f}%",
            "amount": round(monthly_amount * 12 * dp["extra_pct"], 2),
            "reason": dp["note"], "priority": 2,
        })

    # ── 4) 机器学习回撤风险（可选）
    if use_ml and mapping:
        try:
            import ml_risk as ML
            ml_info = {}
            for key in sorted(set(mapping.values())):
                cur = ML.current_risk(key)
                ml_info[key] = cur
                if not cur.get("ok") or not cur.get("usable"):
                    continue
                mul = ML.risk_multiplier(cur.get("probability"),
                                         cur.get("base_rate"))
                if mul["multiplier"] == 1.0:
                    continue
                actions.append({
                    "action": ("减投" if mul["multiplier"] < 1.0 else "加投"),
                    "target": cur.get("name") or key,
                    "detail": f"未来 20 天回撤概率 {cur['probability']*100:.1f}%"
                              f"（历史均值 {cur['base_rate']*100:.1f}%），"
                              f"按风险规则投 {mul['multiplier']:.1f} 倍",
                    "amount": round(monthly_amount * mul["multiplier"], 2),
                    "reason": f"ML 风险信号（AUC {cur.get('mean_auc')}）",
                    "priority": 1,
                })
            details["ml_risk"] = ml_info
        except Exception as ex:
            details["ml_risk_error"] = str(ex)[:150]

    actions.sort(key=lambda x: x["priority"])

    if not actions:
        verdict = "今天不用动。没有触发任何规则——对长线定投来说这是常态。"
    else:
        verdict = f"有 {len(actions)} 条动作，按优先级从高到低执行。"

    return {"ok": True, "n_actions": len(actions),
            "actions": actions[:3], "details": details,
            "mapping": mapping, "monthly_amount": monthly_amount,
            "verdict": verdict}


def format_report(res: dict) -> str:
    L = ["=" * 68, "今日决策", "=" * 68]
    if not res.get("ok"):
        L.append(f"  无法生成建议：{res.get('reason')}")
        L.append("=" * 68)
        return "\n".join(L)

    L.append(f"  月定投基数 {res['monthly_amount']:,.0f} 元")
    L.append(f"  {res['verdict']}")
    L.append("-" * 68)

    if not res["actions"]:
        L.append("  [不动] 没有规则被触发")
    for i, a in enumerate(res["actions"], 1):
        L.append(f"  {i}. [{a['action']}] {a['target']}")
        L.append(f"     {a['detail']}")
        if a.get("amount"):
            L.append(f"     金额参考 {a['amount']:,.0f} 元")
        L.append(f"     依据 {a['reason']}")

    det = res.get("details") or {}
    if det.get("index_percentiles"):
        L.append("-" * 68)
        L.append("  指数价格分位：")
        for k, v in det["index_percentiles"].items():
            if v.get("ok"):
                L.append(f"    {v['name']:<12}{v['current']:>10,.2f}   "
                         f"{v['percentile']:>5.1f}% 分位"
                         f"   （{v['n']} 天，{v['source']}）")
            else:
                L.append(f"    {k:<12} 取数失败：{str(v.get('reason'))[:40]}")

    if det.get("rebalance", {}).get("ok"):
        rb = det["rebalance"]
        L.append("-" * 68)
        L.append(f"  再平衡（阈值 {rb['band']*100:.0f}%）：{rb['verdict']}")
        for r in rb["rows"]:
            L.append(f"    {r['code']:<10}{r['current']*100:>6.1f}%"
                     f" vs 目标 {r['target']*100:>5.1f}%   {r['action']}")

    if det.get("ml_risk"):
        L.append("-" * 68)
        L.append("  机器学习回撤风险（未来 20 天）：")
        for k, v in det["ml_risk"].items():
            if v.get("ok") and v.get("usable"):
                L.append(f"    {v.get('name', k):<12}"
                         f"P={v['probability']*100:>5.1f}%  "
                         f"均值 {v['base_rate']*100:>4.1f}%  "
                         f"{v['level']}  （AUC {v['mean_auc']}）")
            elif v.get("ok"):
                L.append(f"    {v.get('name', k):<12}不出信号："
                         f"{str(v.get('reason'))[:44]}")
            else:
                L.append(f"    {k:<12}失败：{str(v.get('reason'))[:40]}")
    if det.get("ml_risk_error"):
        L.append(f"  ⚠ ML 风险模块异常：{det['ml_risk_error']}")

    if det.get("drawdown", {}).get("ok"):
        dp = det["drawdown"]
        L.append("-" * 68)
        L.append(f"  回撤纪律：当前 {dp['drawdown']*100:.1f}%   {dp['note']}")

    L.append("=" * 68)
    L.append("  ⚠ 本建议基于公开数据与固定规则，不预测方向、不保证收益。")
    L.append("     按分位调整定投长期约改善 0.5~1.5 个百分点年化，单边上行时会跑输。")
    L.append("=" * 68)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 规则表自检 ===")
    for pct in (10, 30, 50, 70, 90, None):
        m = dca_multiplier(pct)
        print(f"  分位 {str(pct):<5} → {m['multiplier']:.2f}x  {m['tier']}")
    assert dca_multiplier(10)["multiplier"] == 1.5
    assert dca_multiplier(90)["multiplier"] == 0.5
    assert dca_multiplier(None)["multiplier"] == 1.0
    # 容错：传 0.98 也要当成 98%
    assert dca_multiplier(0.98)["multiplier"] == 0.5, dca_multiplier(0.98)
    assert dca_multiplier(98.3)["multiplier"] == 0.5

    print("\n=== 再平衡 ===")
    rb = rebalance_plan({"A": 0.55, "B": 0.25, "C": 0.20})
    print(" ", rb["verdict"])
    for r in rb["rows"]:
        print(f"    {r['code']} {r['current']*100:.0f}% vs {r['target']*100:.0f}% → {r['action']}")
    assert rb["need_action"]

    print("\n=== 回撤纪律 ===")
    for dd in (-0.05, -0.15, -0.30, -0.45):
        dp = drawdown_playbook(dd)
        print(f"  回撤 {dd*100:>6.1f}% → 额外 {dp['extra_pct']*100:>3.0f}%  {dp['note']}")
    assert drawdown_playbook(-0.45)["extra_pct"] == 1.0

    print("\nOK v" + __version__)
