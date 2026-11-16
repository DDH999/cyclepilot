# -*- coding: utf-8 -*-
"""
动态风控（M13）
====================================================
四道闸门，任何一道触发都会改变交易动作：

1. **策略拥挤度**——你买的和别人买的是不是同一批票？
   拥挤策略在踩踏时没有对手盘。用集中度、换手、同质化代理衡量。
2. **回撤上限**——亏到线就减仓，而不是"再等等"。
3. **风格暴露上限**——不许把仓位全压在某一类（行业/风格）上。
4. **暂停交易（Kill Switch）**——多个红灯同时亮时直接停手。

输出统一为动作：**NORMAL / REDUCE / HALT**
  NORMAL  → 按原计划
  REDUCE  → 目标仓位 × scale（默认 0.5）
  HALT    → 禁止开新仓，等待冷却

设计原则
--------
* 全部是**纯函数**：给定状态返回决策，不含内部状态（冷却期由调用方传天数）
* 触发原因必须逐条列出，便于复盘「当时为什么停」
* 宁可误停，不可在极端情况继续加仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["RiskLimits", "DEFAULTS", "crowding_score", "drawdown_state",
           "style_exposure", "kill_switch", "format_report"]


class RiskLimits:
    """风控阈值（经验值，按自己的风险承受度调整）。"""

    def __init__(self,
                 max_drawdown: float = 0.20,
                 warn_drawdown: float = 0.12,
                 max_top5_share: float = 0.50,
                 max_style_exposure: float = 0.40,
                 max_turnover_annual: float = 6.0,
                 max_crowding: float = 70.0,
                 reduce_scale: float = 0.5,
                 cooldown_days: int = 20):
        self.max_drawdown = max_drawdown
        self.warn_drawdown = warn_drawdown
        self.max_top5_share = max_top5_share
        self.max_style_exposure = max_style_exposure
        self.max_turnover_annual = max_turnover_annual
        self.max_crowding = max_crowding
        self.reduce_scale = reduce_scale
        self.cooldown_days = cooldown_days


DEFAULTS = RiskLimits()


# ══════════════════════════════════════════════════════════
# 1. 策略拥挤度
# ══════════════════════════════════════════════════════════
def crowding_score(weights: pd.DataFrame | None = None,
                   turnover_annual: float | None = None,
                   bench_r2: float | None = None,
                   participation: float | None = None,
                   limits: RiskLimits = DEFAULTS) -> dict:
    """
    拥挤度评分 0~100（越高越拥挤）。

    四个维度（各 25 分）：
      * 持仓集中度：前 5 大权重占比（50% 以上接近满扣）
      * 换手率：年换手越高，越依赖持续的资金接力
      * 同质化：与基准 R² 越高，说明和大家拿的是同一批票
      * 流动性占用：委托额 / 日均成交额，参与率高则退出困难

    任一维度数据缺失时，该维度不计分，总分按已给维度等比例放大
    （并在 breakdown 里标明"已归一"）。
    """
    parts: dict[str, float | None] = {}

    if weights is not None and len(weights):
        w = weights.fillna(0.0)
        rows = []
        for _, row in w.iterrows():
            a = row.to_numpy(dtype="float64")
            s = a.sum()
            if s > 0:
                p = np.sort(a / s)[::-1]
                rows.append(float(p[:5].sum()))
        top5 = float(np.median(rows)) if rows else None
        if top5 is not None:
            parts["集中度"] = float(np.clip((top5 - 0.2) / (0.7 - 0.2), 0, 1) * 25)
    else:
        parts["集中度"] = None

    parts["换手率"] = (None if turnover_annual is None
                     else float(np.clip((turnover_annual - 2) / (12 - 2), 0, 1) * 25))

    parts["同质化"] = (None if bench_r2 is None or not np.isfinite(bench_r2)
                     else float(np.clip((bench_r2 - 0.3) / (0.9 - 0.3), 0, 1) * 25))

    parts["流动性"] = (None if participation is None or not np.isfinite(participation)
                     else float(np.clip(participation / 0.10, 0, 1) * 25))

    valid = {k: v for k, v in parts.items() if v is not None}
    if not valid:
        return {"ok": False, "score": None, "reason": "拥挤度所需数据全部缺失"}

    # 不做「按可用维度归一化」。
    # 归一化会把「2 个维度都打满」放大成 100 分，
    # 实测导致只有 3 只基金的组合被判「高度拥挤」——纯属误报。
    # 正确做法：分数只用已知维度累加（每维上限 25），并如实报告覆盖率。
    raw = sum(valid.values())
    coverage = len(valid) / 4.0
    score = float(np.clip(raw, 0, 100))

    if score < 40:
        level, note = "低", "拥挤度低，踩踏风险有限"
    elif score < limits.max_crowding:
        level, note = "中", "有一定拥挤，注意换手与集中度"
    else:
        level, note = "高", "高度拥挤：一旦资金撤离，可能缺乏对手盘"

    if coverage <= 0.5:
        note = note + f"（仅 {len(valid)}/{4} 项可算，结论仅供参考）"
    elif coverage < 1.0:
        note = note + f"（{len(valid)}/{4} 项可算）"

    return {"ok": True, "score": round(score, 2), "level": level, "note": note,
            "breakdown": {k: (None if v is None else round(v, 2))
                          for k, v in parts.items()},
            "n_available": len(valid), "coverage": round(coverage, 2),
            "low_confidence": coverage <= 0.5,
            "max_possible": round(25.0 * len(valid), 1)}


# ══════════════════════════════════════════════════════════
# 2. 回撤状态
# ══════════════════════════════════════════════════════════
def drawdown_state(nav, limits: RiskLimits = DEFAULTS) -> dict:
    """当前回撤、峰值、距上限还有多少、是否触发。"""
    s = pd.Series(nav).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if len(s) < 2:
        return {"ok": False, "reason": "净值不足 2 个点"}
    peak = np.maximum.accumulate(s.to_numpy())
    cur_nav = float(s.iloc[-1])
    cur_peak = float(peak[-1]) if peak[-1] > 0 else cur_nav
    dd = cur_nav / cur_peak - 1 if cur_peak > 0 else 0.0
    mdd = float(np.nanmin(s.to_numpy() / np.where(peak <= 0, np.nan, peak) - 1))

    if dd <= -limits.max_drawdown:
        status = "HALT"
    elif dd <= -limits.warn_drawdown:
        status = "REDUCE"
    else:
        status = "NORMAL"

    return {"ok": True, "current_drawdown": round(dd, 6),
            "max_drawdown": round(mdd, 6),
            "peak": round(cur_peak, 2), "nav": round(cur_nav, 2),
            "breach_limit": -limits.max_drawdown,
            "buffer": round(abs(dd) - limits.max_drawdown, 6),
            "status": status}


# ══════════════════════════════════════════════════════════
# 3. 风格暴露
# ══════════════════════════════════════════════════════════
def style_exposure(weights: pd.DataFrame, groups: dict[str, str],
                   limits: RiskLimits = DEFAULTS,
                   top: int = 5) -> dict:
    """
    按分组（行业/风格）统计暴露，检查是否超过上限。

    weights: 行=日期，列=代码；groups: 代码 → 分组名
    未在 groups 里的代码归为「未知」。
    """
    if weights is None or len(weights) == 0:
        return {"ok": False, "reason": "无持仓数据"}
    w = weights.fillna(0.0)
    last = w.iloc[-1]
    total = float(last.sum())
    if total <= 0:
        return {"ok": False, "reason": "最后一期空仓"}

    shares: dict[str, float] = {}
    for code, val in last.items():
        if val <= 0:
            continue
        g = groups.get(str(code), "未知")
        shares[g] = shares.get(g, 0.0) + float(val) / total

    ranked = sorted(shares.items(), key=lambda x: -x[1])
    breaches = [{"group": g, "share": round(v, 4)}
                for g, v in ranked if v > limits.max_style_exposure]

    return {"ok": True, "n_groups": len(ranked),
            "top": [{"group": g, "share": round(v, 4)} for g, v in ranked[:top]],
            "breaches": breaches,
            "max_share": round(ranked[0][1], 4) if ranked else 0.0,
            "limit": limits.max_style_exposure,
            "status": "REDUCE" if breaches else "NORMAL"}


# ══════════════════════════════════════════════════════════
# 4. 暂停交易（Kill Switch）
# ══════════════════════════════════════════════════════════
def kill_switch(nav, weights: pd.DataFrame | None = None,
                groups: dict[str, str] | None = None,
                turnover_annual: float | None = None,
                bench_r2: float | None = None,
                participation: float | None = None,
                data_ok: bool | None = None,
                code_cheat: bool = False,
                days_since_halt: int | None = None,
                limits: RiskLimits = DEFAULTS) -> dict:
    """
    综合决策：NORMAL / REDUCE / HALT。

    优先级（任一满足即升级）：
      HALT  ← 代码作弊 / 数据不可用 / 回撤超上限
      REDUCE← 回撤警戒 / 拥挤过高 / 风格超限 / 换手超限
      另外：HALT 后需冷却 cooldown_days 天才能恢复（由调用方传 days_since_halt）
    """
    reasons: list[dict] = []
    action = "NORMAL"
    scale = 1.0

    def escalate(a: str, why: str, level: str = "mid") -> None:
        nonlocal action, scale
        reasons.append({"action": a, "level": level, "why": why})
        if a == "HALT":
            action, scale = "HALT", 0.0
        elif a == "REDUCE" and action != "HALT":
            action, scale = "REDUCE", limits.reduce_scale

    if code_cheat:
        escalate("HALT", "代码检出未来函数/作弊，结果不可信", "high")
    if data_ok is False:
        escalate("HALT", "数据质量不合格，禁止据此交易", "high")

    dd = drawdown_state(nav, limits) if nav is not None else None
    if dd and dd.get("ok"):
        if dd["status"] == "HALT":
            escalate("HALT", f"回撤 {dd['current_drawdown']*100:.1f}% 超过上限 "
                             f"{limits.max_drawdown*100:.0f}%", "high")
        elif dd["status"] == "REDUCE":
            escalate("REDUCE", f"回撤 {dd['current_drawdown']*100:.1f}% 进入警戒区",
                     "mid")

    cr = crowding_score(weights, turnover_annual, bench_r2, participation, limits)
    if cr.get("ok") and cr["score"] >= limits.max_crowding:
        escalate("REDUCE", f"拥挤度 {cr['score']:.0f} 超过上限 "
                           f"{limits.max_crowding:.0f}（{cr['note']}）", "mid")

    if turnover_annual is not None and turnover_annual > limits.max_turnover_annual:
        escalate("REDUCE", f"年换手 {turnover_annual:.1f}x 超过上限 "
                           f"{limits.max_turnover_annual:.1f}x", "mid")

    if weights is not None and groups:
        se = style_exposure(weights, groups, limits)
        if se.get("ok") and se["breaches"]:
            names = "、".join(f"{b['group']}({b['share']*100:.0f}%)"
                              for b in se["breaches"][:3])
            escalate("REDUCE", f"风格暴露超限：{names} > "
                               f"{limits.max_style_exposure*100:.0f}%", "mid")

    cooled = True
    if action == "HALT" and days_since_halt is not None:
        cooled = days_since_halt >= limits.cooldown_days
        if not cooled:
            reasons.append({"action": "HALT", "level": "info",
                            "why": f"距上次暂停 {days_since_halt} 天，"
                                   f"冷却期 {limits.cooldown_days} 天未满"})

    return {"action": action, "scale": round(scale, 4),
            "reasons": reasons,
            "cooldown_remaining": (max(limits.cooldown_days - (days_since_halt or 0), 0)
                                   if action == "HALT" else 0),
            "drawdown": dd, "crowding": cr,
            "limits": {"max_drawdown": limits.max_drawdown,
                       "max_style_exposure": limits.max_style_exposure,
                       "max_crowding": limits.max_crowding,
                       "max_turnover_annual": limits.max_turnover_annual}}


def format_report(res: dict) -> str:
    L = ["=" * 66, "动态风控决策", "=" * 66]
    icon = {"NORMAL": "[正常]", "REDUCE": "[减仓]", "HALT": "[暂停]"}[res["action"]]
    L.append(f"  动作: {icon}   目标仓位系数: {res['scale']:.2f}")
    if res.get("cooldown_remaining"):
        L.append(f"  冷却期剩余: {res['cooldown_remaining']} 天")

    dd = res.get("drawdown") or {}
    if dd.get("ok"):
        L.append("-" * 66)
        L.append(f"  当前回撤 {dd['current_drawdown']*100:>7.2f}%"
                 f"   历史最大 {dd['max_drawdown']*100:>7.2f}%"
                 f"   上限 -{abs(dd['breach_limit'])*100:.0f}%")
    cr = res.get("crowding") or {}
    if cr.get("ok"):
        L.append(f"  拥挤度   {cr['score']:>7.1f}   {cr['level']}  {cr['note']}")

    if res.get("reasons"):
        L.append("-" * 66)
        L.append("  触发原因:")
        for r in res["reasons"]:
            tag = {"high": "[严重]", "mid": "[中等]", "info": "[信息]"}[r["level"]]
            L.append(f"    {tag} {r['why']}")
    else:
        L.append("-" * 66)
        L.append("  无触发项")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    idx = pd.bdate_range("2024-01-02", periods=300)
    codes = [f"{i:06d}" for i in range(20)]
    W = pd.DataFrame(1 / 20, index=idx, columns=codes)
    nav = pd.Series(np.linspace(100000, 108000, 300), index=idx)
    # 4 个风格各 25%，天然不超 40% 上限
    styles = ["银行", "科技", "消费", "医药"]
    groups = {c: styles[i // 5] for i, c in enumerate(codes)}

    r1 = kill_switch(nav, W, groups, turnover_annual=4.0, bench_r2=0.5,
                     participation=0.01, data_ok=True)
    assert r1["action"] == "NORMAL", r1
    print(format_report(r1))

    crash = pd.Series(np.concatenate([np.linspace(100000, 130000, 200),
                                      np.linspace(130000, 95000, 100)]), index=idx)
    r2 = kill_switch(crash, W, groups, turnover_annual=9.0, bench_r2=0.85,
                     participation=0.05, data_ok=True)
    assert r2["action"] == "HALT", r2["action"]
    print(format_report(r2))
    print(f"OK v{__version__}")
