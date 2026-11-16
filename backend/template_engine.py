# -*- coding: utf-8 -*-
"""
本地规则模板引擎（M3 降级兜底）
====================================================
职责：当大模型不可用（无 Key / 断网 / 限流 / 超额 / 返回异常）时，
**仅凭六维体检数据**生成结构合规的诊断结论，绝不让调用方拿到异常。

输出严格包含三部分：
  1. summary  综合定性（一句话：优秀 / 中性 / 高危）
  2. fatal    核心致命伤（扣分最严重的维度 + 原因）
  3. advice   整改建议（可执行的具体方向）

设计约束
--------
* 纯标准库，零外部依赖
* **纯函数**：同样的输入必然得到同样的输出（可测试、可复现）
* 任何畸形输入都不抛异常，缺失字段按"未知"降级
"""
from __future__ import annotations

from typing import Iterable

__version__ = "0.1.0"
__all__ = [
    "DIM_LABEL", "DIM_ORDER", "SEVERITY_BANDS",
    "summarize", "find_fatal", "build_advice", "render", "format_report",
]

DIM_LABEL = {
    "anti_cheat": "防作弊",
    "anti_overfit": "抗过拟合",
    "stress": "抗压能力",
    "purity": "策略纯度",
    "cost": "成本可控",
    "stability": "稳健性",
}
DIM_ORDER = ["anti_cheat", "anti_overfit", "stress", "purity", "cost", "stability"]

# (下限分数, 严重度标签)
SEVERITY_BANDS = [(85.0, "低"), (70.0, "中"), (55.0, "较高"), (0.0, "高")]


def _severity(score: float) -> str:
    for lo, label in SEVERITY_BANDS:
        if score >= lo:
            return label
    return "高"


def _num(v) -> float | None:
    """安全转数值；失败返回 None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _radar(report: dict) -> dict[str, float]:
    """从报告里取出「有分数」的维度，忽略 N/A。"""
    radar = (report or {}).get("radar") or {}
    out: dict[str, float] = {}
    for k in DIM_ORDER:
        v = _num(radar.get(k)) if isinstance(radar, dict) else None
        if v is not None:
            out[k] = v
    return out


def _dims(report: dict) -> dict[str, dict]:
    """按 key 索引维度明细。"""
    out = {}
    for d in (report or {}).get("dimensions") or []:
        if isinstance(d, dict) and d.get("key"):
            out[d["key"]] = d
    return out


# ══════════════════════════════════════════════════════════
# 一、综合定性
# ══════════════════════════════════════════════════════════
def summarize(report: dict) -> str:
    """
    一句话定性。
    优先按裁决结论（REJECTED/FAIL/WARNING/PASS）+ 评级给出，避免"高分但被否决"的矛盾。
    """
    report = report or {}
    verdict = str(report.get("verdict") or "").upper()
    grade = str(report.get("grade") or "")
    capped = bool(report.get("grade_capped"))
    kind = report.get("reject_kind")

    if verdict == "REJECTED":
        if kind == "cheat":
            return ("综合定性：该策略属于**高危**——检出未来函数（数据穿越）嫌疑，"
                    "属主观作弊，不具备任何可用性。")
        return ("综合定性：该策略**无法评估**——核心指标缺失或计算失败，"
                "不能作为有效策略对待。")

    if verdict == "FAIL":
        return f"综合定性：该策略属于**高危**——评级 {grade}，存在未达标项，不具备上线条件。"

    if grade == "A":
        return "综合定性：该策略**优秀**——六维体检全部达标，未发现明显缺陷。"

    if grade == "B":
        note = "（存在数学计算异常，评级已封顶）" if capped else ""
        return f"综合定性：该策略**中性偏好**——评级 {grade}{note}，整体可用但有改进空间。"

    if grade == "C":
        note = "（存在数学计算异常，评级已封顶）" if capped else ""
        return (f"综合定性：该策略**中性偏谨慎**——评级 {grade}{note}，"
                "需针对性整改后方可考虑使用。")

    return f"综合定性：该策略属于**高危**——评级 {grade}，多项指标未达标。"


# ══════════════════════════════════════════════════════════
# 二、核心致命伤
# ══════════════════════════════════════════════════════════
_FATAL_TEXT = {
    "anti_cheat": ("策略存在未来函数（数据穿越），信号使用了当时不可得的信息，"
                   "回测结论完全不可采信。"),
    "anti_overfit": ("过拟合特征明显：样本外表现无法支撑样本内结论，"
                     "该策略大概率是对历史噪声的拟合。"),
    "stress": ("极端行情与高摩擦环境下表现失控：扣除成本后收益不足，"
               "或区间回撤显著放大，抗压能力不足。"),
    "purity": ("策略纯度存疑：收益可能并非来自选股能力，"
               "而是市场 beta、少数标的押注或长期空仓所致。"),
    "cost": ("交易成本侵蚀严重：换手过高导致滑点与手续费吞掉大部分收益。"),
    "stability": ("结论对参数高度敏感：参数邻域内表现剧烈波动，"
                  "说明当前结果是特定参数点的巧合，而非稳健规律。"),
}


_MILD_TEXT = {
    "anti_cheat": "代码真实性存在疑点，建议人工复核。",
    "anti_overfit": "样本外表现略弱于样本内，需持续观察其稳定性。",
    "stress": "极端行情下的抗压表现一般，建议补充风控缓冲。",
    "purity": "策略纯度尚可，但仍有 beta 或集中度方面的改进空间。",
    "cost": "交易成本略高，长期看会侵蚀收益。",
    "stability": "结论对参数有一定敏感性，建议关注参数邻域表现。",
}


def find_fatal(report: dict) -> str:
    """
    指出扣分最严重的维度并解释。
    一票否决时直接给出否决原因；否则取可用维度中得分最低者。
    """
    report = report or {}
    if str(report.get("verdict") or "").upper() == "REJECTED":
        if report.get("reject_kind") == "cheat":
            return "核心致命伤：" + _FATAL_TEXT["anti_cheat"]
        return ("核心致命伤：核心指标缺失（净值或收益序列无法计算），"
                "策略未跑通，无法进入评估流程。")

    radar = _radar(report)
    if not radar:
        return "核心致命伤：缺乏可用的维度得分，无法定位主要问题。"

    worst = min(radar, key=lambda k: radar[k])
    score = radar[worst]
    label = DIM_LABEL.get(worst, worst)

    # 全部达标时不得虚构"致命伤"
    if score >= _HEALTHY:
        return (f"核心致命伤：未发现致命缺陷（最低维度为{label}，"
                f"得分 {score:.0f} 分，仍在健康区间）。")
    sev = _severity(score)
    # 按严重度选措辞：>=55 分为中度，用缓和表述，避免小问题被说成致命伤
    if score >= 55.0:
        detail = _MILD_TEXT.get(worst, "该维度表现一般。")
    else:
        detail = _FATAL_TEXT.get(worst, "该维度表现不达标。")

    # 优先用维度自身给的说明补充
    dim = _dims(report).get(worst) or {}
    note = str(dim.get("note") or "").strip()
    tail = f"（体检说明：{note}）" if note else ""

    return (f"核心致命伤：**{label}**得分 {score:.0f} 分，严重度{sev}。"
            f"{detail}{tail}")


# ══════════════════════════════════════════════════════════
# 三、整改建议
# ══════════════════════════════════════════════════════════
_FIX = {
    "anti_cheat": "删除全部未来函数与数据穿越写法，严格执行 T 日生成信号、T+1 日成交。",
    "anti_overfit": ("减少参数搜索次数并记录全部尝试，改用 Purged Walk-Forward 切分；"
                     "扩大样本外区间，同时报告 Deflated Sharpe 与参数敏感性。"),
    "stress": ("增加组合级回撤制动与个股止损，保留现金缓冲；"
               "在高滑点假设下重跑，确认收益仍成立。"),
    "purity": ("提高持仓数量至 20 只以上并限制单票权重；"
               "若与基准 R² 过高，应做 beta 中性化后再评估超额收益。"),
    "cost": "降低换手率、延长持仓周期，设置最小调仓幅度，改用限价单控制滑点。",
    "stability": "在参数邻域内选平台区而非尖峰点，并分年度、分市场状态分别验证。",
}
_THRESHOLD = 75.0      # 低于该分数才给整改建议
_HEALTHY = 85.0        # 所有维度都不低于此值 → 视为无致命伤


def build_advice(report: dict, max_items: int = 4) -> str:
    """针对低分维度给出可执行建议；全部达标时给出运维建议。"""
    report = report or {}
    radar = _radar(report)

    if str(report.get("verdict") or "").upper() == "REJECTED":
        if report.get("reject_kind") == "cheat":
            return ("整改建议：该策略不可通过修改参数挽救，必须重写信号生成逻辑，"
                    "确保所有输入在决策时点均已可得；完成后重新提交体检。")
        return ("整改建议：先修复数据管道（收益序列、净值、成本口径），"
                "确保基础指标可计算，再进入六维评估。")

    weak = sorted([(k, v) for k, v in radar.items() if v < _THRESHOLD],
                  key=lambda x: x[1])
    if not weak:
        notes = [str(n) for n in (report.get("notes") or []) if n]
        extra = f"另需关注：{'；'.join(notes[:2])}" if notes else ""
        return ("整改建议：各维度均已达标，建议保持现有风控阈值，"
                "并定期用新样本复检以避免环境漂移。" + extra)

    items = []
    for k, v in weak[:max_items]:
        label = DIM_LABEL.get(k, k)
        fix = _FIX.get(k, "针对该维度做专项优化。").rstrip("。；")
        items.append(f"{label}（{v:.0f} 分）：{fix}")
    return "整改建议：" + "；".join(items) + "。"


# ══════════════════════════════════════════════════════════
# 组合输出
# ══════════════════════════════════════════════════════════
def render(report: dict) -> dict:
    """
    生成完整诊断（三部分 + 结构化 issues）。

    返回
    ----
    summary / fatal / advice : 三段文本
    issues                   : [{dimension, severity, score, fix}]
    source                   : 固定 "template"
    """
    report = report or {}
    radar = _radar(report)
    weak = sorted([(k, v) for k, v in radar.items() if v < _THRESHOLD],
                  key=lambda x: x[1])

    issues = [{
        "dimension": k,
        "label": DIM_LABEL.get(k, k),
        "score": round(v, 2),
        "severity": _severity(v),
        "fix": _FIX.get(k, ""),
    } for k, v in weak]

    return {
        "summary": summarize(report),
        "fatal": find_fatal(report),
        "advice": build_advice(report),
        "issues": issues,
        "source": "template",
        "engine_version": __version__,
    }


def format_report(res: dict) -> str:
    L = ["=" * 66, "策略风险诊断（本地规则模板）", "=" * 66]
    L.append("  " + str(res.get("summary")))
    L.append("")
    L.append("  " + str(res.get("fatal")))
    L.append("")
    L.append("  " + str(res.get("advice")))
    if res.get("issues"):
        L.append("-" * 66)
        L.append("  问题清单:")
        for i in res["issues"]:
            L.append(f"    - [{i['severity']}] {i['label']} "
                     f"{i['score']:.0f} 分 → {i['fix']}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    # 三种典型输入的自测
    good = {"verdict": "PASS", "grade": "A", "radar": {
        k: 95.0 for k in DIM_ORDER}}
    mid = {"verdict": "WARNING", "grade": "C", "grade_capped": True,
           "radar": {"anti_cheat": 100, "anti_overfit": 0, "stress": 90,
                     "purity": 100, "cost": 80, "stability": 80},
           "dimensions": [{"key": "anti_overfit", "note": "PBO 无法计算"}]}
    bad = {"verdict": "REJECTED", "reject_kind": "cheat", "radar": {}}

    for name, rep in [("优秀", good), ("封顶", mid), ("作弊", bad)]:
        r = render(rep)
        assert r["summary"] and r["fatal"] and r["advice"]
        assert r["summary"].startswith("综合定性")
        assert r["fatal"].startswith("核心致命伤")
        assert r["advice"].startswith("整改建议")
        print(f"[{name}]")
        print(format_report(r))
        print()

    print(f"OK v{__version__}  模板引擎自测通过")
