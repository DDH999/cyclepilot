# -*- coding: utf-8 -*-
"""
回测陷阱清单（M17）
====================================================
把「让回测变好看」的常见手法一次列全，逐条判定通过与否。

十条陷阱
--------
1. **未来函数**：用了当时不可得的信息
2. **幸存者偏差**：只用了今天还活着的标的
3. **过拟合**：样本外站不住（PBO / DSR）
4. **多重检验**：试了很多次只报最好的
5. **交易成本**：未计佣金/印花税/滑点/冲击
6. **样本太短**：不足以覆盖一轮牛熊
7. **无样本外**：全样本调参，没有留出验证段
8. **数据质量**：停牌/跳变/缺失未处理
9. **分段不稳**：滚动检验正负参半
10. **策略拥挤**：换手/集中度/同质化过高

输出：每条 PASS / WARN / FAIL / UNKNOWN + 证据 + 整改方向，
以及 0~100 的"回测可信度"总分。
"""
from __future__ import annotations

import numpy as np

__version__ = "0.1.0"
__all__ = ["PITFALLS", "check_pitfalls", "format_report"]

PITFALLS = [
    ("future_function", "未来函数 / 数据穿越",
     "信号使用了当时不可得的信息", "删除 shift(-1) 等写法，T 日信号 T+1 成交"),
    ("survivorship", "幸存者偏差",
     "只用今天还存续的标的，退市样本被剔除", "并入退市股，或使用 point-in-time 成分"),
    ("overfit", "过拟合",
     "样本外表现无法支撑样本内结论", "减少参数搜索，做 Purged Walk-Forward"),
    ("multiple_testing", "多重检验",
     "尝试次数多但未做校正", "记录全部试验次数并计入 DSR，或用 BH-FDR"),
    ("costs", "交易成本缺失",
     "未计佣金/印花税/滑点/冲击", "按真实费率与换手率扣减，并做成本压力测试"),
    ("short_sample", "样本过短",
     "不足以覆盖一轮完整牛熊", "至少覆盖 3 年以上，含至少一次极端行情"),
    ("no_oos", "无样本外",
     "全样本调参，没有独立验证段", "留出最后 2~3 年不参与任何调参"),
    ("data_quality", "数据质量",
     "停牌、跳变、缺失未处理", "先跑数据体检，修复拆股与冻结行情"),
    ("unstable", "分段不稳",
     "滚动检验正负参半", "报告分段结果，而非单一年化数字"),
    ("crowding", "策略拥挤",
     "换手过高、集中度过高、同质化严重", "降低换手与集中度，监控拥挤度"),
]


def _mk(key: str, status: str, evidence: str) -> dict:
    label, name, why, fix = next((k, n, w, f) for k, n, w, f in PITFALLS if k == key)
    return {"key": key, "name": name, "status": status,
            "evidence": evidence, "why": why, "fix": fix}


def check_pitfalls(code_ok: bool | None = None,
                   findings: list | None = None,
                   survivorship_ok: bool | None = None,
                   dsr: float | None = None,
                   pbo: float | None = None,
                   n_trials: int | None = None,
                   costs_included: bool | None = None,
                   n_days: int | None = None,
                   has_oos: bool | None = None,
                   data_score: float | None = None,
                   wf_hit_rate: float | None = None,
                   turnover: float | None = None,
                   max_risk_contrib: float | None = None) -> dict:
    """
    逐条判定回测陷阱。任何参数缺失 → 该项 UNKNOWN（不猜）。

    宁可标 UNKNOWN，也不要把"没检查"说成"没问题"。
    """
    items: list[dict] = []

    # 1 未来函数
    if code_ok is None:
        items.append(_mk("future_function", "UNKNOWN", "未提供策略代码"))
    elif code_ok:
        items.append(_mk("future_function", "PASS", "代码扫描未命中"))
    else:
        n = len(findings or [])
        items.append(_mk("future_function", "FAIL", f"命中 {n} 条可疑写法"))

    # 2 幸存者偏差
    if survivorship_ok is None:
        items.append(_mk("survivorship", "UNKNOWN", "未提供票池构成信息"))
    elif survivorship_ok:
        items.append(_mk("survivorship", "PASS", "票池含退市样本"))
    else:
        items.append(_mk("survivorship", "FAIL", "票池仅含存续标的，收益可能虚高"))

    # 3 过拟合
    bad = []
    if pbo is not None and np.isfinite(pbo) and pbo > 0.5:
        bad.append(f"PBO={pbo:.2f}>0.5")
    if dsr is not None and np.isfinite(dsr) and dsr < 0.95:
        bad.append(f"DSR={dsr:.2f}<0.95")
    if pbo is None and dsr is None:
        items.append(_mk("overfit", "UNKNOWN", "未做过拟合检验"))
    elif bad:
        items.append(_mk("overfit", "FAIL", "；".join(bad)))
    else:
        items.append(_mk("overfit", "PASS",
                         f"PBO={pbo if pbo is not None else '—'} "
                         f"DSR={dsr if dsr is not None else '—'}"))

    # 4 多重检验
    if n_trials is None:
        items.append(_mk("multiple_testing", "UNKNOWN", "未记录试验次数"))
    elif n_trials <= 1:
        items.append(_mk("multiple_testing", "PASS", "仅 1 次试验，无选择偏差"))
    elif n_trials <= 10:
        items.append(_mk("multiple_testing", "WARN",
                         f"试验 {n_trials} 次，需核对是否已计入 DSR 惩罚"))
    else:
        items.append(_mk("multiple_testing", "FAIL",
                         f"试验 {n_trials} 次且未校正，最好结果极可能是噪声"))

    # 5 交易成本
    if costs_included is None:
        items.append(_mk("costs", "UNKNOWN", "未确认是否计入成本"))
    elif costs_included:
        items.append(_mk("costs", "PASS", "已计入佣金/印花税/滑点"))
    else:
        items.append(_mk("costs", "FAIL", "未计成本，年化被高估（可高达数个百分点）"))

    # 6 样本长度
    if n_days is None:
        items.append(_mk("short_sample", "UNKNOWN", "未提供样本长度"))
    elif n_days < 252:
        items.append(_mk("short_sample", "FAIL", f"仅 {n_days} 天（不足 1 年）"))
    elif n_days < 756:
        items.append(_mk("short_sample", "WARN", f"{n_days} 天（约 {n_days/252:.1f} 年）"))
    else:
        items.append(_mk("short_sample", "PASS", f"{n_days} 天（约 {n_days/252:.1f} 年）"))

    # 7 样本外
    if has_oos is None:
        items.append(_mk("no_oos", "UNKNOWN", "未确认是否有独立样本外"))
    elif has_oos:
        items.append(_mk("no_oos", "PASS", "已留出独立样本外"))
    else:
        items.append(_mk("no_oos", "FAIL", "全样本调参，无独立验证段"))

    # 8 数据质量
    if data_score is None:
        items.append(_mk("data_quality", "UNKNOWN", "未做数据体检"))
    elif data_score >= 85:
        items.append(_mk("data_quality", "PASS", f"数据体检 {data_score:.0f} 分"))
    elif data_score >= 70:
        items.append(_mk("data_quality", "WARN", f"数据体检 {data_score:.0f} 分"))
    else:
        items.append(_mk("data_quality", "FAIL", f"数据体检 {data_score:.0f} 分，不可用"))

    # 9 分段稳定性
    if wf_hit_rate is None:
        items.append(_mk("unstable", "UNKNOWN", "未做滚动样本外"))
    elif wf_hit_rate >= 0.8:
        items.append(_mk("unstable", "PASS", f"分段命中率 {wf_hit_rate*100:.0f}%"))
    elif wf_hit_rate >= 0.6:
        items.append(_mk("unstable", "WARN", f"分段命中率 {wf_hit_rate*100:.0f}%"))
    else:
        items.append(_mk("unstable", "FAIL", f"分段命中率 {wf_hit_rate*100:.0f}%"))

    # 10 拥挤
    if turnover is None and max_risk_contrib is None:
        items.append(_mk("crowding", "UNKNOWN", "未提供换手/集中度"))
    else:
        bad = []
        if turnover is not None and turnover > 6:
            bad.append(f"年换手 {turnover:.1f}x")
        if max_risk_contrib is not None and max_risk_contrib > 0.4:
            bad.append(f"最大风险贡献 {max_risk_contrib*100:.0f}%")
        if bad:
            items.append(_mk("crowding", "WARN", "；".join(bad)))
        else:
            items.append(_mk("crowding", "PASS", "换手与集中度在合理区间"))

    counts = {s: sum(1 for x in items if x["status"] == s)
              for s in ("PASS", "WARN", "FAIL", "UNKNOWN")}

    score = 100.0 - counts["FAIL"] * 18 - counts["WARN"] * 7 - counts["UNKNOWN"] * 3
    score = float(np.clip(score, 0, 100))

    if counts["FAIL"]:
        verdict = f"不可信：{counts['FAIL']} 条硬伤，按此回测上钱会亏"
    elif counts["UNKNOWN"] >= 3:
        verdict = "证据不足：多项未检查，不能据此下结论"
    elif score >= 85:
        verdict = "基本可信：主要陷阱均已排除"
    elif score >= 70:
        verdict = "有条件可信：存在若干隐患，需补测"
    else:
        verdict = "存疑：多处需整改后才能采信"

    return {"ok": True, "score": round(score, 2), "counts": counts,
            "items": items, "verdict": verdict,
            "n_checked": sum(1 for x in items if x["status"] != "UNKNOWN")}


def format_report(res: dict) -> str:
    L = ["=" * 66, "回测陷阱清单", "=" * 66]
    L.append(f"  回测可信度 {res.get('score')}/100   "
             f"（已检查 {res.get('n_checked')}/10 项）")
    c = res.get("counts", {})
    L.append(f"  通过 {c.get('PASS', 0)} / 警告 {c.get('WARN', 0)} / "
             f"硬伤 {c.get('FAIL', 0)} / 未检 {c.get('UNKNOWN', 0)}")
    L.append(f"  结论: {res.get('verdict')}")
    L.append("-" * 66)
    for it in res.get("items", []):
        tag = {"PASS": "[通过]", "WARN": "[警告]",
               "FAIL": "[硬伤]", "UNKNOWN": "[未检]"}[it["status"]]
        L.append(f"  {tag} {it['name']}")
        L.append(f"         证据: {it['evidence']}")
        if it["status"] != "PASS":
            L.append(f"         整改: {it['fix']}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    good = check_pitfalls(code_ok=True, survivorship_ok=True, pbo=0.25, dsr=0.97,
                          n_trials=1, costs_included=True, n_days=2000,
                          has_oos=True, data_score=92, wf_hit_rate=0.85,
                          turnover=3.0, max_risk_contrib=0.25)
    assert good["counts"]["FAIL"] == 0, good
    assert good["score"] >= 85, good["score"]
    print(format_report(good))

    bad = check_pitfalls(code_ok=False, findings=[1, 2], survivorship_ok=False,
                         pbo=0.8, dsr=0.2, n_trials=50, costs_included=False,
                         n_days=120, has_oos=False, data_score=50,
                         wf_hit_rate=0.2, turnover=12.0, max_risk_contrib=0.6)
    assert bad["counts"]["FAIL"] >= 5, bad["counts"]
    assert bad["score"] < good["score"]
    print(format_report(bad))

    unknown = check_pitfalls()
    assert unknown["counts"]["UNKNOWN"] == 10
    print(f"全缺参数 → 可信度 {unknown['score']}  结论：{unknown['verdict']}")
    print(f"OK v{__version__}")
