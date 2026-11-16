# -*- coding: utf-8 -*-
"""
策略体检 · 缺失值分级裁决层（方案 D）
====================================================
统一处理 M1/M2/M3 各检测项返回值里的 NaN，避免"一个缺数就全盘否定"
或"缺数被静默跳过当成通过"。

三级规则
--------
1. **CORE（核心指标缺失）—— 一票否决**
   净值序列为空、原始年化收益算不出来，说明策略根本没跑通
   → 整单判定 REJECTED，得分 0。

2. **DATA_UNAVAILABLE（数据未覆盖）—— 跳过并标注**
   例如策略只有 3 年数据，没有 2015 年切片，导致算不出该区间回撤。
   此时不否决；该项记 "N/A"，从评分分母中剔除，
   并在报告里写入固定说明（见 NA_NOTE）。
   → 不影响评级。

3. **MATH_ABNORMAL（数学计算异常）—— 重罚降级**
   例如波动率为 0 导致夏普无法定义、有效交易次数太少导致 PBO 算不出、
   分母为 0 导致比值为 nan。
   → 该项得 0 分，整体评级**最多 C**（绝不给 A 或 B）。

4. **EXEMPT（指标对该策略不适用）—— 豁免并特别标注**
   典型场景：全程空仓的策略没有交易，夏普比率不适用。
   此时不判 0 分、不降级，记 N/A 并在报告中特别说明
   （空仓同时意味着该策略对参数与成本高度不敏感）。

评级
----
得分 = 有效项（PASS 与 WARN）分数的算术平均，N/A 项不计入分母。
     >= 90 → A      >= 75 → B      >= 60 → C      其余 → D
存在 WARN 项时，最终等级不超过 C。

结论优先级
----------
REJECTED（核心缺失） > FAIL（存在未通过项） > WARNING（存在数学异常） > PASS

依赖：仅标准库（dataclasses / enum）
"""
from __future__ import annotations
from pathlib import Path

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

__version__ = "0.1.0"
__all__ = [
    "POLICY", "NA_NOTE", "MSG_REJECT_DATA", "MSG_REJECT_CHEAT", "Reason", "Status", "ItemResult",
    "classify_nan", "adjudicate", "format_report", "apply_override",
    "OVERRIDE_FORBIDDEN", "AUDIT_LOG", "load_audit_history",
    "audit_stats", "AUDIT_FILE",
    "item_from_code", "item_from_overfit", "item_from_slippage", "item_from_crash",
]

POLICY = "tiered_v1"

# 不允许人工覆盖的情形：主观作弊必须一票否决
OVERRIDE_FORBIDDEN = {"cheat"}

# 审计日志（内存态；生产环境应落库或写文件）
# ── 审计日志：内存 + JSONL 落盘 ──────────────────────────
# 原来只是一个 list，重启即丢 —— 人工复核改过什么、谁改的，
# 下次开机就查不到了。现在每次写入同时追加一条到 .state/audit_log.jsonl。
STATE_DIR = Path(__file__).resolve().parent.parent / ".state"
AUDIT_FILE = STATE_DIR / "audit_log.jsonl"
AUDIT_MAX_MEMORY = 1000


def _append_audit(entry: dict) -> None:
    """写内存（有上限）+ 追加落盘。落盘失败不影响主流程。"""
    AUDIT_LOG.append(entry)
    if len(AUDIT_LOG) > AUDIT_MAX_MEMORY:
        del AUDIT_LOG[: len(AUDIT_LOG) - AUDIT_MAX_MEMORY]
    try:
        import json
        import time as _t
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _t.time(),
               "time": _t.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        with AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def load_audit_history(limit: int = 200) -> list[dict]:
    """读回历史审计记录（最新在前）。"""
    import json
    if not AUDIT_FILE.exists():
        return []
    out: list[dict] = []
    try:
        for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return list(reversed(out))[:limit]


def audit_stats() -> dict:
    """审计日志规模与最近一条。"""
    hist = load_audit_history(limit=1)
    n = 0
    try:
        if AUDIT_FILE.exists():
            with AUDIT_FILE.open("r", encoding="utf-8") as f:
                for _ in f:
                    n += 1
    except Exception:
        n = len(AUDIT_LOG)
    return {"file": str(AUDIT_FILE), "exists": AUDIT_FILE.exists(),
            "records_on_disk": n, "in_memory": len(AUDIT_LOG),
            "latest": hist[0] if hist else None}


AUDIT_LOG: list[dict] = []


# 数据未覆盖时的固定说明（评审核对用，不要随意改）
NA_NOTE = "由于数据未覆盖该历史区间，此项压力测试豁免，但建议结合定性分析"

# REJECTED 的两种结论文案（必须区分：数据缺失 vs 主观作弊）
MSG_REJECT_DATA = "数据异常，无法评估，判定为不合格"
MSG_REJECT_CHEAT = ("检测到未来函数（数据穿越）嫌疑，策略存在主观作弊，"
                    "直接一票否决，严禁上盘")

# 评级门槛与封顶
GRADE_A, GRADE_B, GRADE_C = 90.0, 75.0, 60.0
CAP_ON_WARN = "C"


class Reason(str, Enum):
    """NaN 的成因分类。"""
    CORE = "core"                       # 核心指标缺失 → 一票否决
    DATA_UNAVAILABLE = "data_unavailable"  # 数据未覆盖 → 跳过并标注
    MATH_ABNORMAL = "math_abnormal"     # 数学计算异常 → 重罚降级
    EXEMPT = "exempt"                   # 指标对该策略不适用 → 豁免并特别标注


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    NA = "N/A"
    FAIL = "FAIL"
    REJECT = "REJECT"


@dataclass
class ItemResult:
    """单个检测项的裁决结果。"""
    name: str
    status: Status
    score: float
    value: float | None = None
    reason: Reason | None = None
    note: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["reason"] = self.reason.value if self.reason else None
        return d


# ============================================================
# 工具
# ============================================================
def _is_nan(v) -> bool:
    """None / nan / inf / 非数值 都视为缺失。绝不抛异常。"""
    if v is None:
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(f)


def classify_nan(value, *, reason: Reason | str | None = None) -> Reason | None:
    """
    判定一个 NaN 属于哪一类。

    - 显式传入 reason 时直接采用（适配器通常已知成因）
    - 未传时按最保守的 MATH_ABNORMAL 处理（可计算但算不出来）
    """
    if not _is_nan(value):
        return None
    if reason is None:
        return Reason.MATH_ABNORMAL
    return reason if isinstance(reason, Reason) else Reason(str(reason))


def _grade_from_score(score: float) -> str:
    if score >= GRADE_A:
        return "A"
    if score >= GRADE_B:
        return "B"
    if score >= GRADE_C:
        return "C"
    return "D"


def _cap_grade(grade: str, cap: str) -> tuple[str, bool]:
    """把等级压到不高于 cap。返回 (新等级, 是否被压制)。"""
    order = {"A": 4, "B": 3, "C": 2, "D": 1}
    if order.get(grade, 0) > order[cap]:
        return cap, True
    return grade, False


# ============================================================
# 裁决
# ============================================================
def _adjudicate_core(items: Iterable[ItemResult], weights: dict | None = None) -> dict:
    """
    对一组检测项做分级裁决。

    返回 dict
    --------
    verdict        PASS / WARNING / FAIL / REJECTED
    grade          A / B / C / D（REJECTED 时为 "REJECTED"）
    score          0~100
    grade_capped   是否因 WARN 被封顶
    items          各项明细
    notes          需要写进报告的说明（含 N/A 固定文案）
    counts         各类计数
    summary        一句话结论
    policy         当前策略标识
    """
    items = list(items)
    counts = {"pass": 0, "warn": 0, "na": 0, "fail": 0, "reject": 0}
    notes: list[str] = []
    scored: list[float] = []

    for it in items:
        counts[{"PASS": "pass", "WARN": "warn", "N/A": "na",
                "FAIL": "fail", "REJECT": "reject"}[it.status.value]] += 1
        if it.status in (Status.PASS, Status.WARN, Status.FAIL):
            scored.append(float(it.score))
        if it.status == Status.NA:
            notes.append(f"[{it.name}] {it.note or NA_NOTE}")
        elif it.note:
            # PASS 项也可能带说明（例如「部分子项豁免」），一并保留
            notes.append(f"[{it.name}] {it.note}")

    # ① 一票否决：区分「数据异常」与「主观作弊」
    if counts["reject"] > 0:
        kinds = {i.detail.get("reject_kind") for i in items
                 if i.status is Status.REJECT}
        if "cheat" in kinds:                 # 作弊优先，文案必须严厉
            reject_kind, summary = "cheat", MSG_REJECT_CHEAT
        else:
            reject_kind, summary = "data", MSG_REJECT_DATA
        return {
            "verdict": "REJECTED", "grade": "REJECTED", "score": 0.0,
            "grade_capped": False, "reject_kind": reject_kind,
            "items": [i.to_dict() for i in items],
            "notes": notes, "counts": counts, "policy": POLICY,
            "summary": summary,
        }

    # 加权平均：weights 以检测项名称为键；未给权重按 1.0 处理
    w_map = weights or {}
    num = den = 0.0
    for it in items:
        if it.status in (Status.PASS, Status.WARN, Status.FAIL):
            w = float(w_map.get(it.name, 1.0))
            num += float(it.score) * w
            den += w
    score = (num / den) if den > 0 else 0.0

    # ② 有未通过项 → FAIL
    if counts["fail"] > 0:
        return {
            "verdict": "FAIL", "grade": "D", "score": round(score, 2),
            "grade_capped": False, "items": [i.to_dict() for i in items],
            "notes": notes, "counts": counts, "policy": POLICY,
            "summary": f"不通过：{counts['fail']} 项未达标，评级 D。",
        }

    grade = _grade_from_score(score)
    capped = False

    # ③ 数学异常 → 封顶 C（只要存在 WARN 就视为封顶规则已触发）
    if counts["warn"] > 0:
        grade, _lowered = _cap_grade(grade, CAP_ON_WARN)
        capped = True
        verdict = "WARNING"
        summary = (f"警告：{counts['warn']} 项因数学计算异常记 0 分，"
                   f"整体评级封顶 {CAP_ON_WARN}（实际 {grade}）。")
    elif counts["na"] > 0 and counts["pass"] == 0:
        verdict = "WARNING"
        summary = "警告：所有项目均因数据未覆盖而豁免，无法给出可靠评级。"
    else:
        verdict = "PASS"
        summary = f"通过：{counts['pass']} 项达标，评级 {grade}。"

    return {
        "verdict": verdict, "grade": grade, "score": round(score, 2),
        "grade_capped": capped, "items": [i.to_dict() for i in items],
        "notes": list(dict.fromkeys(notes)), "counts": counts, "policy": POLICY,
        "summary": summary,
    }


# ============================================================
# M1 输出的适配器
# ============================================================
def item_from_code(ok: bool, findings: list | None = None) -> ItemResult:
    """代码真实性检测。发现未来函数属于作弊，直接否决。"""
    if ok:
        return ItemResult("代码真实性", Status.PASS, 100.0, value=1.0)
    return ItemResult(
        "代码真实性", Status.REJECT, 0.0, value=0.0, reason=Reason.CORE,
        note="检测到未来函数/作弊写法，结果不可采信",
        detail={"reject_kind": "cheat", "findings": len(findings or [])})


def item_from_overfit(res: dict) -> ItemResult:
    """过拟合评估。PBO 无法计算属于数学异常（WARN，封顶 C）。"""
    level = res.get("diagnosis_level")
    pbo = res.get("pbo")
    if level == "UNKNOWN" or _is_nan(pbo):
        return ItemResult(
            "过拟合(DSR/PBO)", Status.WARN, 0.0, value=pbo,
            reason=Reason.MATH_ABNORMAL,
            note="PBO 无法计算（样本内夏普≤0 或有效样本不足），该项记 0 分")
    if level == "REBUILD":
        return ItemResult("过拟合(DSR/PBO)", Status.FAIL, 0.0, value=pbo,
                          reason=None, note="PBO 过高，样本外不可信")
    if level == "REVIEW":
        return ItemResult("过拟合(DSR/PBO)", Status.WARN, 0.0, value=pbo,
                          reason=Reason.MATH_ABNORMAL,
                          note="PBO 异常偏低，需人工复核是否存在泄漏")
    dsr = res.get("dsr")
    score = 100.0 if (not _is_nan(dsr) and dsr >= 0.95) else 70.0
    return ItemResult("过拟合(DSR/PBO)", Status.PASS, score, value=pbo)


def item_from_slippage(res: dict) -> ItemResult:
    """滑点压力测试。输入为空属于核心缺失；净值归零属于未通过。"""
    n = res.get("n_days", 0)
    ann_net = res.get("annual_return_net")
    if not n:
        return ItemResult("滑点压力", Status.REJECT, 0.0,
                          reason=Reason.CORE, note="收益序列为空，策略未跑通",
                          detail={"reject_kind": "data"})
    if _is_nan(ann_net):
        return ItemResult("滑点压力", Status.WARN, 0.0, value=ann_net,
                          reason=Reason.MATH_ABNORMAL,
                          note="净值归零或无法计算年化，该项记 0 分")
    if not res.get("survive"):
        return ItemResult("滑点压力", Status.FAIL, 0.0, value=ann_net,
                          note=f"扣费后年化 {ann_net * 100:.2f}%，未达存活线")
    score = 100.0 if ann_net >= 0.10 else 80.0
    return ItemResult("滑点压力", Status.PASS, score, value=ann_net)


def item_from_crash(res: dict) -> ItemResult:
    """极端行情切片。区间无数据属于数据未覆盖（N/A，不否决）。"""
    if not res.get("ok"):
        return ItemResult("极端行情", Status.NA, 0.0, value=None,
                          reason=Reason.DATA_UNAVAILABLE, note=NA_NOTE)
    ann = res.get("annual_return")
    ratio = res.get("vs_full")
    if _is_nan(ratio):
        return ItemResult("极端行情", Status.WARN, 0.0, value=ratio,
                          reason=Reason.MATH_ABNORMAL,
                          note="全样本回撤过小，回撤倍数无法比较，该项记 0 分")
    if _is_nan(ann):
        return ItemResult("极端行情", Status.WARN, 0.0, value=res.get("max_drawdown"),
                          reason=Reason.MATH_ABNORMAL,
                          note="区间交易日过少，年化无法外推，该项记 0 分")
    if ratio > 2.0:
        return ItemResult("极端行情", Status.FAIL, 0.0, value=ratio,
                          note=f"区间回撤是全样本的 {ratio:.2f} 倍，风险失控")
    score = 100.0 if ratio <= 1.0 else 80.0
    return ItemResult("极端行情", Status.PASS, score, value=ratio)


# ============================================================
# 人工复核（override）与审计
# ============================================================
def apply_override(result: dict, override: dict | None) -> dict:
    """
    对裁决结果施加人工复核（manual override），并写入审计日志。

    override 字段
    -------------
    decision   : 目标结论 PASS / WARNING / FAIL / REJECTED
    reason     : **必填**，复核理由
    operator   : 操作人（工号/姓名）
    timestamp  : ISO 时间，缺省取当前 UTC 时间
    grade      : 可选，同时指定等级

    安全边界
    --------
    * 缺理由 → 拒绝覆盖
    * reject_kind == "cheat"（未来函数/主观作弊） → **绝不接受覆盖**
    * 目标结论非法 → 拒绝覆盖
    所有尝试（无论成功或拒绝）都会写入 result["audit"] 与 AUDIT_LOG。
    """
    if not override:
        return {**result, "audit": list(result.get("audit", [])),
                "override_applied": False}

    ts = str(override.get("timestamp")
             or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    operator = str(override.get("operator") or "unknown")
    reason = str(override.get("reason") or "").strip()
    target = str(override.get("decision") or "").upper()

    audit = list(result.get("audit", []))
    base = {"operator": operator, "timestamp": ts, "reason": reason,
            "from": result.get("verdict"), "to": target}

    def _refuse(why: str) -> dict:
        entry = {**base, "action": "override_refused", "why": why}
        audit.append(entry)
        _append_audit({"verdict": result.get("verdict"), **entry})
        return {**result, "audit": audit, "override_applied": False,
                "override_refused_reason": why}

    if not reason:
        return _refuse("缺少复核理由（reason 必填）")
    if result.get("reject_kind") in OVERRIDE_FORBIDDEN:
        return _refuse("检测到未来函数/主观作弊，不允许人工覆盖")
    if target not in ("PASS", "WARNING", "FAIL", "REJECTED"):
        return _refuse(f"目标结论非法：{target or '(空)'}")

    entry = {**base, "action": "override_applied"}
    audit.append(entry)
    _append_audit({"verdict": result.get("verdict"), **entry})

    out = dict(result)
    out["original_verdict"] = result.get("verdict")
    out["original_grade"] = result.get("grade")
    out["verdict"] = target
    if override.get("grade"):
        out["grade"] = str(override["grade"])
    out["override_applied"] = True
    out["override"] = {"operator": operator, "reason": reason, "timestamp": ts}
    out["audit"] = audit
    out["summary"] = f"[人工复核] {result.get('summary', '')} → 复核为 {target}（{operator}）"
    return out


def adjudicate(items: Iterable[ItemResult], weights: dict | None = None,
               override: dict | None = None) -> dict:
    """裁决入口：先按规则出结论，再（可选）施加人工复核。"""
    result = _adjudicate_core(items, weights)
    result.setdefault("audit", [])
    result.setdefault("override_applied", False)
    return apply_override(result, override)


# ============================================================
# 报告
# ============================================================
def format_report(result: dict) -> str:
    L = ["=" * 66, "策略体检裁决（" + str(result.get("policy")) + "）", "=" * 66]
    c = result.get("counts", {})
    L.append(f"  结论: {result.get('verdict')}   评级: {result.get('grade')}"
             f"   得分: {result.get('score')}")
    if result.get("grade_capped"):
        L.append(f"  注意: 存在数学计算异常项，评级封顶至 C（本次实得 {result.get('grade')}）")
    L.append(f"  统计: 通过 {c.get('pass', 0)} / 警告 {c.get('warn', 0)} / "
             f"豁免 {c.get('na', 0)} / 未通过 {c.get('fail', 0)} / "
             f"否决 {c.get('reject', 0)}")
    L.append("-" * 66)
    for it in result.get("items", []):
        v = it.get("value")
        vs = "N/A" if _is_nan(v) else f"{float(v):.4f}"
        L.append(f"  [{it.get('status'):5}] {it.get('name'):12} 值={vs:>10} "
                 f"得分={it.get('score'):>5.0f}")
        if it.get("note"):
            L.append(f"          {it['note']}")
    if result.get("notes"):
        L.append("-" * 66)
        L.append("  说明:")
        for n in result["notes"]:
            L.append(f"    - {n}")
    if result.get("override_applied") or result.get("audit"):
        L.append("-" * 66)
        L.append(f"  人工复核: {'已生效' if result.get('override_applied') else '未被采纳'}")
        for a in result.get("audit", []):
            L.append(f"    [{a.get('action')}] {a.get('from')} -> {a.get('to')}"
                     f"  操作人={a.get('operator')}  理由={a.get('reason')}"
                     + (f"  拒绝原因={a.get('why')}" if a.get("why") else ""))
    L.append("=" * 66)
    return "\n".join(L)


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ok = [item_from_code(True),
          item_from_overfit({"diagnosis_level": "PASS", "pbo": 0.2, "dsr": 0.97}),
          item_from_slippage({"n_days": 500, "annual_return_net": 0.12, "survive": True})]
    r1 = adjudicate(ok)
    assert r1["verdict"] == "PASS" and r1["grade"] == "A", r1["grade"]

    na = ok + [item_from_crash({"ok": False})]
    r2 = adjudicate(na)
    assert r2["verdict"] == "PASS" and r2["grade"] == "A", r2["grade"]
    assert any(NA_NOTE in n for n in r2["notes"])

    warn = ok + [item_from_overfit({"diagnosis_level": "UNKNOWN", "pbo": None})]
    r3 = adjudicate(warn)
    assert r3["verdict"] == "WARNING" and r3["grade"] == "C", (r3["verdict"], r3["grade"])
    assert r3["grade_capped"] is True

    rej = ok + [item_from_code(False)]
    r4 = adjudicate(rej)
    assert r4["verdict"] == "REJECTED" and r4["score"] == 0.0

    logging.info(format_report(r3))
    print(f"OK v{__version__}  A={r1['grade']} N/A={r2['grade']} "
          f"WARN={r3['grade']} REJECT={r4['verdict']}")
