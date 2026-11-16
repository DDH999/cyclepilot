# -*- coding: utf-8 -*-
"""
六维评分与体检总报告（M2）
====================================================
把 M1 的检测结果聚合成一张「六维雷达图」+ 一个总评级。

六个维度
--------
  anti_cheat    防作弊      代码真实性（未来函数/数据穿越）
  anti_overfit  抗过拟合    DSR / PBO / 样本内外落差
  stress        抗压能力    滑点 + 极端行情切片
  purity        策略纯度    beta 伪装 / 押注 / 空仓 / 收益集中
  cost          成本可控    滑点与换手带来的年化拖累
  stability     稳健性      参数敏感性与子区间一致性（数据不足可豁免）

评分与评级交给 adjudicator（方案 D：核心缺失否决 / 数据未覆盖豁免 / 数学异常封顶 C）。
本模块只负责：维度 → 分值 → 加权 → 雷达图 → 报告。

依赖：标准库 + 本项目 adjudicator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import adjudicator as A

__version__ = "0.1.0"
__all__ = [
    "DIMENSIONS", "DIM_WEIGHT", "DIM_ORDER",
    "dim_anti_cheat", "dim_anti_overfit", "dim_stress", "dim_purity",
    "dim_cost", "dim_stability",
    "build_radar", "full_report", "format_report",
]

log = logging.getLogger(__name__)


# (key, 中文名, 权重)
DIMENSIONS = [
    ("anti_cheat", "防作弊", 0.25),
    ("anti_overfit", "抗过拟合", 0.20),
    ("stress", "抗压能力", 0.20),
    ("purity", "策略纯度", 0.15),
    ("cost", "成本可控", 0.10),
    ("stability", "稳健性", 0.10),
]
DIM_ORDER = [d[0] for d in DIMENSIONS]
FUNDAMENTALS_WEIGHT = 0.15

DIM_WEIGHT = {d[0]: d[2] for d in DIMENSIONS}
# (key, 中文名, 权重) —— 基本面是「加分项」，不在 DIMENSIONS 里，
# 只有真的传入数据时才计入权重，否则会稀释原有六维。
_LABEL = {d[0]: d[1] for d in DIMENSIONS}
_LABEL["fundamentals"] = "基本面质量"
_W = {d[0]: d[2] for d in DIMENSIONS}


def _item(key: str, status: A.Status, score: float, *,
          value=None, reason=None, note: str = "") -> A.ItemResult:
    """按维度构造裁决项；name 用中文，detail 里带 dim key。"""
    return A.ItemResult(
        name=_LABEL[key], status=status, score=float(score), value=value,
        reason=reason, note=note, detail={"dim": key})


def _reject(key: str, note: str, kind: str = "data") -> A.ItemResult:
    """构造否决项（kind=cheat 表示主观作弊）。"""
    return A.ItemResult(
        name=_LABEL[key], status=A.Status.REJECT, score=0.0,
        reason=A.Reason.CORE, note=note,
        detail={"dim": key, "reject_kind": kind})


# ============================================================
# 各维度打分
# ============================================================
def dim_anti_cheat(code_ok: bool | None, findings: list | None = None) -> A.ItemResult:
    """
    防作弊：发现未来函数直接否决。

    code_ok 为 None 表示**没有拿到代码**——这只能说明无法评估，
    绝不能推断为"存在未来函数"（那是凭空指控）。按方案 D 判豁免。
    """
    if code_ok is None:
        return _item("anti_cheat", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE,
                     note="未提供策略代码，无法进行未来函数扫描，本项豁免")
    if code_ok:
        return _item("anti_cheat", A.Status.PASS, 100.0, value=1.0)
    return _reject("anti_cheat",
                   "检测到未来函数/作弊写法，结果不可采信", kind="cheat")


def dim_anti_overfit(overfit: dict | None) -> A.ItemResult:
    """抗过拟合：直接沿用 PBO/DSR 的分级。"""
    if not overfit:
        return _item("anti_overfit", A.Status.WARN, 0.0,
                     reason=A.Reason.MATH_ABNORMAL,
                     note="未提供过拟合检验结果，记 0 分")
    base = A.item_from_overfit(overfit)
    return _item("anti_overfit", base.status, base.score, value=base.value,
                 reason=base.reason, note=base.note)


def dim_stress(slippage: dict | None, crash: dict | None) -> A.ItemResult:
    """抗压能力：滑点存活 + 极端行情回撤，两个子项取可用项均值。"""
    subs: list[A.ItemResult] = []
    if slippage:
        subs.append(A.item_from_slippage(slippage))
    if crash:
        subs.append(A.item_from_crash(crash))
    if not subs:
        return _item("stress", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE, note=A.NA_NOTE)

    rejects = [s for s in subs if s.status is A.Status.REJECT]
    if rejects:
        return _reject("stress", rejects[0].note, kind="data")

    fails = [s for s in subs if s.status is A.Status.FAIL]
    na = [s for s in subs if s.status is A.Status.NA]
    scored = [s for s in subs if s.status in (A.Status.PASS, A.Status.WARN)]

    if fails:
        return _item("stress", A.Status.FAIL, 0.0, note=fails[0].note)
    if not scored:
        return _item("stress", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE, note=A.NA_NOTE)

    score = float(np.mean([s.score for s in scored]))
    if any(s.status is A.Status.WARN for s in scored):
        note = "；".join(s.note for s in scored if s.status is A.Status.WARN)
        return _item("stress", A.Status.WARN, 0.0,
                     reason=A.Reason.MATH_ABNORMAL, note=note)
    note = "" if not na else f"部分子项豁免：{na[0].note}"
    return _item("stress", A.Status.PASS, score, note=note)


def dim_purity(purity: dict | None) -> A.ItemResult:
    """策略纯度。"""
    if not purity or not purity.get("ok"):
        tag = (purity or {}).get("reason_tag")
        if tag == "exempt":
            # 指标对该策略不适用（如全程空仓）→ 豁免并特别标注
            return _item("purity", A.Status.NA, 0.0, reason=A.Reason.EXEMPT,
                         note=(purity or {}).get("verdict") or "该策略无交易，纯度指标豁免")
        if tag == "data_unavailable":
            return _item("purity", A.Status.NA, 0.0,
                         reason=A.Reason.DATA_UNAVAILABLE, note=A.NA_NOTE)
        return _item("purity", A.Status.WARN, 0.0, reason=A.Reason.MATH_ABNORMAL,
                     note=(purity or {}).get("verdict") or "纯度无法计算")
    score = float(purity["score"])
    if score >= 80:
        return _item("purity", A.Status.PASS, score, value=score)
    if score >= 60:
        return _item("purity", A.Status.PASS, score, value=score,
                     note=purity.get("verdict", ""))
    return _item("purity", A.Status.WARN, 0.0, reason=A.Reason.MATH_ABNORMAL,
                 value=score, note=purity.get("verdict", "纯度偏低"))


def dim_cost(slippage: dict | None) -> A.ItemResult:
    """成本可控：按扣费后的年化拖累给分。"""
    if not slippage:
        return _item("cost", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE, note=A.NA_NOTE)
    if not slippage.get("n_days"):
        return _reject("cost", "收益序列为空", kind="data")
    drag = slippage.get("annual_cost_drag")
    if drag is None or not np.isfinite(drag):
        return _item("cost", A.Status.WARN, 0.0, reason=A.Reason.MATH_ABNORMAL,
                     note="成本拖累无法计算，记 0 分")
    if drag <= 0.02:
        return _item("cost", A.Status.PASS, 100.0, value=drag)
    if drag <= 0.05:
        return _item("cost", A.Status.PASS, 80.0, value=drag)
    if drag <= 0.10:
        return _item("cost", A.Status.PASS, 55.0, value=drag,
                     note=f"年化成本拖累 {drag * 100:.1f}%")
    return _item("cost", A.Status.WARN, 0.0, reason=A.Reason.MATH_ABNORMAL,
                 value=drag, note=f"年化成本拖累 {drag * 100:.1f}%，交易成本吃掉大部分收益")


def dim_stability(sensitivity: dict | None) -> A.ItemResult:
    """稳健性：来自参数敏感性网格（区间越窄越稳）。"""
    if not sensitivity:
        return _item("stability", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE, note=A.NA_NOTE)
    lo, hi = sensitivity.get("min"), sensitivity.get("max")
    median = sensitivity.get("median")
    if median is None or not np.isfinite(median):
        return _item("stability", A.Status.WARN, 0.0, reason=A.Reason.MATH_ABNORMAL,
                     note="参数敏感性结果缺少数值，记 0 分")
    if median <= 0:
        return _item("stability", A.Status.WARN, 0.0, reason=A.Reason.MATH_ABNORMAL,
                     value=median, note=f"参数滚动中位数夏普 {median:.2f} ≤ 0，结论不稳健")
    spread = (hi - lo) if (lo is not None and hi is not None) else None
    if spread is None:
        score = 70.0
    elif spread <= 0.2:
        score = 100.0
    elif spread <= 0.5:
        score = 80.0
    elif spread <= 1.0:
        score = 60.0
    else:
        score = 40.0
    return _item("stability", A.Status.PASS, score, value=spread,
                 note=f"参数夏普区间 [{lo:.2f}, {hi:.2f}]" if spread is not None else "")


# ============================================================
# 聚合
# ============================================================
def build_radar(items: list[A.ItemResult]) -> dict:
    """生成雷达图数据：{维度key: 分数 or None}。N/A 项为 None。"""
    radar = {k: None for k in DIM_ORDER}
    for it in items:
        key = (it.detail or {}).get("dim")
        if key in radar:
            radar[key] = None if it.status is A.Status.NA else float(it.score)
    return radar


def dim_fundamentals(fund: dict | None) -> A.ItemResult:
    """
    基本面质量（第七维，美股）。

    fund 来自 quality.quality_report 的返回值。
    * 没数据 → N/A（不否决，也不假装配平）
    * 覆盖率不足 → 上层已封顶 60，这里如实转述
    """
    if not fund:
        return _item("fundamentals", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE,
                     note="未提供基本面数据（目前仅支持美股 ROIC.ai）")
    if not fund.get("ok"):
        return _item("fundamentals", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE,
                     note=str(fund.get("reason") or "基本面数据不可用")[:120])
    score = fund.get("score")
    if score is None:
        return _item("fundamentals", A.Status.NA, 0.0,
                     reason=A.Reason.DATA_UNAVAILABLE,
                     note="基本面评分缺失")
    note = f"{fund.get('level')}：{fund.get('note')}"
    if fund.get("capped_by_coverage"):
        note += "（可算子项不足，评分已封顶 60）"
    status = A.Status.PASS if score > 0 else A.Status.WARN
    return _item("fundamentals", status, float(score),
                 value=score, note=note)


def full_report(code_ok: bool | None = None,
                findings: list | None = None,
                overfit: dict | None = None,
                slippage: dict | None = None,
                crash: dict | None = None,
                purity: dict | None = None,
                sensitivity: dict | None = None,
                fundamentals: dict | None = None) -> dict:
    """
    【主接口】生成六维体检总报告。

    任何一项传 None 时按"该维度不可用"处理（不会崩溃）。
    返回 dict，可直接 json 序列化给 M4 前端渲染雷达图。
    """
    items = [
        dim_anti_cheat(code_ok, findings),
        dim_anti_overfit(overfit),
        dim_stress(slippage, crash),
        dim_purity(purity),
        dim_cost(slippage),
        dim_stability(sensitivity),
        dim_fundamentals(fundamentals),
    ]
    weights = {_LABEL[k]: v for k, v in DIM_WEIGHT.items()}
    # 基本面是「加分项」而非「必检项」：只有真的给了数据才计入权重，
    # 否则会稀释原有六维的权重，让历史结论发生漂移。
    if fundamentals and fundamentals.get("ok"):
        weights[_LABEL["fundamentals"]] = FUNDAMENTALS_WEIGHT
    res = A.adjudicate(items, weights=weights)

    dims = []
    for it in items:
        key = (it.detail or {}).get("dim")
        dims.append({
            "key": key, "label": it.name, "weight": DIM_WEIGHT.get(key, 0.0),
            "score": None if it.status is A.Status.NA else float(it.score),
            "status": it.status.value,
            "reason": it.reason.value if it.reason else None,
            "note": it.note,
        })

    # ── 风险封顶 ──────────────────────────────────────────
    # 校准发现的漏洞：90% 押单一标的的组合，六维加权后仍能拿 A。
    # 对一个「风险体检」平台来说这是错的方向——单项致命缺陷
    # 不应该被其他维度的满分稀释。
    caps = []
    pm = (purity or {}).get("metrics") or {}
    if purity and purity.get("ok"):
        t1 = pm.get("top1_share")
        en = pm.get("effective_n")
        if t1 is not None and float(t1) > 0.5:
            caps.append(f"单一持仓占 {float(t1) * 100:.0f}%（>50%），评级封顶 C")
        elif en is not None and float(en) < 2.0:
            caps.append(f"等效持仓仅 {float(en):.1f} 只（<2），评级封顶 C")
    drag = (slippage or {}).get("annual_cost_drag")
    if drag is not None and np.isfinite(drag) and float(drag) > 0.10:
        caps.append(f"年化成本拖累 {float(drag) * 100:.0f}%（>10%），评级封顶 C")
    if caps:
        _order = {"A": 0, "B": 1, "C": 2, "D": 3, "REJECTED": 4}
        cur = res.get("grade")
        if cur in _order and _order[cur] < _order["C"]:
            res["grade"] = "C"
            res["verdict"] = str(res.get("verdict") or "") + "（评级已封顶）"
        res["grade_caps"] = caps

    res.update(
        dimensions=dims,
        radar=build_radar(items),
        score_breakdown={"weighted_score": res["score"],
                         "grade": res["grade"],
                         "verdict": res["verdict"]},
        version=__version__,
    )
    return res


def format_report(res: dict) -> str:
    L = ["=" * 66, "策略体检总报告", "=" * 66]
    L.append(f"  结论: {res['verdict']}   评级: {res['grade']}   "
             f"加权得分: {res['score']}")
    if res.get("grade_capped"):
        L.append("  注意: 存在数学计算异常项，评级已封顶至 C")
    L.append(f"  {res.get('summary', '')}")
    L.append("-" * 66)
    L.append(f"  {'维度':<10}{'权重':>6}{'得分':>8}   状态")
    for d in res.get("dimensions", []):
        s = "N/A" if d["score"] is None else f"{d['score']:.0f}"
        L.append(f"  {d['label']:<10}{d['weight']:>6.2f}{s:>8}   {d['status']}")
        if d.get("note"):
            L.append(f"      {d['note']}")
    L.append("-" * 66)
    L.append(f"  雷达数据: {res.get('radar')}")
    if res.get("notes"):
        L.append("  说明:")
        for n in res["notes"]:
            L.append(f"    - {n}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import pandas as pd
    import purity_tester as P

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-02", periods=400)
    codes = [f"{i:06d}" for i in range(30)]
    w = pd.DataFrame(1 / 30, index=dates, columns=codes)
    r = pd.Series(rng.normal(0.0006, 0.011, len(dates)), index=dates)
    b = pd.Series(rng.normal(0.0004, 0.010, len(dates)), index=dates)

    rep = full_report(
        code_ok=True,
        overfit={"diagnosis_level": "PASS", "pbo": 0.25, "dsr": 0.97},
        slippage={"n_days": 400, "annual_return_net": 0.12, "survive": True,
                  "annual_cost_drag": 0.03},
        crash={"ok": True, "annual_return": 0.05, "vs_full": 1.2},
        purity=P.purity_report(w, r, b),
        sensitivity={"min": 0.6, "max": 0.85, "median": 0.72},
    )
    log.info(format_report(rep))
    assert rep["grade"] in ("A", "B"), rep["grade"]
    assert len(rep["radar"]) == 6
    print(f"OK v{__version__}  grade={rep['grade']} radar={rep['radar']}")
