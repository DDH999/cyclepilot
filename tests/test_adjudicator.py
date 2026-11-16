# -*- coding: utf-8 -*-
"""
缺失值分级裁决层（方案 D）验证
====================================================================
跑法:  python tests/test_adjudicator.py

覆盖:
  [1] 三类 NaN 的判定与分类
  [2] 核心缺失 → REJECTED
  [3] 数据未覆盖 → N/A，不否决，标注固定文案
  [4] 数学异常 → WARN，得分 0，评级封顶 C
  [5] 评级门槛与封顶逻辑
  [6] M1 适配器映射正确
  [7] 鲁棒性（脏输入不崩溃）
====================================================================
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import adjudicator as A
from adjudicator import (ItemResult, Status, Reason, adjudicate, classify_nan,
                         NA_NOTE, item_from_code, item_from_overfit,
                         item_from_slippage, item_from_crash, format_report)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# 分级裁决层 v{A.__version__}  策略={A.POLICY}")
print("#" * 70)

# ══════════════════════════════════════════════════════════════
print("\n[1] NaN 判定与分类")

check("None 判为缺失", A._is_nan(None))
check("nan 判为缺失", A._is_nan(float("nan")))
check("inf 判为缺失", A._is_nan(float("inf")))
check("字符串判为缺失", A._is_nan("abc"))
check("0 不是缺失", not A._is_nan(0))
check("0.0 不是缺失", not A._is_nan(0.0))
check("负数不是缺失", not A._is_nan(-1.5))

check("显式 reason 被采用", classify_nan(float("nan"), reason=Reason.CORE) == Reason.CORE)
check("默认按数学异常", classify_nan(float("nan")) == Reason.MATH_ABNORMAL)
check("非缺失返回 None", classify_nan(1.23) is None)
check("字符串 reason 可转", classify_nan(float("nan"), reason="data_unavailable")
      == Reason.DATA_UNAVAILABLE)

# ══════════════════════════════════════════════════════════════
print("\n[2] 核心缺失 → REJECTED（规则 1）")

base_ok = [
    item_from_code(True),
    item_from_overfit({"diagnosis_level": "PASS", "pbo": 0.2, "dsr": 0.97}),
    item_from_slippage({"n_days": 500, "annual_return_net": 0.12, "survive": True}),
]

r = adjudicate(base_ok + [item_from_code(False)])
check("未来函数 → REJECTED", r["verdict"] == "REJECTED", r["verdict"])
check("REJECTED 得分 0", r["score"] == 0.0)
check("REJECTED 等级为 REJECTED", r["grade"] == "REJECTED")

r_empty = adjudicate(base_ok + [item_from_slippage({"n_days": 0})])
check("净值序列为空 → REJECTED", r_empty["verdict"] == "REJECTED")

# ══════════════════════════════════════════════════════════════
print("\n[3] 数据未覆盖 → N/A（规则 2）")

r = adjudicate(base_ok + [item_from_crash({"ok": False})])
check("无历史数据不否决", r["verdict"] == "PASS", r["verdict"])
check("无历史数据不降级", r["grade"] == "A", r["grade"])
check("计数中记为 na", r["counts"]["na"] == 1, r["counts"])
check("note 写入固定文案", any(NA_NOTE in n for n in r["notes"]))
check("文案逐字一致", NA_NOTE == "由于数据未覆盖该历史区间，此项压力测试豁免，但建议结合定性分析")

# N/A 不计入评分分母：只留 N/A + 一个满分项，仍应 100 分
r_denom = adjudicate([ItemResult("A", Status.PASS, 100.0),
                      ItemResult("B", Status.NA, 0.0, reason=Reason.DATA_UNAVAILABLE)])
check("N/A 不进分母", r_denom["score"] == 100.0, r_denom["score"])

# ══════════════════════════════════════════════════════════════
print("\n[4] 数学异常 → WARN + 封顶 C（规则 3）")

warn_item = item_from_overfit({"diagnosis_level": "UNKNOWN", "pbo": None})
r = adjudicate(base_ok + [warn_item])
check("PBO 算不出 → WARNING", r["verdict"] == "WARNING", r["verdict"])
check("该项得 0 分", warn_item.score == 0.0)
check("评级封顶为 C", r["grade"] == "C", r["grade"])
check("标记为被封顶", r["grade_capped"] is True)
check("原因记为数学异常", warn_item.reason == Reason.MATH_ABNORMAL)

# 全满分 + 一个数学异常，仍只能 C（绝不 A/B）
r_full = adjudicate([ItemResult(f"项{i}", Status.PASS, 100.0) for i in range(9)]
                    + [ItemResult("坏项", Status.WARN, 0.0, reason=Reason.MATH_ABNORMAL)])
check("其余全满分仍封顶 C", r_full["grade"] == "C", r_full["grade"])
check("得分仍高于门槛", r_full["score"] == 90.0, r_full["score"])

# 零波动导致夏普 nan 的场景
zero_vol = item_from_slippage({"n_days": 300, "annual_return_net": float("nan"),
                               "survive": False})
check("净值归零 → WARN 而非否决", zero_vol.status == Status.WARN, zero_vol.status)
r = adjudicate(base_ok + [zero_vol])
check("净值归零封顶 C", r["grade"] == "C", r["grade"])

# ══════════════════════════════════════════════════════════════
print("\n[5] 评级门槛")

for score, expect in [(100, "A"), (90, "A"), (89.9, "B"), (75, "B"),
                      (74.9, "C"), (60, "C"), (59.9, "D"), (0, "D")]:
    r = adjudicate([ItemResult("x", Status.PASS, score)])
    check(f"得分 {score} → {expect}", r["grade"] == expect, r["grade"])

# 3 个满分 + 1 个数学异常：均值 75 → B，应被封顶到 C
r_cap = adjudicate([ItemResult(f"p{i}", Status.PASS, 100.0) for i in range(3)]
                   + [ItemResult("w", Status.WARN, 0.0, reason=Reason.MATH_ABNORMAL)])
check("封顶后不再是 A/B", r_cap["grade"] == "C", r_cap["grade"])
check("封顶被标记", r_cap["grade_capped"] is True)
check("封顶前本应是 B", r_cap["score"] == 75.0, r_cap["score"])
# 本来就低于 C 的，封顶不会把它拉回 C
check("D 不会被压成 C", adjudicate([ItemResult("x", Status.PASS, 50),
                                ItemResult("y", Status.WARN, 0)])["grade"] == "D")

# 优先级：REJECTED > FAIL > WARNING
r_mix = adjudicate([ItemResult("a", Status.REJECT, 0, reason=Reason.CORE),
                    ItemResult("b", Status.WARN, 0, reason=Reason.MATH_ABNORMAL),
                    ItemResult("c", Status.FAIL, 0)])
check("REJECTED 优先最高", r_mix["verdict"] == "REJECTED")
r_mf = adjudicate([ItemResult("a", Status.WARN, 0, reason=Reason.MATH_ABNORMAL),
                   ItemResult("b", Status.FAIL, 0)])
check("FAIL 优先于 WARNING", r_mf["verdict"] == "FAIL")
check("FAIL 时等级 D", r_mf["grade"] == "D")

# ══════════════════════════════════════════════════════════════
print("\n[6] M1 适配器")

check("代码通过 → PASS", item_from_code(True).status == Status.PASS)
check("代码作弊 → REJECT", item_from_code(False).status == Status.REJECT)

ov_pass = item_from_overfit({"diagnosis_level": "PASS", "pbo": 0.3, "dsr": 0.97})
ov_rebuild = item_from_overfit({"diagnosis_level": "REBUILD", "pbo": 0.9})
ov_unknown = item_from_overfit({"diagnosis_level": "UNKNOWN", "pbo": None})
ov_review = item_from_overfit({"diagnosis_level": "REVIEW", "pbo": 0.005})
check("PBO PASS → PASS", ov_pass.status == Status.PASS)
check("PBO REBUILD → FAIL", ov_rebuild.status == Status.FAIL)
check("PBO UNKNOWN → WARN", ov_unknown.status == Status.WARN)
check("PBO UNKNOWN 记数学异常", ov_unknown.reason == Reason.MATH_ABNORMAL)
check("PBO REVIEW → WARN", ov_review.status == Status.WARN)

sl_ok = item_from_slippage({"n_days": 500, "annual_return_net": 0.115, "survive": True})
sl_dead = item_from_slippage({"n_days": 500, "annual_return_net": -0.02, "survive": False})
check("滑点达标 → PASS", sl_ok.status == Status.PASS)
check("滑点不达标 → FAIL", sl_dead.status == Status.FAIL)

cr_na = item_from_crash({"ok": False})
cr_bad = item_from_crash({"ok": True, "annual_return": 0.1, "vs_full": 3.0})
cr_ok = item_from_crash({"ok": True, "annual_return": 0.05, "vs_full": 0.8})
check("切片无数据 → NA", cr_na.status == Status.NA)
check("切片无数据原因正确", cr_na.reason == Reason.DATA_UNAVAILABLE)
check("切片失控 → FAIL", cr_bad.status == Status.FAIL)
check("切片可控 → PASS", cr_ok.status == Status.PASS)

# ══════════════════════════════════════════════════════════════
print("\n[7] 鲁棒性")

check("空列表不崩溃", adjudicate([])["verdict"] in ("WARNING", "PASS"))
try:
    adjudicate([])
    check("空列表返回合法结果", True)
except Exception as ex:
    check("空列表返回合法结果", False, repr(ex))

for bad in (None, float("nan"), float("inf"), "x", [], {}):
    try:
        A._is_nan(bad)
        ok = True
    except Exception:
        ok = False
    check(f"_is_nan({type(bad).__name__}) 不崩溃", ok)

r_bad_item = adjudicate([ItemResult("怪项", Status.WARN, 0.0, value=float("nan"),
                                    reason=Reason.MATH_ABNORMAL)])
check("nan 值项可渲染", "N/A" in format_report(r_bad_item))

rep = format_report(r)
check("报告含结论与评级", "结论:" in rep and "评级:" in rep)
check("报告含封顶提示", "封顶" in format_report(r_cap))
check("报告可序列化", isinstance(r["items"][0], dict))

# ══════════════════════════════════════════════════════════════
print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  方案 D 已落地")
print("#" * 70)
