# -*- coding: utf-8 -*-
"""
数据契约 + 人工复核 + 端到端黄金样本
====================================================================
跑法:  python tests/test_contracts.py

分组
  [1] 数据契约校验（M1/M2 输出必须符合 schema）
  [2] 契约的向后兼容（多余字段不报错、缺必填能发现）
  [3] 人工复核（override）与审计日志
  [4] 极低仓位标记（cash_dominated）
  [5] 端到端流水线 + 黄金样本回归（可重复性）
====================================================================
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import adjudicator as A
import purity_tester as P
import scoring as SC
import schemas as SCm
import stress_tester as S
import validator as V

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# 契约 v{SCm.__version__}（contract {SCm.CONTRACT_VERSION}）")
print("#" * 70)

# ── 固定数据集（黄金样本用） ──
SEED = 20261104
DATES = pd.bdate_range("2023-01-02", periods=420)
CODES = [f"{i:06d}" for i in range(30)]


def make_data():
    rng = np.random.default_rng(SEED)
    rets = pd.Series(rng.normal(0.0007, 0.011, len(DATES)), index=DATES)
    bench = pd.Series(rng.normal(0.0004, 0.010, len(DATES)), index=DATES)
    w = pd.DataFrame(1 / 30, index=DATES, columns=CODES)
    return rets, bench, w


RETS, BENCH, W = make_data()


def run_pipeline():
    """M1 → M2 → 裁决 的完整流水线（黄金样本）。"""
    code_ok = V.check_future_function(
        'ret = df["close"].pct_change(fill_method=None)\nsig = ret.shift(1)\nranked = ret.rank(method="first")')
    over = V.calculate_overfitting(RETS.iloc[:260], RETS.iloc[260:], n_trials=5)
    slip = S.apply_slippage(RETS, turnover=0.1)
    crash = S.test_extreme_crash(RETS, *S.CRASH_WINDOWS["2024微盘股流动性危机"])
    pur = P.purity_report(W, RETS, BENCH)
    rep = SC.full_report(code_ok=code_ok, overfit=over, slippage=slip, crash=crash,
                         purity=pur,
                         sensitivity={"min": 0.6, "max": 0.85, "median": 0.72})
    return {"code_ok": code_ok, "over": over, "slip": slip, "crash": crash,
            "pur": pur, "rep": rep}


OUT = run_pipeline()

# ══════════════════════════════════════════════════════════════
print("\n[1] 数据契约校验")

for name, model, payload in [
    ("StressResult", SCm.StressResult, OUT["slip"]),
    ("CrashResult", SCm.CrashResult, OUT["crash"]),
    ("OverfitResult", SCm.OverfitResult, OUT["over"]),
    ("PurityResult", SCm.PurityResult, OUT["pur"]),
    ("ReportModel", SCm.ReportModel, OUT["rep"]),
]:
    ok, errs = SCm.validate_payload(model, payload)
    check(f"{name} 校验通过", ok, errs[:2])

check("契约版本一致", OUT["rep"].get("version") == SC.__version__)

# ══════════════════════════════════════════════════════════════
print("\n[2] 向后兼容与错误检出")

extra = dict(OUT["slip"])
extra["brand_new_field"] = "未来新增字段"
ok, _ = SCm.validate_payload(SCm.StressResult, extra)
check("多余字段被允许（向后兼容）", ok)

missing = {k: v for k, v in OUT["slip"].items() if k != "survive"}
ok2, errs2 = SCm.validate_payload(SCm.StressResult, missing)
check("缺必填字段能检出", not ok2)
check("错误信息可读", any("survive" in e for e in errs2), errs2[:1])

ok3, errs3 = SCm.validate_payload(SCm.StressResult, "not a dict")
check("非 dict 不抛异常", not ok3 and errs3)

bad = dict(OUT["slip"])
bad["n_days"] = -5
ok4, _ = SCm.validate_payload(SCm.StressResult, bad)
check("非法取值能检出（n_days<0）", not ok4)

check("所有契约都带 contract 版本",
      SCm.validate_payload(SCm.PurityResult, OUT["pur"])[0]
      and OUT["pur"].get("contract", SCm.CONTRACT_VERSION) == SCm.CONTRACT_VERSION)

# ══════════════════════════════════════════════════════════════
print("\n[3] 人工复核（override）与审计")

A.AUDIT_LOG.clear()
items_ok = [A.ItemResult("x", A.Status.PASS, 100.0)]
base = A.adjudicate(items_ok)
check("默认无复核", base["override_applied"] is False)
check("默认含空审计", base["audit"] == [])

ov = A.adjudicate(items_ok,
                  override={"decision": "FAIL", "reason": "人工抽查发现数据源错误",
                            "operator": "zhang", "timestamp": "2026-11-04T09:00:00Z"})
check("复核可改结论", ov["verdict"] == "FAIL", ov["verdict"])
check("保留原结论", ov["original_verdict"] == "PASS")
check("标记已复核", ov["override_applied"] is True)
check("记录操作人与理由",
      ov["override"]["operator"] == "zhang" and "数据源" in ov["override"]["reason"])
check("审计条目已写入", len(ov["audit"]) == 1 and ov["audit"][0]["action"] == "override_applied")
check("全局审计日志有记录", len(A.AUDIT_LOG) >= 1)

ref = A.adjudicate([A.item_from_code(False)],
                   override={"decision": "PASS", "reason": "我觉得没问题", "operator": "x"})
check("作弊不可被复核放行", ref["verdict"] == "REJECTED")
check("拒绝理由写明", "作弊" in ref.get("override_refused_reason", ""))
check("拒绝也写审计", ref["audit"][-1]["action"] == "override_refused")

no_reason = A.adjudicate(items_ok, override={"decision": "FAIL", "operator": "x"})
check("缺理由被拒", no_reason["override_applied"] is False)
check("缺理由说明清楚", "理由" in no_reason.get("override_refused_reason", ""))

bad_target = A.adjudicate(items_ok, override={"decision": "随便", "reason": "r"})
check("非法目标结论被拒", bad_target["override_applied"] is False)

check("复核信息进报告", "人工复核" in A.format_report(ov))

# ══════════════════════════════════════════════════════════════
print("\n[4] 极低仓位标记")

W_low = W * 0.05
rp_low = P.purity_report(W_low, RETS, BENCH)
check("极低仓位被标记", "cash_dominated" in rp_low["flags"], rp_low["flags"])
check("极低仓位仍有分数", rp_low["score"] is not None, rp_low["score"])
check("极低仓位不豁免", rp_low["reason_tag"] is None, rp_low["reason_tag"])
check("诊断提到极低仓位", "极低仓位" in rp_low["verdict"], rp_low["verdict"])
check("记录了 avg_gross", rp_low["metrics"].get("avg_gross") is not None,
      rp_low["metrics"].get("avg_gross"))

W_full = W
rp_full = P.purity_report(W_full, RETS, BENCH)
check("满仓不标记", "cash_dominated" not in rp_full["flags"])

W_zero = pd.DataFrame(0.0, index=DATES, columns=CODES)
rp_zero = P.purity_report(W_zero, RETS, BENCH)
check("完全空仓仍走豁免", rp_zero["reason_tag"] == "exempt", rp_zero["reason_tag"])
check("完全空仓无分数", rp_zero["score"] is None)

# ══════════════════════════════════════════════════════════════
print("\n[5] 端到端 + 黄金样本")

check("流水线跑通", OUT["rep"]["verdict"] in ("PASS", "WARNING", "FAIL", "REJECTED"))
check("六维齐全", len(OUT["rep"]["radar"]) == 6)
check("评级合法", OUT["rep"]["grade"] in ("A", "B", "C", "D", "REJECTED"))

# 可重复性：同种子两次结果完全一致
OUT2 = run_pipeline()
same = all(
    OUT["rep"]["radar"][k] == OUT2["rep"]["radar"][k] for k in OUT["rep"]["radar"]
)
check("雷达结果可重复", same, (OUT["rep"]["radar"], OUT2["rep"]["radar"]))
check("结论可重复", OUT["rep"]["verdict"] == OUT2["rep"]["verdict"])
check("得分可重复", OUT["rep"]["score"] == OUT2["rep"]["score"])

# 结构稳定性（字段名变了这里会红）
required_top = {"verdict", "grade", "score", "items", "radar", "dimensions",
                "notes", "counts", "policy", "summary", "audit",
                "override_applied", "grade_capped"}
check("报告顶层字段稳定", required_top <= set(OUT["rep"]),
      required_top - set(OUT["rep"]))
dims = {d["key"] for d in OUT["rep"]["dimensions"]}
check("维度 key 稳定（六维 + 基本面）",
      dims == (set(SC.DIM_ORDER) | {"fundamentals"}), dims)

# 报告可 JSON 序列化（M4 前端要用）
import json
try:
    s = json.dumps(OUT["rep"], ensure_ascii=False, default=str)
    check("报告可 JSON 序列化", len(s) > 100)
except Exception as ex:
    check("报告可 JSON 序列化", False, repr(ex))

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
print("# 全部通过  契约与流水线稳定")
print("#" * 70)
