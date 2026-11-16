# -*- coding: utf-8 -*-
"""
M24 验证：基本面质量（第七维）
====================================================================
跑法:  python tests/test_m24_quality.py   （不联网，不消耗 ROIC 配额）
"""
import sys
import types
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import quality as QL
import scoring as SC

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M24 验证  quality v{QL.__version__}")
print("#" + "=" * 69)

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

print("\n[1] 字段抽取")

m = QL._pick({"is_sales_revenue_turnover": 100.0}, "revenue")
check("模糊匹配 revenue", m == 100.0, m)
check("匹配不到返回 None", QL._pick({"foo": 1}, "revenue") is None)
check("不拿别的字段凑数", QL._pick({"total_assets": 5}, "revenue") is None)
check("非数值被跳过", QL._pick({"revenue": "abc"}, "revenue") is None)

ext = QL.extract_metrics(INC, BAL, CF)
check("抽取成功", ext["ok"] is True)
mm = ext["metrics"]
check("净利率正确", abs(mm["net_margin"] - 112000 / 478834) < 1e-6, mm["net_margin"])
check("毛利率正确", abs(mm["gross_margin"] - 233602 / 478834) < 1e-6)
check("ROE 正确", abs(mm["roe"] - 112000 / 56950) < 1e-6)
check("资产负债率正确", abs(mm["debt_ratio"] - 308030 / 364980) < 1e-6)
check("流动比率正确", abs(mm["current_ratio"] - 152987 / 145308) < 1e-6)
check("现金流/净利正确", abs(mm["ocf_to_net"] - 118254 / 112000) < 1e-6)
check("营收增速正确",
      abs(mm["revenue_growth"] - (478834 / 391035 - 1)) < 1e-6)
check("净利增速正确", abs(mm["net_growth"] - (112000 / 93736 - 1)) < 1e-6)
check("期间记录", ext["period"] == "2026-09-30", ext["period"])

no_bal = QL.extract_metrics(INC)
check("缺资产负债表仍可算", no_bal["ok"] is True)
check("缺表时少算 ROE（不补 0）", "roe" not in no_bal["metrics"])
check("缺表时少算负债率", "debt_ratio" not in no_bal["metrics"])

check("空利润表返回 ok=False", QL.extract_metrics([])["ok"] is False)

print("\n[2] 打分")

sc_full = QL.score_quality(mm)
check("可打分", sc_full["ok"] is True)
check("分数在 0~100", 0 <= sc_full["score"] <= 100, sc_full["score"])
check("四项全覆盖", sc_full["coverage"] == 1.0, sc_full["coverage"])
check("四项都有分", all(v is not None for v in sc_full["breakdown"].values()),
      sc_full["breakdown"])
check("含等级", sc_full["level"] in ("优秀", "良好", "一般", "偏弱"),
      sc_full["level"])

sc_inc = QL.score_quality(no_bal["metrics"])
check("半覆盖仍可打分", sc_inc["ok"] is True)
check("覆盖率为 0.5", sc_inc["coverage"] == 0.5, sc_inc["coverage"])
check("半覆盖被封顶 60", sc_inc["score"] <= 60.0, sc_inc["score"])
check("标记受限", sc_inc["capped_by_coverage"] is True)

# 关键回归：看得越少不能分越高
check("半覆盖不高于全覆盖或封顶值",
      sc_inc["score"] <= max(sc_full["score"], 60.0),
      (sc_inc["score"], sc_full["score"]))

# 覆盖 < 50% → 不给分
only_growth = {"revenue_growth": 0.1}
sc_low = QL.score_quality(only_growth)
check("覆盖不足一半不给分", sc_low["ok"] is False, sc_low.get("reason"))
check("给出原因", "证据不足" in str(sc_low.get("reason")), sc_low.get("reason"))

check("空指标不给分", QL.score_quality({})["ok"] is False)

# 极端值
bad = {"roe": -0.3, "net_margin": -0.2, "gross_margin": 0.05,
       "revenue_growth": -0.4, "net_growth": -0.5,
       "debt_ratio": 0.95, "current_ratio": 0.4, "ocf_to_net": -0.5}
sc_bad = QL.score_quality(bad)
check("差公司分数低", sc_bad["score"] < 40, sc_bad["score"])
check("等级偏弱", sc_bad["level"] == "偏弱", sc_bad["level"])

good = {"roe": 0.35, "net_margin": 0.30, "gross_margin": 0.70,
        "revenue_growth": 0.30, "net_growth": 0.35,
        "debt_ratio": 0.15, "current_ratio": 2.5, "ocf_to_net": 1.5}
sc_good = QL.score_quality(good)
check("好公司分数高", sc_good["score"] >= 80, sc_good["score"])
check("等级优秀", sc_good["level"] == "优秀", sc_good["level"])

print("\n[3] 第七维接入")

rep0 = SC.full_report(code_ok=True, findings=[])
dims0 = [d["key"] for d in rep0["dimensions"]]
check("维度数为 7", len(dims0) == 7, dims0)
check("含 fundamentals", "fundamentals" in dims0)
check("原六维仍在", all(k in dims0 for k in
                      ("anti_cheat", "anti_overfit", "stress",
                       "purity", "cost", "stability")))

f0 = rep0["dimensions"][-1]
check("无数据时 N/A", f0["status"] == "N/A", f0["status"])
check("无数据时不计分", f0["score"] is None, f0["score"])

# 无基本面时，总分应与「六维时代」一致
import adjudicator as A
base_items = [SC.dim_anti_cheat(True, []), SC.dim_anti_overfit(None),
              SC.dim_stress(None, None), SC.dim_purity(None),
              SC.dim_cost(None), SC.dim_stability(None)]
w6 = {SC._LABEL[k]: v for k, v in SC.DIM_WEIGHT.items()}
six = A.adjudicate(base_items, weights=w6)
check("无基本面时总分不被稀释",
      abs(six["score"] - rep0["score"]) < 1e-9, (six["score"], rep0["score"]))

fund = {"ok": True, "score": 82.0, "level": "优秀", "note": "四项健康",
        "capped_by_coverage": False}
rep1 = SC.full_report(code_ok=True, findings=[], fundamentals=fund)
f1 = rep1["dimensions"][-1]
check("有数据时参与计分", f1["score"] == 82.0, f1["score"])
check("有数据时为 PASS", f1["status"] == "PASS", f1["status"])
check("有基本面时总分变化", rep1["score"] != rep0["score"],
      (rep1["score"], rep0["score"]))
check("基本面拉高总分", rep1["score"] > rep0["score"],
      (rep1["score"], rep0["score"]))

bad_fund = {"ok": True, "score": 20.0, "level": "偏弱", "note": "弱"}
rep2 = SC.full_report(code_ok=True, findings=[], fundamentals=bad_fund)
check("差基本面拉低总分", rep2["score"] < rep0["score"] or rep2["score"] < rep1["score"],
      (rep2["score"], rep0["score"]))

rep3 = SC.full_report(code_ok=True, findings=[],
                      fundamentals={"ok": False, "reason": "取数失败"})
check("基本面取数失败按 N/A 处理",
      rep3["dimensions"][-1]["status"] == "N/A")
check("取数失败不影响总分", abs(rep3["score"] - rep0["score"]) < 1e-9)

print("\n[4] 报告与接口")

txt = QL.format_report({"ok": False, "ticker": "X", "reason": "拉取失败"})
check("失败报告可渲染", "无法评估" in txt)

txt2 = QL.format_report({
    "ok": True, "ticker": "AAPL", "period": "2026-09-30", "fiscal_year": 2026,
    "currency": "USD", "score": 63.7, "level": "良好", "note": "扎实",
    "breakdown": sc_full["breakdown"], "metrics": mm, "missing": [],
    "low_confidence": False, "capped_by_coverage": False, "warnings": []})
check("成功报告含评分", "63.7" in txt2, txt2[:80])
check("含指标名", "净利率" in txt2)

import api_v2 as API
from fastapi.testclient import TestClient
cli = TestClient(API.app)

# 打桩，避免真实调用消耗配额
_orig = QL.quality_report
QL.quality_report = lambda t, refresh=False: {
    "ok": True, "ticker": t, "score": 75.0, "level": "良好",
    "note": "测试", "breakdown": sc_full["breakdown"], "metrics": mm,
    "missing": [], "warnings": []}
try:
    r = cli.get("/api/quality?ticker=AAPL")
    check("接口 200", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        b = r.json()
        check("接口返回 ok", b.get("ok") is True)
        check("接口含分数", b["report"]["score"] == 75.0)
        check("接口含文本", "基本面质量" in b.get("text", ""))
finally:
    QL.quality_report = _orig

import quantpilot as QP
check("CLI 注册 quality", "quality" in QP.COMMANDS)
pp = QP.build_parser()
check("quality 解析 ticker",
      pp.parse_args(["quality", "--ticker", "MSFT"]).ticker == "MSFT")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  第七维可用")
print("#" * 70)
