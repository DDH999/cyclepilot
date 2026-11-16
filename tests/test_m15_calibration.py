# -*- coding: utf-8 -*-
"""
M15 验证：换手口径 / 风险封顶 / 评分效度校准
====================================================================
这组测试锁住三个「校准发现的真 bug」：
  1. turnover 口径混用 → 成本放大 252 倍
  2. 90% 单票仍能拿 A → 需要封顶
  3. 六维对已知缺陷是否真的敏感

跑法:  python tests/test_m15_calibration.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import calibration as CAL
import purity_tester as PT
import scoring as SC
import stress_tester as ST

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M15 验证  calibration v{CAL.__version__}")
print("#" + "=" * 69)

IDX = pd.bdate_range("2015-01-02", periods=2600)
rng = np.random.default_rng(7)
RET = pd.DataFrame({"ret": rng.normal(0.0004, 0.011, 2600)}, index=IDX)

print("\n[1] 换手口径（回归：曾经放大 252 倍）")

a = ST.apply_slippage(RET, slippage_rate=0.0005, turnover=2.0,
                      turnover_period="year")
b = ST.apply_slippage(RET, slippage_rate=0.0005, turnover=2.0 / 252,
                      turnover_period="day")
check("年/日口径结果一致",
      abs(a["annual_cost_drag"] - b["annual_cost_drag"]) < 1e-4,
      (round(a["annual_cost_drag"], 6), round(b["annual_cost_drag"], 6)))
check("年换手 2 倍成本约 0.2%（非 44%）",
      abs(a["annual_cost_drag"] - 0.002) < 0.0005, round(a["annual_cost_drag"], 4))
check("旧 bug 不会再出现（拖累 < 1%）", a["annual_cost_drag"] < 0.01,
      round(a["annual_cost_drag"], 4))
check("记录换手口径", a.get("turnover_period") == "year", a.get("turnover_period"))
check("正常情况不告警", a.get("sanity_warning") is None)

big = ST.apply_slippage(RET, slippage_rate=0.002, turnover=24.0,
                        turnover_period="year")
check("高换手拖累显著", big["annual_cost_drag"] > 0.05,
      round(big["annual_cost_drag"], 4))
check("高换手仍可存活（不是必然爆仓）", big["annual_cost_drag"] < 0.30,
      round(big["annual_cost_drag"], 4))

try:
    ST.apply_slippage(RET, turnover=2.0, turnover_period="month")
    check("非法周期报错", False)
except ValueError:
    check("非法周期报错", True)

print("\n[2] 风险封顶")

sc_c = CAL.build_scenario("concentrated", seed=7)
pur_c = PT.purity_report(weights=sc_c["weights"], returns=sc_c["returns"],
                         bench=sc_c["bench"])
check("识别出高集中度",
      pur_c["metrics"]["top1_share"] > 0.8,
      round(float(pur_c["metrics"]["top1_share"]), 3))
check("等效持仓 < 2", pur_c["metrics"]["effective_n"] < 2)

slip_ok = ST.apply_slippage(pd.DataFrame({"ret": sc_c["returns"]}),
                            slippage_rate=0.0005, turnover=2.0,
                            turnover_period="year")
rep_c = SC.full_report(code_ok=True, findings=[], purity=pur_c,
                       slippage=slip_ok, sensitivity={"min": 0.5, "max": 0.7,
                                                      "median": 0.6})
check("高集中度被封顶 C 或更低",
      rep_c["grade"] in ("C", "D"), rep_c["grade"])
check("封顶原因被记录", len(rep_c.get("grade_caps") or []) >= 1,
      rep_c.get("grade_caps"))
check("封顶说明含百分比",
      any("%" in x for x in (rep_c.get("grade_caps") or [])))

sc_d = CAL.build_scenario("clean_alpha", seed=7)
pur_d = PT.purity_report(weights=sc_d["weights"], returns=sc_d["returns"],
                         bench=sc_d["bench"])
rep_d = SC.full_report(code_ok=True, findings=[], purity=pur_d,
                       slippage=slip_ok, sensitivity={"min": 0.6, "max": 0.75,
                                                      "median": 0.68})
check("分散组合不被封顶", not rep_d.get("grade_caps"), rep_d.get("grade_caps"))
# 评级高低的比较放到 [4]，用完整管线跑（这里维度不全，比较会失真）

print("\n[3] 场景构造")

for nm in ("clean_alpha", "beta_only", "overfit", "high_turnover",
           "concentrated", "crash_prone", "unstable", "cheat"):
    s = CAL.build_scenario(nm, seed=7)
    check(f"场景 {nm} 可用",
          s["returns"] is not None and len(s["returns"]) > 1000,
          len(s["returns"]) if s["returns"] is not None else 0)

s_cheat = CAL.build_scenario("cheat", seed=7)
check("作弊场景带未来函数代码", "shift(-1)" in s_cheat["code"])

print("\n[4] 校准结果（单种子，较快）")

res = CAL.calibrate(seeds=(7,))
check("校准完成", res["ok"] is True)
check("跑了 8 个场景", res["n_runs"] == 8, res["n_runs"])
check("干净组高分", res["clean_mean"] >= 80, res["clean_mean"])
check("缺陷组低于干净组", res["defect_mean"] < res["clean_mean"],
      (res["defect_mean"], res["clean_mean"]))
check("区分度 AUC ≥ 0.8", res["auc"] is not None and res["auc"] >= 0.8, res["auc"])

by_name = {r["name"]: r for r in res["runs"]}
check("未来函数被判否决",
      by_name["cheat"]["grade"] == "REJECTED" or by_name["cheat"]["score"] < 10,
      (by_name["cheat"]["grade"], by_name["cheat"]["score"]))
_order = ["A", "B", "C", "D", "REJECTED"]
check("分散组合评级高于集中组合",
      _order.index(by_name["clean_alpha"]["grade"])
      < _order.index(by_name["concentrated"]["grade"]),
      (by_name["clean_alpha"]["grade"], by_name["concentrated"]["grade"]))
check("干净 alpha 拿到 A/B",
      by_name["clean_alpha"]["grade"] in ("A", "B"),
      by_name["clean_alpha"]["grade"])

check("六类缺陷全部检出", not res["missed"],
      [m["defect"] for m in res["missed"]])
check("检出数 = 6", len(res["detected"]) == 6, len(res["detected"]))
for d in ("overfit", "cost", "purity", "stress", "stability", "cheat"):
    check(f"检出 {d}", res["sensitivity"].get(d) is True,
          res["sensitivity"].get(d))

check("纯 beta 被标注为盲区场景",
      res["blindspot_mean"] is not None, res["blindspot_mean"])
check("报告可渲染", "评分效度校准" in CAL.format_report(res))
check("报告含免责说明",
      "不能证明" in CAL.format_report(res))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  校准层可用，三个 bug 已被锁住")
print("#" * 70)
