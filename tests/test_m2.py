# -*- coding: utf-8 -*-
"""
M2 验证：六维评分 + 策略纯度
====================================================================
跑法:  python tests/test_m2.py

分组
  [1] 纯度指标（等效持仓 / 暴露度 / 基准回归 / 收益集中度）
  [2] 纯度报告（分散 vs 单票 vs 数据缺失）
  [3] 六个维度的打分与状态
  [4] 雷达图数据
  [5] 总报告（评级 / 加权 / 否决 / 豁免 / 封顶）
  [6] 与 M1 的端到端联调
  [7] 鲁棒性
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
import validator as V
import stress_tester as S

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M2 验证  purity={P.__version__}  scoring={SC.__version__}")
print("#" * 70)

rng = np.random.default_rng(2024)
DATES = pd.bdate_range("2023-01-02", periods=400)
CODES = [f"{i:06d}" for i in range(30)]


# ══════════════════════════════════════════════════════════════
print("\n[1] 纯度指标")

eq = np.full(30, 1 / 30)
one = np.zeros(30)
one[0] = 1.0

check("等权 30 只等效持仓=30", abs(P.effective_n(eq) - 30) < 1e-9, P.effective_n(eq))
check("单票等效持仓=1", abs(P.effective_n(one) - 1) < 1e-9)
check("空仓等效持仓=0", P.effective_n(np.zeros(30)) == 0.0)
check("含 NaN 返回 0", P.effective_n(np.full(30, np.nan)) == 0.0)

check("等权 top1 = 1/30", abs(P.top_share(eq, 1) - 1 / 30) < 1e-9)
check("单票 top1 = 1", abs(P.top_share(one, 1) - 1) < 1e-9)
check("top5 不超过 1", P.top_share(eq, 5) <= 1.0)

W_full = pd.DataFrame(1 / 30, index=DATES, columns=CODES)
W_half = W_full.copy()
W_half.iloc[: len(W_half) // 2] = 0.0
check("满仓暴露度=1", abs(P.exposure_ratio(W_full) - 1.0) < 1e-9)
check("半仓暴露度=0.5", abs(P.exposure_ratio(W_half) - 0.5) < 0.02,
      P.exposure_ratio(W_half))
check("空权重暴露度=0", P.exposure_ratio(pd.DataFrame(0.0, index=DATES, columns=CODES)) == 0.0)

bsp = pd.Series(rng.normal(0.0004, 0.010, len(DATES)), index=DATES)
y_perfect = bsp * 2.0
beta, r2 = P.bench_beta_r2(y_perfect, bsp)
check("完美跟随 beta=2", abs(beta - 2.0) < 1e-6, beta)
check("完美跟随 R²=1", abs(r2 - 1.0) < 1e-6, r2)
y_noise = pd.Series(rng.normal(0, 0.01, len(DATES)), index=DATES)
_, r2n = P.bench_beta_r2(y_noise, bsp)
check("无关序列 R² 很低", r2n < 0.2, r2n)
check("基准缺失返回 nan", not np.isfinite(P.bench_beta_r2(bsp, None)[1]))

rc = P.return_concentration(bsp)
check("集中度有界 [0,1]", 0 <= rc <= 1, rc)
spiky = pd.Series(0.0, index=DATES)
spiky.iloc[:5] = 0.05
check("尖峰序列集中度高", P.return_concentration(spiky) > 0.6,
      P.return_concentration(spiky))
check("样本过短返回 nan", not np.isfinite(P.return_concentration(bsp.iloc[:10])))

# ══════════════════════════════════════════════════════════════
print("\n[2] 纯度报告")

good_ret = pd.Series(rng.normal(0.0006, 0.011, len(DATES)), index=DATES)
rp_good = P.purity_report(W_full, good_ret, bsp)
check("分散策略可评估", rp_good["ok"] is True)
check("分散策略得分高", rp_good["score"] >= 80, rp_good["score"])
check("无 reason_tag", rp_good["reason_tag"] is None)

W_one = pd.DataFrame(np.tile(one, (len(DATES), 1)), index=DATES, columns=CODES)
rp_one = P.purity_report(W_one, good_ret, bsp)
check("单票得分低于分散", rp_one["score"] < rp_good["score"],
      (rp_one["score"], rp_good["score"]))
check("单票诊断提到押注", "押注" in rp_one["verdict"], rp_one["verdict"])

W_empty = pd.DataFrame(0.0, index=DATES, columns=CODES)
rp_empty = P.purity_report(W_empty, good_ret, bsp)
check("全空仓 → 豁免而非 0 分", rp_empty["reason_tag"] == "exempt", rp_empty["reason_tag"])
check("全空仓不给分数", rp_empty["score"] is None, rp_empty["score"])
check("全空仓标注夏普不适用", "夏普" in rp_empty["verdict"], rp_empty["verdict"])
check("全空仓说明参数不敏感", "不敏感" in rp_empty["verdict"], rp_empty["verdict"])
na_empty = SC.dim_purity(rp_empty)
check("空仓维度记为 N/A", na_empty.status is A.Status.NA, na_empty.status)
check("空仓不成惩罚项", na_empty.score == 0.0 and na_empty.reason is A.Reason.EXEMPT)

rp_mirror = P.purity_report(W_full, bsp * 1.0, bsp)
check("完全复制基准得分低", rp_mirror["score"] < 70, rp_mirror["score"])
check("复制基准诊断提到 beta", "beta" in rp_mirror["verdict"], rp_mirror["verdict"])

rp_none = P.purity_report(None, None, None)
check("全缺失 → data_unavailable", rp_none["reason_tag"] == "data_unavailable")
check("全缺失不可评估", rp_none["ok"] is False and rp_none["score"] is None)

rp_short = P.purity_report(W_full, good_ret.iloc[:10], bsp)
check("样本过短 → math_abnormal", rp_short["reason_tag"] == "math_abnormal",
      rp_short["reason_tag"])

# ══════════════════════════════════════════════════════════════
print("\n[3] 六维打分")

it = SC.dim_anti_cheat(True)
check("防作弊通过 → PASS", it.status is A.Status.PASS and it.score == 100)
it_bad = SC.dim_anti_cheat(False, findings=[object()])
check("防作弊失败 → REJECT", it_bad.status is A.Status.REJECT)
check("防作弊标记 cheat", it_bad.detail.get("reject_kind") == "cheat")
check("防作弊 detail 带 dim", it_bad.detail.get("dim") == "anti_cheat")

check("过拟合 PASS", SC.dim_anti_overfit(
    {"diagnosis_level": "PASS", "pbo": 0.2, "dsr": 0.97}).status is A.Status.PASS)
check("过拟合 REBUILD → FAIL", SC.dim_anti_overfit(
    {"diagnosis_level": "REBUILD", "pbo": 0.9}).status is A.Status.FAIL)
check("过拟合缺失 → WARN", SC.dim_anti_overfit(None).status is A.Status.WARN)
check("过拟合 UNKNOWN → WARN", SC.dim_anti_overfit(
    {"diagnosis_level": "UNKNOWN", "pbo": None}).status is A.Status.WARN)

sl_ok = {"n_days": 400, "annual_return_net": 0.12, "survive": True,
         "annual_cost_drag": 0.03}
cr_ok = {"ok": True, "annual_return": 0.05, "vs_full": 1.2}
cr_na = {"ok": False}
check("抗压 双通过 → PASS", SC.dim_stress(sl_ok, cr_ok).status is A.Status.PASS)
check("抗压 切片无数据仍 PASS", SC.dim_stress(sl_ok, cr_na).status is A.Status.PASS)
check("抗压 全缺失 → NA", SC.dim_stress(None, None).status is A.Status.NA)
check("抗压 滑点空 → REJECT", SC.dim_stress(
    {"n_days": 0}, None).status is A.Status.REJECT)
check("抗压 REJECT 标记 data", SC.dim_stress(
    {"n_days": 0}, None).detail.get("reject_kind") == "data")

check("成本 低拖累 → 80 分", SC.dim_cost(sl_ok).score == 80,
      SC.dim_cost(sl_ok).score)
check("成本 极低拖累 → 100 分", SC.dim_cost(
    {"n_days": 400, "annual_cost_drag": 0.01}).score == 100)
check("成本 高拖累 → 低分", SC.dim_cost(
    {"n_days": 400, "annual_cost_drag": 0.2}).score < 60)
check("成本 缺失 → NA", SC.dim_cost(None).status is A.Status.NA)

check("稳健 窄区间 → 100", SC.dim_stability(
    {"min": 0.6, "max": 0.75, "median": 0.68}).score == 100)
check("稳健 宽区间 → 40", SC.dim_stability(
    {"min": -0.5, "max": 1.5, "median": 0.5}).score == 40)
check("稳健 中位为负 → WARN", SC.dim_stability(
    {"min": -1.0, "max": -0.2, "median": -0.5}).status is A.Status.WARN)
check("稳健 缺失 → NA", SC.dim_stability(None).status is A.Status.NA)

# ══════════════════════════════════════════════════════════════
print("\n[4] 雷达图")

items = [SC.dim_anti_cheat(True), SC.dim_anti_overfit(
    {"diagnosis_level": "PASS", "pbo": 0.2, "dsr": 0.97}),
    SC.dim_stress(sl_ok, cr_ok), SC.dim_purity(rp_good), SC.dim_cost(sl_ok),
    SC.dim_stability({"min": 0.6, "max": 0.8, "median": 0.7})]
radar = SC.build_radar(items)
check("雷达覆盖传入的 6 个维度", len(radar) == 6, len(radar))
check("维度 key 与 DIM_ORDER 一致",
      set(radar) == set(SC.DIM_ORDER), sorted(radar))
check("全部有值", all(v is not None for v in radar.values()))

radar_na = SC.build_radar([SC.dim_stability(None)])
check("N/A 维度为 None", radar_na["stability"] is None)

# ══════════════════════════════════════════════════════════════
print("\n[5] 总报告")

rep = SC.full_report(code_ok=True, overfit={"diagnosis_level": "PASS", "pbo": 0.2,
                                            "dsr": 0.97},
                     slippage=sl_ok, crash=cr_ok, purity=rp_good,
                     sensitivity={"min": 0.6, "max": 0.8, "median": 0.7})
check("全优 → 评级 A", rep["grade"] == "A", rep["grade"])
check("全优 → PASS", rep["verdict"] == "PASS")
check("含 7 个维度明细（六维 + 基本面）",
      len(rep["dimensions"]) == 7, len(rep["dimensions"]))
check("含雷达数据", len(rep["radar"]) == 6)
check("含版本号", rep.get("version") == SC.__version__)

# 作弊 → 严厉文案
rep_cheat = SC.full_report(code_ok=False, overfit={"diagnosis_level": "PASS",
                                                   "pbo": 0.2, "dsr": 0.97},
                           slippage=sl_ok, crash=cr_ok, purity=rp_good)
check("作弊 → REJECTED", rep_cheat["verdict"] == "REJECTED")
check("作弊 → reject_kind=cheat", rep_cheat.get("reject_kind") == "cheat")
check("作弊 → 严厉文案", rep_cheat["summary"] == A.MSG_REJECT_CHEAT,
      rep_cheat["summary"][:20])
check("作弊 → 得分 0", rep_cheat["score"] == 0.0)

# 数据缺失 → 另一种文案
rep_data = SC.full_report(code_ok=True, overfit={"diagnosis_level": "PASS",
                                                 "pbo": 0.2, "dsr": 0.97},
                          slippage={"n_days": 0}, crash=None, purity=None)
check("数据缺失 → REJECTED", rep_data["verdict"] == "REJECTED")
check("数据缺失 → reject_kind=data", rep_data.get("reject_kind") == "data")
check("数据缺失 → 普通文案", rep_data["summary"] == A.MSG_REJECT_DATA)

# 数学异常 → 封顶 C
rep_warn = SC.full_report(code_ok=True, overfit={"diagnosis_level": "UNKNOWN",
                                                 "pbo": None},
                          slippage=sl_ok, crash=cr_ok, purity=rp_good,
                          sensitivity={"min": 0.6, "max": 0.8, "median": 0.7})
check("数学异常 → WARNING", rep_warn["verdict"] == "WARNING", rep_warn["verdict"])
check("数学异常 → 封顶 C", rep_warn["grade"] == "C", rep_warn["grade"])
check("数学异常 → 标记封顶", rep_warn["grade_capped"] is True)

# 数据未覆盖 → 不否决
rep_na = SC.full_report(code_ok=True, overfit={"diagnosis_level": "PASS",
                                               "pbo": 0.2, "dsr": 0.97},
                        slippage=sl_ok, crash={"ok": False}, purity=rp_good,
                        sensitivity=None)
check("数据未覆盖 → 不否决", rep_na["verdict"] == "PASS", rep_na["verdict"])
check("数据未覆盖 → 有豁免说明",
      any(A.NA_NOTE in n for n in rep_na["notes"]), rep_na["notes"])

# 权重确实生效
w_items = [A.ItemResult("x", A.Status.PASS, 100.0),
           A.ItemResult("y", A.Status.PASS, 0.0)]
weighted = A.adjudicate(w_items, weights={"x": 0.25, "y": 0.75})
unweighted = A.adjudicate(w_items)
check("加权评分生效", weighted["score"] < unweighted["score"],
      (weighted["score"], unweighted["score"]))
check("加权值正确", absw := (abs(weighted["score"] - 25.0) < 1e-6), weighted["score"])

# ══════════════════════════════════════════════════════════════
print("\n[6] 与 M1 端到端")

code_ok = V.check_future_function('ret = df["close"].pct_change(fill_method=None)\n'
                                  'sig = ret.shift(1)')
ov = V.calculate_overfitting(rng.normal(0.001, 0.011, 350),
                             rng.normal(0.0009, 0.011, 400), n_trials=5)
sl = S.apply_slippage(good_ret, turnover=0.1)
cr = S.test_extreme_crash(good_ret, *S.CRASH_WINDOWS["2015股灾"], label="2015股灾")
pu = P.purity_report(W_full, good_ret, bsp)

rep_e2e = SC.full_report(code_ok=code_ok, overfit=ov, slippage=sl, crash=cr,
                         purity=pu, sensitivity={"min": 0.6, "max": 0.8, "median": 0.7})
check("端到端能出报告", rep_e2e["grade"] in ("A", "B", "C", "D"))
check("端到端雷达完整", len(rep_e2e["radar"]) == 6)
check("端到端 2015 切片判 NA", rep_e2e["radar"]["stress"] is not None)
check("端到端有豁免说明", any(A.NA_NOTE in n for n in rep_e2e["notes"]))
check("报告可渲染", isinstance(SC.format_report(rep_e2e), str))
check("报告含雷达数据", "雷达数据" in SC.format_report(rep_e2e))

# ══════════════════════════════════════════════════════════════
print("\n[7] 鲁棒性")

rep_empty = SC.full_report()
check("全 None 不崩溃", rep_empty["verdict"] in ("PASS", "WARNING"))
check("全 None 雷达有 None", sum(v is None for v in rep_empty["radar"].values()) >= 1)

try:
    SC.full_report(code_ok=None, overfit={"diagnosis_level": "UNKNOWN"})
    check("半缺参数不崩溃", True)
except Exception as ex:
    check("半缺参数不崩溃", False, repr(ex))

for bad in (None, float("nan"), float("inf"), "x"):
    try:
        P.purity_report(None, pd.Series([bad] * 40), None)
        ok = True
    except Exception:
        ok = False
    check(f"纯度喂入 {type(bad).__name__} 不崩溃", ok)

check("format_report 可调用", isinstance(P.format_report(rp_good), str))

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
print("# 全部通过  M2 可以进入 M3")
print("#" * 70)
