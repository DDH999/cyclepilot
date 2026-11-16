# -*- coding: utf-8 -*-
"""
M14–M17 验证：组合理论 / 资产配置与凯利 / 时间序列 / 回测陷阱
====================================================================
跑法:  python tests/test_m11_theory.py     （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import allocation as AL
import pitfalls as PF
import portfolio_theory as PT
import timeseries as TS

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M14–M17 验证  theory={PT.__version__} alloc={AL.__version__} "
      f"ts={TS.__version__} pitfalls={PF.__version__}")
print("#" * 70)

IDX = pd.bdate_range("2020-01-02", periods=800)
rng = np.random.default_rng(7)
RAND = pd.Series(rng.normal(0, 0.01, 800), index=IDX)

# ══════════════════════════════════════════════════════════════
print("\n[1] EMH 弱式检验")

e_rand = PT.emh_test(RAND)
check("随机序列可评估", e_rand["ok"] is True)
check("随机序列 p 值不显著拒绝", e_rand["ljung_box_p"] > 0.05,
      e_rand["ljung_box_p"])
check("随机序列判为有效", e_rand["efficient"] is True, e_rand["verdict"])
check("ACF 列出多阶", len(e_rand["acf"]) == 5)
check("含方差比三档", len(e_rand["variance_ratio"]) == 3)

ar = np.zeros(800)
for i in range(1, 800):
    ar[i] = 0.35 * ar[i - 1] + rng.normal(0, 0.01)
e_ar = PT.emh_test(pd.Series(ar, index=IDX))
check("强自相关判为无效", e_ar["efficient"] is False, e_ar["verdict"])
check("AR 序列 LB p 显著", e_ar["ljung_box_p"] < 0.05, e_ar["ljung_box_p"])
check("AC1 明显为正", e_ar["acf"]["AC1"] > 0.2, e_ar["acf"]["AC1"])

check("样本过短返回 ok=False", PT.emh_test(RAND.iloc[:30])["ok"] is False)
vr = PT.variance_ratio(RAND, 5)
check("方差比可算", vr["ok"] and abs(vr["vr"] - 1.0) < 0.5, round(vr["vr"], 3))

# ══════════════════════════════════════════════════════════════
print("\n[2] MPT 有效前沿")

R4 = pd.DataFrame({f"A{i}": rng.normal(0.0004 * (i + 1), 0.008 + 0.004 * i, 800)
                   for i in range(4)}, index=IDX)
ef = PT.efficient_frontier(R4)
check("可算有效前沿", ef["ok"] is True)
check("前沿点数正确", len(ef["frontier"]) > 5, len(ef["frontier"]))
check("最小方差权重和为 1",
      abs(sum(ef["min_variance"]["weights"].values()) - 1) < 1e-3)
check("最小方差组合波动为正", ef["min_variance"]["vol"] > 0)
check("切点组合夏普更高",
      ef["max_sharpe"]["sharpe"] >= ef["min_variance"]["sharpe"] - 1e-6,
      (ef["max_sharpe"]["sharpe"], ef["min_variance"]["sharpe"]))
check("前沿波动随收益单调不减（大致）",
      all(a["vol"] <= b["vol"] + 0.05 for a, b in
          zip(ef["frontier"], ef["frontier"][1:])))

# 最小方差波动应低于等权
eq_w = np.full(4, 0.25)
eq_vol = float(np.sqrt(eq_w @ (R4.cov().to_numpy() * 252) @ eq_w))
check("最小方差波动 ≤ 等权波动", ef["min_variance"]["vol"] <= eq_vol + 1e-6,
      (ef["min_variance"]["vol"], round(eq_vol, 4)))

try:
    PT.efficient_frontier(R4.iloc[:, :1])
    check("单资产抛错", False)
except ValueError:
    check("单资产抛错", True)

# ══════════════════════════════════════════════════════════════
print("\n[3] CAPM")

bench = pd.Series(rng.normal(0.0004, 0.01, 800), index=IDX)
port = 1.5 * bench + rng.normal(0, 0.002, 800)          # 已知 beta=1.5
c = PT.capm_regression(pd.Series(np.cumprod(1 + port), index=IDX),
                       pd.Series(np.cumprod(1 + bench), index=IDX))
check("beta 接近 1.5", abs(c["beta"] - 1.5) < 0.15, c["beta"])
check("R² 很高", c["r2"] > 0.8, c["r2"])
check("alpha 不显著", c["alpha_significant"] is False, c["alpha_t"])
check("含定价偏离", "mispricing" in c)

# 加独立 alpha
port2 = 1.0 * bench + 0.0012 + rng.normal(0, 0.002, 800)
c2 = PT.capm_regression(pd.Series(np.cumprod(1 + port2), index=IDX),
                        pd.Series(np.cumprod(1 + bench), index=IDX))
check("注入 alpha 后年化为正", c2["alpha_annual"] > 0, c2["alpha_annual"])
check("alpha t 值更大", c2["alpha_t"] > c["alpha_t"])
check("CAPM 期望收益可算", np.isfinite(c["capm_expected"]))

# ══════════════════════════════════════════════════════════════
print("\n[4] APT / 多因子")

F1 = pd.DataFrame({"mkt": bench.values}, index=IDX)
a1 = PT.apt_regression(pd.Series(np.cumprod(1 + port), index=IDX), F1)
check("单因子 beta ≈ CAPM beta", abs(a1["loadings"]["mkt"] - 1.5) < 0.15,
      a1["loadings"]["mkt"])
check("单因子 R² 接近 CAPM", abs(a1["r2"] - c["r2"]) < 1e-6)

F3 = pd.DataFrame({"mkt": bench.values,
                   "size": rng.normal(0, 0.006, 800),
                   "value": rng.normal(0, 0.006, 800)}, index=IDX)
a3 = PT.apt_regression(pd.Series(np.cumprod(1 + port), index=IDX), F3)
check("三因子回归可算", a3["ok"] and len(a3["loadings"]) == 3)
check("显著因子至少 1 个", len(a3["significant_factors"]) >= 1, a3["significant_factors"])
check("含 t 值字典", "mkt" in a3["t_values"])
check("条件数报告", np.isfinite(a3["condition_number"]))

F_dup = pd.DataFrame({"mkt": bench.values, "mkt2": bench.values * 1.0}, index=IDX)
a_dup = PT.apt_regression(pd.Series(np.cumprod(1 + port), index=IDX), F_dup)
check("共线因子给出警告", a_dup["multicollinearity_warning"] is True,
      a_dup["condition_number"])

# ══════════════════════════════════════════════════════════════
print("\n[5] 主动管理基本法则")

fl = PT.fundamental_law(ic=0.03, breadth=252)
check("IC0.03×252 → IR≈0.476", abs(fl["ir"] - 0.03 * np.sqrt(252)) < 1e-3, fl["ir"])
fl2 = PT.fundamental_law(ir=1.0, breadth=252)
check("反推 IC 合理", abs(fl2["ic"] - 1.0 / np.sqrt(252)) < 1e-4, fl2["ic"])
fl3 = PT.fundamental_law(ir=0.5, ic=0.02)
check("反推宽度 = 625", abs(fl3["breadth"] - 625) < 1e-3, fl3["breadth"])
fl4 = PT.fundamental_law(ir=1.0, breadth=12)
check("低宽度需超高 IC → 判不现实", fl4["ic_feasible"] is False, fl4["ic"])
try:
    PT.fundamental_law(ir=1.0)
    check("参数不足抛错", False)
except ValueError:
    check("参数不足抛错", True)

# ══════════════════════════════════════════════════════════════
print("\n[6] 资产配置与风险平价")

ASSETS = pd.DataFrame({
    "股票": rng.normal(0.0006, 0.018, 800),
    "债券": rng.normal(0.00015, 0.004, 800),
    "黄金": rng.normal(0.00025, 0.010, 800),
    "现金": rng.normal(0.00006, 0.0005, 800),
}, index=IDX)

w_eq = AL.equal_weight(ASSETS.columns)
check("等权各 25%", all(abs(v - 0.25) < 1e-9 for v in w_eq.values))
check("等权和为 1", abs(w_eq.sum() - 1) < 1e-9)

w_iv = AL.inverse_vol(ASSETS)
check("逆波动权重和为 1", abs(w_iv.sum() - 1) < 1e-6)
check("逆波动给低波资产更高权重",
      w_iv["现金"] > w_iv["股票"], (w_iv["现金"], w_iv["股票"]))

w_mv = AL.min_variance(ASSETS)
check("最小方差权重和为 1", abs(w_mv.sum() - 1) < 1e-3)
check("最小方差压注低波资产", w_mv["股票"] < 0.4, w_mv["股票"])

w_rp = AL.risk_parity(ASSETS)
cov = ASSETS.cov().to_numpy() * 252
rc = AL.risk_contributions(w_rp.to_numpy(), cov)
check("风险平价权重和为 1", abs(w_rp.sum() - 1) < 1e-6)
check("风险平价风险贡献相等",
      float(np.max(np.abs(rc.to_numpy() - 0.25))) < 0.02,
      np.round(rc.to_numpy(), 3).tolist())
check("风险贡献合计为 1", abs(rc.sum() - 1) < 1e-6)
check("风险平价≠等权", float(np.max(np.abs(w_rp.to_numpy() - 0.25))) > 0.05)

cmp = AL.compare_allocation(ASSETS)
check("四方案对比", len(cmp["schemes"]) == 4)
check("含最大风险贡献", all(s["max_risk_contrib"] is not None for s in cmp["schemes"]))
rp_row = next(s for s in cmp["schemes"] if s["scheme"] == "风险平价")
eq_row = next(s for s in cmp["schemes"] if s["scheme"] == "等权")
check("风险平价风险更均衡",
      rp_row["max_risk_contrib"] < eq_row["max_risk_contrib"],
      (rp_row["max_risk_contrib"], eq_row["max_risk_contrib"]))
check("报告可渲染", "资产配置方案对比" in AL.format_report(cmp))

# ══════════════════════════════════════════════════════════════
print("\n[7] 凯利公式")

k = AL.kelly_fraction(0.55, 1.5, fraction=1.0)
check("满凯利 = p - q/b = 0.25", abs(k["kelly_full"] - 0.25) < 1e-9, k["kelly_full"])
k2 = AL.kelly_fraction(0.5, 1.0, fraction=1.0)
check("无优势时凯利为 0", abs(k2["kelly_full"]) < 1e-9, k2["kelly_full"])
check("无优势 has_edge=False", k2["has_edge"] is False)
k3 = AL.kelly_fraction(0.55, 1.5, fraction=0.25)
check("1/4 凯利 = 0.0625", abs(k3["kelly_fraction"] - 0.0625) < 1e-9, k3["kelly_fraction"])
check("分数凯利小于满凯利", k3["kelly_fraction"] < k["kelly_full"])
try:
    AL.kelly_fraction(1.5, 1.0)
    check("非法胜率抛错", False)
except ValueError:
    check("非法胜率抛错", True)

KR = AL.kelly_from_returns(ASSETS["股票"])
check("从收益序列算凯利", KR["ok"] is True)
check("含年化收益与波动", np.isfinite(KR["annual_return"]) and KR["annual_vol"] > 0)
check("凯利公式报告可渲染", "凯利" in AL.format_report(KR, kind="kelly"))

# ══════════════════════════════════════════════════════════════
print("\n[8] 时间序列")

ts_rand = TS.ts_report(RAND)
check("随机序列可诊断", ts_rand["ok"] is True)
check("Hurst ≈ 0.5", abs(ts_rand["hurst"]["hurst"] - 0.5) < 0.12,
      ts_rand["hurst"]["hurst"])
check("ACF 六阶", len(ts_rand["acf"]) == 6)
check("含 AR(1)", "ar1" in ts_rand and "phi" in ts_rand["ar1"])
check("含单位根检验", ts_rand["adf"]["ok"] is True)
check("单位根含临界值", "5%" in ts_rand["adf"]["crit"])

mr = np.zeros(800)
for i in range(1, 800):
    mr[i] = -0.4 * mr[i - 1] + rng.normal(0, 0.01)
ts_mr = TS.ts_report(pd.Series(mr, index=IDX))
check("均值回归序列 H<0.45", ts_mr["hurst"]["hurst"] < 0.45,
      ts_mr["hurst"]["hurst"])
check("均值回归 half_life 有值", ts_mr["half_life"]["half_life"] is not None,
      ts_mr["half_life"]["half_life"])
check("AR(1) φ 为负", ts_mr["ar1"]["phi"] < 0, ts_mr["ar1"]["phi"])

tr = np.zeros(800)
for i in range(1, 800):
    tr[i] = 0.5 * tr[i - 1] + rng.normal(0, 0.01)
h_tr = TS.hurst(pd.Series(tr, index=IDX))["hurst"]
check("趋势序列 H>0.55", h_tr > 0.55, h_tr)
check("趋势序列 H 高于随机序列", h_tr > ts_rand["hurst"]["hurst"])

check("方差比可算", TS.variance_ratio(RAND, 5)["ok"] is True)
check("样本过短返回 ok=False", TS.ts_report(RAND.iloc[:40])["ok"] is False)
check("时间序列报告可渲染", "时间序列诊断" in TS.format_report(ts_rand))

# ══════════════════════════════════════════════════════════════
print("\n[9] 回测陷阱清单")

good = PF.check_pitfalls(code_ok=True, survivorship_ok=True, pbo=0.25, dsr=0.97,
                         n_trials=1, costs_included=True, n_days=2000,
                         has_oos=True, data_score=92, wf_hit_rate=0.85,
                         turnover=3.0, max_risk_contrib=0.25)
check("干净回测无硬伤", good["counts"]["FAIL"] == 0, good["counts"])
check("干净回测高分", good["score"] >= 85, good["score"])
check("十条陷阱全检查", good["n_checked"] == 10)

bad = PF.check_pitfalls(code_ok=False, findings=[1, 2], survivorship_ok=False,
                        pbo=0.8, dsr=0.2, n_trials=50, costs_included=False,
                        n_days=120, has_oos=False, data_score=50,
                        wf_hit_rate=0.2, turnover=12.0, max_risk_contrib=0.6)
check("脏回测多硬伤", bad["counts"]["FAIL"] >= 5, bad["counts"])
check("脏回测低分", bad["score"] < good["score"])
check("脏回测判不可信", "不可信" in bad["verdict"], bad["verdict"])

unknown = PF.check_pitfalls()
check("全缺参数全 UNKNOWN", unknown["counts"]["UNKNOWN"] == 10, unknown["counts"])
check("全缺参数判证据不足", "证据不足" in unknown["verdict"], unknown["verdict"])
check("未检不当作通过", unknown["score"] < good["score"])

check("清单含十条定义", len(PF.PITFALLS) == 10)
check("每条含整改方向", all(len(x[3]) > 5 for x in PF.PITFALLS))
check("报告可渲染", "回测陷阱清单" in PF.format_report(bad))

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
print("# 全部通过  理论层已可计算")
print("#" * 70)
