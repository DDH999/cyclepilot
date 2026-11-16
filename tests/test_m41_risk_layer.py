# -*- coding: utf-8 -*-
"""M41 验证：风险预算 / 事前风控 / 熔断 / 截面归因（不联网）"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import attribution as AT
import risk_budget as RB
import risk_control as RC

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
          (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M41 验证  risk_budget v{RB.__version__} / risk_control v{RC.__version__}")
print("#" + "=" * 69)

print("\n[1] 波动率目标")
rng = np.random.default_rng(5)
idx = pd.bdate_range("2023-01-02", periods=400)
r = pd.Series(rng.normal(0.0003, 0.015, 400), index=idx)
cap = RB.vol_target_cap(r, target_vol=0.12, window=20, max_leverage=1.0)
check("cap 非空", len(cap) == len(r), len(cap))
check("cap 在 [0,1]", float(cap.min()) >= 0 and float(cap.max()) <= 1.0 + 1e-9,
      (float(cap.min()), float(cap.max())))
check("cap 无 NaN", int(cap.isna().sum()) == 0)
check("cap 首值为 0（无历史）", float(cap.iloc[0]) == 0.0, cap.iloc[0])
# 无前视：cap_t 只依赖 t-1 及之前
c2 = RB.vol_target_cap(r.copy(), 0.12, 20, 1.0)
check("cap 可复现", np.allclose(cap.to_numpy(), c2.to_numpy()))
r_hi = pd.Series(rng.normal(0, 0.04, 400), index=idx)
cap_hi = RB.vol_target_cap(r_hi, 0.12, 20, 1.0)
check("高波动 → cap 更低", float(cap_hi.mean()) < float(cap.mean()),
      (round(float(cap_hi.mean()), 3), round(float(cap.mean()), 3)))

print("\n[2] 风险预算权重")
vols = pd.Series({"a": 0.10, "b": 0.20, "c": 0.40})
iv = RB.inverse_vol_weights(vols)
check("逆波动合计 1", abs(float(iv.sum()) - 1) < 1e-9, iv.sum())
check("低波权重更大", iv["a"] > iv["b"] > iv["c"], iv.round(3).to_dict())
check("逆波动与波动成反比", abs(iv["a"] / iv["b"] - 2.0) < 1e-6)
check("top_n 生效", len(RB.inverse_vol_weights(vols, top_n=2)) == 2)
check("全零波动返回空", len(RB.inverse_vol_weights(pd.Series({"a": 0, "b": 0}))) == 0)

C = pd.DataFrame([[0.01, 0.002], [0.002, 0.09]], index=["a", "b"],
                 columns=["a", "b"])
erc = RB.risk_parity_weights(C)
rc = RB.risk_contributions(erc, C)
check("ERC 合计 1", abs(float(erc.sum()) - 1) < 1e-6, erc.sum())
check("ERC 风险贡献各半", abs(rc["a"] - 0.5) < 0.01 and abs(rc["b"] - 0.5) < 0.01,
      rc.round(4).to_dict())
check("ERC 低波权重更大", erc["a"] > erc["b"], erc.round(3).to_dict())
check("单资产 ERC = 1", float(RB.risk_parity_weights(
    pd.DataFrame([[0.04]], index=["x"], columns=["x"])).iloc[0]) == 1.0)

print("\n[3] 事前风控")
L = RC.Limits(max_weight=0.15, max_industry=0.35, max_leverage=1.0)
w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
res = RC.check_pre_trade(w, limits=L)
check("单票上限生效", float(res["adjusted"].max()) <= 0.15 + 1e-9,
      float(res["adjusted"].max()))
check("捕获违规", any(v["rule"] == "max_weight" for v in res["violations"]))
check("合计不超 1", float(res["adjusted"].sum()) <= 1.0 + 1e-9)

res2 = RC.check_pre_trade(pd.Series({"A": 0.5, "B": 0.5}),
                          industry={"A": "tech", "B": "tech"},
                          limits=RC.Limits(max_weight=1.0, max_industry=0.4))
check("行业上限生效", float(res2["adjusted"].sum()) <= 0.4 + 1e-6,
      float(res2["adjusted"].sum()))

res3 = RC.check_pre_trade(pd.Series({"A": 0.6, "B": 0.6}),
                          limits=RC.Limits(max_weight=1.0, max_leverage=1.0))
check("杠杆上限生效", float(res3["adjusted"].sum()) <= 1.0 + 1e-9)

prev = pd.Series({"A": 0.5, "B": 0.5})
res4 = RC.check_pre_trade(pd.Series({"C": 0.5, "D": 0.5}),
                          prev_weights=prev,
                          limits=RC.Limits(max_weight=1.0, max_turnover=0.2))
check("换手上限生效", res4["turnover_after"] <= 0.2 + 1e-9,
      (res4["turnover_before"], res4["turnover_after"]))
check("换手降下来了", res4["turnover_after"] < res4["turnover_before"])
check("合规输入无违规", RC.check_pre_trade(
    pd.Series({"A": 0.1}), limits=L)["ok"] is True)

print("\n[4] 回撤熔断")
nav = pd.Series(np.concatenate([np.linspace(1, 1.3, 80),
                                np.linspace(1.3, 1.0, 40),
                                np.linspace(1.0, 1.2, 40)]),
                index=pd.bdate_range("2023-01-02", periods=160))
g = RC.circuit_breaker(nav, RC.Limits(dd_stop=0.15, dd_cooldown=10))
check("闸门长度一致", len(g) == len(nav))
check("闸门只有 0/1", set(np.unique(g.to_numpy())) <= {0.0, 1.0},
      set(np.unique(g.to_numpy())))
check("触发熔断", int((g == 0).sum()) > 0, int((g == 0).sum()))
check("熔断后 shift 一天", g.iloc[0] == 1.0)
flat = pd.Series(1.0, index=pd.bdate_range("2023-01-02", periods=100))
check("无回撤不熔断", int((RC.circuit_breaker(flat, RC.Limits()) == 0).sum()) == 0)

print("\n[5] 敞口与监控")
exp = RC.exposure_report(pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
                         industry={"A": "t", "B": "t", "C": "b", "D": "b"})
check("敞口标的数", exp["n"] == 4, exp["n"])
check("HHI = 0.25", abs(exp["hhi"] - 0.25) < 1e-9, exp["hhi"])
check("有效持仓数 = 4", abs(exp["effective_n"] - 4.0) < 1e-6, exp["effective_n"])
check("行业集中度 0.5", abs(exp["max_industry"] - 0.5) < 1e-9)
check("空仓位", RC.exposure_report(pd.Series(dtype=float))["n"] == 0)
mon = RC.monitor(nav, None, None, RC.Limits(dd_stop=0.15, dd_cooldown=10))
check("监控返回天数", mon["days"] == len(nav))
check("监控含熔断天数", mon["breaker_off_days"] >= 1, mon["breaker_off_days"])

print("\n[6] 截面归因")
contrib = {"600519": 120000, "000001": 40000, "600036": -60000}
st = AT.by_stock(contrib, 1e6)
check("单票归因行数", len(st) == 3, len(st))
check("单位是 pp", abs(st[st["code"] == "600519"]["contrib_pp"].iloc[0] - 12.0) < 1e-6)
check("降序排列", st["contrib_pp"].is_monotonic_decreasing)
grp = AT.by_group(contrib, {"600519": "食品", "000001": "银行", "600036": "银行"}, 1e6)
check("分组数 2", len(grp) == 2, len(grp))
check("分组合计等于总和",
      abs(float(grp["contrib_pp"].sum()) - 10.0) < 1e-6,
      float(grp["contrib_pp"].sum()))

idx2 = pd.bdate_range("2023-01-02", periods=250)
p = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 250)), index=idx2)
b = pd.Series(np.cumprod(1 + rng.normal(0.0002, 0.009, 250)), index=idx2)
vb = AT.vs_benchmark(p, b)
check("基准对比可跑", vb["ok"] is True)
check("有 beta", np.isfinite(vb["beta"]), vb["beta"])
check("TE 非负", vb["tracking_error"] >= 0)
check("区间不足返回失败", AT.vs_benchmark(p.head(5), b.head(5))["ok"] is False)

br = AT.brinson(pd.Series({"x": 0.6, "y": 0.4}), pd.Series({"x": 0.5, "y": 0.5}),
                pd.Series({"x": 0.02, "y": 0.04}), pd.Series({"x": 0.01, "y": 0.03}))
check("Brinson 三效应齐全",
      set(["allocation", "selection", "interaction"]) <= set(br.columns))
check("Brinson 合计等于三者和",
      abs(float((br["allocation"] + br["selection"] + br["interaction"] - br["total"])
                .abs().sum())) < 1e-12)

rec = AT.recombination_check({"a": 100000}, 0.10, 1e6)
check("对平检查通过", rec["ok"] is True, rec["gap_pp"])
rec2 = AT.recombination_check({"a": 300000}, 0.10, 1e6)
check("对平偏离被标记", rec2["ok"] is False, rec2["gap_pp"])

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  第 3 层风控可用")
print("#" * 70)
