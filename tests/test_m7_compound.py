# -*- coding: utf-8 -*-
"""M7 验证：复利测算器"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import compound as C

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def approx(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


print("\n" + "#" * 70)
print(f"# M7 验证  compound v{C.__version__}")
print("#" + "=" * 69)

print("\n[1] 增长测算")
g = C.compound_growth(100000, 0.08, 20)
check("20 年终值 466096", approx(g["final"], 466095.71, 1.0), g["final"])
check("累计投入 = 本金", approx(g["total_in"], 100000, 1e-6))
check("净赚 = 终值 - 投入", approx(g["profit"], g["final"] - g["total_in"], 1e-6))
check("倍数 4.66x", approx(g["multiple"], 4.66, 0.01), g["multiple"])
check("逐年表长度 21", len(g["table"]) == 21, len(g["table"]))
check("第 0 年等于本金", approx(g["table"][0]["end_balance"], 100000, 1e-6))
check("逐年单调递增",
      all(a["end_balance"] < b["end_balance"] for a, b in
          zip(g["table"], g["table"][1:])))
check("年化自洽", approx(g["cagr"], 0.08, 1e-4), g["cagr"])

z = C.compound_growth(100000, 0.08, 0)
check("0 年终值等于本金", approx(z["final"], 100000, 1e-6))

neg = C.compound_growth(100000, -0.05, 10)
check("负收益也会缩减", neg["final"] < 100000, neg["final"])

dca = C.compound_growth(100000, 0.08, 10, monthly_add=1000)
check("定投累计投入增加", approx(dca["total_in"], 100000 + 12000 * 10, 1e-6),
      dca["total_in"])
check("定投终值高于不加投", dca["final"] > C.compound_growth(100000, 0.08, 10)["final"])
check("定投倍数低于纯本金", dca["multiple"] < C.compound_growth(100000, 0.08, 10)["multiple"])

print("\n[2] 目标反推")
r = C.required_rate(1000000, 100000, 20)
check("10万→100万/20年 需 ~12.2%", approx(r, 0.1220, 0.001), round(r, 4))
r2 = C.required_rate(200000, 100000, 10)
check("翻倍 10 年需 ~7.18%", approx(r2, 0.0718, 0.002), round(r2, 4))
check("目标过低返回负或零附近", C.required_rate(50000, 100000, 10) < 0.01)
# 5% 上限 20 年最多约 3.65e19，超过即不可达
check("不可能达成返回 nan",
      not (C.required_rate(10**21, 100000, 20) == C.required_rate(10**21, 100000, 20)))
check("0 年返回 nan", not (C.required_rate(200000, 100000, 0) == C.required_rate(200000, 100000, 0)))

y = C.years_to_target(1000000, 100000, 0.08)
check("10万按8%到100万需 30 年", y == 30, y)
check("已达标返回 1 或更小", C.years_to_target(50000, 100000, 0.08) == 1)
check("达不到返回 None", C.years_to_target(10**12, 100000, 0.08) is None)

print("\n[3] 通胀折算")
inf = C.inflation_adjust(466095.71, 20, 0.03)
check("20 年 3% 通胀后实际约 25.8 万", approx(inf["real"], 466095.71 / 1.03**20, 1.0),
      inf["real"])
check("购买力损失约 44%", 0.40 < inf["loss_pct"] < 0.50, inf["loss_pct"])
check("0 年不变", approx(C.inflation_adjust(1000, 0)["real"], 1000, 1e-6))

print("\n[4] 成本代价")
f = C.fee_drag_effect(100000, 0.08, 20, 0.021)
check("成本拉低终值", f["net_final"] < f["gross_final"])
check("差额约 15 万", approx(f["difference"], 151379, 2000), f["difference"])
check("差额占比约 32%", 0.28 < f["difference_pct"] < 0.36, f["difference_pct"])
f0 = C.fee_drag_effect(100000, 0.08, 20, 0.0)
check("零成本无差异", approx(f0["difference"], 0, 1e-6))

print("\n[5] 回撤代价")
check("亏10%需约2年", C.drawdown_recovery_years(-0.10, 0.08) == 2)
check("亏30%需约5年", C.drawdown_recovery_years(-0.30, 0.08) == 5)
check("亏50%需约10年", C.drawdown_recovery_years(-0.50, 0.08) == 10)
check("无回撤返回 0", C.drawdown_recovery_years(0, 0.08) == 0)
check("零收益无法补回", C.drawdown_recovery_years(-0.30, 0.0) is None)
check("亏光返回 None", C.drawdown_recovery_years(-1.0, 0.08) is None)

print("\n[6] 渲染")
rep = C.format_growth(g)
for token in ["复利测算", "期末余额", "终值", "倍数"]:
    check(f"增长报告含「{token}」", token in rep, token)

cmp = C.format_compare(100000, 20)
check("对比表含 4 档", cmp.count("%") >= 4)
check("对比表含通胀列", "扣通胀后" in cmp)
check("同等年限下高收益终值更高",
      C.compound_growth(100000, 0.10, 20)["final"] > C.compound_growth(100000, 0.08, 20)["final"])

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for x in FAIL:
        print(f"#   - {x}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  复利测算可用")
print("#" * 70)
