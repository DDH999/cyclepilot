# -*- coding: utf-8 -*-
"""
M14 验证：定投台账 / XIRR / 分析
====================================================================
跑法:  python tests/test_m14_dca.py     （不联网）
"""
import shutil
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import dca as D

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m14"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M14 验证  dca v{D.__version__}")
print("#" + "=" * 69)

NL = chr(10)

print("\n[1] XIRR 正确性（已知答案）")

d0 = date(2024, 1, 2)
x1 = D.xirr([(d0, -1000.0), (d0 + timedelta(days=365), 1100.0)])
check("一年 10% → XIRR=10%", x1["ok"] and abs(x1["rate"] - 0.10) < 1e-4,
      round(x1["rate"] * 100, 4) if x1["ok"] else None)

x2 = D.xirr([(d0, -1000.0), (d0 + timedelta(days=730), 2000.0)])
check("两年翻倍 → 约 41.42%", x2["ok"] and abs(x2["rate"] - (2 ** 0.5 - 1)) < 1e-3,
      round(x2["rate"] * 100, 2) if x2["ok"] else None)

x3 = D.xirr([(d0, -1000.0), (d0 + timedelta(days=365), 800.0)])
check("亏 20% → XIRR=-20%", x3["ok"] and abs(x3["rate"] + 0.20) < 1e-4,
      round(x3["rate"] * 100, 4) if x3["ok"] else None)

x0 = D.xirr([(d0, -1000.0), (d0 + timedelta(days=365), 1000.0)])
check("不赚不亏 → XIRR≈0", x0["ok"] and abs(x0["rate"]) < 1e-4)

x4 = D.xirr([(d0, -1000.0), (d0 + timedelta(days=182), 1100.0)])
check("半年赚 10% 年化更高", x4["ok"] and x4["rate"] > 0.20,
      round(x4["rate"] * 100, 2) if x4["ok"] else None)

check("单笔现金流不支持", D.xirr([(d0, -1000.0)])["ok"] is False)
check("全负现金流不支持",
      D.xirr([(d0, -1000.0), (d0 + timedelta(days=30), -500.0)])["ok"] is False)
check("空输入不支持", D.xirr([])["ok"] is False)

# 定投多笔
flows = [(d0, -1000.0), (d0 + timedelta(days=30), -1000.0),
         (d0 + timedelta(days=60), -1000.0),
         (d0 + timedelta(days=365), 3300.0)]
xm = D.xirr(flows)
check("多笔定投可求解", xm["ok"] is True)
check("多笔定投回报为正", xm["rate"] > 0, round(xm["rate"] * 100, 2))
check("NPV 在解处接近 0", abs(xm["npv_at_rate"]) < 1e-5, xm["npv_at_rate"])

print("\n[2] 台账读取")

f1 = TMP / "l1.csv"
f1.write_text("date,code,amount,nav" + NL + "2024-01-02,050025,1000,2.00" + NL
              + "2024-07-01,050025,1000,1.80" + NL, encoding="utf-8")
led = D.load_ledger(f1)
check("读到 2 条", len(led) == 2, len(led))
check("按日期升序", led["date"].is_monotonic_increasing)
check("代码补零为 6 位", led["code"].iloc[0] == "050025")

f2 = TMP / "l2.csv"
f2.write_text("# 注释行" + NL + "date,code,amount,nav" + NL
              + "2024-01-02,050025,1000," + NL
              + "# 中间注释" + NL
              + "2024-02-01,050025,abc,2.10" + NL
              + "2024-03-01,050025,1000,2.20" + NL, encoding="utf-8")
led2 = D.load_ledger(f2)
check("忽略注释行", len(led2) == 2, len(led2))
check("净值可留空", pd.isna(led2["nav"].iloc[0]))
check("非法金额被丢弃", not led2["amount"].isna().any())

f3 = TMP / "l3.csv"
f3.write_text("# 只有注释" + NL, encoding="utf-8")
try:
    D.load_ledger(f3)
    check("空台账报错", False)
except ValueError:
    check("空台账报错", True)

try:
    D.load_ledger(TMP / "nope.csv")
    check("文件不存在报错", False)
except FileNotFoundError:
    check("文件不存在报错", True)

f4 = TMP / "l4.csv"
f4.write_text("date,code" + NL + "2024-01-02,050025" + NL, encoding="utf-8")
try:
    D.load_ledger(f4)
    check("缺金额列报错", False)
except ValueError:
    check("缺金额列报错", True)

print("\n[3] 补齐净值")

navs = {"050025": pd.Series([2.0, 2.1, 2.2],
                            index=pd.to_datetime(["2024-01-02", "2024-02-01",
                                                  "2024-03-01"]))}
filled = D.fill_missing_nav(led2, navs)
check("缺失净值被补齐", filled["nav"].notna().all(), filled["nav"].tolist())
check("补齐取当日或之前", abs(filled["nav"].iloc[0] - 2.0) < 1e-9,
      filled["nav"].iloc[0])

print("\n[4] 持仓汇总")

res = D.position_summary(led, {"050025": 2.50})
check("可汇总", res["ok"] is True)
check("投入合计 2000", abs(res["invested"] - 2000) < 1e-6)
expect_shares = 1000 / 2.0 + 1000 / 1.8
got_shares = res["positions"][0]["shares"]
check("份额正确", abs(got_shares - expect_shares) < 1e-3,
      (got_shares, round(expect_shares, 4)))
check("平均成本 = 投入/份额",
      abs(res["positions"][0]["avg_cost"] - 2000 / expect_shares) < 1e-4,
      res["positions"][0]["avg_cost"])
check("市值正确",
      abs(res["market_value"] - expect_shares * 2.50) < 1e-2,
      round(res["market_value"], 2))
check("盈亏为正", res["pnl"] > 0)

res_miss = D.position_summary(led, {})
check("缺现价时不估值", res_miss["market_value"] == 0)
check("缺现价有警告", len(res_miss["warnings"]) == 1, res_miss["warnings"])

print("\n[5] 定投全分析")

ana = D.dca_analysis(led, {"050025": 2.50}, as_of="2025-07-01")
check("可分析", ana["ok"] is True)
check("含 XIRR", ana.get("xirr") is not None, ana.get("xirr"))
check("XIRR 为正", ana["xirr"] > 0)
check("跨度计算正确", ana["span_years"] > 1.0, ana["span_years"])
check("含定投次数", ana["n_buys_total"] == 2)
check("含一次性对照", "lump_sum" in ana and ana["lump_sum"]["market_value"] > 0)
check("XIRR ≠ 简单收益率（口径不同）",
      ana["xirr"] != ana["pnl_pct"], (ana["xirr"], ana["pnl_pct"]))
check("报告可渲染", "定投分析" in D.format_report(ana))

print("\n[6] 定投 vs 一次性")

seq = pd.Series([2.0, 2.5])
cmp_ = D.compare_lump_sum(led, seq, code="050025")
check("可对比", cmp_["ok"] is True)
check("含两种份额", cmp_["dca_shares"] > 0 and cmp_["lump_shares"] > 0)
check("含差额", "diff" in cmp_)
check("有结论文案", len(cmp_["verdict"]) > 5)

# 单边上行时，一次性应更优
up = D.load_ledger(f1)
up.loc[:, "nav"] = [1.0, 1.5]     # 越买越贵
cmp_up = D.compare_lump_sum(up, pd.Series([2.0]), code="050025")
check("上涨行情一次性更优", cmp_up["lump_value"] > cmp_up["dca_value"],
      (cmp_up["lump_value"], cmp_up["dca_value"]))

down = D.load_ledger(f1)
down.loc[:, "nav"] = [2.0, 1.0]   # 越买越便宜
cmp_down = D.compare_lump_sum(down, pd.Series([2.0]), code="050025")
check("下跌行情定投更优", cmp_down["dca_value"] > cmp_down["lump_value"],
      (cmp_down["dca_value"], cmp_down["lump_value"]))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  定投分析可用")
print("#" * 70)
