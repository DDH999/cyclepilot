# -*- coding: utf-8 -*-
"""
M30 验证：可转债打新 / 费率核对 / 交易准入
====================================================================
跑法:  python tests/test_m30_yieldhunt.py   （不联网，全部注入假数据）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cb_new as CB
import fee_check as FE
import paper_validate as PV

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print("# M30 验证：可转债 / 费率 / 准入")
print("#" + "=" * 69)

# ══════════════════════════════════════════════════════
print("\n[1] 可转债：市场溢价")

fake = pd.DataFrame({
    "债券代码": ["111001", "111002", "111003"],
    "债券简称": ["高价债", "平价款", "破发款"],
    "正股价": [12.0, 10.0, 8.0],
    "转股价": [10.0, 10.0, 10.0],
    "转股价值": [120.0, 100.0, 80.0],
    "发行规模": [3.0, 15.0, 40.0],
    "信用评级": ["AA", "AA-", "A+"],
    "债现价": [130.0, 115.0, 90.0],
})
pm = CB.market_premium(fake)
check("可算市场溢价", pm["ok"] is True, pm.get("reason"))
check("中位数为 12.5%", abs(pm["premium"] - 0.125) < 1e-6, pm["premium"])
check("样本数正确", pm["n"] == 3, pm["n"])
check("缺字段时拒绝",
      CB.market_premium(pd.DataFrame({"x": [1]}))["ok"] is False)
check("无有效样本时拒绝",
      CB.market_premium(pd.DataFrame({"债现价": [0], "转股价值": [0]}))["ok"]
      is False)

print("\n[2] 可转债：单只评估")

hi = CB.assess_bond(fake.iloc[0].to_dict(), pm["premium"])
mid = CB.assess_bond(fake.iloc[1].to_dict(), pm["premium"])
lo = CB.assess_bond(fake.iloc[2].to_dict(), pm["premium"])
check("高转股价值可评估", hi["ok"] is True)
check("转股价值正确", abs(hi["convert_value"] - 120.0) < 1e-6)
check("风险随转股价值升高而降低",
      hi["score"] < mid["score"] < lo["score"],
      (hi["score"], mid["score"], lo["score"]))
check("低转股价值建议放弃", lo["advice"] == "放弃", lo["advice"])
check("高转股价值正常申购", hi["advice"] == "正常申购", hi["advice"])
check("高转股价值风险等级低", hi["level"] == "低", hi["level"])
check("给出理由", len(hi["reasons"]) >= 2, hi["reasons"])

# 溢价上限：42% 不能直接套到首日
hi_cap = CB.assess_bond(fake.iloc[0].to_dict(), 0.42)
check("溢价被限制在 30% 以内",
      hi_cap["expected_price"] <= 120.0 * 1.301,
      hi_cap["expected_price"])
check("说明了上限", any("上限" in r for r in hi_cap["reasons"]),
      hi_cap["reasons"])

# 缺数据必须拒绝
bad = CB.assess_bond({"债券代码": "x", "债券简称": "x"}, 0.15)
check("缺转股价/正股价 → 拒绝估值", bad["ok"] is False, bad.get("reason"))

# 用 正股价/转股价 现算
calc = CB.assess_bond({"债券代码": "y", "债券简称": "y", "正股价": 11.0,
                       "转股价": 10.0}, 0.15)
check("可由正股价/转股价现算",
      calc["ok"] and abs(calc["convert_value"] - 110.0) < 1e-6,
      calc.get("convert_value"))

print("\n[3] 可转债：扫描")

_orig_load = CB.load_bonds
fake2 = fake.copy()
fake2["申购日期"] = pd.to_datetime(["2026-11-16", "2026-11-30", "2026-12-20"])
fake2["申购代码"] = ["713001", "713002", "713003"]
fake2["正股简称"] = ["甲", "乙", "丙"]
fake2["申购上限"] = [100, 100, 100]
CB.load_bonds = lambda refresh=False: fake2
try:
    res = CB.scan(days_ahead=7, today="2026-11-13")
    check("扫描可跑", res["ok"] is True)
    check("只含 7 天内的", len(res["items"]) == 1, len(res["items"]))
    check("正确识别申购日", res["items"][0]["apply_date"] == "2026-11-16",
          res["items"][0].get("apply_date"))

    res2 = CB.scan(days_ahead=60, today="2026-11-13")
    check("放宽窗口后变多", len(res2["items"]) == 3, len(res2["items"]))
    check("给出整体结论", "共" in res2["verdict"], res2["verdict"])
    check("报告可渲染", "可转债打新扫描" in CB.format_report(res2))
    check("报告含中签率提示", "中签率很低" in CB.format_report(res2))

    empty = CB.scan(days_ahead=1, today="2027-06-01")
    check("无新债时明确说明", "没有可申购" in empty["verdict"], empty["verdict"])
finally:
    CB.load_bonds = _orig_load

# ══════════════════════════════════════════════════════
print("\n[4] 费率：临界点")

be = FE.breakeven_months(0.0015, 0.0040)
check("临界点可算", be["ok"] is True)
check("0.15% vs 0.40% → 4.5 个月", abs(be["months"] - 4.5) < 0.05,
      be["months"])
check("给出判断", "A 类" in be["verdict"], be["verdict"])

be2 = FE.breakeven_months(0.015, 0.0040)
check("标准费率 1.5% → 45 个月", abs(be2["months"] - 45.0) < 0.05,
      be2["months"])

check("销售服务费为 0 时拒绝", FE.breakeven_months(0.001, 0)["ok"] is False)
check("负申购费拒绝", FE.breakeven_months(-0.01, 0.004)["ok"] is False)
check("非数字拒绝", FE.breakeven_months("x", 0.004)["ok"] is False)

print("\n[5] 费率：A/C 对比")

r20 = FE.compare_ac(monthly=4000, years=20)
check("20 年可算", r20["ok"] is True)
check("20 年选 A 类", r20["better"] == "A类", r20["better"])
check("A 类终值更高", r20["fv_a"] > r20["fv_c"],
      (r20["fv_a"], r20["fv_c"]))
check("差额为正", r20["diff"] > 0, r20["diff"])

r_short = FE.compare_ac(monthly=4000, years=0.25)
check("3 个月选 C 类", r_short["better"] == "C类", r_short["better"])

# 长期持有，A 类优势应更大
r30 = FE.compare_ac(monthly=4000, years=30)
check("30 年 A 类仍更好", r30["better"] == "A类", r30["better"])

check("自定义费率生效",
      FE.compare_ac(monthly=4000, years=20, c_service=0.0080)["c_annual_cost"]
      > r20["c_annual_cost"])

print("\n[6] 费率：影响估算")

imp = FE.fee_impact(4000, 20, 0.07, 0.0060, 0.0110)
check("可算", imp["ok"] is True)
check("费率差 0.5pp", abs(imp["fee_gap_pp"] - 0.5) < 1e-6, imp["fee_gap_pp"])
check("高费率终值更低", imp["fv_low_fee"] > imp["fv_high_fee"])
check("差额为正", imp["diff"] > 0, imp["diff"])
check("报告可渲染", "费率核对" in FE.format_report(r20, imp))
check("报告含免责", "以基金合同为准" in FE.format_report(r20, imp))

swapped = FE.fee_impact(4000, 20, 0.07, 0.0110, 0.0060)
check("参数顺序自动纠正", abs(swapped["diff"] - imp["diff"]) < 1.0,
      (swapped["diff"], imp["diff"]))

print("\n[7] 准入：五道闸门")


def mk(mu, sigma, n=400, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + rng.normal(0, sigma, n),
                     index=pd.bdate_range("2024-01-02", periods=n))


good = mk(0.0008, 0.004, seed=7)
# 用比例构造基准，保证超额方向确定（不靠随机数运气）
bench = good * 0.6

g1 = PV.check_sample(good, 60)
check("样本量足够通过", g1["pass"] is True)
check("样本不足不通过", PV.check_sample(good[:20], 60)["pass"] is False)
check("笔数不足不通过", PV.check_sample(good, 10)["pass"] is False)

rng_ = np.random.default_rng(1)
ret_ok = pd.Series(rng_.normal(0.002, 0.005, 300),
                   index=pd.bdate_range("2024-01-02", periods=300))
g2 = PV.check_net(ret_ok, turnover_year=0)
check("无成本时净为正", g2["pass"] is True)
check("高换手会转负",
      PV.check_net(ret_ok, turnover_year=500)["pass"] is False,
      PV.check_net(ret_ok, turnover_year=500).get("net_annual"))

g3 = PV.check_excess(good, bench)
check("有超额通过", g3["pass"] is True, g3.get("excess_annual"))
check("跑输不通过",
      PV.check_excess(bench, good)["pass"] is False)
check("重叠期不足不通过",
      PV.check_excess(good[:10], bench)["pass"] is False)

g4 = PV.check_robust(good)
check("稳健性可算", "positive_rate" in g4, g4.get("positive_rate"))
check("多折为正通过", g4["pass"] is True, g4.get("note"))
bad_seq = pd.concat([good[:100], -good[100:]])
check("前后反转不通过", PV.check_robust(bad_seq)["pass"] is False,
      PV.check_robust(bad_seq).get("note"))

g5 = PV.check_concentration(good)
check("剔除极端日后仍盈利", g5["pass"] is True, g5.get("note"))
lucky = mk(-0.0001, 0.004, seed=5)
lucky.iloc[10] = 0.30
check("靠单日暴利不通过", PV.check_concentration(lucky)["pass"] is False,
      lucky.sum())
check("总收益为负直接不通过",
      PV.check_concentration(pd.Series(-0.001, index=range(200)))["pass"]
      is False)

print("\n[8] 准入：整体判定")

allpass = PV.validate(good, bench, turnover_year=0, n_trades=60)
check("好策略可判定", allpass["ok"] is True)
check("五道闸门齐全", allpass["n_gate"] == 5, allpass["n_gate"])
check("好策略全通过", allpass["all_pass"] is True, allpass["failed"])
check("给出 GO", "GO" in allpass["verdict"], allpass["verdict"])
check("含硬止损规则", "3 个自然月" in allpass["hard_stop_rule"],
      allpass["hard_stop_rule"])
check("建议含试运行上限", "10%" in allpass["advice"], allpass["advice"])

nogo = PV.validate(bench, good, turnover_year=50, n_trades=5)
check("差策略被拦", nogo["all_pass"] is False)
check("列出未通过项", len(nogo["failed"]) >= 1, nogo["failed"])
check("给出 NO-GO", "NO-GO" in nogo["verdict"], nogo["verdict"])

no_bench = PV.validate(good, None)
check("无基准时第三门不通过", any("跑赢基准" in x for x in no_bench["failed"]),
      no_bench["failed"])

check("样本过少返回 ok=False",
      PV.validate(pd.Series([0.01, 0.02]))["ok"] is False)
check("报告可渲染", "主动交易准入判定" in PV.format_report(allpass))
check("报告含免责", "≠ 未来会赚钱" in PV.format_report(allpass))

# ══════════════════════════════════════════════════════
print("\n[9] CLI / API")


def boom(*a, **k):
    raise RuntimeError("fake")


import api_v2 as API
from fastapi.testclient import TestClient
cli = TestClient(API.app)

r = cli.get("/api/fee?monthly=4000&years=20")
check("fee 接口 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    b = r.json()
    check("fee 返回文本", "费率核对" in b.get("text", ""))
    check("fee 返回结论", b["result"].get("better") == "A类")

_orig_scan = CB.scan
CB.scan = lambda **k: {"ok": True, "from": "a", "to": "b", "items": [],
                       "market_premium": {"ok": False},
                       "verdict": "测试"}
try:
    r2 = cli.get("/api/cb_new?days=3")
    check("cb 接口 200", r2.status_code == 200, r2.status_code)
finally:
    CB.scan = _orig_scan

r3 = cli.get("/api/admit?file=nope.csv")
check("admit 文件不存在返回 404", r3.status_code == 404, r3.status_code)

import quantpilot as QP
for c in ("cb", "fee", "admit"):
    check(f"CLI 注册 {c}", c in QP.COMMANDS)
pp = QP.build_parser()
check("解析 cb --days", pp.parse_args(["cb", "--days", "5"]).days == 5)
check("解析 fee --a-purchase",
      abs(pp.parse_args(["fee", "--a-purchase", "0.001"]).a_purchase - 0.001)
      < 1e-12)
check("解析 admit --file", pp.parse_args(["admit", "--file", "x.csv"]).file
      == "x.csv")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  三条实盘路径可用")
print("#" * 70)
