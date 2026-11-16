# -*- coding: utf-8 -*-
"""
M31 验证：可转债双低
====================================================================
跑法:  python tests/test_m31_cb_double_low.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cb_double_low as DL

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M31 验证  cb_double_low v{DL.__version__}")
print("#" + "=" * 69)

print("\n[1] 双低值")

check("100+10 = 110", abs(DL.double_low_value(100, 10) - 110) < 1e-9)
check("105+12 = 117", abs(DL.double_low_value(105, 12) - 117) < 1e-9)
check("低价格更优", DL.double_low_value(100, 5) < DL.double_low_value(120, 5))
check("低溢价更优", DL.double_low_value(100, 5) < DL.double_low_value(100, 15))

print("\n[2] 候选池（注入假数据）")

fake = pd.DataFrame({
    "债券代码": ["111001", "111002", "111003", "111004", "111005", "111006"],
    "债券简称": ["优等", "次优", "高价", "低评级", "强赎区", "无现价"],
    "正股简称": ["甲", "乙", "丙", "丁", "戊", "己"],
    "债现价": [102.0, 108.0, 150.0, 103.0, 120.0, None],
    "转股价": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "正股价": [10.5, 11.0, 13.0, 10.2, 14.0, 10.0],
    "转股价值": [105.0, 110.0, 130.0, 102.0, 140.0, 100.0],
    "转股溢价率": [2.0, 8.0, 15.0, 5.0, 3.0, 5.0],
    "信用评级": ["AAA", "AA", "AAA", "BBB", "AAA", "AA"],
    "发行规模": [5.0, 10.0, 20.0, 3.0, 8.0, 6.0],
})

_orig = DL.load_current
DL.load_current = lambda refresh=False: fake
try:
    sc = DL.screener(top=10)
    check("候选池可生成", sc["ok"] is True, sc.get("reason"))
    names = [x["name"] for x in sc["items"]]
    check("高价被排除", "高价" not in names, names)
    check("低评级被排除", "低评级" not in names, names)
    check("强赎区被排除", "强赎区" not in names, names)
    check("无现价被排除", "无现价" not in names, names)
    check("保留优等与次优", set(names) == {"优等", "次优"}, names)
    check("按双低升序", (sc["items"][0]["double_low"]
                        <= sc["items"][-1]["double_low"]),
          [x["double_low"] for x in sc["items"]])
    check("优等排第一（104 < 116）", sc["items"][0]["name"] == "优等",
          sc["items"][0]["name"])
    check("记录了排除原因", len(sc["excluded"]) >= 3, sc["excluded"])
    check("报告可渲染", "双低候选池" in DL.format_screen(sc))
    check("报告含免责", "候选池 ≠ 买入信号" in DL.format_screen(sc))

    rr = DL.redeem_risk()
    check("强赎可识别", rr["ok"] is True)
    check("识别出强赎区那一只", rr["n"] == 2, rr["n"])
    check("强赎按转股价值降序",
          rr["items"][0]["convert_value"] >= rr["items"][-1]["convert_value"])
finally:
    DL.load_current = _orig

print("\n[3] 面板与回测")


def mk_panel(n_days=400, n_bonds=6, seed=7, drift=0.0004):
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(seed)
    px, pm = {}, {}
    for i in range(n_bonds):
        px[f"C{i}"] = 100 * np.cumprod(
            1 + rng.normal(drift + 0.0002 * i, 0.010, n_days))
        pm[f"C{i}"] = 10 + 4 * i + rng.normal(0, 1.0, n_days)
    return (pd.DataFrame(px, index=idx), pd.DataFrame(pm, index=idx))


px, pm = mk_panel()
hist = {}
for c in px.columns:
    hist[c] = pd.DataFrame({"date": px.index, "close": px[c].values,
                            "premium": pm[c].values,
                            "convert_value": px[c].values})
p2, m2 = DL.build_panel(hist)
check("面板形状正确", p2.shape == px.shape, p2.shape)
check("溢价矩阵形状一致", m2.shape == pm.shape, m2.shape)
check("索引为日期", isinstance(p2.index, pd.DatetimeIndex))

bt = DL.backtest_double_low(px, pm, top_n=3, rebalance_days=20)
check("回测可跑", bt["ok"] is True, bt.get("reason"))
check("有净值序列", len(bt["nav"]) > 50, len(bt["nav"]))
check("年化可算", bt["annual"] is not None)
check("回撤为负", bt["max_drawdown"] <= 0, bt["max_drawdown"])
check("有调仓记录", bt["rebalances"] >= 3, bt["rebalances"])
check("记录样本数", bt["n_bonds_available"] == 6)
check("警告含幸存者偏差",
      any("幸存者偏差" in w for w in bt["warnings"]), bt["warnings"])
check("净值无 NaN", not bt["nav"].isna().any())
check("净值全为正", bool((bt["nav"] > 0).all()))

# 下跌市场应为负收益
pxd, pmd = mk_panel(drift=-0.0010)
btd = DL.backtest_double_low(pxd, pmd, top_n=3, rebalance_days=20)
check("下跌市场年化为负", btd["annual"] < 0, btd["annual"])

# 样本太少要拒绝
few = DL.backtest_double_low(px.iloc[:, :2], pm.iloc[:, :2])
check("转债太少拒绝回测", few["ok"] is False, few.get("reason"))
short = DL.backtest_double_low(px.iloc[:30], pm.iloc[:30])
check("历史太短拒绝回测", short["ok"] is False, short.get("reason"))

# 不调仓（持有到底）应接近买入持有
# 长周期但仍在守卫内（400 天，需 ≥3×周期 → 100 天可过；390 天会被拦）
bh = DL.backtest_double_low(px, pm, top_n=6, rebalance_days=100)
check("长调仓周期可跑", bh["ok"] is True, bh.get("reason"))

check("回测报告可渲染", "双低轮动回测" in DL.format_backtest(bt))
check("回测报告含警告", "必读警告" in DL.format_backtest(bt))
check("失败报告可渲染",
      "无法回测" in DL.format_backtest({"ok": False, "reason": "x"}))

print("\n[4] 换手成本")

bt_cost = DL.backtest_double_low(px, pm, top_n=3, rebalance_days=20,
                                 cost_per_turn=0.0100)
check("成本更高收益更低", bt_cost["final_nav"] <= bt["final_nav"],
      (bt_cost["final_nav"], bt["final_nav"]))

print("\n[5] 常量与闸门")

check("强赎线为 130", DL.REDEEM_LINE == 130.0)
check("价格上限存在", DL.MAX_PRICE == 135.0)
check("评级白名单不含 A+", "A+" not in DL.MIN_RATING_OK, DL.MIN_RATING_OK)

print("\n[6] CLI / API")

import api_v2 as API
from fastapi.testclient import TestClient

DL.load_current = lambda refresh=False: fake
try:
    cli = TestClient(API.app)
    r = cli.get("/api/cb_double_low?top=5")
    check("接口 200", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        b = r.json()
        check("返回候选池", b["screen"].get("ok") is True)
        check("返回强赎名单", "redeem" in b)
        check("含文本", "双低候选池" in b.get("text", ""))
finally:
    DL.load_current = _orig

import quantpilot as QP
check("CLI 注册 cbdl", "cbdl" in QP.COMMANDS)
pp = QP.build_parser()
check("解析 --backtest", pp.parse_args(["cbdl", "--backtest"]).backtest is True)
check("解析 --top", pp.parse_args(["cbdl", "--top", "8"]).top == 8)
check("解析 --bonds", pp.parse_args(["cbdl", "--bonds", "50"]).bonds == 50)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  双低模块可用（回测仍需补退市样本）")
print("#" * 70)
