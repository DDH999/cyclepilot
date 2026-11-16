# -*- coding: utf-8 -*-
"""
M32 验证：滑点 / 冲击成本 / 模拟盘
====================================================================
跑法:  python tests/test_m32_cb_paper.py   （不联网）
"""
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cb_paper as CP

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m32"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M32 验证  cb_paper v{CP.__version__}")
print("#" + "=" * 69)

print("\n[1] 冲击成本函数")

check("参与率 0 → 冲击 0", CP.impact_bp(0) == 0.0)
check("单调递增",
      CP.impact_bp(0.001) < CP.impact_bp(0.01) < CP.impact_bp(0.10))
check("0.05% → 约 2.2bp", abs(CP.impact_bp(0.0005) - 2.236) < 0.05,
      round(CP.impact_bp(0.0005), 3))
check("10% → 约 31.6bp", abs(CP.impact_bp(0.10) - 31.62) < 0.1,
      round(CP.impact_bp(0.10), 2))
check("负参与率不报错", CP.impact_bp(-1) == 0.0)
check("系数 k 可调", CP.impact_bp(0.01, k=2.0) > CP.impact_bp(0.01, k=1.0))

print("\n[2] 单笔成本拆解")

c = CP.order_cost(100_000, 2.0e8)
check("可算", c["ok"] is True)
check("参与率正确", abs(c["participation"] - 0.0005) < 1e-9,
      c["participation"])
check("佣金 = 万二", abs(c["commission"] - 20.0) < 1e-6, c["commission"])
check("半价差 = 5bp", abs(c["half_spread"] - 50.0) < 1e-6, c["half_spread"])
check("总成本 = 三项之和",
      abs(c["total"] - (c["commission"] + c["half_spread"] + c["impact"]))
      < 1e-6)
check("总 bp 合理", 5 < c["total_bp"] < 20, c["total_bp"])
check("未标记过大", c["too_large"] is False)
check("未标记不可执行", c["unexecutable"] is False)

# 小额也要收最低佣金
c_small = CP.order_cost(100, 2.0e8)
check("小额按最低佣金 0.1 元",
      abs(c_small["commission"] - CP.COMMISSION_MIN) < 1e-9,
      c_small["commission"])

# 大单进入不可执行区
c_big = CP.order_cost(2_000_000, 3.0e6)
check("参与率 >10% 标记需拆单", c_big["too_large"] is True,
      c_big["participation"])
check("参与率 >30% 标记不可执行", c_big["unexecutable"] is True,
      c_big["participation"])
check("大单冲击显著", c_big["impact_bp"] > 30, c_big["impact_bp"])

# 成交额未知 → 保守假设并标记
c_unk = CP.order_cost(100_000, None)
check("未知成交额标记假设", c_unk["amount_assumed"] is True)
check("未知时用 3000 万假设",
      abs(c_unk["daily_amount"] - 3.0e7) < 1, c_unk["daily_amount"])
check("下单为 0 拒绝", CP.order_cost(0, 1e8)["ok"] is False)

print("\n[3] 成本随本金放大")


def mk(n_days=400, n_bonds=8, seed=7):
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(seed)
    px = pd.DataFrame({f"C{i}": 100 * np.cumprod(
        1 + rng.normal(0.0005, 0.012, n_days)) for i in range(n_bonds)},
        index=idx)
    pm = pd.DataFrame({f"C{i}": 10 + 3 * i +
                       rng.normal(0, 1, n_days) for i in range(n_bonds)},
                      index=idx)
    amt = pd.DataFrame({c: rng.uniform(1e7, 3e8, n_days) for c in px.columns},
                       index=idx)
    return px, pm, amt


px, pm, amt = mk()
small = CP.simulate_double_low(px, pm, amt, top_n=5, rebalance_days=20,
                               capital=100_000)
big = CP.simulate_double_low(px, pm, amt, top_n=5, rebalance_days=20,
                             capital=20_000_000)
check("小资金可模拟", small["ok"] is True, small.get("reason"))
check("大资金可模拟", big["ok"] is True, big.get("reason"))
check("大资金拖累更大",
      big["cost_drag_annual"] >= small["cost_drag_annual"],
      (big["cost_drag_annual"], small["cost_drag_annual"]))
check("大资金冲击更多需拆单",
      big["too_large"] >= small["too_large"],
      (big["too_large"], small["too_large"]))
check("净收益不高于毛收益",
      small["net"]["annual"] <= small["gross"]["annual"] + 1e-9,
      (small["net"]["annual"], small["gross"]["annual"]))
check("成本三项之和等于总成本",
      abs(small["cost_total"] - (small["fee_total"] + small["spread_total"]
                                 + small["impact_total"])) < 1.0,   # 各自已四舍五入
      (small["cost_total"], small["fee_total"] + small["spread_total"]
       + small["impact_total"]))
check("有成交记录", small["n_trades"] > 0, small["n_trades"])
check("单笔平均成本合理", 0 < small["avg_cost_bp"] < 100,
      small["avg_cost_bp"])

print("\n[4] 模拟盘状态")

CP.PAPER_FILE = TMP / "state.json"
st = CP.init_state(100_000)
check("初始化成功", st["ok"] is True)
check("本金正确", st["capital"] == 100_000)
check("空仓", st["holdings"] == [])
check("现金等于本金", st["cash"] == 100_000)
check("状态已落盘", CP.PAPER_FILE.exists())

loaded = CP.load_state()
check("可读回", loaded.get("capital") == 100_000, loaded.get("capital"))
check("状态报告可渲染", "可转债模拟盘状态" in CP.format_state(loaded))
check("空状态报告可渲染",
      "还没有模拟盘" in CP.format_state({"ok": False, "reason": "没有"}),
      CP.format_state({"ok": False, "reason": "没有"})[:40])

# 没有状态文件时的行为
CP.PAPER_FILE = TMP / "nope.json"
check("无文件时返回 ok=False", CP.load_state()["ok"] is False)

print("\n[5] 前进式推进（不联网：打桩）")

CP.PAPER_FILE = TMP / "adv.json"
CP.init_state(100_000)


def fake_prices():
    return {"111001": {"price": 101.0, "name": "甲债"},
            "111002": {"price": 102.0, "name": "乙债"},
            "111003": {"price": 103.0, "name": "丙债"}}


def fake_screener(top=10):
    return {"ok": True, "items": [
        {"code": "111001", "name": "甲债", "price": 101.0},
        {"code": "111002", "name": "乙债", "price": 102.0}]}


_orig_px = CP._current_prices
import cb_double_low as DL
_orig_sc = DL.screener
CP._current_prices = fake_prices
DL.screener = fake_screener
try:
    r = CP.advance(execute=True, force_rebalance=True)
    check("推进成功", r["ok"] is True)
    check("发生调仓", r["rebalanced"] is True, r["n_executed"])
    check("买到 2 只", len(r["holdings"]) == 2, len(r["holdings"]))
    check("有成交成本", r["cost"] > 0, r["cost"])
    check("净值略低于本金（扣成本）", r["nav_after"] < 100_000, r["nav_after"])
    check("记录了净值", len(r["state"]["nav_history"]) == 1)

    # dry run 不成交
    r2 = CP.advance(execute=False, force_rebalance=True)
    check("dry 模式不成交", r2["n_executed"] == 0, r2["n_executed"])
    check("dry 模式仍记净值", len(r2["state"]["nav_history"]) == 2)

    # 第二次推进不应重复调仓（未到 20 天）
    CP.save_state({**r2["state"], "bars_since_rebalance": 5})
    r3 = CP.advance(execute=True)
    check("未到期不调仓", r3["n_executed"] == 0, r3["n_executed"])
finally:
    CP._current_prices = _orig_px
    DL.screener = _orig_sc

print("\n[6] 报告与常量")

check("佣金万二", CP.COMMISSION_RATE == 0.0002)
check("最低佣金 0.1", CP.COMMISSION_MIN == 0.1)
check("半价差 5bp", CP.HALF_SPREAD_BP == 5.0)
check("拆单阈值 10%", CP.PARTICIPATION_WARN == 0.10)
check("不可执行阈值 30%", CP.PARTICIPATION_MAX == 0.30)

txt = CP.format_report(small)
check("模拟盘报告可渲染", "可转债模拟盘" in txt)
check("报告含成本拆解", "冲击成本" in txt)
check("报告含三项成本", all(k in txt for k in ("佣金", "半价差", "冲击")))
check("报告含警告", "必读" in txt)
check("报告含幸存者偏差警告", "幸存者偏差" in txt)
check("失败报告可渲染",
      "无法模拟" in CP.format_report({"ok": False, "reason": "x"}))

print("\n[7] CLI / API")

import api_v2 as API
from fastapi.testclient import TestClient

CP.PAPER_FILE = TMP / "api.json"
CP.init_state(50_000)
cli = TestClient(API.app)
r = cli.get("/api/cb_paper")
check("接口 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    b = r.json()
    check("返回状态", b["state"].get("capital") == 50_000)
    check("含文本", "可转债模拟盘状态" in b.get("text", ""))

import quantpilot as QP
check("CLI 注册 cbpaper", "cbpaper" in QP.COMMANDS)
pp = QP.build_parser()
check("解析 --init", pp.parse_args(["cbpaper", "--init"]).init is True)
check("解析 --cost", pp.parse_args(["cbpaper", "--cost"]).cost is True)
check("解析 --capital",
      pp.parse_args(["cbpaper", "--init", "--capital", "5000"]).capital == 5000)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  成本模型与模拟盘可用")
print("#" * 70)
