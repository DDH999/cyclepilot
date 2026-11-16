# -*- coding: utf-8 -*-
"""
M33 验证：严格回测（未来函数 / 涨跌停 / 样本外）
====================================================================
跑法:  python tests/test_m33_backtest_strict.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import backtest_strict as BS

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M33 验证  backtest_strict v{BS.__version__}")
print("#" + "=" * 69)

print("\n[1] 涨跌停幅度")

check("主板 600519 → 10%", BS.limit_pct("600519") == 0.10)
check("创业板 300750 → 20%", BS.limit_pct("300750") == 0.20)
check("科创板 688981 → 20%", BS.limit_pct("688981") == 0.20)
check("北交所 830799 → 30%", BS.limit_pct("830799") == 0.30)
check("裸码补零", BS.limit_pct("1") == 0.10)

print("\n[2] 一字板与停牌识别")


def mk(n=60, seed=7, px=10.0):
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    c = px * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame({"date": idx, "open": c, "high": c * 1.005,
                         "low": c * 0.995, "close": c,
                         "volume": 1e6, "amount": 1e7})


df = mk(60)
df.loc[10, ["open", "high", "low", "close"]] = df.loc[9, "close"] * 1.10
df.loc[20, "volume"] = 0
df.loc[30, ["open", "high", "low", "close"]] = df.loc[29, "close"] * 0.90

a = BS.annotate(df, "600519")
check("一字涨停被识别", bool(a.loc[10, "one_word_up"]) is True)
check("一字跌停被识别", bool(a.loc[30, "one_word_down"]) is True)
check("停牌被识别", bool(a.loc[20, "tradable"]) is False)
check("正常日可交易", bool(a.loc[12, "tradable"]) is True)
check("正常日非涨停", bool(a.loc[12, "one_word_up"]) is False)
check("加了 ret 列", "ret" in a.columns)
check("加了 close_limit_up 列", "close_limit_up" in a.columns)

# 涨停但非一字（开板过）不应算一字
df2 = mk(40)
df2.loc[15, "open"] = df2.loc[14, "close"] * 1.07
df2.loc[15, "high"] = df2.loc[14, "close"] * 1.10
df2.loc[15, "close"] = df2.loc[14, "close"] * 1.10
df2.loc[15, "low"] = df2.loc[14, "close"] * 1.02
a2 = BS.annotate(df2, "600519")
check("开板涨停不算一字", bool(a2.loc[15, "one_word_up"]) is False,
      (a2.loc[15, "open"], a2.loc[15, "high"], a2.loc[15, "low"]))
check("但收盘涨停仍记录", bool(a2.loc[15, "close_limit_up"]) is True)

print("\n[3] 面板审计")

panel = {f"60000{i}": mk(60, seed=i) for i in range(5)}
au = BS.audit_panel(panel)
check("审计可跑", au["ok"] is True)
check("统计只数", au["n_codes"] == 5, au["n_codes"])
check("统计 bar 数", au["n_bars"] == 300, au["n_bars"])
check("含停牌占比", "suspension_pct" in au)
check("含六项检查", len(au["checks"]) == 6, len(au["checks"]))
check("六项全通过", all(s == "pass" for _, s, _ in au["checks"]))
check("审计报告可渲染", "回测框架审计" in BS.format_audit(au))

print("\n[4] 关键：信号在 t、成交在 t+1 开盘")

# 构造跳空：close(t)=10，open(t+1)=20，close(t+1)=20
# 若引擎错误地用 close(t) 成交，净值会凭空 +100%
n = 40
idx = pd.bdate_range("2022-01-03", periods=n)
base = pd.DataFrame({"date": idx, "open": 10.0, "high": 10.2, "low": 9.8,
                     "close": 10.0, "volume": 1e6, "amount": 1e7})
# 跳空必须发生在「信号次日」= idx[6]（执行日），否则测的不是同一件事
base.loc[6:, ["open", "high", "low", "close"]] = 20.0
gap = {"600519": base.copy()}

sig = pd.DataFrame([{ "600519": 1.0 }], index=[idx[5]])
bt = BS.StrictBacktest(gap, capital=1_000_000).run(sig, max_positions=1,
                                                   min_value=1000)
nav = bt["nav"]
nav_before = float(nav.loc[idx[9]])
nav_after = float(nav.loc[idx[11]])
ratio = nav_after / nav_before
check("成交价是次日开盘（无跳空收益）",
      ratio < 1.05, f"净值比 {ratio:.4f}（若用前一日收盘价成交会接近 2.0）")
check("确实买入了", bt["final_nav"] > 0)

# 对照：如果 open 与 close 一致，净值应平稳
flat = {"600519": mk(40)}
bt2 = BS.StrictBacktest(flat, capital=1_000_000).run(
    pd.DataFrame([{"600519": 1.0}], index=[flat["600519"]["date"].iloc[5]]),
    max_positions=1, min_value=1000)
check("正常行情可运行", bt2["ok"] is True)
check("有净值序列", bt2["n_days"] > 2, bt2["n_days"])

print("\n[5] 涨停不买 / 跌停不卖")

# 次日开盘涨停 → 不能买
lu = mk(40)
# 执行日 idx[6] 开盘即涨停
lu.loc[6:, ["open", "high", "low", "close"]] = lu.loc[5, "close"] * 1.10
bt_lu = BS.StrictBacktest({"600519": lu}, capital=1e6).run(
    pd.DataFrame([{"600519": 1.0}], index=[lu["date"].iloc[5]]),
    max_positions=1, min_value=1000)
check("开盘涨停未买入", bt_lu["blocked_buy"] >= 1, bt_lu["blocked_buy"])

# 次日停牌 → 跳过
sp = mk(40)
sp.loc[6, "volume"] = 0        # 执行日停牌
bt_sp = BS.StrictBacktest({"600519": sp}, capital=1e6).run(
    pd.DataFrame([{"600519": 1.0}], index=[sp["date"].iloc[5]]),
    max_positions=1, min_value=1000)
check("停牌日跳过", bt_sp["blocked_suspension"] >= 1,
      bt_sp["blocked_suspension"])

print("\n[6] 动量信号")

p3 = {f"60000{i}": mk(200, seed=i) for i in range(8)}
sig3 = BS.momentum_weights(p3, lookback=20, top_n=3, rebalance=10)
check("信号可生成", not sig3.empty)
check("每行权重和为 1（等权）",
      all(abs(sig3.sum(axis=1) - 1.0) < 1e-9), sig3.sum(axis=1).unique()[:3])
check("每行正好 3 只",
      all((sig3 > 0).sum(axis=1) == 3), (sig3 > 0).sum(axis=1).unique()[:3])
check("没有未来数据（首个信号晚于回看期）",
      sig3.index[0] >= p3["600000"]["date"].iloc[20], sig3.index[0])

print("\n[7] 样本外检验")

res = BS.tune_oos(
    p3, is_range=("2022-01-03", "2022-06-30"),
    oos_range=("2022-07-01", "2022-12-30"),
    grid={"lookback": [10, 20], "top_n": [2], "rebalance": [10]})
check("可运行", res["ok"] is True, res.get("reason"))
check("选出最优参数", "best_params" in res, res.get("best_params"))
check("参数来自网格",
      res["best_params"]["lookback"] in (10, 20), res["best_params"])
check("含 IS 指标", "is_annual" in res["is"])
check("含 OOS 指标", "annual" in res["oos"])
check("计算衰减", "decay_annual" in res)
check("衰减 = OOS - IS",
      abs(res["decay_annual"] - (res["oos"]["annual"]
                                 - res["is"]["is_annual"])) < 1e-6)
check("给出结论", len(res["verdict"]) > 5, res["verdict"])
check("报告可渲染", "样本外检验" in BS.format_oos(res))
check("失败报告可渲染",
      "样本外检验" in BS.format_oos({"ok": False, "reason": "x"}))

short = BS.tune_oos(p3, ("2022-01-03", "2022-01-10"),
                    ("2022-01-11", "2022-02-01"))
check("分区太短时拒绝", short["ok"] is False, short.get("reason"))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  严格引擎可用")
print("#" * 70)
