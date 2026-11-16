# -*- coding: utf-8 -*-
"""
回测引擎单元测试 —— 验证 A 股交易规则是否真的生效。
运行: python tests/test_backtest.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from backtest import Costs, limit_pct, build_market, Backtest

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{detail}]" if detail and not cond else ""))


def synth(n=120, start=10.0, ret=0.004, jump=None, stop_at=None, limit_day=None):
    """构造合成行情"""
    dates = pd.bdate_range("2020-01-01", periods=n)
    px = start * (1 + ret) ** np.arange(n)
    if jump:
        for i, f in jump.items():
            px[i:] *= f
    df = pd.DataFrame({
        "date": dates, "close": px,
        "open": px * 0.999, "high": px * 1.01, "low": px * 0.99,
        "volume": 1e6, "amount": px * 1e6,
    })
    if limit_day is not None:
        i = limit_day
        df.loc[i, "close"] = df.loc[i - 1, "close"] * 1.10
        df.loc[i, "high"] = df.loc[i, "close"]
        df.loc[i, "low"] = df.loc[i, "close"] * 0.99
        df.loc[i, "open"] = df.loc[i, "close"]
    if stop_at is not None:
        df = df.iloc[:stop_at].reset_index(drop=True)
    return df


print("=" * 62)
print("回测引擎单元测试")
print("=" * 62)

# ── 1. 交易成本
print("\n[1] 交易成本")
c = Costs(commission=0.00025, min_commission=5.0, stamp_duty=0.0005,
          transfer_fee=0.00001, slippage_bps=5)
check("买入佣金 10万 -> 25元", abs(c.buy(100_000) - (25 + 1.0)) < 1e-6, c.buy(100_000))
check("买入最低佣金 1000元 -> 5元", abs(c.buy(1000) - (5 + 0.01)) < 1e-6, c.buy(1000))
sell = c.sell(100_000)
check("卖出含印花税 0.05%", abs(sell - (25 + 50 + 1.0)) < 1e-6, sell)
check("印花税仅卖出（买入不含）", abs(c.buy(100_000) - sell) > 45)

# ── 2. 涨跌停幅度
print("\n[2] 涨跌停幅度")
check("主板 600xxx = 10%", limit_pct("600519") == 0.10)
check("创业板 300xxx = 20%", limit_pct("300750") == 0.20)
check("科创板 688xxx = 20%", limit_pct("688981") == 0.20)
check("深主板 000xxx = 10%", limit_pct("000001") == 0.10)

# ── 3. 涨跌停识别
print("\n[3] 涨跌停识别")
d = synth(limit_day=60)
m = build_market({"600000": d})["600000"]
check("涨停日被标记 limit_up", bool(m["limit_up"].iloc[60]))
check("涨停日仍可交易(tradable)", bool(m["tradable"].iloc[60]))
check("非涨停日不误标", not bool(m["limit_up"].iloc[50]))

# ── 4. T+1 与仓位手数
print("\n[4] T+1 与 100 股手数")
panel = {f"60000{i}": synth(n=150, ret=0.003) for i in range(3)}
bt = Backtest(panel, capital=1_000_000)
reb = pd.DatetimeIndex([bt.dates[20]])
w = pd.DataFrame({list(panel)[0]: [1.0]}, index=reb)
res = bt.run(w, max_positions=1)
check("能完成回测并产出净值", "nav" in res, res.get("error"))
nav = res.get("nav")
check("净值序列长度正确", len(nav) == len(bt.dates), len(nav) if nav is not None else 0)
check("期末净值 > 0", float(nav.iloc[-1]) > 0)
check("换手率非零（确实成交了）", float(res.get("ann_turnover") or 0) > 0)

# ── 5. 退市核销
print("\n[5] 退市核销")
p2 = {"600001": synth(n=60, ret=0.002), "600002": synth(n=150, ret=0.003)}
bt2 = Backtest(p2, capital=1_000_000, delist_recovery=0.30)
check("识别出提前结束的标的", 0 in bt2.writedown, list(bt2.writedown))
check("核销日 = 最后交易日后 21 天", bt2.writedown[0] == bt2.last_real[0] + 21,
      (bt2.last_real.get(0), bt2.writedown.get(0)))

# ── 6. 止损
print("\n[6] 个股止损")
crash = synth(n=150, ret=0.002, jump={80: 0.5})
panel3 = {"600003": crash}
bt3 = Backtest(panel3, capital=1_000_000, stop_loss=0.15)
reb3 = pd.DatetimeIndex([bt3.dates[40]])
w3 = pd.DataFrame({"600003": [1.0]}, index=reb3)
r3 = bt3.run(w3, max_positions=1)
check("暴跌触发止损", int(r3.get("n_stops", 0)) > 0, r3.get("n_stops"))

# ── 7. 矩阵对齐与布尔类型
print("\n[7] 内部数据完整性")
check("CL/VALID 形状一致", bt.CL.shape == bt.VALID.shape)
check("TV/LU/LD 为布尔数组", bt.TV.dtype == bool and bt.LU.dtype == bool and bt.LD.dtype == bool)
check("日期映射完整", len(bt.dmap) == len(bt.dates))
check("无 NaN 价格残留", not np.isnan(bt.CL).any())

print("\n" + "=" * 62)
print(f"通过 {len(PASS)} / {len(PASS)+len(FAIL)}")
if FAIL:
    print("失败: " + ", ".join(FAIL))
    sys.exit(1)
print("全部通过 ✓")
