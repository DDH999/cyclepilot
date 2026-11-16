# -*- coding: utf-8 -*-
"""
M6 验证：收益归因与成本核算
====================================================================
跑法:  python tests/test_m6_attribution.py      （不联网，缓存用临时文件）

分组
  [1] 回本计算
  [2] 成本核算
  [3] 换手敏感性
  [4] 收益归因
  [5] 基准加载与缓存
  [6] 报告渲染
====================================================================
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

import attribution as AT

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m6"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def approx(a, b, tol=1e-6):
    return a is not None and np.isfinite(a) and abs(a - b) <= tol


print("\n" + "#" * 70)
print(f"# M6 验证  attribution v{AT.__version__}")
print("#" * 70)

# ══════════════════════════════════════════════════════════════
print("\n[1] 回本计算")

check("-30% 需涨 42.86%", approx(AT.recovery_needed(-0.30), 0.4286, 1e-3),
      AT.recovery_needed(-0.30))
check("-50% 需涨 100%", approx(AT.recovery_needed(-0.50), 1.0, 1e-9))
check("-10% 需涨 11.11%", approx(AT.recovery_needed(-0.10), 0.1111, 1e-3))
check("正数入参等价", approx(AT.recovery_needed(0.30), AT.recovery_needed(-0.30), 1e-12))
check("0 回撤返回 0", AT.recovery_needed(0) == 0.0)
check("亏光返回 inf", AT.recovery_needed(-1.0) == float("inf"))

# ══════════════════════════════════════════════════════════════
print("\n[2] 成本核算")

c0 = AT.cost_analysis(100000, 0.0)
check("零换手零成本", c0["total"] == 0.0)
check("零换手无成交额", c0["amount_both"] == 0.0)

c12 = AT.cost_analysis(100000, 12.0)
check("成交额 = 2×T×本金", approx(c12["amount_both"], 2 * 12 * 100000, 1e-6))
check("佣金按双边", approx(c12["commission"], 2400000 * 2.5 / 1e4, 1e-6))
check("印花税只按卖出", approx(c12["stamp_duty"], 1200000 * 5 / 1e4, 1e-6))
check("过户费按双边", approx(c12["transfer_fee"], 2400000 * 0.1 / 1e4, 1e-6))
check("滑点按双边", approx(c12["slippage"], 2400000 * 5 / 1e4, 1e-6))
check("合计等于各项之和",
      approx(c12["total"], c12["commission"] + c12["stamp_duty"]
             + c12["transfer_fee"] + c12["slippage"], 1e-6))
check("占本金约 2.42%", approx(c12["total_pct"], 0.02424, 1e-4), c12["total_pct"])

c_big = AT.cost_analysis(200000, 12.0)
check("成本随本金线性放大", approx(c_big["total"], c12["total"] * 2, 1e-6))

c_free = AT.cost_analysis(100000, 12.0, commission_bps=0, stamp_bps=0,
                          transfer_bps=0, slippage_bps=0)
check("零费率时成本为 0", c_free["total"] == 0.0)

check("负换手被夹到 0", AT.cost_analysis(100000, -5)["total"] == 0.0)

# ══════════════════════════════════════════════════════════════
print("\n[3] 换手敏感性")

df = AT.turnover_sensitivity(100000)
check("默认 5 档", len(df) == 5, len(df))
check("成本单调递增", all(a < b for a, b in
                     zip(df["总成本"].tolist(), df["总成本"].tolist()[1:])))
check("首档为买入持有", df.iloc[0]["总成本"] == 0.0)
check("含占本金列", "占本金%" in df.columns)
check("周频成本接近 10%", 9.0 < df.iloc[-1]["占本金%"] < 11.0,
      df.iloc[-1]["占本金%"])

df2 = AT.turnover_sensitivity(100000, turnovers=[0, 1, 2])
check("可自定义档位", len(df2) == 3)

# ══════════════════════════════════════════════════════════════
print("\n[4] 收益归因")

rng = np.random.default_rng(11)
days = pd.bdate_range("2023-01-02", periods=500)
rb = rng.normal(0.0004, 0.010, len(days))
rp = 0.6 * rb + rng.normal(0.0002, 0.006, len(days))

nav = pd.Series(100000 * np.cumprod(1 + rp), index=days)
bench = pd.Series(3000 * np.cumprod(1 + rb), index=days)
a = AT.performance_attribution(nav, bench)

check("beta 接近 0.6", approx(a["beta"], 0.6, 0.15), round(a["beta"], 3))
check("R² 较高", a["r2"] > 0.4, round(a["r2"], 3))
check("天数正确", a["n_days"] == 499, a["n_days"])

# 拆分恒等式
check("市场+选股 ≈ 总收益",
      approx(a["market_contribution"] + a["selection_contribution"],
             a["total_return"], 1e-9),
      (a["market_contribution"], a["selection_contribution"], a["total_return"]))
check("市场贡献 = beta × 基准收益",
      approx(a["market_contribution"], a["beta"] * a["bench_return"], 1e-9))
check("回本涨幅与回撤匹配",
      approx(a["recovery_needed"], AT.recovery_needed(a["max_drawdown"]), 1e-9))
check("波动率为正", a["volatility"] > 0)
check("胜率在 [0,1]", 0 <= a["win_rate"] <= 1)
check("上涨日胜率在 [0,1]", 0 <= a["win_rate_on_up_days"] <= 1)
check("有结论文本", isinstance(a["verdict"], str) and len(a["verdict"]) > 5)

# 完美跟随基准：beta≈1, 选股贡献≈0
nav2 = pd.Series(100000 * np.cumprod(1 + rb), index=days)
a2 = AT.performance_attribution(nav2, bench)
check("复制基准 beta≈1", approx(a2["beta"], 1.0, 0.05), round(a2["beta"], 3))
check("复制基准 R²≈1", a2["r2"] > 0.95, round(a2["r2"], 3))
check("复制基准选股贡献≈0", abs(a2["selection_contribution"]) < 0.02,
      round(a2["selection_contribution"], 4))
check("复制基准结论提及市场", "市场" in a2["verdict"], a2["verdict"][:30])

# 完全不相关
nav3 = pd.Series(100000 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(days))), index=days)
a3 = AT.performance_attribution(nav3, bench)
check("弱相关时 R² 低", a3["r2"] < 0.3, round(a3["r2"], 3))

try:
    AT.performance_attribution(nav.iloc[:10], bench.iloc[:10])
    check("样本过短报错", False)
except ValueError:
    check("样本过短报错", True)

check("最大回撤已知值",
      approx(AT.max_drawdown(np.array([1.0, 1.2, 0.9, 1.1])), 0.9 / 1.2 - 1, 1e-12))
check("回撤空数组返回 0", AT.max_drawdown(np.array([])) == 0.0)

# ══════════════════════════════════════════════════════════════
print("\n[5] 基准加载与缓存")

cache = TMP / "bench.csv"
bdays = pd.bdate_range("2024-01-01", periods=80)
pd.DataFrame({"date": bdays, "close": np.linspace(3000, 3200, 80)}).to_csv(
    cache, index=False)
s = AT.load_bench(cache=cache)
check("缓存可读取", len(s) == 80, len(s))
check("索引为日期", isinstance(s.index, pd.DatetimeIndex))
check("数值正确", approx(s.iloc[-1], 3200, 1e-6))

s_sub = AT.load_bench(start="2024-02-01", cache=cache)
check("可按起始日截取", len(s_sub) < 80 and s_sub.index.min() >= pd.Timestamp("2024-02-01"))

# 模拟 akshare 不可用：把模块置 None 使其 import 失败
bad = TMP / "none.csv"
_ak = sys.modules.get("akshare")
sys.modules["akshare"] = None
try:
    AT.load_bench(cache=bad)
    check("akshare 不可用且无缓存时报错", False)
except RuntimeError:
    check("akshare 不可用且无缓存时报错", True)
finally:
    if _ak is not None:
        sys.modules["akshare"] = _ak
    else:
        sys.modules.pop("akshare", None)

# ══════════════════════════════════════════════════════════════
print("\n[6] 报告渲染")

rep = AT.format_report(a, AT.cost_analysis(100000, 12.0))
for token in ["收益归因", "市场贡献", "选股贡献", "成本核算", "回本需涨", "印花税"]:
    check(f"报告含「{token}」", token in rep, token)

rep2 = AT.format_cost_report(AT.turnover_sensitivity(100000), 100000)
check("成本报告含表头", "双边成交额" in rep2 and "占本金" in rep2)
check("成本报告含 5 行", rep2.count("%") >= 5)

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
print("# 全部通过  归因与成本核算可用")
print("#" * 70)
