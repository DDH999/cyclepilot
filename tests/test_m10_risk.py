# -*- coding: utf-8 -*-
"""
M10–M13 验证：数据清洗 / 滚动样本外 / 成本与容量 / 动态风控
====================================================================
跑法:  python tests/test_m10_risk.py     （不联网）

分组
  [1] 数据质量体检 datacheck
  [2] 滚动样本外 walkforward
  [3] 成本与容量 execution
  [4] 动态风控 riskcontrol
  [5] 端到端串联
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import blackswan as BW
import datacheck as DC
import execution as EX
import riskcontrol as RC
import walkforward as WF

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M10–M13 验证  datacheck={DC.__version__} wf={WF.__version__} "
      f"exec={EX.__version__} risk={RC.__version__}")
print("#" * 70)

IDX = pd.bdate_range("2024-01-02", periods=300)
rng = np.random.default_rng(7)
GOOD_NAV = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 300)), index=IDX)
CODES = [f"{i:06d}" for i in range(20)]
W_EQ = pd.DataFrame(1 / 20, index=IDX, columns=CODES)
STYLES = ["银行", "科技", "消费", "医药"]
GROUPS = {c: STYLES[i // 5] for i, c in enumerate(CODES)}

# ══════════════════════════════════════════════════════════════
print("\n[1] 数据质量体检")

g = DC.check_series(GOOD_NAV, name="干净净值")
check("干净序列可用", g["ok"] is True)
check("干净序列高分", g["score"] >= 85, g["score"])
check("识别为净值", g["metrics"]["kind"] == "净值/价格", g["metrics"]["kind"])

dirty = GOOD_NAV.copy()
dirty.iloc[100:130] = dirty.iloc[100]          # 冻结 30 天
dirty.iloc[200] = dirty.iloc[199] * 1.8        # 跳变
dirty = dirty.drop(dirty.index[list(range(10, 60))])   # 缺 50 天 ≈ 17%
b = DC.check_series(dirty, name="脏数据")
kinds = {x["kind"] for x in b["issues"]}
check("脏数据不可用", b["ok"] is False, b["level"])
check("抓到冻结行情", "frozen" in kinds, kinds)
check("抓到异常跳变", "jump" in kinds, kinds)
check("抓到缺失交易日", "missing_days" in kinds, kinds)
check("脏数据分数更低", b["score"] < g["score"], (b["score"], g["score"]))

check("空数据不可用", DC.check_series(pd.Series(dtype="float64"))["ok"] is False)
check("单点不可用", DC.check_series(pd.Series([1.0]))["ok"] is False)

dup = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 100)),
                index=pd.bdate_range("2024-01-02", periods=100))
dup2 = pd.concat([dup, dup.iloc[[5]]])
check("抓到重复日期", any(x["kind"] == "duplicate" for x in
                       DC.check_series(dup2)["issues"]))

neg = GOOD_NAV.copy()
neg.iloc[150] = -1.0
check("抓到非正数值", any(x["kind"] == "non_positive" for x in
                       DC.check_series(neg)["issues"]))

# 收益序列（不是净值）也要能识别
rets = pd.Series(rng.normal(0.0004, 0.01, 300), index=IDX)
rr = DC.check_series(rets)
check("识别为收益", rr["metrics"]["kind"] == "收益", rr["metrics"]["kind"])
check("收益序列不含 frozen 检查", "max_frozen_run" not in rr["metrics"])

short = DC.check_series(GOOD_NAV.iloc[:20])
check("样本过短报错点", any(x["kind"] == "too_short" for x in short["issues"]))

check("CV 判别：净值", DC.is_level_series(GOOD_NAV) is True)
check("CV 判别：收益", DC.is_level_series(rets) is False)
check("报告可渲染", "数据质量体检" in DC.format_report(b))

# ══════════════════════════════════════════════════════════════
print("\n[2] 滚动样本外")

wf = WF.purged_walk_forward(GOOD_NAV, WF.WFCfg(n_folds=5, purge=5, embargo=5))
check("可评估", wf["ok"] is True)
check("切成 5 段", wf["n_folds"] == 5, wf["n_folds"])
check("每段含夏普", all("sharpe" in f for f in wf["folds"]))
check("命中率在 [0,1]", 0 <= wf["hit_rate"] <= 1)
check("记录 purge/embargo", wf["purge"] == 5 and wf["embargo"] == 5)
check("段天数之和小于总天数（有隔离带）",
      sum(f["days"] for f in wf["folds"]) < wf["n"], 
      (sum(f["days"] for f in wf["folds"]), wf["n"]))
check("分段日期不重叠",
      all(a["end"] < b["start"] for a, b in zip(wf["folds"], wf["folds"][1:])))
check("有结论与稳定标记", isinstance(wf["verdict"], str) and isinstance(wf["stable"], bool))

# 前后两半截然不同 → 应判不稳定
mixed = np.concatenate([rng.normal(0.0015, 0.010, 500), rng.normal(-0.0015, 0.010, 500)])
s_mixed = pd.Series(np.cumprod(1 + mixed), index=pd.bdate_range("2020-01-02", periods=1000))
wf2 = WF.purged_walk_forward(s_mixed)
check("前后矛盾被判不稳定", wf2["stable"] is False, wf2["verdict"])
check("不稳定的命中率不满", wf2["hit_rate"] < 1.0, wf2["hit_rate"])

check("样本过短返回 ok=False",
      WF.purged_walk_forward(GOOD_NAV.iloc[:50])["ok"] is False)
check("夏普零方差返回 0", WF.sharpe_of([0.01] * 20) == 0.0)
check("报告可渲染", "滚动样本外" in WF.format_report(wf))

# ══════════════════════════════════════════════════════════════
print("\n[3] 成本与容量")

check("参与率计算", abs(EX.participation_rate(1e6, 1e8) - 0.01) < 1e-9)
check("ADV 为 0 返回 inf", EX.participation_rate(1e6, 0) == float("inf"))
check("冲击随规模递增",
      EX.impact_bps(1e6, 1e8, 0.012) < EX.impact_bps(1e7, 1e8, 0.012))
check("冲击随波动递增",
      EX.impact_bps(1e7, 1e8, 0.010) < EX.impact_bps(1e7, 1e8, 0.020))
check("冲击有上限", EX.impact_bps(1e12, 1e6, 0.5) <= 500.0)
check("零委托无冲击", EX.impact_bps(0, 1e8, 0.012) == 0.0)

small = EX.cost_breakdown(100000, 12.0, 5e8, 0.012)
check("小资金冲击低于摩擦", small["impact_pct"] < small["friction_pct"],
      (small["impact_pct"], small["friction_pct"]))
big = EX.cost_breakdown(5e8, 12.0, 5e8, 0.012)
check("大资金冲击高于摩擦", big["impact_pct"] > big["friction_pct"],
      (big["impact_pct"], big["friction_pct"]))
check("参与率按单笔算（非年成交额）",
      abs(big["participation"] - 1.0) < 1e-6, big["participation"])
check("单笔委托额 = 资金", abs(big["order_value"] - 5e8) < 1, big["order_value"])
check("成本占比随规模上升", big["total_pct"] > small["total_pct"])

zero_t = EX.cost_breakdown(100000, 0.0, 5e8, 0.012)
check("零换手零成本", zero_t["total"] == 0.0)

curve = EX.capacity_curve([1e5, 1e6, 1e7, 1e8], 12.0, 5e8, 0.012)
check("容量曲线 4 行", len(curve) == 4)
check("成本随规模单调上升",
      all(a <= b for a, b in zip(curve["成本占比"], curve["成本占比"].tolist()[1:])))

cap = EX.max_capacity(12.0, 5e8, 0.012, cost_budget_pct=0.03)
check("容量上限为正", cap > 0, f"{cap:,.0f}")
check("容量上限处成本≈预算",
      abs(EX.cost_breakdown(cap, 12.0, 5e8, 0.012)["total_pct"] - 0.03) < 0.002,
      EX.cost_breakdown(cap, 12.0, 5e8, 0.012)["total_pct"])
check("低换手容量更大",
      EX.max_capacity(1.0, 5e8, 0.012) > EX.max_capacity(12.0, 5e8, 0.012))
check("报告可渲染", "交易成本与容量" in EX.format_report(big, cap))

# ══════════════════════════════════════════════════════════════
print("\n[4] 动态风控")

# 拥挤度
c_low = RC.crowding_score(W_EQ, turnover_annual=1.0, bench_r2=0.3,
                          participation=0.001)
check("分散低换手 → 拥挤度低", c_low["score"] < 40, c_low["score"])
c_high = RC.crowding_score(None, turnover_annual=15.0, bench_r2=0.95,
                           participation=0.20)
check("高换手+高相关 → 拥挤度高", c_high["score"] >= 70, c_high["score"])
check("拥挤度有等级", c_high["level"] == "高", c_high["level"])
check("拥挤度含分解", len(c_high["breakdown"]) == 4)
check("全缺数据返回 ok=False", RC.crowding_score()["ok"] is False)

# 回撤
d_ok = RC.drawdown_state(pd.Series(np.linspace(100, 120, 100)))
check("无回撤 → NORMAL", d_ok["status"] == "NORMAL", d_ok["current_drawdown"])
d_warn = RC.drawdown_state(pd.Series(np.concatenate([np.linspace(100, 130, 80),
                                                     np.linspace(130, 112, 20)])))
check("中度回撤 → REDUCE", d_warn["status"] == "REDUCE",
      d_warn["current_drawdown"])
d_halt = RC.drawdown_state(pd.Series(np.concatenate([np.linspace(100, 130, 80),
                                                     np.linspace(130, 95, 20)])))
check("深度回撤 → HALT", d_halt["status"] == "HALT", d_halt["current_drawdown"])

# 风格暴露
se_ok = RC.style_exposure(W_EQ, GROUPS)
check("4 风格各 25% 不超限", se_ok["status"] == "NORMAL", se_ok["top"])
W_skew = W_EQ.copy()
W_skew.iloc[:, :14] = 0.0
W_skew.iloc[:, 14:] = 1 / 6
se_bad = RC.style_exposure(W_skew, GROUPS)
check("集中单一风格 → REDUCE", se_bad["status"] == "REDUCE")
check("列出超限分组", len(se_bad["breaches"]) >= 1, se_bad["breaches"])
check("未知分组被归类", "未知" in dict(
    (x["group"], x["share"]) for x in RC.style_exposure(W_EQ, {})["top"]))

# Kill switch
r_norm = RC.kill_switch(pd.Series(np.linspace(100, 108, 300)), W_EQ, GROUPS,
                        turnover_annual=4.0, bench_r2=0.5, participation=0.01,
                        data_ok=True)
check("一切正常 → NORMAL", r_norm["action"] == "NORMAL", r_norm["action"])
check("正常时仓位系数 1", r_norm["scale"] == 1.0)

r_reduce = RC.kill_switch(pd.Series(np.linspace(100, 108, 300)), W_EQ, GROUPS,
                          turnover_annual=9.0, bench_r2=0.5, participation=0.01,
                          data_ok=True)
check("换手超限 → REDUCE", r_reduce["action"] == "REDUCE", r_reduce["action"])
check("减仓系数 0.5", r_reduce["scale"] == 0.5)
check("原因含换手", any("换手" in x["why"] for x in r_reduce["reasons"]))

crash = pd.Series(np.concatenate([np.linspace(100, 130, 200),
                                  np.linspace(130, 95, 100)]),
                  index=pd.bdate_range("2024-01-02", periods=300))
r_halt = RC.kill_switch(crash, W_EQ, GROUPS, turnover_annual=4.0, data_ok=True)
check("回撤超限 → HALT", r_halt["action"] == "HALT", r_halt["action"])
check("HALT 时仓位系数 0", r_halt["scale"] == 0.0)

r_cheat = RC.kill_switch(pd.Series(np.linspace(100, 108, 300)), W_EQ, GROUPS,
                         code_cheat=True, data_ok=True)
check("代码作弊 → HALT", r_cheat["action"] == "HALT")
check("作弊原因列出", any("作弊" in x["why"] for x in r_cheat["reasons"]))

r_bad = RC.kill_switch(pd.Series(np.linspace(100, 108, 300)), W_EQ, GROUPS,
                       data_ok=False)
check("数据不合格 → HALT", r_bad["action"] == "HALT")

r_cool = RC.kill_switch(crash, W_EQ, GROUPS, data_ok=True, days_since_halt=3)
check("冷却期未满有提示",
      any(x["level"] == "info" for x in r_cool["reasons"]), r_cool["reasons"])
check("冷却期剩余为正", r_cool["cooldown_remaining"] > 0, r_cool["cooldown_remaining"])

check("报告可渲染", "动态风控决策" in RC.format_report(r_halt))

# ══════════════════════════════════════════════════════════════
print("\n[5] 端到端串联")

dq = DC.check_series(GOOD_NAV)
wfr = WF.purged_walk_forward(GOOD_NAV)
tail = BW.tail_statistics(GOOD_NAV)
frag = BW.fragility_score(tail)
cost = EX.cost_breakdown(100000, 12.0, 5e8, 0.011)
rising = pd.Series(np.linspace(100000, 118000, 300), index=IDX)
ks = RC.kill_switch(rising, W_EQ, GROUPS, turnover_annual=12.0,
                    bench_r2=0.5, participation=cost["participation"],
                    data_ok=dq["ok"])

check("数据合格", dq["ok"] is True)
check("样本外可评估", wfr["ok"] is True)
check("脆弱度可算", frag["ok"] is True)
check("换手 12 触发减仓", ks["action"] == "REDUCE", ks["action"])
check("端到端各环节都有结论",
      all(isinstance(x, str) and len(x) >= 1 for x in
          (dq["level"], wfr["verdict"], frag["level"], ks["action"])))

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
print("# 全部通过  数据/样本外/成本/风控四层已就绪")
print("#" * 70)
