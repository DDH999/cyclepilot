# -*- coding: utf-8 -*-
"""
M19 验证：扰动测试与压力测试
====================================================================
跑法:  python tests/test_m19_robustness.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import robustness as RB

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M19 验证  robustness v{RB.__version__}")
print("#" + "=" * 69)

IDX = pd.bdate_range("2020-01-02", periods=1000)
rng = np.random.default_rng(7)
BENCH_R = rng.normal(0.0003, 0.010, 1000)
BENCH = pd.Series(np.cumprod(1 + BENCH_R), index=IDX)
STABLE = pd.Series(0.0006 + rng.normal(0, 0.006, 1000), index=IDX)

lucky = np.full(1000, -0.0004)
lucky[[200, 500, 800]] = 0.12
LUCKY = pd.Series(lucky, index=IDX)

print("\n[1] 基础工具")

_nav = pd.Series(np.cumprod(1 + STABLE.to_numpy()), index=IDX)
check("净值→收益（真净值）",
      abs(float(RB.to_returns(_nav).mean()) - 0.0006) < 1e-4,
      float(RB.to_returns(_nav).mean()))
check("已是收益则原样返回（长度不变即未转换）",
      len(RB.to_returns(STABLE)) == len(STABLE),
      (len(RB.to_returns(STABLE)), len(STABLE)))
r_ret = RB.to_returns(STABLE)
check("已是收益则不变", abs(len(r_ret) - 1000) <= 1)

m = RB.metrics_of(STABLE)
check("指标可算", m["ok"] is True)
check("年化为正", m["cagr"] > 0, round(m["cagr"], 4))
check("夏普为正", m["sharpe"] > 0, round(m["sharpe"], 2))
check("回撤为负", m["max_drawdown"] < 0)
check("样本过短返回 ok=False", RB.metrics_of([0.01, 0.02])["ok"] is False)

print("\n[2] 扰动：可复现")

p1 = RB.perturbation_suite(STABLE, RB.PerturbCfg(seed=7))
p2 = RB.perturbation_suite(STABLE, RB.PerturbCfg(seed=7))
check("同种子结果一致", abs(p1["score"] - p2["score"]) < 1e-9,
      (p1["score"], p2["score"]))

p3 = RB.perturbation_suite(STABLE, RB.PerturbCfg(seed=99))
check("不同种子结果变化不大(稳健策略)",
      abs(p3["score"] - p1["score"]) < 15, (p3["score"], p1["score"]))

check("五类扰动齐全", len(p1["families"]) == 5, len(p1["families"]))
for fam in ("加噪声", "块重抽样", "起点平移", "随机删日", "追加成本"):
    check(f"含扰动族 {fam}", fam in p1["families"])

check("含基准指标", "base" in p1)
check("含综合分", 0 <= p1["score"] <= 100, p1["score"])
check("含等级", p1["grade"] in ("A", "B", "C", "D"), p1["grade"])
check("指出最脆弱扰动", p1.get("worst_family") in p1["families"],
      p1.get("worst_family"))

print("\n[3] 扰动：能区分稳健与脆弱")

pl = RB.perturbation_suite(LUCKY, RB.PerturbCfg(seed=7))
check("脆弱策略可评估", pl["ok"] is True)
check("稳健策略分数更高", p1["score"] > pl["score"],
      (p1["score"], pl["score"]))
check("脆弱策略等级更低",
      ["A", "B", "C", "D"].index(p1["grade"]) < ["A", "B", "C", "D"].index(pl["grade"]),
      (p1["grade"], pl["grade"]))
check("稳健策略等级为 A/B", p1["grade"] in ("A", "B"), p1["grade"])

check("样本过短返回 ok=False",
      RB.perturbation_suite(STABLE.iloc[:50])["ok"] is False)
check("扰动报告可渲染", "扰动测试" in RB.format_perturbation(p1))
check("报告含最脆弱项", "最脆弱" in RB.format_perturbation(pl))

print("\n[4] 压力：情景与存活")

s1 = RB.stress_suite(STABLE, bench=BENCH)
check("可评估", s1["ok"] is True)
check("情景数 >= 6", s1["n_total"] >= 6, s1["n_total"])
names = [x["scenario"] for x in s1["scenarios"]]
for need in ("尾部注入", "单边下跌 40%", "流动性枯竭", "相关性断裂"):
    check(f"含情景 {need}", need in names, names)
check("波动情景齐全",
      sum(1 for n in names if "波动" in n) == 3, names)
check("存活率在 [0,1]", 0 <= s1["survival_rate"] <= 1)
check("含最差情景", s1.get("worst") is not None)
check("每个情景含判决",
      all("verdict" in x for x in s1["scenarios"] if x.get("ok")))
check("单边下跌应被击穿",
      next(x for x in s1["scenarios"]
           if x["scenario"] == "单边下跌 40%")["survived"] is False)
check("流动性枯竭成本拖累可见",
      next(x for x in s1["scenarios"]
           if x["scenario"] == "流动性枯竭")["cagr"]
      <= s1["base"]["cagr"] + 1e-9)

s2 = RB.stress_suite(STABLE, bench=None)
check("无基准时不崩", s2["ok"] is True)
check("无基准时相关性情景标注不可构造",
      any((not x.get("ok")) and "基准" in str(x.get("reason"))
          for x in s2["scenarios"]),
      [x.get("reason") for x in s2["scenarios"] if not x.get("ok")])

check("样本过短返回 ok=False", RB.stress_suite(STABLE.iloc[:50])["ok"] is False)
check("压力报告可渲染", "压力测试" in RB.format_stress(s1))
check("报告含存活判据", "存活判据" in RB.format_stress(s1))

print("\n[5] 接口与 CLI")


def _fake_nav():
    return STABLE


import api_v2 as API
from fastapi.testclient import TestClient

cli = TestClient(API.app)

r = cli.get("/api/perturbation")
check("扰动接口 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    b = r.json()
    check("扰动返回 ok", b.get("ok") is True)
    check("扰动含文本", "扰动测试" in b.get("text", ""))
    check("扰动含分数", b["result"].get("score") is not None)

r2 = cli.get("/api/stress")
check("压力接口 200", r2.status_code == 200, r2.status_code)
if r2.status_code == 200:
    b2 = r2.json()
    check("压力返回 ok", b2.get("ok") is True)
    check("压力含文本", "压力测试" in b2.get("text", ""))
    check("压力标注基准可用性", "has_bench" in b2)

import quantpilot as QP
check("CLI 注册 perturb", "perturb" in QP.COMMANDS)
check("CLI 注册 stress", "stress" in QP.COMMANDS)
parser = QP.build_parser()
a1 = parser.parse_args(["perturb", "--seed", "3"])
check("perturb 参数解析", a1.seed == 3)
a2 = parser.parse_args(["stress", "--seed", "5"])
check("stress 参数解析", a2.seed == 5)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  扰动与压力测试可用")
print("#" * 70)
