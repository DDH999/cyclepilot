# -*- coding: utf-8 -*-
"""
M26 验证：实盘决策引擎
====================================================================
跑法:  python tests/test_m26_advisor.py   （不联网，全部注入假数据）
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

import advisor as AD

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m26"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M26 验证  advisor v{AD.__version__}")
print("#" + "=" * 69)

print("\n[1] 定投倍数规则")

cases = [(5, 1.5), (19.9, 1.5), (20, 1.2), (39, 1.2), (40, 1.0),
         (59, 1.0), (60, 0.8), (79, 0.8), (80, 0.5), (98, 0.5)]
for pct, want in cases:
    got = AD.dca_multiplier(pct)["multiplier"]
    check(f"分位 {pct} → {want}x", got == want, got)

check("分位 None → 1.0（不猜）", AD.dca_multiplier(None)["multiplier"] == 1.0)
check("分位 nan → 1.0", AD.dca_multiplier(float("nan"))["multiplier"] == 1.0)
check("极端高位仍给 0.5 不归零",
      AD.dca_multiplier(100)["multiplier"] == 0.5)
check("有档位说明", len(AD.dca_multiplier(10)["tier"]) > 3)

print("\n[2] 单位容错（历史 bug：98% 被当成未知）")

check("传 0.98 被识别为 98%",
      AD.dca_multiplier(0.98)["multiplier"] == 0.5,
      AD.dca_multiplier(0.98))
check("传 98.3 正常", AD.dca_multiplier(98.3)["multiplier"] == 0.5)
check("传 0.05 被识别为 5%",
      AD.dca_multiplier(0.05)["multiplier"] == 1.5,
      AD.dca_multiplier(0.05))
check("容错后仍记录分位",
      AD.dca_multiplier(0.98).get("percentile") == 98.0,
      AD.dca_multiplier(0.98).get("percentile"))

print("\n[3] 再平衡")

w = {"A": 0.55, "B": 0.25, "C": 0.20}
rb = AD.rebalance_plan(w)
check("可计算", rb["ok"] is True)
check("识别出偏离", bool(rb["need_action"]))
check("A 应减仓",
      next(r for r in rb["rows"] if r["code"] == "A")["action"] == "减仓")
check("C 应加仓",
      next(r for r in rb["rows"] if r["code"] == "C")["action"] == "加仓")
check("阈值内不动",
      not AD.rebalance_plan({"A": 0.34, "B": 0.33, "C": 0.33})["need_action"])
check("自定义阈值生效",
      AD.rebalance_plan({"A": 0.40, "B": 0.30, "C": 0.30}, band=0.15)[
          "need_action"] == [])
check("自定义目标权重",
      AD.rebalance_plan({"A": 0.5, "B": 0.5},
                        targets={"A": 0.7, "B": 0.3})["need_action"][0]["action"]
      == "加仓")

# 现金措辞
rb_cash = AD.rebalance_plan({"A": 0.3, "CASH": 0.7})
acts = {r["code"]: r["action"] for r in rb_cash["rows"]}
check("现金过多是「投出现金」", acts.get("CASH") == "投出现金", acts)
check("现金过少是「补充现金」",
      {r["code"]: r["action"] for r in
       AD.rebalance_plan({"A": 0.9, "CASH": 0.1})["rows"]}.get("CASH")
      == "补充现金")
check("空权重返回 ok=False", AD.rebalance_plan({})["ok"] is False)

print("\n[4] 回撤纪律")

for dd, want in [(-0.05, 0.0), (-0.09, 0.0), (-0.15, 0.25),
                 (-0.30, 0.5), (-0.50, 1.0)]:
    got = AD.drawdown_playbook(dd)["extra_pct"]
    check(f"回撤 {dd*100:.0f}% → 额外 {want*100:.0f}%", got == want, got)
check("缺数据返回 ok=False", AD.drawdown_playbook(None)["ok"] is False)
check("有纪律说明", len(AD.drawdown_playbook(-0.2)["note"]) > 5)

print("\n[5] 相关性（假分散识别）")

idx = pd.bdate_range("2023-01-02", periods=500)
rng = np.random.default_rng(7)
base = rng.normal(0.0004, 0.01, 500)
navs = {
    "A": pd.Series(np.cumprod(1 + base), index=idx),
    "B": pd.Series(np.cumprod(1 + base * 0.95 + rng.normal(0, 0.001, 500)), index=idx),
    "C": pd.Series(np.cumprod(1 + rng.normal(0.0002, 0.012, 500)), index=idx),
}
c = AD.portfolio_correlation(navs)
check("可计算", c["ok"] is True, c.get("reason"))
check("发现高相关对", len(c["high_pairs"]) >= 1, c["high_pairs"])
check("A/B 相关性高",
      max(p["corr"] for p in c["pairs"] if {p["a"], p["b"]} == {"A", "B"}) > 0.9)
check("给出结论", "同质" in c["verdict"], c["verdict"])
check("序列不足返回 ok=False",
      AD.portfolio_correlation({"A": navs["A"]})["ok"] is False)

print("\n[6] 汇总建议")

names = {"F1": "博时标普500ETF联接C", "F2": "华安纳斯达克100"}
per = {
    "sp500": {"ok": True, "name": "标普500", "current": 100.0,
              "percentile": 10.0, "source": "test", "n": 300, "years": 3,
              "low": 80.0, "high": 110.0},
    "ndx": {"ok": True, "name": "纳斯达克100", "current": 200.0,
            "percentile": 95.0, "source": "test", "n": 300, "years": 3,
            "low": 150.0, "high": 205.0},
}
res = AD.advise(["F1", "F2"], names,
                weights={"F1": 0.6, "F2": 0.4},
                monthly_amount=1000.0, portfolio_drawdown=-0.25,
                index_percentiles=per)
check("生成建议", res["ok"] is True)
check("动作不超过 3 条", len(res["actions"]) <= 3, len(res["actions"]))
check("有动作", len(res["actions"]) >= 1)
acts = [a["action"] for a in res["actions"]]
check("低位指数建议加投", "加投" in acts, acts)
check("高位指数建议减投", "减投" in acts, acts)
check("回撤到位建议额外加仓", "额外加仓" in acts, acts)
check("每条都有依据", all(a.get("reason") for a in res["actions"]))
check("加投金额按倍数算",
      any(a.get("amount") == 1500.0 for a in res["actions"]),
      [a.get("amount") for a in res["actions"]])

# 什么都不该动的情形
quiet = AD.advise(["F1"], {"F1": "博时标普500ETF联接C"},
                  weights={"F1": 0.5}, monthly_amount=1000.0,
                  portfolio_drawdown=-0.02,
                  index_percentiles={"sp500": {**per["sp500"],
                                               "percentile": 50.0}})
check("中性时不动", quiet["n_actions"] == 0, [a["action"] for a in quiet["actions"]])
check("明确说不用动", "不用动" in quiet["verdict"], quiet["verdict"])

check("报告可渲染", "今日决策" in AD.format_report(res))
check("报告含免责", "不构成" in AD.format_report(res) or
      "不保证" in AD.format_report(res))
check("失败报告可渲染",
      "无法生成" in AD.format_report({"ok": False, "reason": "x"}))

print("\n[7] 接口与 CLI")

import api_v2 as API
from fastapi.testclient import TestClient
cli = TestClient(API.app)

_orig_advise = AD.advise
AD.advise = lambda *a, **k: {"ok": True, "n_actions": 0, "actions": [],
                             "details": {}, "mapping": {},
                             "monthly_amount": 1000.0,
                             "verdict": "今天不用动"}
try:
    r = cli.get("/api/advice?monthly=2000")
    check("接口 200", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        check("返回文本", "今日决策" in r.json().get("text", ""))
finally:
    AD.advise = _orig_advise

import quantpilot as QP
check("CLI 注册 advice", "advice" in QP.COMMANDS)
pp = QP.build_parser()
check("解析 --monthly", pp.parse_args(["advice", "--monthly", "3000"]).monthly == 3000)
check("解析 --drawdown",
      abs(pp.parse_args(["advice", "--drawdown", "-0.2"]).drawdown + 0.2) < 1e-9)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  决策引擎可用")
print("#" * 70)
