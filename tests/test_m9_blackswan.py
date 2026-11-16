# -*- coding: utf-8 -*-
"""M9 验证：黑天鹅与事件风险（不联网）"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import blackswan as B

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M9 验证  blackswan v{B.__version__}")
print("#" + "=" * 69)

print("\n[1] 选举日期（规则化，可验证）")
el = {d["year"]: d for d in B.us_election_days(2020, 2032)}
check("2024 总统大选 = 11-05", el[2024]["date"] == "2024-11-05", el[2024]["date"])
check("2026 中期选举 = 11-03", el[2026]["date"] == "2026-11-03", el[2026]["date"])
check("2028 总统大选 = 11-07", el[2028]["date"] == "2028-11-07", el[2028]["date"])
check("2024 标为总统大选", el[2024]["kind"] == "美国大选")
check("2026 标为中期选举", el[2026]["kind"] == "美国中期选举")
check("只含偶数年", all(y % 2 == 0 for y in el))
for y in (2024, 2026, 2028, 2030):
    check(f"{y} 落在星期二", pd.Timestamp(el[y]["date"]).weekday() == 1,
          el[y]["date"])
check("全部为 11 月", all(d["date"][5:7] == "11" for d in el.values()))

print("\n[2] A 股披露截止日")
dl = B.a_share_report_deadlines(2026, 2026)
check("一年三条", len(dl) == 3, len(dl))
check("年报 4-30", any(x["date"] == "2026-04-30" for x in dl))
check("半年报 8-31", any(x["date"] == "2026-08-31" for x in dl))
check("三季报 10-31", any(x["date"] == "2026-10-31" for x in dl))

print("\n[3] 事件日历")
cal = B.event_calendar("2026-11-08", months_ahead=12)
check("返回列表", isinstance(cal, list))
check("按日期升序", all(a["date"] <= b["date"] for a, b in zip(cal, cal[1:])))
check("都带剩余天数", all("days_until" in x for x in cal))
check("剩余天数非负", all(x["days_until"] >= 0 for x in cal))
check("12 个月内不含大选（2028 在窗口外）",
      not any("选举" in x["kind"] for x in cal), [x["kind"] for x in cal])

cal_long = B.event_calendar("2026-11-08", months_ahead=30)
check("拉长窗口能覆盖 2028 大选",
      any("大选" in x["kind"] for x in cal_long), [x["kind"] for x in cal_long])
check("事件含说明", all(x.get("note") for x in cal_long))

print("\n[4] 尾部风险")
rng = np.random.default_rng(11)
idx = pd.bdate_range("2023-01-02", periods=600)
base_r = rng.normal(0.0005, 0.011, 600)
nav = pd.Series(np.cumprod(1 + base_r), index=idx)
t = B.tail_statistics(nav)
check("可评估", t["ok"] is True)
check("VaR95 为负", t["var95"] < 0, round(t["var95"], 4))
check("VaR99 比 VaR95 更极端", t["var99"] <= t["var95"], (t["var99"], t["var95"]))
check("CVaR 不优于 VaR95", t["cvar95"] <= t["var95"])
check("最差单日等于最小值", abs(t["worst_day"] - base_r.min()) < 1e-6,
      (t["worst_day"], base_r.min()))
check("最大回撤为负", t["max_drawdown"] < 0)
check("含偏度峰度", np.isfinite(t["skew"]) and np.isfinite(t["kurtosis"]))
check("样本过短返回 ok=False", B.tail_statistics(base_r[:10])["ok"] is False)

# 净值/收益判别：归一化净值必须被识别为净值
t_nav = B.tail_statistics(nav)
t_ret = B.tail_statistics(pd.Series(base_r, index=idx))
check("净值与收益口径一致",
      abs(t_nav["worst_day"] - t_ret["worst_day"]) < 1e-6,
      (t_nav["worst_day"], t_ret["worst_day"]))
check("归一化净值不被误判为收益", t_nav["worst_day"] < -0.01, t_nav["worst_day"])
check("显式 is_nav=True 生效",
      abs(B.tail_statistics(nav, is_nav=True)["worst_day"] - base_r.min()) < 1e-6)
check("显式 is_nav=False 生效",
      abs(B.tail_statistics(pd.Series(base_r), is_nav=False)["worst_day"] - base_r.min()) < 1e-6)

print("\n[5] 历史情景回放")
nav2 = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.012, 900)),
                 index=pd.bdate_range("2023-06-01", periods=900))
stress = B.stress_windows(nav2)
check("覆盖全部冲击窗口", len(stress) == len(B.HISTORICAL_SHOCKS), len(stress))
covered = [x for x in stress if x["ok"]]
check("至少覆盖 2024 微盘股", any("2024 微盘股" in x["name"] for x in covered),
      [x["name"] for x in covered])
check("未覆盖窗口有原因", all("reason" in x for x in stress if not x["ok"]))
check("已覆盖含收益与回撤",
      all("return" in x and "max_drawdown" in x for x in covered))
check("按收益升序（最惨在前）",
      all(a.get("return", 0) <= b.get("return", 0) for a, b in zip(covered, covered[1:])))

try:
    B.stress_windows(list(nav2.to_numpy()))
    check("非日期索引报错", False)
except ValueError:
    check("非日期索引报错", True)

print("\n[6] 脆弱度")
r_fat = rng.normal(0.0005, 0.011, 800)
r_fat[[100, 300, 500, 700]] = -0.11
fat = pd.Series(np.cumprod(1 + r_fat), index=pd.bdate_range("2023-01-02", periods=800))
normal = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.011, 800)),
                   index=pd.bdate_range("2023-01-02", periods=800))
f_n = B.fragility_score(B.tail_statistics(normal))
f_f = B.fragility_score(B.tail_statistics(fat))
check("肥尾序列更脆弱", f_f["score"] > f_n["score"] + 5,
      (round(f_n["score"], 1), round(f_f["score"], 1)))
check("分数在 [0,100]", 0 <= f_n["score"] <= 100 and 0 <= f_f["score"] <= 100)
check("含等级", f_f["level"] in ("低", "中", "较高", "高"), f_f["level"])
check("含建议", isinstance(f_f["advice"], str) and len(f_f["advice"]) > 5)
check("含量化分解", isinstance(f_f["breakdown"], dict) and len(f_f["breakdown"]) == 5)
check("分解之和等于总分",
      abs(sum(f_f["breakdown"].values()) - f_f["score"]) < 0.05,
      (sum(f_f["breakdown"].values()), f_f["score"]))
check("不可评估时返回 ok=False",
      B.fragility_score({"ok": False, "reason": "太短"})["ok"] is False)

print("\n[7] 报告")
txt = B.format_report(t_nav, cal[:5], stress, f_f)
for token in ["黑天鹅", "未来事件", "尾部风险", "VaR95", "历史情景", "脆弱度"]:
    check(f"报告含「{token}」", token in txt, token)
check("报告含建议", "建议:" in txt)
check("空输入不崩溃", isinstance(B.format_report({}, None, None, None), str))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  黑天鹅与事件风险可用")
print("#" * 70)
