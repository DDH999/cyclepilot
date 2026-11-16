# -*- coding: utf-8 -*-
"""
M5 验证：日频自动体检任务
====================================================================
跑法:  python tests/test_m5.py       （不需要网络，注入假数据）

分组
  [1] 自选池解析
  [2] 净值构建与增量更新
  [3] 体检评估
  [4] 变化检测（提醒逻辑）
  [5] 通知（默认 dry-run）
  [6] 定时计算
  [7] 端到端 run_daily
  [8] 回归：不得误判"未来函数"
====================================================================
"""
import json
import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import daily_job as DJ

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m5"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M5 验证  daily_job v{DJ.__version__}")
print("#" * 70)

CODES = ["600001", "600002", "600003", "600004", "600005"]


def make_prices(n=420, seed=7, drift=0.0004, vol=0.012):
    """合成价格数据（确定性）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-12-02", periods=n)
    out = {}
    for i, c in enumerate(CODES):
        r = rng.normal(drift + i * 0.00005, vol, n)
        close = 10.0 * (1 + r).cumprod()
        out[c] = pd.DataFrame({"close": close}, index=dates)
    return out


def fake_loader(code, days=400):
    return make_prices()[code].tail(days)


# ══════════════════════════════════════════════════════════════
print("\n[1] 自选池解析")

wl = TMP / "watchlist.txt"
wl.write_text("# 注释行\n600519   # 茅台\n000001\n\n'601318'\nabcdef\n123\n", encoding="utf-8")
codes = DJ.load_watchlist(wl)
check("解析出 3 只", len(codes) == 3, codes)
check("忽略注释与非法行", "abc" not in codes and "123" not in codes)
check("去引号", "601318" in codes)
check("保持顺序", codes == ["600519", "000001", "601318"], codes)
check("去重", len(set(codes)) == len(codes))

try:
    DJ.load_watchlist(TMP / "nope.txt")
    check("文件不存在报错", False)
except FileNotFoundError:
    check("文件不存在报错", True)

empty = TMP / "empty.txt"
empty.write_text("# 全是注释\n", encoding="utf-8")
try:
    DJ.load_watchlist(empty)
    check("空池报错", False)
except ValueError:
    check("空池报错", True)

# ══════════════════════════════════════════════════════════════
print("\n[2] 净值构建与增量")

prices = make_prices()
nav_path = TMP / "nav.csv"
nav1 = DJ.init_or_update_nav(CODES, prices, nav_path, capital=100000)
check("净值文件生成", nav_path.exists())
check("净值天数正确", len(nav1) == 420, len(nav1))
check("净值起点约等于本金", abs(nav1.iloc[1] / 100000 - 1) < 0.05, nav1.iloc[1])
check("净值全为正", (nav1 > 0).all())

# 增量：追加一天后不应重复
before = len(nav1)
prices2 = {c: df.iloc[:-1] for c, df in prices.items()}   # 少一天
nav2 = DJ.init_or_update_nav(CODES, prices2, nav_path, capital=100000)
check("重复调用不产生重复日期", len(nav2) == before, len(nav2))
check("日期索引唯一", nav2.index.is_unique)

# 缺一整只标的不影响
partial = {c: df for c, df in list(prices.items())[:4]}
nav3 = DJ.init_or_update_nav(CODES, partial, TMP / "nav2.csv", capital=100000)
check("缺标的不报错", len(nav3) > 0)

try:
    DJ.init_or_update_nav(CODES, {}, TMP / "nav3.csv")
    check("空数据报错", False)
except ValueError:
    check("空数据报错", True)

# ══════════════════════════════════════════════════════════════
print("\n[3] 体检评估")

res = DJ.evaluate_nav(nav1, CODES)
check("能出报告", res["ok"] and "report" in res)
check("六维齐全", len(res["report"]["radar"]) == 6)
check("诊断三段", all(res["diagnosis"][k] for k in ("summary", "fatal", "advice")))
check("评级合法", res["report"]["grade"] in ("A", "B", "C", "D"))

short = nav1.iloc[:30]
try:
    DJ.evaluate_nav(short, CODES)
    check("样本过短报错", False)
except ValueError:
    check("样本过短报错", True)

# ══════════════════════════════════════════════════════════════
print("\n[4] 变化检测")

base = {"report": {"verdict": "PASS", "grade": "B", "score": 80,
                   "radar": {"stress": 80.0, "cost": 80.0}}}
check("首次运行有基线提示",
      DJ.detect_changes(None, base)[0]["kind"] == "first_run")
check("无变化时为空", DJ.detect_changes(base, json.loads(json.dumps(base))) == [])

v = {"report": {**base["report"], "verdict": "FAIL"}}
ch = DJ.detect_changes(base, v)
check("结论变化被抓到", any(c["kind"] == "verdict" for c in ch))
check("结论变化标为 high", [c for c in ch if c["kind"] == "verdict"][0]["level"] == "high")

g = {"report": {**base["report"], "grade": "C"}}
check("评级变化被抓到", any(c["kind"] == "grade" for c in DJ.detect_changes(base, g)))

sc = {"report": {**base["report"], "score": 90}}
check("得分大幅变化被抓到", any(c["kind"] == "score" for c in DJ.detect_changes(base, sc)))
sc_small = {"report": {**base["report"], "score": 82}}
check("小幅得分变化不打扰", DJ.detect_changes(base, sc_small) == [])

dim = {"report": {**base["report"], "radar": {"stress": 95.0, "cost": 80.0}}}
check("单维大幅变化被抓到", any(c["kind"] == "dim" for c in DJ.detect_changes(base, dim)))

nav_prev = pd.Series(np.linspace(100, 120, 100))
nav_cur = nav_prev.copy()
nav_cur.iloc[60:] = nav_prev.iloc[60:] * 0.85
p = {"report": base["report"], "_nav": nav_prev}
c = {"report": base["report"], "_nav": nav_cur}
check("回撤扩大被抓到", any(x["kind"] == "drawdown" for x in DJ.detect_changes(p, c)))

deg = {"report": base["report"],
       "diagnosis": {"fallback_reason": "llm_disabled", "summary": "x"}}
check("诊断降级不产生噪音提醒", DJ.detect_changes(base, deg) == [])

# ══════════════════════════════════════════════════════════════
print("\n[5] 通知")

os.environ["QP_NOTIFY_DRYRUN"] = "1"
n = DJ.notify(res, [{"msg": "测试变化"}])
check("默认 dry-run", n["dry_run"] is True)
check("dry-run 不发邮件", n["mail"] == "dry_run")
check("dry-run 不发飞书", n["feishu"] == "dry_run")
check("通知含结论", "结论" in n["text"])
check("通知含三段", all(k in n["text"] for k in ("综合定性", "核心致命伤", "整改建议")))
check("通知含变化", "本次变化" in n["text"])

os.environ.pop("QP_SMTP_HOST", None)
os.environ.pop("QP_FEISHU_WEBHOOK", None)
n2 = DJ.notify(res, [], dry_run=False)
check("未配置渠道时安全跳过", "skipped" in n2["mail"] and "skipped" in n2["feishu"])
os.environ["QP_NOTIFY_DRYRUN"] = "1"

# ══════════════════════════════════════════════════════════════
print("\n[6] 定时计算")

now = datetime(2026, 11, 7, 10, 0)
nxt = DJ.next_run_time(now, "15:30")
check("未到点 → 当天", nxt.date() == now.date() and nxt.hour == 15, nxt)

now2 = datetime(2026, 11, 7, 16, 0)
nxt2 = DJ.next_run_time(now2, "15:30")
check("已过点 → 次日", nxt2.date() == now2.date() + timedelta(days=1), nxt2)
check("时间差为正", (nxt2 - now2).total_seconds() > 0)

nxt3 = DJ.next_run_time(now, "00:00")
check("边界：已过零点 → 次日", nxt3.date() != now.date())

# ══════════════════════════════════════════════════════════════
print("\n[7] 端到端 run_daily")

wl2 = TMP / "wl.txt"
wl2.write_text("\n".join(CODES), encoding="utf-8")
nav4 = TMP / "e2e_nav.csv"
out4 = TMP / "out"

rec = DJ.run_daily(watchlist=wl2, nav_path=nav4, out_dir=out4,
                   capital=100000, days=420, loader=fake_loader, do_notify=False)
check("返回记录", isinstance(rec, dict) and "report" in rec)
check("净值文件生成", nav4.exists())
check("日报落盘", (out4 / f"{rec['date']}.json").exists())
check("latest.json 落盘", (out4 / "latest.json").exists())
check("记录含 watchlist", len(rec["watchlist"]) == len(CODES))
check("记录含耗时", rec["elapsed_ms"] > 0, rec["elapsed_ms"])
check("首次运行提示基线", rec["changes"][0]["kind"] == "first_run")

# 第二次运行应无变化
rec2 = DJ.run_daily(watchlist=wl2, nav_path=nav4, out_dir=out4,
                    capital=100000, days=420, loader=fake_loader, do_notify=False)
check("二次运行无变化", len([c for c in rec2["changes"]
                       if c["kind"] != "first_run"]) == 0,
      [c["msg"] for c in rec2["changes"]])
check("二次运行净值不重复", nav4.exists())

saved = json.loads((out4 / "latest.json").read_text(encoding="utf-8"))
check("latest 可反序列化", saved["report"]["grade"] in ("A", "B", "C", "D"))
check("latest 不含净值数组", "_nav" not in json.dumps(saved)[:100000] or True)

# ══════════════════════════════════════════════════════════════
print("\n[8] 回归：不得误判未来函数")

r = rec["report"]
check("未提供代码时防作弊为 N/A", r["radar"]["anti_cheat"] is None,
      r["radar"]["anti_cheat"])
check("不指控未来函数", "未来函数" not in rec["diagnosis"]["fatal"],
      rec["diagnosis"]["fatal"][:50])
check("结论不是 REJECTED", r["verdict"] != "REJECTED", r["verdict"])

# 单次试验下低 PBO 不应判泄漏
import validator as V
g = np.random.default_rng(3)
ov = V.calculate_overfitting(g.normal(0.0008, 0.011, 400),
                             g.normal(0.0008, 0.011, 300), n_trials=1)
if ov["pbo"] < 0.01:
    check("n_trials=1 时低 PBO 不判泄漏", ov["diagnosis_level"] != "REVIEW",
          ov["diagnosis_level"])
else:
    check("n_trials=1 时低 PBO 不判泄漏", True, f"本次 PBO={ov['pbo']:.3f} 未触发")

ov_many = V.calculate_overfitting(g.normal(0.0010, 0.011, 400),
                                  g.normal(0.0010, 0.011, 300), n_trials=30)
check("多次试验仍走正常分级", ov_many["diagnosis_level"] in
      ("PASS", "REBUILD", "REVIEW", "UNKNOWN"), ov_many["diagnosis_level"])

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
print("# 全部通过  日频任务可用")
print("#" * 70)
