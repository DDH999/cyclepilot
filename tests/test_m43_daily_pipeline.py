# -*- coding: utf-8 -*-
"""M43 验证：日更流水线（dry-run / 报告渲染 / 跳过语义 / 任务注册），不联网"""
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import daily_pipeline as DP
import opshealth as OH
import opsjobs as OJ

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m43"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
          (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M43 验证  daily_pipeline v{DP.__version__}")
print("#" + "=" * 69)

DP.REPORT_DIR = TMP / "reports"
OJ.STATE = TMP / "state.json"
OJ.LOCKS = TMP / "locks"
OJ.JOBS.clear()

print("\n[1] dry-run")
r = DP.run_daily(dry_run=True)
check("dry-run 成功", r["ok"] is True)
check("返回全部步骤", r["steps"] == DP.STEPS, r["steps"])
check("步骤顺序正确",
      DP.STEPS == ["ingest", "health", "research", "report", "notify"],
      DP.STEPS)

print("\n[2] 入库跳过语义")
r2 = DP.step_ingest(0)
check("limit=0 跳过", r2.get("skipped") is True, r2)
check("不产生调用", r2.get("done") == 0)
r3 = DP.step_ingest(-5)
check("负数也跳过", r3.get("skipped") is True)
check("CTX 记录跳过", DP.CTX["ingest"]["ok"] is True)

print("\n[3] 报告渲染")
DP.CTX.clear()
fake_health = {"ok": True, "n": 2, "n_fail": 0,
               "results": [{"check": "a", "ok": True, "level": "info",
                            "msg": "fine", "ms": 1.0},
                           {"check": "b", "ok": True, "level": "info",
                            "msg": "ok", "ms": 2.0}]}
DP._mark("health", True, {"n_fail": 0, "alert": None, "report": fake_health})
DP._mark("research", True, {"n_codes": 133, "window": "2023~2026",
                            "ic_mean": 0.021, "wf_oos_annual": 5.93,
                            "wf_oos_sharpe": 0.599, "wf_folds": 3})
rr = DP.step_report()
check("报告生成", rr["ok"] is True, rr.get("error"))
fp = Path(rr["path"])
check("文件存在", fp.exists())
txt = fp.read_text(encoding="utf-8")
check("含标题", "QuantPilot 日更" in txt)
check("含体检段", "系统体检" in txt)
check("含研究段", "Walk-forward" in txt)
check("含模块状态表", "| 步骤 |" in txt)
check("报告不报 KeyError", "KeyError" not in txt)

print("\n[4] 缺失字段容错")
DP.CTX.clear()
DP._mark("health", False, {"n_fail": 2, "alert": False, "report": {
    "ok": False, "n": 1, "n_fail": 1,
    "results": [{"check": "x", "ok": False, "level": "warn",
                 "msg": "bad", "ms": 1.0}]}})
rr2 = DP.step_report()
check("缺研究段也能出报告", rr2["ok"] is True, rr2.get("error"))
check("报告标记失败项", "FAIL" in Path(rr2["path"]).read_text(encoding="utf-8"))

print("\n[5] 任务注册")
DP.register_jobs(notify=False)
check("注册 1 个任务", len(OJ.JOBS) == 1, len(OJ.JOBS))
check("任务名正确", "daily_pipeline" in OJ.JOBS, list(OJ.JOBS))
cfg = OJ.JOBS["daily_pipeline"]
check("调度表达式", cfg["expr"] == "weekday@15:30", cfg["expr"])
check("有重试", cfg["retries"] >= 1, cfg["retries"])
parsed = OJ.parse_expr(cfg["expr"])
check("表达式可解析", parsed["type"] == "weekday" and parsed["h"] == 15)

print("\n[6] 通知步骤容错")
n = DP.step_notify(None)
check("未配置邮件不抛异常", isinstance(n, dict) and "ok" in n, n)
check("CTX 有 notify", "notify" in DP.CTX, list(DP.CTX))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  日更流水线可用")
print("#" * 70)
