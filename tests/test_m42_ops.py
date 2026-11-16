# -*- coding: utf-8 -*-
"""M42 验证：运维层（审计日志 / 调度 / 体检告警 / 备份恢复），不联网"""
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import opshealth as OH
import opsjobs as OJ
import opslog as OL

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m42"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
          (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M42 验证  opslog v{OL.__version__} / opsjobs v{OJ.__version__}"
      f" / opshealth v{OH.__version__}")
print("#" + "=" * 69)

# 全部重定向到临时目录
OL.LOGDIR = TMP / "logs"
OL.LOG = OL.LOGDIR / "audit.jsonl"
OJ.STATE = TMP / "jobs" / "state.json"
OJ.LOCKS = TMP / "jobs" / "locks"
OH.DATA = TMP / "data"
OH.BACKUPS = OH.DATA / "backups"
OH.ALERT_STATE = OH.DATA / "alerts" / "state.json"
OJ.JOBS.clear()

print("\n[1] 审计日志")
rid = OL.new_run("t", note="x")
OL.log("e1", status="ok", kind="t", rows=5)
OL.log("e2", status="fail", kind="t", error="boom")
df = OL.query(run=rid)
check("run 内 3 条", len(df) == 3, len(df))
check("按 status 过滤", len(OL.query(status="fail")) == 1)
check("按 kind 过滤", len(OL.query(kind="t")) == 3)
check("倒序（新在前）", df.iloc[0]["event"] == "e2", df.iloc[0]["event"])
check("带 pid", isinstance(df.iloc[0].get("pid"), (int, np.integer)))
check("tail 可用", len(OL.tail(2)) == 2)
check("run 汇总有 1 行", len(OL.runs_summary()) == 1)
check("空查询不报错", len(OL.query(kind="nope")) == 0)
r = OL.rotate(0)
check("滚动生效", r["rotated"] is True and not OL.LOG.exists())

print("\n[2] 调度与任务")
cnt = {"a": 0, "b": 0, "boom": 0}


@OJ.job("a", "every@1m")
def _a():
    cnt["a"] += 1
    return "ok"


@OJ.job("b", "every@1m", deps=("a",))
def _b():
    cnt["b"] += 1
    return "ok"


@OJ.job("boom", "every@1m", retries=2)
def _boom():
    cnt["boom"] += 1
    raise RuntimeError("故意")


check("注册 3 个任务", len(OJ.JOBS) == 3, len(OJ.JOBS))
check("解析 daily", OJ.parse_expr("daily@09:30")["h"] == 9)
check("解析 weekday", OJ.parse_expr("weekday@15:05")["m"] == 5)
check("解析 every", OJ.parse_expr("every@15m")["minutes"] == 15)
try:
    OJ.parse_expr("nonsense")
    check("非法表达式报错", False)
except ValueError:
    check("非法表达式报错", True)

r1 = OJ.run_job("a")
check("首次执行成功", r1["ok"] is True and cnt["a"] == 1, r1)
check("重复执行被幂等跳过", OJ.run_job("a").get("skipped") is True)
check("force 可强制", OJ.run_job("a", force=True)["ok"] and cnt["a"] == 2)
check("依赖满足后可跑", OJ.run_job("b")["ok"] and cnt["b"] == 1)

rfail = OJ.run_job("boom")
check("失败任务返回 False", rfail["ok"] is False)
check("重试 3 次", cnt["boom"] == 3, cnt["boom"])
st = OJ.state()
check("状态记录失败", st["boom"]["last_status"] == "fail")

check("未注册任务报错", OJ.run_job("nope")["ok"] is False)
check("任务清单 3 行", len(OJ.list_jobs()) == 3)
due = OJ.run_due()
check("run_due 返回 dict", isinstance(due, dict))

print("\n[3] 体检与告警")


@OH.check("always_ok", "info")
def _okc():
    return True, "good"


@OH.check("always_bad", "warn")
def _badc():
    return False, "bad thing"


res = OH.run_checks()
check("体检跑通", res["n"] >= 2, res["n"])
check("识别失败项", res["ok"] is False and res["n_fail"] >= 1, res["n_fail"])
names = [r["check"] for r in res["results"]]
check("注册项都执行", "always_ok" in names and "always_bad" in names)
check("报告可渲染", "系统体检" in OH.format_report(res))

sent = []


def _send(sub, body):
    sent.append((sub, body))
    return True


a1 = OH.alert(res, _send)
check("失败时告警发出", a1["sent"] is True, a1)
check("邮件函数被调用", len(sent) == 1)
a2 = OH.alert(res, _send)
check("同类告警去重", a2["sent"] is False and len(sent) == 1, a2)
ok_res = {"ok": True, "n": 1, "n_fail": 0, "results": []}
check("全通过时不告警", OH.alert(ok_res, _send)["sent"] is False)

print("\n[4] 备份与恢复")
(OH.DATA / "parquet" / "cn").mkdir(parents=True, exist_ok=True)
(OH.DATA / "versions").mkdir(parents=True, exist_ok=True)
(OH.DATA / "parquet" / "cn" / "x.parquet").write_bytes(b"PAR1" + b"0" * 100)
(OH.DATA / "versions" / "v1.json").write_text('{"a":1}', encoding="utf-8")

b = OH.backup(keep=2)
check("备份成功", b["ok"] is True, b.get("error"))
check("清单文件数正确", b["n_files"] == 2, b["n_files"])
check("有 sha1", len(b.get("names_sha1", "")) == 40, b.get("names_sha1"))
check("zip 存在", Path(b["path"]).exists())

v = OH.verify_backup(b["path"])
check("校验通过", v["ok"] is True, v)
check("校验文件数一致", v["n_files"] == 2, v["n_files"])

bad = TMP / "bad.zip"
bad.write_bytes(b"not a zip")
check("坏包校验失败", OH.verify_backup(str(bad))["ok"] is False)
check("不存在文件校验失败", OH.verify_backup(str(TMP / "nope.zip"))["ok"] is False)

tgt = TMP / "restore"
rr = OH.restore(b["path"], str(tgt))
check("恢复成功", rr["ok"] is True, rr.get("error"))
check("恢复出文件", (tgt / "parquet" / "cn" / "x.parquet").exists())
check("恢复内容一致",
      (tgt / "versions" / "v1.json").read_text(encoding="utf-8") == '{"a":1}')

OH.backup(keep=1)
OH.backup(keep=1)
zips = list(OH.BACKUPS.glob("backup_*.zip"))
check("保留数量生效", len(zips) <= 1, len(zips))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  第 5 层运维可用")
print("#" * 70)
