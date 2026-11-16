# -*- coding: utf-8 -*-
"""
基础设施与运维层（M67）：调度与任务
====================================================
为什么不用 Airflow
------------------
Airflow 要起 Web 服务 + 元数据库 + Worker，本地单机是过度工程。
这里实现一个**够用的轻量调度**，保留 Airflow 的核心语义：

| 语义 | 本模块 |
|---|---|
| DAG 依赖 | `deps`：前置任务当天成功才跑 |
| 重试 | `retries` + 指数退避 |
| 幂等 | 同一任务同一周期只跑一次（状态文件） |
| 并发保护 | 锁文件（含 PID，陈旧锁自动清理） |
| 执行记录 | 全部写 `opslog` 审计 |
| 调度表达式 | `daily@HH:MM` / `weekday@HH:MM` / `every@Nm` |

用法
----
```python
@job("ingest_cn", "weekday@15:30", retries=2)
def ingest_cn():
    ...
run_due()          # 跑所有到点的任务
run_job("ingest_cn", force=True)
```
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path

import pandas as pd

import opslog as OL

__version__ = "0.1.0"
__all__ = ["JOBS", "job", "list_jobs", "should_run", "run_job", "run_due",
           "parse_expr", "state"]

_ROOT = Path(__file__).resolve().parent.parent
STATE = _ROOT / ".data" / "jobs" / "state.json"
LOCKS = _ROOT / ".data" / "jobs" / "locks"
STALE_LOCK_H = 3.0

JOBS: dict[str, dict] = {}


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1),
                     encoding="utf-8")


def state() -> dict:
    return _load_state()


def job(name: str, expr: str = "daily@00:00", retries: int = 1,
        deps: tuple = (), timeout_s: int | None = None):
    """注册任务装饰器。"""
    def deco(fn):
        JOBS[name] = {"fn": fn, "expr": str(expr), "retries": int(retries),
                      "deps": tuple(deps), "timeout_s": timeout_s}
        return fn
    return deco


def parse_expr(expr: str) -> dict:
    """
    解析调度表达式：
      daily@09:30 / weekday@09:30 / every@15m
    """
    e = str(expr).strip().lower()
    m = re.match(r"^(daily|weekday)@(\d{1,2}):(\d{2})$", e)
    if m:
        h, mi = int(m.group(2)), int(m.group(3))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            raise ValueError(f"时间越界 {expr}")
        return {"type": m.group(1), "h": h, "m": mi}
    m = re.match(r"^every@(\d+)m$", e)
    if m:
        return {"type": "every", "minutes": max(1, int(m.group(1)))}
    raise ValueError(f"无法解析调度表达式 {expr!r}")


def _last_success(st: dict, name: str) -> str | None:
    return (st.get(name) or {}).get("last_success")


def should_run(name: str, now: pd.Timestamp | None = None,
               st: dict | None = None) -> bool:
    """判断任务此刻是否该跑（当日/当周期是否已成功过）。"""
    if name not in JOBS:
        return False
    st = st if st is not None else _load_state()
    cfg = parse_expr(JOBS[name]["expr"])
    now = now or pd.Timestamp.now()
    last = _last_success(st, name)
    if last is None:
        return True
    last_ts = pd.Timestamp(last)
    if cfg["type"] in ("daily", "weekday"):
        if cfg["type"] == "weekday" and now.dayofweek >= 5:
            return False
        target = now.normalize() + pd.Timedelta(hours=cfg["h"],
                                                minutes=cfg["m"])
        return now >= target and last_ts.normalize() < now.normalize()
    if cfg["type"] == "every":
        return (now - last_ts).total_seconds() >= cfg["minutes"] * 60
    return False


def _lock_path(name: str) -> Path:
    LOCKS.mkdir(parents=True, exist_ok=True)
    return LOCKS / f"{name}.lock"


def _acquire(name: str) -> bool:
    """取锁；陈旧锁（>3h 或进程已死）自动清理。"""
    fp = _lock_path(name)
    if fp.exists():
        try:
            info = json.loads(fp.read_text(encoding="utf-8"))
            age_h = (time.time() - float(info.get("ts", 0))) / 3600
            pid = int(info.get("pid", 0))
            alive = True
            if pid:
                try:
                    os.kill(pid, 0)
                except Exception:
                    alive = False
            if age_h < STALE_LOCK_H and alive:
                return False
        except Exception:
            pass
        try:
            fp.unlink()
        except Exception:
            pass
    try:
        fp.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}),
                      encoding="utf-8")
        return True
    except Exception:
        return False


def _release(name: str) -> None:
    try:
        _lock_path(name).unlink()
    except Exception:
        pass


def run_job(name: str, force: bool = False,
            now: pd.Timestamp | None = None) -> dict:
    """执行一个任务（含依赖、重试、锁、审计）。"""
    if name not in JOBS:
        return {"ok": False, "error": f"未注册任务 {name}"}
    cfg = JOBS[name]
    st = _load_state()

    # 依赖检查
    for d in cfg.get("deps") or ():
        last = _last_success(st, d)
        today = pd.Timestamp.now().normalize()
        if last is None or pd.Timestamp(last).normalize() < today:
            OL.log(name, status="skip", kind="job", reason=f"依赖 {d} 未成功")
            return {"ok": False, "skipped": True, "reason": f"依赖 {d} 未成功"}

    if not force and not should_run(name, now, st):
        return {"ok": True, "skipped": True, "reason": "本周期已跑过"}

    if not _acquire(name):
        OL.log(name, status="skip", kind="job", reason="已有实例在跑")
        return {"ok": False, "skipped": True, "reason": "已有实例在跑"}

    attempts = int(cfg.get("retries", 0)) + 1
    t0 = time.time()
    last_err = None
    for k in range(attempts):
        OL.log(name, status="start", kind="job", attempt=k + 1)
        try:
            res = cfg["fn"]()
            dt = round(time.time() - t0, 2)
            st[name] = {"last_success": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "last_status": "ok", "seconds": dt,
                        "attempt": k + 1}
            _save_state(st)
            OL.log(name, status="ok", kind="job", seconds=dt,
                   attempt=k + 1, result=str(res)[:200] if res else None)
            _release(name)
            return {"ok": True, "attempt": k + 1, "seconds": dt,
                    "result": res}
        except Exception as ex:
            last_err = f"{type(ex).__name__}: {ex}"
            OL.log(name, status="retry" if k + 1 < attempts else "fail",
                   kind="job", attempt=k + 1, error=last_err[:300],
                   tb=traceback.format_exc()[-400:])
            if k + 1 < attempts:
                time.sleep(min(30, 2 ** k))
    dt = round(time.time() - t0, 2)
    st[name] = {"last_fail": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "last_status": "fail", "seconds": dt,
                "error": last_err[:200]}
    _save_state(st)
    _release(name)
    return {"ok": False, "error": last_err, "seconds": dt}


def run_due(now: pd.Timestamp | None = None) -> dict:
    """跑所有到点的任务（按注册顺序）。"""
    st = _load_state()
    out = {}
    for name in list(JOBS):
        if should_run(name, now, st):
            out[name] = run_job(name, now=now)
    return out


def list_jobs() -> pd.DataFrame:
    st = _load_state()
    rows = []
    for n, c in JOBS.items():
        s = st.get(n) or {}
        rows.append({"task": n, "expr": c["expr"], "retries": c["retries"],
                     "deps": ",".join(c.get("deps") or ()),
                     "last_status": s.get("last_status") or "-",
                     "last_success": s.get("last_success") or "-"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import shutil

    TMP = _ROOT / ".tmp" / "test_opsjobs"
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)
    STATE = TMP / "state.json"
    LOCKS = TMP / "locks"
    OL.LOG = TMP / "audit.jsonl"
    OL.LOGDIR = TMP
    JOBS.clear()

    cnt = {"a": 0, "b": 0, "fail": 0}

    @job("a", "every@1m")
    def _a():
        cnt["a"] += 1
        return "a done"

    @job("b", "every@1m", deps=("a",))
    def _b():
        cnt["b"] += 1
        return "b done"

    @job("boom", "every@1m", retries=2)
    def _boom():
        cnt["fail"] += 1
        raise RuntimeError("故意失败")

    r1 = run_job("a")
    print("首次跑 a:", r1)
    assert r1["ok"] and cnt["a"] == 1
    r2 = run_job("a")
    print("立刻再跑 a:", r2)
    assert r2.get("skipped") is True
    r3 = run_job("a", force=True)
    assert r3["ok"] and cnt["a"] == 2

    rb = run_job("b")
    print("b（依赖 a 今日已成功）:", rb)
    assert rb["ok"] and cnt["b"] == 1

    rf = run_job("boom")
    print("boom 重试后:", rf["ok"], "调用次数", cnt["fail"])
    assert rf["ok"] is False and cnt["fail"] == 3

    print("解析:", parse_expr("weekday@09:30"), parse_expr("every@15m"))
    try:
        parse_expr("bad")
        raise AssertionError("应报错")
    except ValueError:
        print("非法表达式正确报错")

    print(list_jobs().to_string(index=False))
    print("OK v" + __version__)
