# -*- coding: utf-8 -*-
"""
基础设施与运维层（M66）：全链路审计日志
====================================================
为什么需要
----------
回测/入库/调度如果只 print，出问题时无法回答：
* 这次结果算的哪个数据版本？
* 那次失败是第几次重试？
* 昨天 22:00 的任务到底跑了没？

本模块把每个动作写成**结构化 JSONL**，一行一条，可 grep、可 SQL（DuckDB）。

事件模型
--------
```
run_id  一次任务执行（如 today_ingest_20261116）
seq     同一 run 内自增
ts      ISO 时间
kind    ingest / backtest / job / alert / backup
status  start / ok / fail / retry / skip
fields  任意附加字段（code、rows、annual、error…）
```
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"
__all__ = ["LOGDIR", "new_run", "log", "query", "tail", "runs_summary",
           "rotate", "current_run"]

_ROOT = Path(__file__).resolve().parent.parent
LOGDIR = _ROOT / ".data" / "logs"
LOG = LOGDIR / "audit.jsonl"
MAX_MB = 32

_RUN = {"id": None, "seq": 0}


def _ensure() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)


def new_run(kind: str = "job", **meta) -> str:
    """开启一次 run，返回 run_id。"""
    _ensure()
    rid = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    _RUN["id"] = rid
    _RUN["seq"] = 0
    log("start", status="start", kind=kind, run=rid, **meta)
    return rid


def current_run() -> str | None:
    return _RUN.get("id")


def log(event: str, status: str = "ok", kind: str = "event",
        run: str | None = None, **fields) -> dict:
    """追加一条审计记录。"""
    _ensure()
    _RUN["seq"] = int(_RUN.get("seq", 0)) + 1
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run or _RUN.get("id"),
        "seq": _RUN["seq"],
        "kind": kind,
        "event": event,
        "status": status,
        "pid": os.getpid(),
    }
    for k, v in fields.items():
        if isinstance(v, (pd.Timestamp,)):
            v = str(v)
        rec[k] = v
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def query(kind: str | None = None, status: str | None = None,
          since: str | None = None, run: str | None = None,
          limit: int = 200) -> pd.DataFrame:
    """查询审计记录（新的在前）。"""
    if not LOG.exists():
        return pd.DataFrame()
    rows = []
    with LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if kind and r.get("kind") != kind:
                continue
            if status and r.get("status") != status:
                continue
            if run and r.get("run_id") != run:
                continue
            if since and str(r.get("ts", "")) < str(since):
                continue
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[::-1].head(int(limit))
    return df.reset_index(drop=True)


def tail(n: int = 20) -> list[dict]:
    if not LOG.exists():
        return []
    lines = LOG.read_text(encoding="utf-8").strip().splitlines()[-int(n):]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def runs_summary() -> pd.DataFrame:
    """按 run 汇总：起止时间、事件数、失败数、耗时。"""
    if not LOG.exists():
        return pd.DataFrame()
    rows = []
    cur = None
    start_ts = None
    # 首条记录可能没有 run_id（cur 与 rid 同为 None），此时不能引用未定义变量
    n_ev, n_f, last_ts = 0, 0, None
    for ln in LOG.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        rid = r.get("run_id")
        if rid != cur:
            if cur:
                rows.append({"run_id": cur, "start": start_ts,
                             "events": n_ev, "fails": n_f,
                             "end": last_ts})
            cur, start_ts, n_ev, n_f, last_ts = rid, r.get("ts"), 0, 0, r.get("ts")
        n_ev += 1
        if r.get("status") == "fail":
            n_f += 1
        last_ts = r.get("ts")
    if cur:
        rows.append({"run_id": cur, "start": start_ts, "events": n_ev,
                     "fails": n_f, "end": last_ts})
    df = pd.DataFrame(rows)
    return df.iloc[::-1].reset_index(drop=True) if len(df) else df


def rotate(max_mb: float = MAX_MB) -> dict:
    """日志超过上限就滚动（保留一份 .1）。"""
    if not LOG.exists():
        return {"ok": True, "rotated": False}
    mb = LOG.stat().st_size / 1024 / 1024
    if mb < float(max_mb):
        return {"ok": True, "rotated": False, "mb": round(mb, 2)}
    bak = LOG.with_suffix(".jsonl.1")
    try:
        if bak.exists():
            bak.unlink()
        LOG.rename(bak)
        return {"ok": True, "rotated": True, "mb": round(mb, 2)}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    LOGDIR = _ROOT / ".tmp" / "test_opslog"
    LOG = LOGDIR / "audit.jsonl"
    if LOGDIR.exists():
        import shutil
        shutil.rmtree(LOGDIR)

    rid = new_run("selftest", note="自测")
    log("step1", status="ok", kind="selftest", rows=100)
    log("step2", status="fail", kind="selftest", error="模拟失败")
    print("run_id:", rid)
    print("记录数:", len(query(run=rid)))
    assert len(query(run=rid)) == 3
    print("失败数:", len(query(status="fail")))
    assert len(query(status="fail")) == 1
    print("tail:", len(tail(5)))
    print(runs_summary().to_string(index=False))
    assert rotate(0)["rotated"] is True
    assert not LOG.exists()
    print("OK v" + __version__)
