# -*- coding: utf-8 -*-
"""
基础设施与运维层（M68）：健康检查 / 告警 / 备份恢复
====================================================
三件事
------
1. **体检**：数据新鲜度、入库失败率、审计错误、磁盘、存储完整性、版本库
2. **告警**：失败就发（复用 `mailer`），带**签名去重**，同类问题 60 分钟内只发一次
3. **备份**：把 Parquet / 版本库 / 元数据打成一个 zip，带清单，可校验、可恢复

设计取舍
--------
* 体检项用注册制，新增一项只需 `@check("name")`
* 告警去重按「问题签名」而不是时间：同一批问题不重复轰炸邮箱
* 备份校验用「文件数 + 总字节 + 文件名列表 sha1」，比逐文件 sha1 快得多，
  足以发现"少文件/被截断"，且不占大量时间
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

import opslog as OL

__version__ = "0.1.0"
__all__ = ["CHECKS", "check", "run_checks", "format_report", "alert",
           "backup", "verify_backup", "restore", "rotate_backups"]

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / ".data"
BACKUPS = DATA / "backups"
ALERT_STATE = DATA / "alerts" / "state.json"

CHECKS: dict[str, dict] = {}


def check(name: str, level: str = "warn"):
    """注册体检项。函数返回 (ok: bool, msg: str)。"""
    def deco(fn):
        CHECKS[name] = {"fn": fn, "level": level}
        return fn
    return deco


# ══════════════════════════════════════════════════════════
@check("data_freshness", "warn")
def _freshness(max_lag_days: int = 10) -> tuple[bool, str]:
    try:
        import store
    except Exception as ex:
        return False, f"store 不可用 {ex}"
    bad = []
    for m in ("cn", "hk", "us"):
        cov = store.coverage(m)
        if not len(cov):
            bad.append(f"{m}: 无数据")
            continue
        last = pd.to_datetime(cov["end"]).max()
        lag = int((pd.Timestamp.now().normalize() - last.normalize()).days)
        if lag > int(max_lag_days):
            bad.append(f"{m}: 滞后 {lag} 天")
    return (len(bad) == 0), ("；".join(bad) if bad else "各市场数据新鲜")


@check("ingest_failures", "warn")
def _ingest_fail(max_ratio: float = 0.35) -> tuple[bool, str]:
    p = DATA / "bulk_status.json"
    if not p.exists():
        return True, "无批量入库记录"
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return True, "状态文件损坏，跳过"
    done = len(st.get("done") or [])
    fail = len(st.get("failed") or {})
    tot = done + fail
    if tot == 0:
        return True, "无记录"
    ratio = fail / tot
    return (ratio <= float(max_ratio)), \
        f"失败 {fail}/{tot} = {ratio:.0%}"


@check("audit_failures", "warn")
def _audit_fail(lookback: int = 200, max_ratio: float = 0.25,
                max_abs: int = 40) -> tuple[bool, str]:
    """
    按**失败比例**判断，而不是绝对条数。

    调试期集中跑测试会产生大量失败记录，绝对阈值会误报；
    系统性故障的特征是"持续高比例失败"。
    """
    df = OL.query(limit=int(lookback))
    if not len(df):
        return True, "无审计记录"
    n = int((df["status"] == "fail").sum())
    ratio = n / len(df)
    ok = (n <= int(max_abs)) and (ratio <= float(max_ratio))
    return ok, f"最近 {len(df)} 条中失败 {n} 条（{ratio:.0%}）"


@check("disk_space", "warn")
def _disk(min_free_gb: float = 1.0) -> tuple[bool, str]:
    try:
        u = shutil.disk_usage(str(_ROOT))
        free_gb = u.free / 1024 ** 3
        return free_gb >= float(min_free_gb), f"剩余 {free_gb:.1f} GB"
    except Exception as ex:
        return True, f"无法读取磁盘 {ex}"


@check("store_integrity", "warn")
def _store() -> tuple[bool, str]:
    try:
        import store
    except Exception as ex:
        return False, f"store 不可用 {ex}"
    st = store.stats()
    if st.get("total_codes", 0) == 0:
        return False, "存储为空"
    return True, f"{st['total_codes']} 只 {st['total_rows']:,} 行 {st['mb']} MB"


@check("versions", "info")
def _versions() -> tuple[bool, str]:
    try:
        import versions as V
    except Exception as ex:
        return False, f"versions 不可用 {ex}"
    nf = len(V.list_items("factor"))
    ns = len(V.list_items("strategy"))
    return True, f"因子 {nf} 个、策略 {ns} 个"


# ══════════════════════════════════════════════════════════
def run_checks() -> dict:
    """跑全部体检。"""
    results = []
    for name, c in CHECKS.items():
        t0 = time.time()
        try:
            ok, msg = c["fn"]()
        except Exception as ex:
            ok, msg = False, f"{type(ex).__name__}: {ex}"
        results.append({"check": name, "ok": bool(ok), "level": c["level"],
                        "msg": str(msg)[:200],
                        "ms": round((time.time() - t0) * 1000, 1)})
        OL.log(name, status="ok" if ok else "fail", kind="health",
               msg=str(msg)[:200])
    n_fail = sum(1 for r in results if not r["ok"])
    return {"ok": n_fail == 0, "n": len(results), "n_fail": n_fail,
            "results": results}


def format_report(res: dict) -> str:
    L = ["=" * 72, "系统体检", "=" * 72]
    for r in res.get("results") or []:
        mark = "OK  " if r["ok"] else "FAIL"
        L.append(f"  [{mark}] {r['check']:<18}{r['msg']}")
    L.append("-" * 72)
    L.append(f"  {res['n'] - res['n_fail']}/{res['n']} 项通过")
    L.append("=" * 72)
    return "\n".join(L)


def _sig(res: dict) -> str:
    bad = sorted(r["check"] for r in res.get("results") or [] if not r["ok"])
    return hashlib.sha1("|".join(bad).encode()).hexdigest()[:12]


def alert(res: dict, send_fn=None, min_interval_min: int = 60) -> dict:
    """
    体检失败时告警（带签名去重）。

    send_fn: 可选的发送函数，签名 fn(subject: str, body: str) -> bool
    """
    if res.get("ok"):
        return {"sent": False, "reason": "全部通过"}
    sig = _sig(res)
    now = time.time()
    st = {}
    if ALERT_STATE.exists():
        try:
            st = json.loads(ALERT_STATE.read_text(encoding="utf-8"))
        except Exception:
            st = {}
    last = st.get(sig)
    if last and (now - float(last)) / 60 < float(min_interval_min):
        return {"sent": False, "reason": f"同类告警 {min_interval_min} 分钟内已发"}

    body = format_report(res)
    sent = False
    if send_fn is not None:
        try:
            sent = bool(send_fn("QuantPilot 体检告警", body))
        except Exception as ex:
            OL.log("alert", status="fail", kind="alert", error=str(ex)[:200])
            sent = False
    ALERT_STATE.parent.mkdir(parents=True, exist_ok=True)
    st[sig] = now
    ALERT_STATE.write_text(json.dumps(st), encoding="utf-8")
    OL.log("alert", status="ok" if sent else "fail", kind="alert",
           sig=sig, sent=sent, n_fail=res.get("n_fail"))
    return {"sent": sent, "sig": sig, "body": body}


# ══════════════════════════════════════════════════════════
INCLUDE = ("parquet", "versions", "meta", "jobs")


def _iter_files():
    for sub in INCLUDE:
        base = DATA / sub
        if not base.exists():
            continue
        for fp in base.rglob("*"):
            if fp.is_file():
                yield fp


def backup(keep: int = 7, tag: str = "") -> dict:
    """打包数据成 zip（带清单），并清理旧备份。"""
    BACKUPS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"backup_{ts}{('_' + tag) if tag else ''}.zip"
    out = BACKUPS / name
    files = list(_iter_files())
    manifest = {"created": ts, "n_files": len(files), "total_bytes": 0,
                "names": []}
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for fp in files:
                rel = fp.relative_to(DATA).as_posix()
                z.write(fp, arcname=rel)
                manifest["total_bytes"] += fp.stat().st_size
                manifest["names"].append(rel)
            manifest["names_sha1"] = hashlib.sha1(
                "\n".join(sorted(manifest["names"])).encode()).hexdigest()
            z.writestr("manifest.json",
                       json.dumps(manifest, ensure_ascii=False, indent=1))
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
    mb = round(out.stat().st_size / 1024 / 1024, 2)
    OL.log("backup", status="ok", kind="backup", n_files=len(files),
           mb=mb, path=str(out))
    removed = rotate_backups(keep)
    return {"ok": True, "path": str(out), "mb": mb, "n_files": len(files),
            "names_sha1": manifest["names_sha1"], "removed": removed}


def verify_backup(path: str) -> dict:
    """校验备份：zip 可读 + 文件数/清单 sha1 一致。"""
    fp = Path(path)
    if not fp.exists():
        return {"ok": False, "error": "文件不存在"}
    try:
        with zipfile.ZipFile(fp) as z:
            names = [n for n in z.namelist() if n != "manifest.json"]
            man = json.loads(z.read("manifest.json")) if \
                "manifest.json" in z.namelist() else {}
            bad = z.testzip()
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
    sha = hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()
    ok = (bad is None and len(names) == man.get("n_files")
          and sha == man.get("names_sha1"))
    return {"ok": bool(ok), "n_files": len(names),
            "expect": man.get("n_files"), "corrupt": bad,
            "sha_match": sha == man.get("names_sha1")}


def restore(path: str, target: str | None = None) -> dict:
    """从备份恢复到指定目录（默认 .data 旁边的 .data_restore，避免覆盖）。"""
    v = verify_backup(path)
    if not v["ok"]:
        return {"ok": False, "error": "校验未通过", "verify": v}
    dst = Path(target) if target else (_ROOT / ".data_restore")
    dst.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as z:
            z.extractall(dst)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
    n = sum(1 for _ in dst.rglob("*") if _.is_file())
    OL.log("restore", status="ok", kind="backup", to=str(dst), n_files=n)
    return {"ok": True, "target": str(dst), "n_files": n}


def rotate_backups(keep: int = 7) -> int:
    """只保留最近 N 个备份。"""
    if not BACKUPS.exists():
        return 0
    zips = sorted(BACKUPS.glob("backup_*.zip"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for fp in zips[int(keep):]:
        try:
            fp.unlink()
            removed += 1
        except Exception:
            pass
    return removed


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    res = run_checks()
    print(format_report(res))
    a = alert(res)
    print("告警:", {k: v for k, v in a.items() if k != "body"})
    print("立即再告警:", alert(res)["reason"])
    print("OK v" + __version__)
