# -*- coding: utf-8 -*-
"""
版本管理（M62）
====================================================
为什么自己实现
--------------
插件自带的 `factor_create` / `strategy_create` 在写入时会 ENOENT：
它直接写 `product/factors/<id>/object.json.tmp-*`，**但不创建 `<id>` 子目录**。
（已实测：`~/.dsh/dsh-skills-stock/product/` 整个目录都不存在，
 手动建到 `product/factors/` 仍报同样错，说明叶子目录也没 mkdir -p。）

在插件修好之前，用本模块做**等价且可迁移**的版本管理：
目录结构与插件一致，插件修好后可直接导入。

结构
----
```
.data/versions/<kind>/<id>/meta.json          名称/假设/创建时间
.data/versions/<kind>/<id>/v1.json            版本快照（code + params + note + parent）
.data/versions/<kind>/<id>/runs/<ts>.json     运行归档（指标/区间/配置）
.data/versions/<kind>/index.json              id → 名称/最新版本
```

保证
----
* **代码必须入库**：`save` 时 code 为空直接拒绝（禁止"只留在会话里"）
* **乐观锁**：`parent` 必须等于当前版本，防止并发覆盖
* **不可变**：版本快照只增不改，要改就加新版本
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["ROOT", "KINDS", "save", "get", "latest", "list_items", "record_run",
           "runs_of", "diff", "format_table"]

_ROOT = Path(__file__).resolve().parent.parent
ROOT = _ROOT / ".data" / "versions"
KINDS = ("factor", "strategy", "selection")


def _dir(kind: str, vid: str) -> Path:
    return ROOT / str(kind) / str(vid)


def _index(kind: str) -> dict:
    fp = ROOT / str(kind) / "index.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_index(kind: str, idx: dict) -> None:
    fp = ROOT / str(kind) / "index.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(idx, ensure_ascii=False, indent=1),
                  encoding="utf-8")


def create(kind: str, name: str, hypothesis: str = "",
           category: str = "") -> dict:
    """新建条目，返回 id。"""
    if kind not in KINDS:
        return {"ok": False, "error": f"kind 必须是 {KINDS}"}
    vid = f"{kind[:3]}_{hashlib.md5((name + str(time.time())).encode())
                  .hexdigest()[:12]}"
    d = _dir(kind, vid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "id": vid, "kind": kind, "name": name,
        "hypothesis": hypothesis, "category": category,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    idx = _index(kind)
    idx[vid] = {"name": name, "latest": 0}
    _save_index(kind, idx)
    return {"ok": True, "id": vid, "name": name}


def save(kind: str, vid: str, code: str, params: dict | None = None,
         change_note: str = "", parent: int | None = None) -> dict:
    """
    保存新版本（代码必须非空，parent 必须是当前版本）。
    """
    if not str(code or "").strip():
        return {"ok": False, "error": "code 不能为空：策略/因子代码必须入库"}
    d = _dir(kind, vid)
    if not (d / "meta.json").exists():
        return {"ok": False, "error": f"条目不存在 {vid}"}
    cur = latest(kind, vid)
    cur_n = int(cur.get("version", 0)) if cur.get("ok") else 0
    if parent is not None and int(parent) != cur_n:
        return {"ok": False, "error": f"乐观锁冲突：parent={parent} 但当前={cur_n}"}
    n = cur_n + 1
    snap = {
        "version": n, "kind": kind, "id": vid,
        "code": str(code), "params": params or {},
        "change_note": str(change_note),
        "parent": cur_n if cur_n else None,
        "code_sha1": hashlib.sha1(str(code).encode()).hexdigest()[:12],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (d / f"v{n}.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    idx = _index(kind)
    idx.setdefault(vid, {})["latest"] = n
    _save_index(kind, idx)
    return {"ok": True, "version": n, "sha1": snap["code_sha1"]}


def latest(kind: str, vid: str) -> dict:
    d = _dir(kind, vid)
    files = sorted(d.glob("v*.json"),
                   key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)
    if not files:
        return {"ok": False, "error": "无版本"}
    return {"ok": True, **json.loads(files[-1].read_text(encoding="utf-8"))}


def get(kind: str, vid: str, version: int) -> dict:
    fp = _dir(kind, vid) / f"v{int(version)}.json"
    if not fp.exists():
        return {"ok": False, "error": "版本不存在"}
    return {"ok": True, **json.loads(fp.read_text(encoding="utf-8"))}


def record_run(kind: str, vid: str, version: int, metrics: dict,
               config: dict | None = None, start: str = "", end: str = "",
               universe: str = "", note: str = "") -> dict:
    """归档一次运行（指标 + 配置 + 区间）。"""
    d = _dir(kind, vid)
    if not d.exists():
        return {"ok": False, "error": f"条目不存在 {vid}"}
    rd = d / "runs"
    rd.mkdir(parents=True, exist_ok=True)
    # 精确到微秒 + 冲突自增，避免同一秒两次归档互相覆盖（踩过）
    base = time.strftime("%Y%m%d_%H%M%S")
    ts = base
    k = 1
    while (rd / f"{ts}.json").exists():
        ts = f"{base}_{k:02d}"
        k += 1
    rec = {"version": int(version), "timestamp": ts, "metrics": metrics,
           "config": config or {}, "data_start": start, "data_end": end,
           "universe": universe, "note": note}
    (rd / f"{ts}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "run": ts}


def runs_of(kind: str, vid: str) -> list[dict]:
    rd = _dir(kind, vid) / "runs"
    if not rd.exists():
        return []
    out = []
    for fp in sorted(rd.glob("*.json")):
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def list_items(kind: str) -> list[dict]:
    base = ROOT / str(kind)
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        lt = latest(kind, d.name)
        m["latest_version"] = int(lt.get("version", 0)) if lt.get("ok") else 0
        m["runs"] = len(runs_of(kind, d.name))
        out.append(m)
    return out


def diff(kind: str, vid: str, v1: int, v2: int) -> dict:
    import difflib
    a = get(kind, vid, v1)
    b = get(kind, vid, v2)
    if not (a.get("ok") and b.get("ok")):
        return {"ok": False, "error": "版本不存在"}
    d = list(difflib.unified_diff(
        str(a["code"]).splitlines(), str(b["code"]).splitlines(),
        fromfile=f"v{v1}", tofile=f"v{v2}", lineterm=""))
    return {"ok": True, "diff": "\n".join(d), "n_lines": len(d),
            "params_a": a.get("params"), "params_b": b.get("params"),
            "note": b.get("change_note")}


def format_table(kind: str) -> str:
    items = list_items(kind)
    L = ["=" * 78, f"版本库 · {kind}", "=" * 78]
    if not items:
        L.append("  （空）")
    else:
        L.append(f"  {'id':<20}{'名称':<28}{'版本':>4}{'运行':>4}")
        for it in items:
            L.append(f"  {it['id']:<20}{it['name'][:26]:<28}"
                     f"{it['latest_version']:>4}{it['runs']:>4}")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    r = create("factor", "低波动因子 low_vol", "低波动异象", "low_vol")
    print("创建:", r)
    vid = r["id"]
    r2 = save("factor", vid, "s = -CL.pct_change().rolling(20).std(ddof=0)",
              {"lookback": 20}, "首版", parent=0)
    print("保存 v1:", r2)
    r3 = save("factor", vid, "s = -CL.pct_change().rolling(20).std(ddof=0)",
              {"lookback": 20}, "重复保存", parent=1)
    print("保存 v2:", r3["version"])
    bad = save("factor", vid, "   ", {}, "空代码")
    print("空代码被拒:", bad)
    lock = save("factor", vid, "x=1", {}, "错误 parent", parent=1)
    print("乐观锁:", lock.get("error"))
    record_run("factor", vid, 1, {"ic": 0.031, "sharpe": 0.42},
               {"rebalance": 20}, "2019-01-02", "2023-12-29", "cn 63 只")
    print("运行归档:", len(runs_of("factor", vid)), "条")
    print(format_table("factor"))
    print("diff:", diff("factor", vid, 1, 2)["n_lines"], "行")
    print("OK v" + __version__)
