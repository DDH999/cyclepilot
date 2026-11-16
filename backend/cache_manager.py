# -*- coding: utf-8 -*-
"""
缓存治理（M28）
====================================================
长期跑会踩的坑：`.cache/` 无限增长（实测已到 1.1 GB），
其中两个 panel 缓存就占 425 MB，还有上千个单票 CSV。

本模块提供：统计、按龄清理、全清。

安全约束
--------
* **只在 `.cache/` 内操作**，绝不碰 `reports/`、`.state/`、持仓台账
* 默认 dry-run，必须显式 `dry_run=False` 才真删
* 删除前把清单返回给调用方，可核对
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["CACHE_ROOT", "PROTECTED", "stats", "trim", "clean_all",
           "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = _ROOT / ".cache"

# 这些不属于缓存，删了会丢数据
PROTECTED = {".state", "reports", "holdings.csv", "dca_ledger.csv",
             "watchlist.txt", "backend", "frontend", "tests", "docs"}

CATEGORIES = {
    "panel 缓存": "*.pkl",
    "行情 CSV": "*_cn_*.csv",
    "海外行情": "*_us_*.csv",
    "基金净值": "fund/**/*.csv",
    "基金名录": "fund/catalog.json",
    "估值缓存": "valuation/*.csv",
    "基准缓存": "bench_*.csv",
    "价格缓存": "last_price.json",
}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _safe_under_cache(p: Path, root: Path | None = None) -> bool:
    """
    安全检查：p 必须位于 root（默认全局 CACHE_ROOT）之内。

    注意 root 必须参与判断 —— 否则传入自定义 root 做演练时，
    所有文件都会被「不在 .cache 内」挡掉，永远删不掉（踩过）。
    """
    base = (root or CACHE_ROOT).resolve()
    try:
        p.resolve().relative_to(base)
        return True
    except Exception:
        return False


def stats(root: Path | str | None = None) -> dict:
    """统计缓存体积，按类别拆分。"""
    root = Path(root) if root else CACHE_ROOT
    if not root.exists():
        return {"ok": True, "exists": False, "total_bytes": 0,
                "total_human": "0 B", "files": 0, "categories": {}}

    cats: dict[str, dict] = {}
    total_files = 0
    total_bytes = 0

    for name, pat in CATEGORIES.items():
        files = [f for f in root.glob(pat) if f.is_file()]
        if not files:
            continue
        size = sum(f.stat().st_size for f in files)
        oldest = min((f.stat().st_mtime for f in files), default=time.time())
        cats[name] = {"files": len(files), "bytes": size,
                      "human": _human(size),
                      "oldest_days": round((time.time() - oldest) / 86400, 1)}
        total_files += len(files)
        total_bytes += size

    # 其余未归类文件
    known = set()
    for pat in CATEGORIES.values():
        for f in root.glob(pat):
            known.add(f.resolve())
    others = [f for f in root.rglob("*")
              if f.is_file() and f.resolve() not in known]
    if others:
        size = sum(f.stat().st_size for f in others)
        cats["其他"] = {"files": len(others), "bytes": size,
                        "human": _human(size),
                        "oldest_days": round(
                            (time.time() - min(f.stat().st_mtime
                                               for f in others)) / 86400, 1)}
        total_files += len(others)
        total_bytes += size

    return {"ok": True, "exists": True, "root": str(root),
            "total_bytes": total_bytes, "total_human": _human(total_bytes),
            "files": total_files, "categories": cats}


def trim(max_age_days: float = 30.0, root: Path | str | None = None,
         dry_run: bool = True) -> dict:
    """
    删除超过 max_age_days 未访问的缓存文件。

    默认 dry-run：只报告会删什么，不动手。
    """
    root = Path(root) if root else CACHE_ROOT
    if not root.exists():
        return {"ok": True, "removed": 0, "freed_bytes": 0,
                "freed_human": "0 B", "dry_run": dry_run, "files": []}

    cutoff = time.time() - float(max_age_days) * 86400
    victims: list[Path] = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if not _safe_under_cache(f, root):
            continue
        try:
            if f.stat().st_mtime < cutoff:
                victims.append(f)
        except Exception:
            continue

    freed = sum(f.stat().st_size for f in victims)
    removed = 0
    if not dry_run:
        for f in victims:
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass

    return {"ok": True, "dry_run": dry_run,
            "cutoff_days": max_age_days,
            "candidates": len(victims),
            "removed": removed if not dry_run else 0,
            "freed_bytes": freed, "freed_human": _human(freed),
            "files": [str(f.relative_to(root)) for f in victims[:50]]}


def clean_all(root: Path | str | None = None, dry_run: bool = True) -> dict:
    """清空整个缓存目录（保留目录本身）。"""
    root = Path(root) if root else CACHE_ROOT
    if not root.exists():
        return {"ok": True, "removed": 0, "freed_human": "0 B", "dry_run": dry_run}

    total = 0
    n = 0
    for f in root.rglob("*"):
        if f.is_file() and _safe_under_cache(f, root):
            try:
                total += f.stat().st_size
                n += 1
            except Exception:
                pass

    if not dry_run:
        for child in list(root.iterdir()):
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except Exception:
                pass

    return {"ok": True, "dry_run": dry_run,
            "files": n, "freed_bytes": total, "freed_human": _human(total)}


def format_report(st: dict) -> str:
    L = ["=" * 66, "缓存统计", "=" * 66]
    if not st.get("exists"):
        L.append("  缓存目录不存在（还没跑过取数）")
        L.append("=" * 66)
        return "\n".join(L)

    L.append(f"  位置 {st['root']}")
    L.append(f"  合计 {st['total_human']}   文件 {st['files']} 个")
    L.append("-" * 66)
    L.append("  " + f"{'类别':<12}{'文件':>8}{'体积':>12}{'最旧(天)':>12}")
    L.append("  " + "-" * 56)
    rows = sorted(st["categories"].items(), key=lambda kv: -kv[1]["bytes"])
    for name, c in rows:
        L.append("  " + f"{name:<12}{c['files']:>8}{c['human']:>12}"
                 f"{c['oldest_days']:>12.1f}")
    L.append("-" * 66)
    L.append("  建议：超过 30 天未用的缓存可清理（`quantpilot.py cache --trim 30`）")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    st = stats()
    print(format_report(st))

    d = trim(30, dry_run=True)
    print(f"\n干跑清理（>30 天）：候选 {d['candidates']} 个，"
          f"可释放 {d['freed_human']}")
    assert d["dry_run"] is True and d["removed"] == 0

    # 保护性检查：绝不在 .cache 之外删东西
    outside = _ROOT / "reports" / "paper_nav.csv"
    assert not _safe_under_cache(outside), "不得被认为在缓存内"
    print("保护性检查：reports/ 不在可删范围 ✓")

    # 小目录演练
    (_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    demo = _ROOT / ".tmp" / "_cache_demo"
    demo.mkdir(exist_ok=True)
    (demo / "a.csv").write_text("x", encoding="utf-8")
    import os
    old = time.time() - 40 * 86400
    os.utime(demo / "a.csv", (old, old))
    r1 = trim(30, root=demo, dry_run=True)
    assert r1["candidates"] == 1, r1
    assert (demo / "a.csv").exists(), "dry-run 不应删除文件"
    r2 = trim(30, root=demo, dry_run=False)
    assert r2["removed"] == 1
    print(f"演练：dry-run 候选 {r1['candidates']} → 实删 {r2['removed']} ✓")
    shutil.rmtree(demo, ignore_errors=True)
    print(f"OK v{__version__}")
