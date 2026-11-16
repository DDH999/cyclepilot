# -*- coding: utf-8 -*-
"""
数据存储层（M55）：Parquet 分区 + DuckDB 查询
====================================================
为什么不用 ClickHouse
--------------------
* 本地单机，数据量约 4000 只 × 5000 天 ≈ 2000 万行 → Parquet + DuckDB 完全够
* 不需要常驻服务、不需要 Docker，`pip install duckdb` 即用
* DuckDB 能**直接查 Parquet**，零导入

目录结构
--------
```
.data/parquet/<market>/<year>/<code>.parquet   逐票逐年分区
.data/meta/<market>.json                        清单（代码/行数/区间/来源）
```

写入口径（幂等）
----------------
`write_bars` 按 `date` 去重合并（keep=last），重复 ingest 不会翻倍，
且新数据覆盖旧数据同一日期。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["ROOT", "STD_COLS", "write_bars", "read_bars", "list_codes",
           "coverage", "query", "delete_code", "stats"]

_ROOT = Path(__file__).resolve().parent.parent
ROOT = _ROOT / ".data"
PARQUET = ROOT / "parquet"
LONG = ROOT / "parquet_long"   # 长表：一年一文件，含全部代码
META = ROOT / "meta"

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
# pct_chg 可选：Tushare 来源会带，用于**读取时**重建复权价
EXTRA_COLS = ["pct_chg"]
FULL_COLS = STD_COLS + EXTRA_COLS + ["source"]


def _market_dir(market: str) -> Path:
    return PARQUET / str(market).lower()


def _meta_path(market: str) -> Path:
    META.mkdir(parents=True, exist_ok=True)
    return META / f"{str(market).lower()}.json"


def _load_meta(market: str) -> dict:
    fp = _meta_path(market)
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"market": market, "codes": {}, "updated": None}


def _save_meta(market: str, meta: dict) -> None:
    meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _meta_path(market).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ══════════════════════════════════════════════════════════
def write_bars(market: str, code: str, df: pd.DataFrame,
               source: str = "unknown") -> dict:
    """
    写入/更新一只标的的日线（按 date 幂等合并）。

    返回 {ok, n, path, added}
    """
    if df is None or len(df) == 0:
        return {"ok": False, "error": "空数据"}
    d = df.copy()
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c not in d.columns:
            d[c] = np.nan
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    cols = list(STD_COLS)
    for c in EXTRA_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            cols.append(c)
    d = d.dropna(subset=["date", "close"])[cols]
    d["date"] = d["date"].dt.normalize()
    d["source"] = source
    d = d.sort_values("date").drop_duplicates("date", keep="last")

    prev_n = len(read_bars(market, code))
    md = _market_dir(market)
    md.mkdir(parents=True, exist_ok=True)
    # 按年分区写
    d["_y"] = d["date"].dt.year
    added = 0
    paths = []
    for y, g in d.groupby("_y"):
        fp = md / str(int(y)) / f"{code}.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        g = g.drop(columns=["_y"])
        if fp.exists():
            try:
                old = pd.read_parquet(fp)
                n0 = len(old)
                merged = pd.concat([old, g], ignore_index=True)
                merged = merged.drop_duplicates("date", keep="last") \
                    .sort_values("date")
                added += max(0, len(merged) - n0)
                g = merged
            except Exception:
                pass
        g.to_parquet(fp, index=False, compression="snappy")
        paths.append(str(fp))
    total = len(read_bars(market, code))
    added = max(0, total - prev_n)

    meta = _load_meta(market)
    sub = read_bars(market, code)
    meta["codes"][code] = {
        "n": total,
        "start": str(sub["date"].min().date()) if len(sub) else None,
        "end": str(sub["date"].max().date()) if len(sub) else None,
        "source": source,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_meta(market, meta)
    return {"ok": True, "n": total, "added": added, "paths": paths}


def read_bars(market: str, code: str, start: str | None = None,
              end: str | None = None, adjust: bool = True) -> pd.DataFrame:
    """
    读回一只标的（跨年份分区合并）。

    ⚠ 复权在**读取时**计算，不在写入时
    ----------------------------------
    写入只存"不复权价 + pct_chg"，读取时按
    `复权价 = 首日收盘 × cumprod(1 + pct_chg/100)` 重建。

    这样增量落盘天然正确：追加新数据不会改变已有行，
    也不需要"每块锚点一致"——之前按块锚定导致接缝错价。

    adjust=False 返回原始不复权价（调试用）。
    """
    # 优先读长表（批量灌库的新布局）
    _long = read_long(market, code)
    if len(_long):
        d = _long
        if adjust and "pct_chg" in d.columns and d["pct_chg"].notna().any():
            base0 = float(d["close"].iloc[0])
            chain = (1.0 + d["pct_chg"].fillna(0.0) / 100.0).cumprod()
            adj = base0 * chain
            raw = d["close"].astype(float)
            d = d.copy()
            d["close_raw"] = raw
            ratio = (adj / raw.replace(0, np.nan)).fillna(1.0)
            for c in ("open", "high", "low"):
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce") * ratio
            d["close"] = adj
        if start:
            d = d[d["date"] >= pd.Timestamp(start)]
        if end:
            d = d[d["date"] <= pd.Timestamp(end)]
        return d.reset_index(drop=True)

    md = _market_dir(market)
    if not md.exists():
        return pd.DataFrame(columns=FULL_COLS)
    frames = []
    for fp in sorted(md.glob(f"*/{code}.parquet")):
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=FULL_COLS)
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d.drop_duplicates("date", keep="last").sort_values("date")

    if adjust and "pct_chg" in d.columns and d["pct_chg"].notna().any():
        base = float(d["close"].iloc[0])
        chain = (1.0 + d["pct_chg"].fillna(0.0) / 100.0).cumprod()
        adj = base * chain
        raw = d["close"].astype(float)
        d["close_raw"] = raw
        ratio = (adj / raw.replace(0, np.nan)).fillna(1.0)
        for c in ("open", "high", "low"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce") * ratio
        d["close"] = adj

    if start:
        d = d[d["date"] >= pd.Timestamp(start)]
    if end:
        d = d[d["date"] <= pd.Timestamp(end)]
    return d.reset_index(drop=True)


def list_codes(market: str) -> list[str]:
    md = _market_dir(market)
    if not md.exists():
        return []
    return sorted({p.stem for p in md.glob("*/*.parquet")})


def coverage(market: str) -> pd.DataFrame:
    """覆盖度清单：每只行数/区间/来源/文件大小。"""
    md = _market_dir(market)
    rows = []
    if not md.exists():
        return pd.DataFrame(columns=["code", "n", "start", "end",
                                     "source", "kb"])
    # 同一只标的会分多个年份分区存，这里必须按 code 聚合，
    # 否则 300 天跨两年会显示成 2 只（踩过）。
    agg: dict[str, dict] = {}
    for fp in sorted(md.glob("*/*.parquet")):
        try:
            d = pd.read_parquet(fp)
        except Exception:
            continue
        c = fp.stem
        a = agg.setdefault(c, {"code": c, "n": 0, "start": None, "end": None,
                               "source": "?", "kb": 0.0})
        a["n"] += len(d)
        lo, hi = d["date"].min(), d["date"].max()
        a["start"] = lo if a["start"] is None else min(a["start"], lo)
        a["end"] = hi if a["end"] is None else max(a["end"], hi)
        if "source" in d.columns and len(d):
            a["source"] = d["source"].iloc[-1]
        a["kb"] = round(a["kb"] + fp.stat().st_size / 1024, 1)
    return pd.DataFrame(list(agg.values()))


def query(sql: str) -> pd.DataFrame:
    """
    用 DuckDB 直接查 Parquet（零导入），SQL 里用 `bars` 指代全市场。

    SELECT * FROM bars WHERE market='cn' AND code='600519' LIMIT 5
    """
    import duckdb
    con = duckdb.connect()
    pat = str(PARQUET / "*" / "*" / "*.parquet")
    # Windows 的 filename 是反斜杠，正则前统一成正斜杠（chr(92) 避免转义）
    norm = "replace(filename, chr(92), '/')"
    con.execute(
        "CREATE VIEW bars AS SELECT *, "
        f"regexp_extract({norm}, 'parquet/([^/]+)/', 1) AS market, "
        f"regexp_extract({norm}, '/([^/]+)[.]parquet$', 1) AS code "
        f"FROM read_parquet('{pat}', filename=true)"
    )
    return con.execute(sql).df()


def delete_code(market: str, code: str) -> dict:
    """删除一只标的的所有分区（谨慎使用）。"""
    md = _market_dir(market)
    n = 0
    for fp in md.glob(f"*/{code}.parquet"):
        try:
            fp.unlink()
            n += 1
        except Exception:
            pass
    meta = _load_meta(market)
    meta["codes"].pop(code, None)
    _save_meta(market, meta)
    return {"ok": True, "removed_files": n}


def stats() -> dict:
    """全局统计：市场数、标的数、总行数、磁盘占用。"""
    out = {"markets": {}, "total_codes": 0, "total_rows": 0, "mb": 0.0}
    if not PARQUET.exists():
        return out
    for md in sorted(PARQUET.iterdir()):
        if not md.is_dir():
            continue
        files = list(md.glob("*/*.parquet"))
        n_rows = 0
        size = 0
        for fp in files:
            size += fp.stat().st_size
            try:
                import pyarrow.parquet as pq
                n_rows += pq.ParquetFile(fp).metadata.num_rows
            except Exception:
                pass
        out["markets"][md.name] = {"codes": len({f.stem for f in files}),
                                   "rows": n_rows,
                                   "mb": round(size / 1024 / 1024, 2)}
        out["total_codes"] += len({f.stem for f in files})
        out["total_rows"] += n_rows
        out["mb"] += size / 1024 / 1024
    out["mb"] = round(out["mb"], 2)
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # 自测
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(7)
    c = 10 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300))
    df = pd.DataFrame({"date": idx, "open": c, "high": c * 1.01,
                       "low": c * 0.99, "close": c,
                       "volume": 1e6, "amount": 1e7})
    r1 = write_bars("test", "000001", df, source="selftest")
    print("写入:", r1["n"], "行，新增", r1["added"])
    r2 = write_bars("test", "000001", df, source="selftest")
    print("重复写入:", r2["n"], "行，新增", r2["added"], "（幂等，应为 0）")
    assert r2["added"] == 0 and r2["n"] == 300

    back = read_bars("test", "000001", start="2023-06-01")
    print("条件读取:", len(back), "行，起始", back["date"].min().date())
    assert len(back) < 300

    q = query("SELECT market, code, count(*) n FROM bars "
              "WHERE market='test' GROUP BY 1,2")
    print("DuckDB 查询:\n", q.to_string(index=False))
    assert len(q) == 1

    print("统计:", stats())
    delete_code("test", "000001")
    print("OK v" + __version__)

# ══════════════════════════════════════════════════════════
# 长表读写（批量灌库用）：一年一个 parquet，含所有代码
# 为什么需要：逐票一文件时 5400 只 × 7 年 = 3.8 万个小文件，
# 单次落盘 9 分钟；长表只有 7 个文件，写入几秒。
# ══════════════════════════════════════════════════════════
def write_long(market: str, df: pd.DataFrame) -> dict:
    """按年写入长表（累加合并，按 date+code 去重）。"""
    if df is None or len(df) == 0:
        return {"ok": False, "error": "空数据"}
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "close"])
    if "code" not in d.columns:
        return {"ok": False, "error": "缺 code 列"}
    d["code"] = d["code"].astype(str).str.zfill(6)
    base = LONG / str(market).lower()
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    for y, g in d.groupby(d["date"].dt.year):
        fp = base / f"{int(y)}.parquet"
        if fp.exists():
            try:
                old = pd.read_parquet(fp)
                g = pd.concat([old, g], ignore_index=True)
            except Exception:
                pass
        g = g.drop_duplicates(["date", "code"], keep="last").sort_values(
            ["date", "code"])
        g.to_parquet(fp, index=False, compression="snappy")
        n += len(g)
    return {"ok": True, "rows": n}


def read_long(market: str, code: str | None = None) -> pd.DataFrame:
    """读长表；给 code 则过滤。"""
    base = LONG / str(market).lower()
    if not base.exists():
        return pd.DataFrame()
    frames = []
    for fp in sorted(base.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    if code is not None:
        d = d[d["code"] == str(code).zfill(6)]
    return d.sort_values(["date", "code"]).reset_index(drop=True)

def load_panel(market: str = "cn", min_rows: int = 0,
               max_codes: int | None = None,
               adjust: bool = True) -> dict:
    """
    一次性加载整市场面板（{code: DataFrame}）。

    ⚠ 不要循环调 `read_bars`：长表下每次调用都会读全部年份文件，
    5000 只 = 5000 次全量读，几小时都跑不完。
    这里只读一遍、按 code 分组，10 秒量级。
    """
    base = LONG / str(market).lower()
    if not base.exists():
        return {}
    frames = []
    for fp in sorted(base.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:
            continue
    if not frames:
        return {}
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(["date", "code"], keep="last").sort_values(
        ["code", "date"])
    if adjust and "pct_chg" in d.columns and d["pct_chg"].notna().any():
        # 逐票复权：锚定各自首日收盘
        d["close_raw"] = d["close"].astype(float)
        g = d.groupby("code", sort=False)
        chain = (1.0 + d["pct_chg"].fillna(0.0) / 100.0)
        d["_chain"] = chain.groupby(d["code"]).cumprod()
        base_px = g["close_raw"].transform("first")
        adj = base_px * d["_chain"]
        ratio = (adj / d["close_raw"].replace(0, np.nan)).fillna(1.0)
        for c in ("open", "high", "low"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce") * ratio
        d["close"] = adj
        d = d.drop(columns=["_chain"])

    out = {}
    for code, gg in d.groupby("code", sort=True):
        if len(gg) < int(min_rows):
            continue
        out[str(code)] = gg.drop(columns=["code"]).reset_index(drop=True)
        if max_codes and len(out) >= int(max_codes):
            break
    return out
