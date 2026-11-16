# -*- coding: utf-8 -*-
"""
数据质量监控（M56）
====================================================
逐票体检 + 全库汇总。**只标记，不自动删改**——数据是资产。

检查项
------
| 级别 | 检查 | 阈值 |
|---|---|---|
| 致命 | 空数据 / 行数过少 | < 30 行 |
| 致命 | 日期重复 | > 0 |
| 致命 | 日期非单调 | 递减 |
| 致命 | 非正价格 | close <= 0 |
| 严重 | OHLC 逻辑错（high < max(o,c) 等） | > 0 |
| 严重 | 单日跳变（拆股/错价） | abs(ret) > 21% |
| 严重 | 缺口过大 | > 15 自然日 |
| 严重 | 数据陈旧 | 末日期早于 7 天 |
| 一般 | 零成交占比 | > 10% |
| 一般 | 字段缺失（amount） | 全空 |
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["LIMITS", "check_frame", "check_market", "format_report",
           "check_store"]

_ROOT = Path(__file__).resolve().parent.parent

LIMITS = {
    "min_rows": 30,
    "jump_pct": 0.21,
    "max_gap_days": 15,
    "max_stale_days": 7,
    "zero_vol_ratio": 0.10,
}

LEVEL_ORDER = {"fatal": 0, "severe": 1, "minor": 2}


def _issue(kind: str, level: str, msg: str) -> dict:
    return {"kind": kind, "level": level, "msg": msg}


def check_frame(df: pd.DataFrame, code: str = "", market: str = "") -> dict:
    """单票体检。返回 {ok, issues, n, start, end}。"""
    issues: list[dict] = []
    if df is None or len(df) == 0:
        return {"ok": False, "code": code, "market": market, "n": 0,
                "issues": [_issue("empty", "fatal", "空数据")]}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    n = len(d)

    if n < LIMITS["min_rows"]:
        issues.append(_issue("few_rows", "fatal", f"仅 {n} 行"))

    # 日期
    dup = int(d["date"].duplicated().sum())
    if dup:
        issues.append(_issue("dup_date", "fatal", f"重复日期 {dup} 条"))
    dd = d["date"].dropna()
    if len(dd) > 1 and not dd.is_monotonic_increasing:
        issues.append(_issue("unsorted", "fatal", "日期非单调递增"))

    # 价格
    for c in ("open", "high", "low", "close"):
        if c not in d.columns:
            issues.append(_issue("missing_col", "fatal", f"缺列 {c}"))
    if all(c in d.columns for c in ("open", "high", "low", "close")):
        px = d[["open", "high", "low", "close"]].apply(
            pd.to_numeric, errors="coerce")
        bad = int((px["close"] <= 0).sum())
        if bad:
            issues.append(_issue("nonpos", "fatal", f"收盘价 <=0 共 {bad} 条"))
        hi_bad = int((px["high"] < px[["open", "close"]].max(axis=1) - 1e-6).sum())
        lo_bad = int((px["low"] > px[["open", "close"]].min(axis=1) + 1e-6).sum())
        if hi_bad or lo_bad:
            issues.append(_issue("ohlc_logic", "severe",
                                 f"OHLC 逻辑错 high {hi_bad} / low {lo_bad} 条"))

        # 跳变
        r = px["close"].pct_change(fill_method=None)
        jumps = r.abs() > LIMITS["jump_pct"]
        nj = int(jumps.sum())
        if nj:
            worst = float(r[jumps].abs().max())
            issues.append(_issue("jump", "severe",
                                 f"单日跳变 >21% 共 {nj} 条，最大 {worst*100:.1f}%"))

    # 缺口
    if len(dd) > 1:
        gaps = dd.sort_values().diff().dt.days.dropna()
        mg = int(gaps.max()) if len(gaps) else 0
        if mg > LIMITS["max_gap_days"]:
            issues.append(_issue("gap", "severe", f"最长缺口 {mg} 天"))

    # 陈旧
    if len(dd):
        stale = int((pd.Timestamp.now().normalize()
                     - dd.max().normalize()).days)
        if stale > LIMITS["max_stale_days"]:
            issues.append(_issue("stale", "severe", f"末日期滞后 {stale} 天"))

    # 零成交
    if "volume" in d.columns:
        v = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
        zr = float((v == 0).mean())
        if zr > LIMITS["zero_vol_ratio"]:
            issues.append(_issue("zero_vol", "minor",
                                 f"零成交占 {zr*100:.1f}%"))

    # amount
    if "amount" in d.columns:
        a = pd.to_numeric(d["amount"], errors="coerce")
        if a.notna().sum() == 0:
            issues.append(_issue("no_amount", "minor", "amount 全空"))

    ok = not any(i["level"] == "fatal" for i in issues)
    return {"ok": ok, "code": code, "market": market, "n": n,
            "start": (str(dd.min().date()) if len(dd) else None),
            "end": (str(dd.max().date()) if len(dd) else None),
            "issues": issues}


def check_market(market: str) -> dict:
    """全市场体检（读 Parquet 存储层）。"""
    import store
    codes = store.list_codes(market)
    if not codes:
        return {"ok": False, "market": market, "reason": "该市场无数据"}
    rows = []
    for c in codes:
        df = store.read_bars(market, c)
        r = check_frame(df, c, market)
        r["levels"] = [i["level"] for i in r["issues"]]
        rows.append(r)
    t = pd.DataFrame([{
        "code": r["code"], "n": r["n"], "ok": r["ok"],
        "fatal": r["levels"].count("fatal"),
        "severe": r["levels"].count("severe"),
        "minor": r["levels"].count("minor"),
        "start": r.get("start"), "end": r.get("end"),
    } for r in rows])
    return {"ok": True, "market": market, "n_codes": len(t),
            "n_ok": int(t["ok"].sum()),
            "n_fatal": int((t["fatal"] > 0).sum()),
            "n_severe": int((t["severe"] > 0).sum()),
            "detail": t, "raw": rows}


def check_store() -> dict:
    """全库体检。"""
    import store
    markets = [p.name for p in (store.PARQUET).iterdir() if p.is_dir()] \
        if store.PARQUET.exists() else []
    out = {}
    for m in sorted(markets):
        out[m] = check_market(m)
    return {"ok": True, "markets": out, "stats": store.stats()}


def format_report(res: dict) -> str:
    L = ["=" * 74, "数据质量报告", "=" * 74]
    if res.get("market") and not res.get("markets"):
        # 单市场
        if not res.get("ok"):
            L.append(f"  {res.get('reason')}")
            L.append("=" * 74)
            return "\n".join(L)
        L.append(f"  {res['market']}: {res['n_codes']} 只，"
                 f"健康 {res['n_ok']}，有致命 {res['n_fatal']}，"
                 f"有严重 {res['n_severe']}")
        d = res["detail"]
        bad = d[(d["fatal"] > 0) | (d["severe"] > 0)]
        if len(bad):
            L.append("-" * 74)
            L.append(bad.to_string(index=False))
        L.append("=" * 74)
        return "\n".join(L)

    st = res.get("stats") or {}
    L.append(f"  总标的 {st.get('total_codes', 0)}，"
             f"总行数 {st.get('total_rows', 0):,}，"
             f"占用 {st.get('mb', 0)} MB")
    L.append("-" * 74)
    for m, r in (res.get("markets") or {}).items():
        if not r.get("ok"):
            L.append(f"  {m:<8} {r.get('reason')}")
            continue
        L.append(f"  {m:<8} {r['n_codes']:>4} 只   健康 {r['n_ok']:>4}   "
                 f"致命 {r['n_fatal']:>3}   严重 {r['n_severe']:>3}")
    L.append("=" * 74)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=200)
    c = np.linspace(10, 12, 200)
    good = pd.DataFrame({"date": idx, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c,
                         "volume": 1e6, "amount": 1e7})
    r = check_frame(good, "000001")
    print(f"正常数据: ok={r['ok']} 问题 {len(r['issues'])} 项")
    assert r["ok"] and not r["issues"]

    bad = good.copy()
    bad.loc[10, "close"] = -1
    bad.loc[20, "close"] = bad.loc[19, "close"] * 1.5      # 跳变
    bad = pd.concat([bad, bad.iloc[[30]]], ignore_index=True)  # 重复
    bad.loc[40, "volume"] = 0
    r2 = check_frame(bad, "000002")
    kinds = {i["kind"] for i in r2["issues"]}
    print(f"问题数据: ok={r2['ok']} 命中 {sorted(kinds)}")
    assert not r2["ok"]
    for k in ("nonpos", "dup_date", "jump"):
        assert k in kinds, k

    empty = check_frame(pd.DataFrame(), "x")
    assert not empty["ok"]
    print("空数据正确判死 ✓")
    print("OK v" + __version__)
