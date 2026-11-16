# -*- coding: utf-8 -*-
"""
采集入库流水线（M57）
====================================================
多源采集 → 质量体检 → Parquet 入库，全程记录来源与拒收原因。

策略
----
* **致命问题** → 拒收（不入库），记录原因
* **严重/一般问题** → 入库但标记，写进质量报告
* **来源** 逐票记录到 Parquet 的 `source` 列与 meta 清单
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"
__all__ = ["ingest", "ingest_market", "format_result"]

_ROOT = Path(__file__).resolve().parent.parent


def ingest(market: str, code: str, refresh: bool = False) -> dict:
    """采集一只并入库。"""
    import data_quality as DQ
    import feed as FD
    import store

    t0 = time.time()
    r = FD.get_daily(market, code, refresh=refresh)
    if not r.get("ok"):
        return {"market": market, "code": code, "ok": False,
                "stage": "fetch", "error": (r.get("error") or "")[:120],
                "attempts": r.get("attempts") or [],
                "elapsed": round(time.time() - t0, 1)}

    df = r["df"]
    q = DQ.check_frame(df, code, market)
    if not q["ok"]:
        return {"market": market, "code": code, "ok": False,
                "stage": "quality", "error": "致命问题，拒收",
                "issues": q["issues"], "source": r.get("source"),
                "elapsed": round(time.time() - t0, 1)}

    w = store.write_bars(market, code, df, source=str(r.get("source")))
    if not w.get("ok"):
        return {"market": market, "code": code, "ok": False,
                "stage": "store", "error": str(w.get("error"))[:120],
                "elapsed": round(time.time() - t0, 1)}

    trimmed = int(getattr(df, "attrs", {}).get("trimmed_rows", 0) or 0)
    return {"market": market, "code": code, "ok": True,
            "n": w["n"], "added": w["added"], "source": r.get("source"),
            "trimmed": trimmed,
            "trimmed_before": getattr(df, "attrs", {}).get("trimmed_before"),
            "issues": q["issues"],
            "warn": len([i for i in q["issues"] if i["level"] != "minor"]),
            "elapsed": round(time.time() - t0, 1)}


def ingest_market(market: str, codes: list[str], refresh: bool = False,
                  stop_after_fails: int = 20) -> dict:
    """
    批量采集。

    stop_after_fails：连续失败 N 只就停（防止对着挂掉的源空转）。
    """
    results = []
    consec = 0
    for i, c in enumerate(codes, 1):
        r = ingest(market, c, refresh=refresh)
        results.append(r)
        if r["ok"]:
            consec = 0
        else:
            consec += 1
            if consec >= stop_after_fails:
                results.append({"market": market, "code": "-", "ok": False,
                                "stage": "abort",
                                "error": f"连续失败 {consec} 只，中止"})
                break
        if i % 20 == 0:
            print(f"    {i}/{len(codes)} …", flush=True)
    ok = [r for r in results if r.get("ok")]
    return {"ok": len(ok) > 0, "market": market,
            "n_req": len(codes), "n_ok": len(ok),
            "n_fail": len(results) - len(ok),
            "added": sum(r.get("added", 0) for r in ok),
            "detail": results}


def format_result(res: dict) -> str:
    L = ["=" * 74, f"采集结果：{res.get('market')}", "=" * 74]
    if not res.get("ok"):
        L.append(f"  全部失败（{res.get('n_fail')} 只）")
    else:
        L.append(f"  请求 {res['n_req']}，成功 {res['n_ok']}，"
                 f"失败 {res['n_fail']}，新增 {res['added']} 行")
    bad = [r for r in res.get("detail") or [] if not r.get("ok")]
    if bad:
        L.append("-" * 74)
        L.append("  失败明细（前 10 条）")
        for r in bad[:10]:
            L.append(f"    {r.get('code'):<10} [{r.get('stage')}] "
                     f"{str(r.get('error'))[:60]}")
    src = {}
    for r in res.get("detail") or []:
        if r.get("ok"):
            src[r.get("source")] = src.get(r.get("source"), 0) + 1
    if src:
        L.append("-" * 74)
        L.append("  来源分布：" + "，".join(f"{k}×{v}" for k, v in src.items()))
    L.append("=" * 74)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== A股 5 只 ===")
    a = ingest_market("cn", ["600519", "600036", "000001", "000858", "601318"])
    print(format_result(a))

    print("\n=== 港股 3 只 ===")
    b = ingest_market("hk", ["00700", "09988", "03690"])
    print(format_result(b))

    print("\n=== 美股 3 只 ===")
    c = ingest_market("us", ["AAPL", "MSFT", "NVDA"])
    print(format_result(c))
