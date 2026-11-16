# -*- coding: utf-8 -*-
"""
全市场批量入库（M58）
====================================================
目标：把时点票池（5566 只，含 342 只退市股）灌进 Parquet 存储层。

设计（针对实测的网络不稳定）
----------------------------
1. **断点续传**：已入库的 code 自动跳过，中断后重跑继续
2. **限速**：默认每只间隔 0.25s，避免触发东财限流
3. **自适应退避**：连续失败 → 指数退避（最长 60s），避免对着挂掉的源空转
4. **源顺序可调**：A股默认 adata → baostock → zzshare → efinance → akshare
5. **进度落盘**：`.data/bulk_status.json`，随时可查
6. **预算控制**：`--limit N` 只跑 N 只，方便分批

用法
----
```
python backend/bulk_ingest.py --limit 200          # 先跑 200 只试吞吐
python backend/bulk_ingest.py                      # 跑完全部未入库的
python backend/bulk_ingest.py --delisted-only      # 只补退市股
python backend/bulk_ingest.py --status             # 看进度
```
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_STATUS = _ROOT / ".data" / "bulk_status.json"

# A股批量采集的源顺序（efinance/akshare 走东财，批量时最容易限流，放后面）
CN_ORDER = ["tushare", "efinance", "adata", "baostock", "akshare", "zzshare"]
OTHER_ORDER = {
    "hk": ["efinance", "akshare", "adata"],
    "us": ["efinance", "akshare", "yfinance", "roicai"],
}


def _load_status() -> dict:
    if _STATUS.exists():
        try:
            return json.loads(_STATUS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": [], "failed": {}, "started": None, "updated": None}


def _save_status(st: dict) -> None:
    _STATUS.parent.mkdir(parents=True, exist_ok=True)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _STATUS.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def _targets(market: str = "cn", delisted_only: bool = False,
             listed_only: bool = False) -> list[str]:
    """待入库清单（时点票池全量）。"""
    import universe_pit as UP
    basic = UP.load_basic()
    b = basic.copy()
    if delisted_only:
        b = b[b["out_date"].notna()]
    elif listed_only:
        b = b[b["out_date"].isna()]
    return sorted(b["code"].astype(str).str.zfill(6).tolist())


def run(market: str = "cn", limit: int | None = None,
        sleep_s: float = 0.25, delisted_only: bool = False,
        listed_only: bool = False, reset: bool = False) -> dict:
    """执行批量入库（可续跑）。"""
    import feed as FD
    import store

    if reset:
        _save_status({"done": [], "failed": {}, "started": None})
    st = _load_status()
    done = set(st.get("done") or [])

    codes = _targets(market, delisted_only, listed_only)
    # 已有数据的也算完成（避免重复拉）
    have = set(store.list_codes(market))
    todo = [c for c in codes if c not in done and c not in have]
    if limit:
        todo = todo[:int(limit)]

    print(f"目标 {len(codes)} 只；已入库 {len(have)}；本轮待办 {len(todo)}",
          flush=True)
    if not todo:
        return {"ok": True, "done": 0, "msg": "无待办"}

    # A股源顺序覆盖
    if market == "cn":
        FD.MARKETS["cn"]["sources"] = CN_ORDER
        FD.SOURCES["cn"] = CN_ORDER

    ok = fail = 0
    consec_fail = 0
    t0 = time.time()
    failed_this = {}

    def _progress(i: int) -> None:
        el = time.time() - t0
        rate = i / el if el > 0 else 0
        eta = (len(todo) - i) / rate / 60 if rate > 0 else 0
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ok={ok} fail={fail}  "
                  f"{rate*60:.0f} 只/分  剩余约 {eta:.0f} 分", flush=True)

    for i, code in enumerate(todo, 1):
        try:
            r = FD.get_daily(market, code, refresh=True)
            if r.get("ok"):
                w = store.write_bars(market, code, r["df"],
                                     source=str(r.get("source")))
                if w.get("ok"):
                    ok += 1
                    consec_fail = 0
                    st.setdefault("done", []).append(code)
                else:
                    fail += 1
                    failed_this[code] = "store"
            else:
                fail += 1
                consec_fail += 1
                failed_this[code] = (r.get("error") or "")[:80]
        except Exception as ex:
            fail += 1
            consec_fail += 1
            failed_this[code] = f"{type(ex).__name__}"

        # 自适应退避
        if consec_fail >= 8:
            back = min(60, 2 ** min(consec_fail - 6, 6))
            print(f"    连续失败 {consec_fail}，退避 {back}s", flush=True)
            time.sleep(back)
        else:
            time.sleep(float(sleep_s))

        if i % 25 == 0:
            st["failed"] = failed_this
            _save_status(st)
        _progress(i)

    st["failed"] = failed_this
    _save_status(st)
    el = time.time() - t0
    return {"ok": True, "done": ok, "fail": fail, "elapsed": round(el, 1),
            "rate_per_min": round(ok / el * 60, 1) if el > 0 else 0,
            "todo": len(todo)}


def status() -> dict:
    """当前进度。"""
    import store
    st = _load_status()
    cnt = {}
    for m in ("cn", "hk", "us"):
        try:
            cnt[m] = len(store.list_codes(m))
        except Exception:
            cnt[m] = 0
    return {"stored": cnt, "done": len(st.get("done") or []),
            "failed": len(st.get("failed") or {}),
            "updated": st.get("updated")}


def run_threaded(market: str = "cn", limit: int | None = None,
                 workers: int = 4, sleep_s: float = 0.1,
                 delisted_only: bool = False,
                 reset: bool = False) -> dict:
    """
    并发版：HTTP 源（efinance/adata）用线程池，失败后再串行兜底。

    为什么分两段：baostock 是**全局会话**，不能并发；
    而 efinance/adata 是普通 HTTP，可以并行。
    """
    import feed as FD
    import store
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if reset:
        _save_status({"done": [], "failed": {}, "started": None})
    st = _load_status()
    done = set(st.get("done") or [])
    have = set(store.list_codes(market))
    codes = _targets(market, delisted_only)
    todo = [c for c in codes if c not in done and c not in have]
    if limit:
        todo = todo[:int(limit)]
    print(f"目标 {len(codes)}；已入库 {len(have)}；本轮 {len(todo)}；"
          f"并发 {workers}", flush=True)
    if not todo:
        return {"ok": True, "done": 0, "msg": "无待办"}

    ok = 0
    leftovers: list[str] = []
    t0 = time.time()

    def _one(code: str):
        for src in ("efinance", "adata"):
            fn = FD.SRC_FN.get(src)
            if fn is None:
                continue
            try:
                df = fn(market, code)
                if df is not None and len(df) > 20:
                    return code, df, src
            except Exception:
                continue
        return code, None, None

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(_one, c): c for c in todo}
        i = 0
        for fu in as_completed(futs):
            i += 1
            try:
                code, df, src = fu.result()
            except Exception:
                code, df, src = futs[fu], None, None
            if df is not None:
                w = store.write_bars(market, code, df, source=str(src))
                if w.get("ok"):
                    ok += 1
                    st.setdefault("done", []).append(code)
                else:
                    leftovers.append(code)
            else:
                leftovers.append(code)
            if i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ok={ok} 剩余 {len(leftovers)}  "
                      f"{i/el*60:.0f} 只/分", flush=True)
            time.sleep(float(sleep_s))
            if i % 100 == 0:
                _save_status(st)

    print(f"第一轮完成：成功 {ok}，待串行兜底 {len(leftovers)}", flush=True)

    # 串行兜底（baostock/akshare/zzshare）
    ok2 = 0
    for j, code in enumerate(leftovers, 1):
        try:
            r = FD.get_daily(market, code, refresh=True)
            if r.get("ok"):
                w = store.write_bars(market, code, r["df"], source=str(r.get("source")))
                if w.get("ok"):
                    ok2 += 1
                    st.setdefault("done", []).append(code)
                    st.setdefault("failed", {}).pop(code, None)
                    continue
            st.setdefault("failed", {})[code] = "all-sources-failed"
        except Exception as ex:
            st.setdefault("failed", {})[code] = type(ex).__name__
        if j % 25 == 0:
            print(f"  兜底 {j}/{len(leftovers)} 补回 {ok2}", flush=True)
        time.sleep(0.2)

    _save_status(st)
    el = time.time() - t0
    return {"ok": True, "done": ok + ok2, "first_pass": ok,
            "fallback": ok2, "failed": len(st.get("failed") or {}),
            "elapsed": round(el, 1),
            "rate_per_min": round((ok + ok2) / el * 60, 1) if el > 0 else 0}



if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(_ROOT / "backend"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="cn", choices=["cn", "hk", "us"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--delisted-only", action="store_true")
    ap.add_argument("--listed-only", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--threaded", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    if a.status:
        print(status())
    elif a.threaded:
        r = run_threaded(a.market, a.limit, a.workers, a.sleep,
                         a.delisted_only, a.reset)
        print(r)
        print("进度:", status())
    else:
        r = run(a.market, a.limit, a.sleep, a.delisted_only,
                a.listed_only, a.reset)
        print(r)
        print("进度:", status())

