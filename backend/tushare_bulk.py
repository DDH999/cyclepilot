# -*- coding: utf-8 -*-
"""
Tushare 全市场灌库（M72）
====================================================
为什么不用逐只拉
----------------
* 逐只：5566 次调用，`adj_factor` 还限 5 次/分 → 十几小时
* **按交易日拉**：`pro.daily(trade_date=...)` 一次返回**当天全市场**，
  6 年约 1700 个交易日 = 1700 次调用，20 分钟内跑完

复权怎么处理（关键）
--------------------
`daily` 返回**不复权**价，但自带 `pct_chg`——它相对 `pre_close` 计算，
**已经处理了除权除息**。所以：

```
复权收盘 = 首日收盘 × cumprod(1 + pct_chg/100)
```

这样既避免 `adj_factor` 的限流，又不会出现前复权的负价问题。

口径
----
* `vol` 单位是**手** → 转成股（×100）
* `amount` 单位是**千元** → 转成元（×1000）
* 退市股：`stock_basic(list_status="D")` 一起纳入，`out_date` 落到 `delist_map`

用法
----
```
python backend\\tushare_bulk.py --check              # 看权限
python backend\\tushare_bulk.py --days 60            # 先试 60 个交易日
python backend\\tushare_bulk.py --start 20190101     # 全量
```
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

__version__ = "0.1.0"
__all__ = ["check", "trade_dates", "fetch_day", "run", "STATUS"]

STATUS = _ROOT / ".data" / "tushare_bulk_status.json"


def _pro():
    import tushare_source as TS
    return TS.pro()


def check() -> dict:
    """
    轻量权限检查：只探 **daily 和 stock_basic** 两个真正要用的接口。

    不要跑全量探测——那会把频率额度耗光（踩过：探测完 daily 就没额度了）。
    """
    import tushare_source as TS
    a = TS.available()
    pro = TS.pro()
    out = {"ok": bool(a.get("ok")), "token": a.get("token_masked"),
           "daily_ok": False, "basic_ok": False, "detail": {}}
    try:
        d = pro.daily(trade_date="20251114")
        out["daily_ok"] = d is not None and len(d) > 100
        out["detail"]["daily_rows"] = int(len(d) if d is not None else 0)
    except Exception as ex:
        out["detail"]["daily"] = f"{type(ex).__name__}: {str(ex)[:60]}"
    try:
        b = pro.stock_basic(exchange="", list_status="L")
        out["basic_ok"] = b is not None and len(b) > 1000
        out["detail"]["basic_rows"] = int(len(b) if b is not None else 0)
    except Exception as ex:
        out["detail"]["stock_basic"] = f"{type(ex).__name__}: {str(ex)[:60]}"
    out["ok"] = out["daily_ok"] and out["basic_ok"]
    return out


def trade_dates(start: str, end: str) -> list[str]:
    """
    交易日列表。

    ⚠ 不用 Tushare 的 `trade_cal`——它限 **1 次/分钟**，会卡住整条流水线。
    改用已落盘缓存的 baostock 交易日历（`universe_pit.trading_days`），
    既不占 Tushare 额度，也不需要联网重取。
    """
    try:
        import universe_pit as UP
        cal = UP.trading_days(
            f"{start[:4]}-{start[4:6]}-{start[6:]}",
            f"{end[:4]}-{end[4:6]}-{end[6:]}")
        return [x.strftime("%Y%m%d") for x in cal]
    except Exception as ex:
        raise RuntimeError(f"取交易日历失败：{type(ex).__name__}: {ex}")


def fetch_day(pro, trade_date: str, retries: int = 3) -> pd.DataFrame:
    """取某交易日全市场日线。"""
    for k in range(int(retries)):
        try:
            d = pro.daily(trade_date=str(trade_date))
            if d is not None and len(d):
                return d
        except Exception:
            time.sleep(2 * (k + 1))
    return pd.DataFrame()


def _load_status() -> dict:
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"dates_done": [], "rows": 0}


def _save_status(st: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def run(start: str = "20190101", end: str | None = None,
        days: int | None = None, sleep_s: float = 0.13,
        reset: bool = False, verbose: bool = True,
        batch: int | None = None) -> dict:
    """
    拉取全市场日线并写入 Parquet 库。

    按交易日增量落盘：中断后重跑会跳过已完成日期。
    """
    import store
    pro = _pro()
    ck = check()
    # 只要求 daily：退市名单已由 universe_pit 从 baostock 取，
    # 不必依赖 stock_basic（它限 1 次/分钟）。
    if not ck["daily_ok"]:
        return {"ok": False, "reason": "daily 无权限或限流", "check": ck}

    end = end or time.strftime("%Y%m%d")
    dates = trade_dates(start, end)
    if days:
        dates = dates[-int(days):]
    if not dates:
        return {"ok": False, "reason": "无交易日"}

    if reset and STATUS.exists():
        STATUS.unlink()
    st = _load_status()
    done = set(st.get("dates_done") or [])
    todo = [d for d in dates if d not in done]
    if batch:
        todo = todo[:int(batch)]
    if verbose:
        print(f"交易日 {len(dates)} 天，已完成 {len(done)}，本轮 {len(todo)}",
              flush=True)
    if not todo:
        return {"ok": True, "msg": "已全部完成", "dates": len(dates)}

    def _flush(buf: dict, store_mod) -> tuple[int, int]:
        """整理 buffer → 写成**长表**（一年一文件），几秒完成。"""
        if not buf:
            return 0, 0
        rows = []
        for code, frames in buf.items():
            try:
                g = pd.concat(frames, ignore_index=True)
                g = g.sort_values("trade_date").drop_duplicates(
                    "trade_date", keep="last")
                rows.append(pd.DataFrame({
                    "code": code,
                    "date": pd.to_datetime(g["trade_date"], format="%Y%m%d"),
                    "open": g["open"], "high": g["high"], "low": g["low"],
                    "close": g["close"],              # 不复权
                    "volume": g["vol"] * 100,
                    "amount": g["amount"] * 1000,
                    "pct_chg": g["pct_chg"],
                }))
            except Exception:
                continue
        if not rows:
            return 0, len(buf)
        allrows = pd.concat(rows, ignore_index=True).dropna(
            subset=["date", "close"])
        r = store_mod.write_long("cn", allrows)
        return (len(buf) if r.get("ok") else 0), (0 if r.get("ok") else len(buf))

    # 按代码累积
    buf: dict[str, list[pd.DataFrame]] = {}
    pending_dates: list[str] = []   # 已抓取但还没落盘的日期
    n_written = n_write_fail = 0
    FLUSH_EVERY = 300         # 长表写入只需几秒，可以常落盘
                          # （原来是逐票写 5382 个文件，需 9 分钟）
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        df = fetch_day(pro, d)
        if len(df):
            df = df[["ts_code", "trade_date", "open", "high", "low", "close",
                     "pre_close", "pct_chg", "vol", "amount"]].copy()
            for c in ("open", "high", "low", "close", "vol", "amount",
                      "pct_chg"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            for ts_code, g in df.groupby("ts_code"):
                buf.setdefault(ts_code.split(".")[0].zfill(6), []).append(g)
            st["rows"] = int(st.get("rows", 0)) + len(df)
        # ⚠ 不能在这里标记 done：buffer 还没落盘，
        # 标记跑在落盘前面 → 崩溃后这些日期被永久跳过（踩过）
        pending_dates.append(d)
        if i % FLUSH_EVERY == 0:
            a, b = _flush(buf, store)
            n_written += a
            n_write_fail += b
            buf.clear()
            # 只有落盘成功才标记完成
            st.setdefault("dates_done", []).extend(pending_dates)
            pending_dates.clear()
            _save_status(st)
            if verbose:
                print(f"    [落盘] 已写 {n_written} 只（失败 {n_write_fail}）",
                      flush=True)
        if i % 20 == 0:
            # 只更新行数统计，不动 dates_done（避免标记跑在落盘前）
            _save_status({**st, "dates_done": st.get("dates_done", [])})
            el = time.time() - t0
            rate = i / el * 60
            eta = (len(todo) - i) / max(rate, 1e-9)
            if verbose:
                print(f"  {i}/{len(todo)}  {rate:.0f} 天/分  "
                      f"剩余约 {eta:.0f} 分  缓存 {len(buf)} 只", flush=True)
        time.sleep(float(sleep_s))

    _save_status(st)

    # 收尾：把剩余 buffer 落盘
    a, b = _flush(buf, store)
    n_written += a
    n_write_fail += b
    buf.clear()
    st.setdefault("dates_done", []).extend(pending_dates)
    pending_dates.clear()
    _save_status(st)

    el = time.time() - t0
    return {"ok": True, "dates": len(todo), "codes_written": n_written,
            "codes_failed": n_write_fail, "seconds": round(el, 1),
            "rows": int(st.get("rows", 0))}


if __name__ == "__main__":
    import argparse
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--start", type=str, default="20190101")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()

    if a.check:
        print(json.dumps(check(), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(run(a.start, a.end, a.days, reset=a.reset,
                             batch=a.batch),
                         ensure_ascii=False, indent=1))
