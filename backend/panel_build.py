# -*- coding: utf-8 -*-
"""
全市场日线面板：新浪（存续股）+ baostock（退市股）
====================================================
为什么混合：
* 新浪 `stock_zh_a_daily` 一次调用返回全历史前复权（含 amount），~1.3s，
  但不覆盖**已退市**股票（实测 187 只全部 JSONDecodeError）
* baostock 覆盖退市股且带 tradestatus/isST，但单只 ~4s，5000 只不可行

因此：存续股走向新浪（可并发），退市股（187 只）逐只走 baostock。
两个源都是**前复权**，日收益口径一致；退市股的涨跌停/停牌标记由 baostock 提供，
存续股没有 isST/tradestatus → 引擎按 volume>0 判停牌、按代码前缀判涨跌停幅度
（ST 的 5% 板会被当成 10%，已知近似，A/B/C 同等适用）。

缓存：`.cache/panel/{code}.csv`（列：date,open,high,low,close,volume,amount[,tradestatus,is_st]）
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

__version__ = "0.1.0"
__all__ = ["CACHE", "fetch_sina", "fetch_bs", "build_panel", "load"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / ".cache" / "panel"
CACHE.mkdir(parents=True, exist_ok=True)

START, END = "2018-06-01", "2024-12-31"
COLS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _sina_symbol(c6: str) -> str:
    return ("sh" if c6.startswith(("60", "68", "90")) else "sz") + c6


def fetch_sina(c6: str, retries: int = 3) -> pd.DataFrame | None:
    """新浪前复权日线（存续股）。带缓存。"""
    fp = CACHE / f"{c6}.csv"
    if fp.exists():
        try:
            d = pd.read_csv(fp, parse_dates=["date"])
            if len(d) > 30:
                return d
        except Exception:
            pass
    import akshare as ak
    for k in range(retries):
        try:
            d = ak.stock_zh_a_daily(symbol=_sina_symbol(c6),
                                    start_date="20180601", end_date="20241231",
                                    adjust="qfq")
            if d is None or len(d) == 0:
                return None
            d = d.rename(columns=str.lower)
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date", "close"])
            d = d[d["close"] > 0].sort_values("date").reset_index(drop=True)
            if len(d) < 30:
                return None
            d[COLS].to_csv(fp, index=False)
            return d[COLS]
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(0.5 * (k + 1))
    return None


def fetch_bs(c6: str) -> pd.DataFrame | None:
    """baostock 前复权日线（退市股），带 tradestatus/isST。"""
    fp = CACHE / f"{c6}.csv"
    if fp.exists():
        try:
            d = pd.read_csv(fp, parse_dates=["date"])
            if len(d) > 30:
                return d
        except Exception:
            pass
    import baostock_source as BSS
    BSS._login()
    parts = []
    for a, b in ((START, "2021-07-01"), ("2021-07-01", END)):
        try:
            seg = BSS.stock_daily(c6, start=a, end=b, with_flags=True)
            if seg is not None and len(seg):
                parts.append(seg)
        except Exception:
            pass
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    d = d[(d["date"] >= pd.Timestamp(START)) & (d["date"] <= pd.Timestamp(END))]
    if len(d) < 30:
        return None
    keep = COLS + [c for c in ("tradestatus", "is_st") if c in d.columns]
    d[keep].to_csv(fp, index=False)
    return d[keep]


def _tx_sym(c6: str) -> str:
    return ("sh" if c6.startswith(("60", "68", "90")) else "sz") + c6


_TX_WINDOWS = [("2018-06-01", "2020-06-01"), ("2020-06-01", "2022-06-01"),
               ("2022-06-01", "2024-12-31")]


def _tx_window(sym: str, a: str, b: str) -> list:
    """腾讯 fqkline 单窗口（≤640 行；触顶则再二分）。"""
    import requests
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    p = {"param": f"{sym},day,{a},{b},2000,qfq"}
    r = requests.get(url, params=p, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    j = r.json()
    d = (j.get("data") or {}).get(sym) or {}
    arr = d.get("qfqday") or d.get("day") or []
    if len(arr) >= 640 and a < b:
        # 触顶：二分再取，避免被截断
        mid = (pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
               ).strftime("%Y-%m-%d")
        return (_tx_window(sym, a, mid) + _tx_window(sym, mid, b))
    return arr


def fetch_tx(c6: str, retries: int = 2) -> pd.DataFrame | None:
    """腾讯前复权日线（存续股 + 退市股都覆盖）。amount 用 close×volume×100 估算。"""
    fp = CACHE / f"{c6}.csv"
    if fp.exists():
        try:
            d = pd.read_csv(fp, parse_dates=["date"])
            if len(d) > 30:
                return d
        except Exception:
            pass
    sym = _tx_sym(c6)
    for k in range(retries):
        try:
            rows = []
            for a, b in _TX_WINDOWS:
                rows += _tx_window(sym, a, b)
            if not rows:
                return None
            d = pd.DataFrame([r[:6] for r in rows],
                             columns=["date", "open", "close", "high",
                                      "low", "volume"])
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            for col in ("open", "high", "low", "close", "volume"):
                d[col] = pd.to_numeric(d[col], errors="coerce")
            d = (d.dropna(subset=["date", "close"])
                  .drop_duplicates("date").sort_values("date")
                  .reset_index(drop=True))
            d = d[d["close"] > 0]
            # 腾讯的 volume 单位按板块不同：科创板(68x)返回**股**，其余返回**手**
            mult = 1.0 if c6.startswith("68") else 100.0
            d["amount"] = d["close"] * d["volume"] * mult
            if len(d) < 30:
                return None
            d[COLS].to_csv(fp, index=False)
            return d[COLS]
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(0.5 * (k + 1))
    return None


def build_panel(codes: list[str], delisted: set | None = None,
                workers: int = 8, verbose: bool = True) -> dict:
    """
    单进程内构建面板：存续股走新浪（akshare，py_mini_racer 线程不安全 → 串行），
    退市股走 baostock。**多进程请用 shard 脚本**（每个进程内部再串行）。
    """
    delisted = set(delisted or ())
    codes = [str(c).zfill(6) for c in codes]
    panel, miss = {}, []
    t0 = time.time()
    for i, c in enumerate(codes, 1):
        d = fetch_bs(c) if c in delisted else fetch_sina(c)
        if d is None:
            miss.append(c)
        else:
            panel[c] = d
        if verbose and i % 50 == 0:
            print(f"    {i}/{len(codes)}  成功 {len(panel)}  失败 {len(miss)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    if verbose:
        print(f"  面板 {len(panel)} 只 / 缺失 {len(miss)}  "
              f"耗时 {time.time()-t0:.0f}s", flush=True)
    return {"ok": len(panel) > 0, "panel": panel, "missing": miss,
            "n_ok": len(panel), "n_missing": len(miss)}


def load(codes: list[str]) -> dict:
    out = {}
    for c in codes:
        c6 = str(c).zfill(6)
        fp = CACHE / f"{c6}.csv"
        if fp.exists():
            try:
                d = pd.read_csv(fp, parse_dates=["date"])
                if len(d) > 30:
                    out[c6] = d
            except Exception:
                pass
    return out
