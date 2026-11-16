# -*- coding: utf-8 -*-
"""
统一取数层（M54）：四市场分层路由
====================================================
分层原则
--------
| 市场 | 主源 | 备源 | 兜底 | 状态 |
|---|---|---|---|---|
| **A股** | adata（0.4s） | efinance（1.7s） | akshare / zzshare / baostock | ✅ |
| **港股** | efinance（1.1s） | akshare | adata | ✅ |
| **美股** | efinance（0.8s） | akshare | yfinance（需代理）/ roicai | ✅ |
| **加密货币** | ccxt（需代理） | akshare（数据陈旧） | — | ⚠️ 当前网络不可用 |

设计要点
--------
1. **统一列名**：date/open/high/low/close/volume/amount
2. **逐源重试**：主源失败自动降级，记录 `attempts`
3. **熔断**：某源连续失败 N 次，冷却期内不再尝试
4. **落盘缓存**：`.cache/feed/{market}/{code}.csv`
5. **来源可追溯**：返回 `source`，写进报告

⚠ 复权
------
efinance / adata / akshare 默认均为**前复权**。本模块不做二次复权。
加密货币无复权概念。

⚠ 加密货币
----------
当前网络下 okx/binance/bybit/kraken/gate/kucoin/mexc/bitget/coinbase/htx
全部连接超时或拒绝。设了 `HTTPS_PROXY` 才会走 ccxt 路径。
akshare 的 `crypto_js_spot` 数据停在 2023-10，**不建议使用**。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["MARKETS", "SOURCES", "get_daily", "smoke_test",
           "source_status", "COLUMN_MAP"]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = _ROOT / ".cache" / "feed"
CACHE_HOURS = 12

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

COLUMN_MAP = {
    "日期": "date", "trade_date": "date", "时间": "date",
    "开盘": "open", "开盘价": "open", "open": "open",
    "最高": "high", "最高价": "high", "high": "high",
    "最低": "low", "最低价": "low", "low": "low",
    "收盘": "close", "收盘价": "close", "close": "close",
    "成交量": "volume", "volume": "volume", "vol": "volume",
    "成交额": "amount", "amount": "amount", "turnover": "amount",
}

MARKETS = {
    "cn": {"name": "A股", "sources": ["tushare", "efinance", "adata", "baostock", "akshare", "zzshare"]},
    "hk": {"name": "港股", "sources": ["efinance", "akshare", "adata"]},
    "us": {"name": "美股", "sources": ["efinance", "akshare", "yfinance", "roicai"]},
    "crypto": {"name": "加密货币", "sources": ["ccxt", "akshare"]},
}
SOURCES = {k: v["sources"] for k, v in MARKETS.items()}

# 熔断状态：source -> (失败次数, 冷却截止)
_BREAK: dict[str, tuple[int, float]] = {}
FAIL_THRESHOLD = 3
COOLDOWN = 300.0


def _norm(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    d.columns = [COLUMN_MAP.get(str(c).strip(), str(c).strip().lower())
                 for c in d.columns]
    # 去重：同名列只保留第一个（adata 同时有 trade_date/trade_time）
    if d.columns.duplicated().any():
        d = d.loc[:, ~d.columns.duplicated()]
    if "date" not in d.columns:
        for c in d.columns:
            if "date" in c or "日期" in c:
                d = d.rename(columns={c: "date"})
                break
    if "date" not in d.columns:
        return None
    for c in ("open", "high", "low", "close"):
        if c not in d.columns:
            return None
    if "volume" not in d.columns:
        d["volume"] = np.nan
    if "amount" not in d.columns:
        d["amount"] = pd.to_numeric(d["close"], errors="coerce") * \
            pd.to_numeric(d["volume"], errors="coerce")
    out = d[STD_COLS].copy()
    for c in ("open", "high", "low", "close", "volume", "amount"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates("date").reset_index(drop=True)

    # 前复权在超长历史上会产生**负价**（累计分红 > 早期股价）。
    # 实测 AAPL 1984 年 close=-10.75、00700 2004 年 close=-41.37。
    # 负价下 pct_change 无意义，因此截断到首个「四价全为正」的连续区间，
    # 并记录被截掉的起点，供上层如实报告。
    if len(out):
        px = out[["open", "high", "low", "close"]]
        good = ((px > 0).all(axis=1)).to_numpy()
        if not good.all():
            first = int(np.argmax(good)) if good.any() else -1
            if first > 0:
                out.attrs["trimmed_rows"] = first
                out.attrs["trimmed_before"] = str(out["date"].iloc[first - 1].date())
                out = out.iloc[first:].reset_index(drop=True)
            elif first < 0:
                return None

        # 兜底：前复权序列里仍可能夹着零星 0/负价（停牌占位、复权瑕疵），
        # 这些行不能进回测。直接剔除并计数。
        px2 = out[["open", "high", "low", "close"]]
        mask = (px2 > 0).all(axis=1)
        if not bool(mask.all()):
            out.attrs["dropped_nonpos"] = int((~mask).sum())
            out = out[mask].reset_index(drop=True)
    return out if len(out) else None


# ══════════════════════════════════════════════════════════
def _src_adata(market: str, code: str) -> pd.DataFrame | None:
    if market != "cn":
        return None
    import adata
    return _norm(adata.stock.market.get_market(code))


def _src_efinance(market: str, code: str) -> pd.DataFrame | None:
    import efinance as ef
    return _norm(ef.stock.get_quote_history(code))


def _src_akshare(market: str, code: str) -> pd.DataFrame | None:
    import akshare as ak
    if market == "cn":
        raw = ak.stock_zh_a_hist(symbol=code, period="daily",
                                 adjust="qfq")
    elif market == "hk":
        raw = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")
    elif market == "us":
        raw = ak.stock_us_hist(symbol=code, period="daily", adjust="qfq")
    else:
        return None
    return _norm(raw)


def _src_zzshare(market: str, code: str) -> pd.DataFrame | None:
    if market != "cn":
        return None
    import zz_source
    return _norm(zz_source.stock_daily(code, days=6000))


def _src_baostock(market: str, code: str) -> pd.DataFrame | None:
    if market != "cn":
        return None
    import baostock_source as BS
    # days=0 → 不截断，返回全历史（默认 800 会只剩近 3 年，不够回测）
    return _norm(BS.stock_daily(code, days=0))


def _src_yfinance(market: str, code: str) -> pd.DataFrame | None:
    if market != "us":
        return None
    import yfinance as yf
    raw = yf.Ticker(code).history(period="10y", auto_adjust=True)
    if raw is None or len(raw) == 0:
        return None
    raw = raw.reset_index()
    return _norm(raw)


def _src_roicai(market: str, code: str) -> pd.DataFrame | None:
    if market != "us":
        return None
    import us_source
    r = us_source.roic_prices(code)
    return _norm(r.get("df") if isinstance(r, dict) else r)


def _src_tushare(market: str, code: str) -> pd.DataFrame | None:
    """Tushare 后复权日线（无权限会快速失败，由上层降级）。"""
    if market != "cn":
        return None
    import tushare_source as TS
    if TS._PERM.get("daily") is False:
        raise RuntimeError("tushare 无权限")
    ts_code = f"{code}.SH" if str(code).startswith(("6", "9")) else f"{code}.SZ"
    start = "20050101"
    import datetime as _dt
    end = _dt.date.today().strftime("%Y%m%d")
    return _norm(TS.daily_hfq(ts_code, start, end))


def _src_ccxt(market: str, code: str) -> pd.DataFrame | None:
    if market != "crypto":
        return None
    if not (os.environ.get("HTTPS_PROXY") or os.environ.get("QP_PROXY")):
        raise RuntimeError("无代理，交易所不可达")
    import ccxt
    ex = os.environ.get("QP_CRYPTO_EXCHANGE", "okx")
    c = getattr(ccxt, ex)({"enableRateLimit": True, "timeout": 15000})
    ohlcv = c.fetch_ohlcv(code, "1d", limit=1000)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low",
                                      "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df["amount"] = df["close"] * df["volume"]
    return _norm(df)


SRC_FN = {
    "adata": _src_adata, "efinance": _src_efinance, "akshare": _src_akshare,
    "zzshare": _src_zzshare, "baostock": _src_baostock,
    "yfinance": _src_yfinance, "roicai": _src_roicai, "ccxt": _src_ccxt,
    "tushare": _src_tushare,
}


def _breaker_ok(src: str) -> bool:
    n, until = _BREAK.get(src, (0, 0.0))
    return not (n >= FAIL_THRESHOLD and time.time() < until)


_CONN_HINTS = ("Connection", "Timeout", "HTTP", "Network", "Proxy", "SSL")


def _is_conn_error(ex: BaseException) -> bool:
    """连接类错误才熔断；"该代码无数据"不该禁用整个源。"""
    if isinstance(ex, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(ex).__name__
    return any(k in name for k in _CONN_HINTS)


def _breaker_bump(src: str, ok: bool, source_level: bool = True) -> None:
    if ok:
        _BREAK[src] = (0, 0.0)
        return
    if not source_level:
        return          # 数据缺失，不熔断源
    n, _ = _BREAK.get(src, (0, 0.0))
    _BREAK[src] = (n + 1, time.time() + COOLDOWN)


def _cache_path(market: str, code: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / market).mkdir(parents=True, exist_ok=True)
    return CACHE / market / f"{code}.csv"


def get_daily(market: str, code: str, refresh: bool = False,
              allow_stale: bool = False) -> dict:
    """
    取日线（自动降级）。

    返回 {ok, df, source, attempts, cached, error}
    """
    market = str(market).lower()
    if market not in MARKETS:
        return {"ok": False, "error": f"未知市场 {market!r}",
                "attempts": []}

    fp = _cache_path(market, code)
    if not refresh and fp.exists():
        try:
            if (time.time() - fp.stat().st_mtime) / 3600 <= CACHE_HOURS:
                d = pd.read_csv(fp, parse_dates=["date"])
                if len(d) > 20:
                    return {"ok": True, "df": _norm(d), "source": "cache",
                            "attempts": [], "cached": True}
        except Exception:
            pass

    attempts = []
    best = None                    # (src, df, 耗时)
    MIN_ACCEPT = 1500              # 行数达到这个值就不再试后面的源
    for src in SOURCES[market]:
        if not _breaker_ok(src):
            attempts.append(f"{src}: 熔断中")
            continue
        t0 = time.time()
        df = None
        last_ex = None
        for attempt in range(2):
            try:
                df = SRC_FN[src](market, code)
                if df is not None and len(df) > 20:
                    break
            except Exception as ex:
                last_ex = ex
            time.sleep(0.8 * (attempt + 1))
        try:
            dt = time.time() - t0
            if df is not None and len(df) > 20:
                _breaker_bump(src, True)
                if best is None or len(df) > len(best[1]):
                    best = (src, df, dt)
                attempts.append(f"{src}: {len(df)}行 {dt:.1f}s")
                # 已有足够长的历史，不必再试更慢的源
                if len(df) >= MIN_ACCEPT:
                    break
            else:
                why = (f"{type(last_ex).__name__}: {str(last_ex)[:50]}"
                       if last_ex is not None else "空数据/行数不足")
                attempts.append(f"{src}: {why}")
                _breaker_bump(src, False,
                              source_level=bool(last_ex is not None
                                                and _is_conn_error(last_ex)))
        except Exception as ex:
            last_ex = last_ex or ex
            attempts.append(f"{src}: {type(last_ex).__name__}")
            _breaker_bump(src, False,
                          source_level=_is_conn_error(last_ex))

    if best is not None:
        src, df, dt = best
        try:
            df.to_csv(fp, index=False)
        except Exception:
            pass
        return {"ok": True, "df": df, "source": src,
                "attempts": attempts, "cached": False}

    err = "；".join(attempts) if attempts else "无可用数据源"
    return {"ok": False, "df": None, "source": None, "attempts": attempts,
            "cached": False, "error": err}


def source_status() -> dict:
    """当前各源熔断状态。"""
    out = {}
    for src, (n, until) in _BREAK.items():
        out[src] = {"fails": n,
                    "cooling": bool(n >= FAIL_THRESHOLD and time.time() < until)}
    return out


def smoke_test(cases: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """四市场冒烟测试。"""
    cases = cases or [("cn", "600519"), ("hk", "00700"),
                      ("us", "AAPL"), ("crypto", "BTC/USDT")]
    rows = []
    for market, code in cases:
        t0 = time.time()
        r = get_daily(market, code, refresh=True)
        rows.append({
            "市场": MARKETS[market]["name"], "代码": code,
            "OK": r["ok"], "来源": r.get("source"),
            "行数": (len(r["df"]) if r.get("ok") else 0),
            "耗时s": round(time.time() - t0, 1),
            "降级链": " > ".join(r.get("attempts") or [])[:80],
            "错误": (r.get("error") or "")[:60],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 四市场冒烟测试 ===")
    df = smoke_test()
    print(df.to_string(index=False))
    print("\n=== 熔断状态 ===")
    st = source_status()
    print(st if st else "  （无熔断）")
    n_ok = int(df["OK"].sum())
    print(f"\n{n_ok}/{len(df)} 市场可用")
    if n_ok < len(df):
        bad = df[~df["OK"]]["市场"].tolist()
        print(f"不可用：{', '.join(bad)}")
