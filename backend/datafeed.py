# -*- coding: utf-8 -*-
"""
QuantPilot · 数据层
A股：Sina 主源 -> 腾讯 -> 东财（东财易限流，仅作兜底）
美股：Sina (stock_us_daily)
全部本地 CSV 缓存。
"""
from __future__ import annotations
import time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _cpath(key: str) -> Path:
    return CACHE / f"{hashlib.md5(key.encode()).hexdigest()[:12]}_{key}.csv"


def _to_sina(code: str) -> str:
    c = str(code).zfill(6)
    return ("sh" if c[0] in "65" else "sz") + c


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    m = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
         "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover",
         "Date": "date", "Open": "open", "Close": "close", "High": "high", "Low": "low",
         "Volume": "volume", "Amount": "amount"}
    df = df.rename(columns={k: v for k, v in m.items() if k in df.columns})
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "date"})
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in keep:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def _retry(fn, tries=3, base=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base * (i + 1))
    raise last


def load_ashare(symbol: str, refresh: bool = False) -> pd.DataFrame:
    code = str(symbol).zfill(6)
    cp = _cpath(f"cn_{code}")
    if cp.exists() and not refresh:
        return pd.read_csv(cp, parse_dates=["date"])
    import akshare as ak
    sk = _to_sina(code)
    loaders = [
        ("sina",    lambda: ak.stock_zh_a_daily(symbol=sk, adjust="qfq")),
        ("tencent", lambda: ak.stock_zh_a_hist_tx(symbol=sk, adjust="qfq")),
        ("em",      lambda: ak.stock_zh_a_hist(symbol=code, period="daily",
                                               start_date="20050101",
                                               end_date=time.strftime("%Y%m%d"),
                                               adjust="qfq")),
    ]
    df, used = None, None
    for name, fn in loaders:
        try:
            df = _norm(_retry(fn, tries=2))
            if len(df) > 250:
                used = name
                break
            df = None
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"all A-share loaders failed for {code}")
    df["src"] = used
    df.to_csv(cp, index=False, encoding="utf-8")
    return df


def load_us(symbol: str, refresh: bool = False) -> pd.DataFrame:
    cp = _cpath(f"us_{symbol}")
    if cp.exists() and not refresh:
        return pd.read_csv(cp, parse_dates=["date"])
    import akshare as ak
    df = _norm(_retry(lambda: ak.stock_us_daily(symbol=symbol, adjust="qfq"), tries=3))
    df.to_csv(cp, index=False, encoding="utf-8")
    return df


def load_index(symbol: str = "000300", refresh: bool = False) -> pd.DataFrame:
    """指数日线（中证/上证系列），用于真实基准"""
    cp = _cpath(f"idx_{symbol}")
    if cp.exists() and not refresh:
        return pd.read_csv(cp, parse_dates=["date"])
    import akshare as ak
    raw = _retry(lambda: ak.index_zh_a_hist(symbol=symbol, period="daily",
                                            start_date="20050101",
                                            end_date=time.strftime("%Y%m%d")), tries=3)
    df = _norm(raw)
    df.to_csv(cp, index=False, encoding="utf-8")
    return df


def load(symbol: str, market: str = "cn", refresh: bool = False) -> pd.DataFrame:
    return load_ashare(symbol, refresh) if market == "cn" else load_us(symbol, refresh)


def get_panel(symbols: dict[str, str], refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for code, mkt in symbols.items():
        try:
            df = load(code, mkt, refresh)
            if len(df) > 250:
                out[code] = df
        except Exception as e:
            print(f"[datafeed] skip {code} ({mkt}): {type(e).__name__}: {str(e)[:80]}")
    return out


def repair_splits(s: pd.Series, thresh: float = 0.45) -> pd.Series:
    """
    修复未复权价格序列中的拆股跳变。
    检测单日 |收益| > thresh，若价格比值接近整数倍(>=2, 误差<4%)，判定为拆股并回溯调整。
    美股源 akshare.stock_us_daily 的 adjust 参数并不可靠，必须用这一层兜底。
    """
    v = s.astype(float).copy()
    v[v <= 0] = np.nan
    v = v.ffill()
    for _ in range(12):
        r = v.pct_change(fill_method=None)
        bad = np.where(r.abs() > thresh)[0]
        if not len(bad):
            break
        i = int(bad[0])
        prev, cur = v.iloc[i - 1], v.iloc[i]
        if not (prev > 0 and cur > 0):
            v.iloc[i] = prev
            continue
        ratio = cur / prev
        k = 1.0 / ratio if ratio < 1 else ratio
        kr = round(k)
        if kr >= 2 and abs(k - kr) / kr < 0.04:
            v.iloc[:i] = v.iloc[:i] * ratio      # 把历史价格拉到同一口径
            continue
        v.iloc[i] = prev                          # 无法解释的跳变 -> 视为数据错误
    return v


def sanitize_panel(panel: dict[str, pd.DataFrame], start: str = "2016-01-01",
                   clip: float = 0.35) -> dict[str, pd.DataFrame]:
    """统一修复拆股、剔除零价、截断异常收益、对齐时间窗口。"""
    import numpy as _np
    out = {}
    for s, d in panel.items():
        d = d.copy()
        d["close"] = repair_splits(d["close"])
        for col in ("open", "high", "low"):
            if col in d.columns:
                d[col] = repair_splits(d[col])
        d = d[d["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        r = d["close"].pct_change(fill_method=None)
        med = float(r.abs().median() or 0)
        d = d[(r.abs() <= clip) | r.isna()].reset_index(drop=True)
        if len(d) < 300 or med <= 0 or med > 0.15:
            continue
        out[s] = d
    return out


UNIVERSE_CN = ["601919", "600519", "000858", "600036", "300750",
               "601318", "000001", "600030", "002594", "600900"]
UNIVERSE_US = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN",
               "GOOGL", "META", "AMD", "AVGO", "JPM"]


if __name__ == "__main__":
    for s in UNIVERSE_CN[:3]:
        d = load_ashare(s)
        print(f"CN {s}: {len(d):5d} rows  {d['date'].min().date()} -> {d['date'].max().date()}  "
              f"last={d['close'].iloc[-1]:>9.2f}  src={d['src'].iloc[-1]}")
    for s in UNIVERSE_US[:3]:
        d = load_us(s)
        print(f"US {s}: {len(d):5d} rows  {d['date'].min().date()} -> {d['date'].max().date()}  "
              f"last={d['close'].iloc[-1]:>9.2f}")
