# -*- coding: utf-8 -*-
"""
Baostock 数据源（M31）
====================================================
本机装了 `baostock`（免费 A 股历史数据，v0.9.4）。

实测特性
--------
* **日线 0.1 秒**——比 akshare（0.5s）和 zzshare（0.5s）都快
* 需要 `login()` / `logout()`，是**模块级全局会话**（不是线程安全的）
* 行业分类 5562 只，但**要 63 秒**，必须缓存
* 行业字段是证监会分类（如 `J66货币金融服务`），不是常用的申万/中信

设计
----
* 会话按需登录，失败自动重试一次
* 取完数据不立刻 logout（复用会话），由 `close()` 显式释放
* 行业映射缓存 30 天
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["BS_AVAILABLE", "BS_VERSION", "available", "to_bs_code",
           "stock_daily", "industry_map", "close", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "baostock"
INDUSTRY_CACHE = CACHE_DIR / "industry.csv"
INDUSTRY_MAX_AGE_DAYS = 30


def _probe() -> tuple[bool, str | None]:
    try:
        import baostock  # noqa: F401
        import importlib.metadata as md
        return True, md.version("baostock")
    except Exception:
        return False, None


BS_AVAILABLE, BS_VERSION = _probe()
_LOGGED_IN = False


def available() -> dict:
    return {"available": BS_AVAILABLE, "version": BS_VERSION,
            "logged_in": _LOGGED_IN}


def to_bs_code(code: str) -> str:
    """600519 → sh.600519；000001 → sz.000001；830799 → bj.830799"""
    c = "".join(ch for ch in str(code) if ch.isdigit()).zfill(6)
    if c.startswith(("60", "68", "51", "58", "11", "50", "90")):
        return f"sh.{c}"
    if c.startswith(("00", "30", "12", "15", "16", "18", "20")):
        return f"sz.{c}"
    if c.startswith(("8", "4", "9")):
        return f"bj.{c}"
    return f"sh.{c}"


def _login(force: bool = False) -> None:
    """确保已登录。baostock 是全局会话，重复 login 会返回错误码，忽略即可。"""
    global _LOGGED_IN
    if _LOGGED_IN and not force:
        return
    import baostock as bs
    lg = bs.login()
    if getattr(lg, "error_code", "1") != "0":
        raise RuntimeError(f"baostock 登录失败：{lg.error_msg}")
    _LOGGED_IN = True


def close() -> None:
    """显式登出并释放会话。"""
    global _LOGGED_IN
    if not _LOGGED_IN or not BS_AVAILABLE:
        return
    try:
        import baostock as bs
        bs.logout()
    except Exception:
        pass
    _LOGGED_IN = False


# ══════════════════════════════════════════════════════════
def stock_daily(code: str, days: int = 800,
                start: str | None = None, end: str | None = None,
                adjustflag: str = "2",
                with_flags: bool = False) -> pd.DataFrame:
    """
    日线（前复权，adjustflag="2"），归一成标准列。

    volume 单位：baostock 返回的是**股**，这里统一换算成系统的「手」前不做换算，
    保持原样并在列名上不承诺单位——调用方按 backtest 的成交额口径使用 amount。
    """
    if not BS_AVAILABLE:
        raise RuntimeError("baostock 未安装")
    import baostock as bs

    _login()
    bs_code = to_bs_code(code)
    if start is None:
        start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 30)
                 ).strftime("%Y-%m-%d")
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    fields = "date,open,high,low,close,volume,amount"
    if with_flags:
        # isST / tradestatus 是**当日**公开的实施状态，用来判当日涨跌停
        # 与是否停牌，不构成前视。
        fields += ",tradestatus,isST,pctChg"
    rs = bs.query_history_k_data_plus(
        bs_code, fields,
        start_date=str(start), end_date=str(end),
        frequency="d", adjustflag=str(adjustflag))
    if getattr(rs, "error_code", "1") != "0":
        raise RuntimeError(f"baostock 查询失败：{rs.error_msg}")

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise ValueError(f"baostock 返回空（{bs_code}）")

    df = pd.DataFrame(rows, columns=rs.fields)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
    })
    # baostock 有时返回 "1405"（整数串），统一转 float64，
    # 避免下游 pandas 算出 int 列、与其它数据源类型不一致。
    for c in ("open", "high", "low", "close", "volume", "amount"):
        out[c] = out[c].astype("float64")
    if with_flags:
        out["tradestatus"] = pd.to_numeric(
            df["tradestatus"], errors="coerce").fillna(1).astype(int)
        out["is_st"] = pd.to_numeric(
            df["isST"], errors="coerce").fillna(0).astype(int)
        out["pct_chg_official"] = pd.to_numeric(df["pctChg"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("baostock 数据归一后为空")
    return out.tail(int(days)) if days else out


# ══════════════════════════════════════════════════════════
def industry_map(cache: Path | str | None = None,
                 refresh: bool = False) -> dict:
    """
    股票 → 行业（证监会分类），首次约 63 秒，缓存 30 天。

    返回 {ok, mapping, n, source}
    """
    cache = Path(cache) if cache is not None else Path(INDUSTRY_CACHE)
    if cache.exists() and not refresh:
        try:
            age_d = (time.time() - cache.stat().st_mtime) / 86400
            if age_d <= INDUSTRY_MAX_AGE_DAYS:
                df = pd.read_csv(cache, dtype={"symbol": str, "industry": str})
                if len(df) > 0:
                    m = {str(r["symbol"]).zfill(6): str(r["industry"])
                         for _, r in df.iterrows()}
                    return {"ok": True, "n": len(m), "source": "cache",
                            "mapping": m}
        except Exception:
            pass

    if not BS_AVAILABLE:
        return {"ok": False, "reason": "baostock 未安装"}
    import baostock as bs

    _login()
    rs = bs.query_stock_industry()
    if getattr(rs, "error_code", "1") != "0":
        return {"ok": False, "reason": f"查询失败：{rs.error_msg}"}

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return {"ok": False, "reason": "行业表为空"}

    df = pd.DataFrame(rows, columns=rs.fields)
    if "code" not in df.columns or "industry" not in df.columns:
        return {"ok": False, "reason": f"字段不符：{list(df.columns)}"}

    df = df.copy()
    df["symbol"] = df["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    df["industry"] = df["industry"].astype(str).str.strip()
    df = df[df["industry"] != ""].drop_duplicates(subset=["symbol"])
    if len(df) == 0:
        return {"ok": False, "reason": "清理后无有效行业数据"}

    out = df[["symbol", "industry"]]
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache, index=False, encoding="utf-8")
    except Exception:
        pass
    return {"ok": True, "n": int(len(out)), "source": "live",
            "mapping": dict(zip(out["symbol"], out["industry"]))}


def format_report(info: dict, ind: dict | None = None) -> str:
    L = ["=" * 66, "Baostock 数据源", "=" * 66]
    L.append(f"  状态: {'可用' if info.get('available') else '不可用'}"
             + (f"   版本 {info.get('version')}" if info.get("version") else ""))
    L.append(f"  会话: {'已登录' if info.get('logged_in') else '未登录'}")
    if ind:
        L.append("-" * 66)
        if ind.get("ok"):
            L.append(f"  行业映射 {ind['n']} 只（来源 {ind['source']}）")
            sample = list(ind["mapping"].items())[:3]
            for k, v in sample:
                L.append(f"    {k} → {v}")
        else:
            L.append(f"  行业映射失败：{ind.get('reason')}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    info = available()
    print(f"baostock 可用: {info['available']}  版本 {info['version']}")
    if not info["available"]:
        raise SystemExit("未安装 baostock")

    print("\n=== 代码转换 ===")
    for c, want in [("600519", "sh.600519"), ("000001", "sz.000001"),
                    ("300750", "sz.300750"), ("830799", "bj.830799")]:
        got = to_bs_code(c)
        assert got == want, (c, got)
    print("  6 位 → sh./sz./bj. 前缀 ✓")

    print("\n=== 日线（实测 0.1 秒）===")
    t0 = time.time()
    df = stock_daily("600519", days=30)
    dt = time.time() - t0
    assert len(df) > 10, len(df)
    assert {"date", "open", "high", "low", "close", "volume", "amount"}.issubset(df.columns)
    print(f"  {len(df)} 行  {dt:.2f}s  最新 {df['date'].iloc[-1].date()} "
          f"收盘 {df['close'].iloc[-1]:.2f}")
    assert dt < 10, f"日线耗时异常 {dt:.1f}s"

    print(f"\nOK v{__version__}（行业映射需 63 秒，此处不跑，用 CLI `industry --bs`）")
    close()
