# -*- coding: utf-8 -*-
"""
数据源冗余与熔断（M26）
====================================================
问题：整个系统押在 akshare 上，东财接口一挂，回测、体检、风控、定投全停。

做法：
1. **多源**：同一个数据，按优先级依次尝试
2. **熔断**：某个源连续失败 N 次，冷却期内直接跳过（不再浪费时间等超时）
3. **归一**：不同源的列名不同（日期/date、收盘/close），统一成一套列
4. **如实报告**：哪个源成功了、降级了几次、有没有全挂

实测可用源
----------
A 股日线：东财 `stock_zh_a_hist`（快，0.5s）→ 新浪 `stock_zh_a_daily`（慢，1.6s，但历史长）
基金净值：场外 `fund_open_fund_info_em` → ETF `fund_etf_fund_info_em`

⚠ 不编数据：所有源都失败就抛错，绝不返回填充值或上一次的结果。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["Breaker", "market_prefix", "normalize_daily", "get_stock_daily",
           "get_fund_nav", "sources_status", "reset", "SOURCES"]

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

# 列名归一映射（不同源的叫法）
_COL_MAP = {
    "日期": "date", "date": "date",
    "开盘": "open", "open": "open",
    "最高": "high", "high": "high",
    "最低": "low", "low": "low",
    "收盘": "close", "close": "close",
    "成交量": "volume", "volume": "volume", "vol": "volume",
    "成交额": "amount", "amount": "amount",
}


# ══════════════════════════════════════════════════════════
class Breaker:
    """
    轻量熔断器。

    * 连续 fail_threshold 次失败 → 冷却 cooldown 秒内直接拒绝
    * 成功一次即清零计数
    """

    def __init__(self, fail_threshold: int = 3, cooldown: float = 300.0):
        self.fail_threshold = int(fail_threshold)
        self.cooldown = float(cooldown)
        self._state: dict[str, dict] = {}

    def _s(self, key: str) -> dict:
        return self._state.setdefault(
            key, {"fails": 0, "open_until": 0.0, "ok": 0, "fail": 0,
                  "last_error": None, "last_ok_at": None})

    def allow(self, key: str) -> bool:
        st = self._s(key)
        return time.time() >= st["open_until"]

    def record(self, key: str, ok: bool, error: str | None = None) -> None:
        st = self._s(key)
        if ok:
            st["fails"] = 0
            st["open_until"] = 0.0
            st["ok"] += 1
            st["last_ok_at"] = time.time()
        else:
            st["fails"] += 1
            st["fail"] += 1
            st["last_error"] = str(error)[:200] if error else None
            if st["fails"] >= self.fail_threshold:
                st["open_until"] = time.time() + self.cooldown

    def snapshot(self) -> dict:
        now = time.time()
        out = {}
        for k, st in self._state.items():
            out[k] = {"ok": st["ok"], "fail": st["fail"],
                      "consecutive_fails": st["fails"],
                      "open": now < st["open_until"],
                      "cooldown_left_s": max(round(st["open_until"] - now, 1), 0.0),
                      "last_error": st["last_error"],
                      "last_ok_at": (time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(st["last_ok_at"]))
                                     if st["last_ok_at"] else None)}
        return out


BREAKER = Breaker()
_STATS: dict[str, dict] = {}


def reset() -> None:
    """重置熔断与统计（测试用）。"""
    global BREAKER, _STATS
    BREAKER = Breaker()
    _STATS = {}


def _bump(kind: str, source: str, ok: bool) -> None:
    d = _STATS.setdefault(kind, {"attempts": 0, "ok": 0, "fail": 0,
                                 "fallbacks": 0, "by_source": {}})
    d["attempts"] += 1
    d["ok" if ok else "fail"] += 1
    s = d["by_source"].setdefault(source, {"ok": 0, "fail": 0})
    s["ok" if ok else "fail"] += 1


# ══════════════════════════════════════════════════════════
def market_prefix(code: str) -> str:
    """6 位代码 → 带市场前缀（sh/sz/bj）。"""
    c = "".join(ch for ch in str(code) if ch.isdigit()).zfill(6)
    if c.startswith(("60", "68", "51", "58", "11", "50")):
        return "sh" + c
    if c.startswith(("00", "30", "12", "15", "16", "18")):
        return "sz" + c
    if c.startswith(("8", "4", "9")):
        return "bj" + c
    return "sh" + c


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """把不同源的日线统一成 STD_COLS。缺 amount 时用 close×volume 估算。"""
    if df is None or len(df) == 0:
        raise ValueError("空数据")
    d = df.copy()
    d.columns = [_COL_MAP.get(str(c).strip(), str(c).strip().lower())
                 for c in d.columns]
    if "date" not in d.columns:
        d = d.reset_index()
        d.columns = [_COL_MAP.get(str(c).strip(), str(c).strip().lower())
                     for c in d.columns]
    if "date" not in d.columns:
        d = d.rename(columns={d.columns[0]: "date"})
    for c in ("open", "high", "low", "close"):
        if c not in d.columns:
            raise ValueError(f"缺少列 {c}")
    if "volume" not in d.columns:
        d["volume"] = np.nan
    if "amount" not in d.columns:
        d["amount"] = d["close"] * d["volume"]

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    out = d[STD_COLS].copy()
    for c in ("open", "high", "low", "close", "volume", "amount"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("归一后为空")
    return out


# ══════════════════════════════════════════════════════════
# 提供者
# ══════════════════════════════════════════════════════════
def _src_em_daily(code: str, days: int = 800) -> pd.DataFrame:
    """东财：快，默认首选。"""
    import akshare as ak
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 30)
             ).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=str(code).zfill(6), period="daily",
                            start_date=start, end_date=end, adjust="qfq")
    return normalize_daily(df)


def _src_sina_daily(code: str, days: int = 800) -> pd.DataFrame:
    """新浪：慢，但历史长、字段全。"""
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=market_prefix(code), adjust="qfq")
    out = normalize_daily(df)
    return out.tail(int(days)) if days else out


def _src_zzshare_daily(code: str, days: int = 800) -> pd.DataFrame:
    """
    zzshare（独立第三方 SDK）。不经过 akshare，所以东财/新浪同时挂了它还有戏。
    实测 0.4~0.5s，字段已归一。
    """
    import zz_source as ZZ
    if not ZZ.ZZ_AVAILABLE:
        raise RuntimeError("zzshare 未安装")
    df = ZZ.stock_daily(code, days=days)
    return normalize_daily(df)


def _src_baostock_daily(code: str, days: int = 800) -> pd.DataFrame:
    """
    baostock（免费 A 股）。实测登录后 0.1s/只，是最快的一个。
    首次调用含 2 秒登录开销。
    """
    import baostock_source as BS
    if not BS.BS_AVAILABLE:
        raise RuntimeError("baostock 未安装")
    return normalize_daily(BS.stock_daily(code, days=days))


SOURCE_STOCK = [
    ("eastmoney", _src_em_daily),
    ("zzshare", _src_zzshare_daily),
    ("baostock", _src_baostock_daily),
    ("sina", _src_sina_daily),
]

SOURCE_FUND = [
    ("fund_open", None),   # 延迟绑定（避免 import 循环）
    ("fund_etf", None),
]


def _src_fund_open(code: str, **kw) -> pd.Series:
    import akshare as ak
    raw = ak.fund_open_fund_info_em(symbol=str(code).zfill(6),
                                    indicator="单位净值走势")
    return _nav_series(raw, "净值日期")


def _src_fund_etf(code: str, **kw) -> pd.Series:
    import akshare as ak
    raw = ak.fund_etf_fund_info_em(
        fund=str(code).zfill(6), start_date="20150101",
        end_date=pd.Timestamp.today().strftime("%Y%m%d"))
    return _nav_series(raw, "净值日期")


def _nav_series(raw, dcol: str) -> pd.Series:
    if raw is None or len(raw) == 0:
        raise ValueError("空数据")
    col = "单位净值" if "单位净值" in raw.columns else raw.columns[-1]
    s = pd.Series(pd.to_numeric(raw[col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(raw[dcol], errors="coerce"))
    s = s.dropna().sort_index()
    if len(s) < 5:
        raise ValueError(f"净值点太少（{len(s)}）")
    return s


def _try_sources(kind: str, sources: list, *args, **kw):
    """按优先级尝试，返回 (结果, 使用的源, 失败明细)。"""
    errors = []
    for idx, (name, fn) in enumerate(sources):
        key = f"{kind}:{name}"
        if not BREAKER.allow(key):
            errors.append(f"{name}: 熔断中，跳过")
            continue
        try:
            res = fn(*args, **kw)
            BREAKER.record(key, True)
            _bump(kind, name, True)
            if idx > 0:
                _STATS.setdefault(kind, {}).setdefault("fallbacks", 0)
                _STATS[kind]["fallbacks"] += 1
            return res, name, errors
        except Exception as ex:
            BREAKER.record(key, False, str(ex))
            _bump(kind, name, False)
            errors.append(f"{name}: {str(ex)[:120]}")
    raise RuntimeError(
        f"所有数据源均失败（{kind}）：" + "；".join(errors))


def get_stock_daily(code: str, days: int = 800) -> dict:
    """A 股日线（多源）。返回 {df, source, downgraded, errors}。"""
    res, src, errs = _try_sources("stock_daily", SOURCE_STOCK, code, days=days)
    return {"df": res, "source": src, "downgraded": len(errs) > 0,
            "errors": errs}


def get_fund_nav(code: str, kind: str = "open") -> dict:
    """基金净值（多源）。"""
    sources = [("fund_open", _src_fund_open)]
    if kind == "etf":
        sources.append(("fund_etf", _src_fund_etf))
    else:
        sources.append(("fund_etf", _src_fund_etf))
    res, src, errs = _try_sources("fund_nav", sources, code)
    return {"nav": res, "source": src, "downgraded": len(errs) > 0,
            "errors": errs}


def _zz_ok() -> bool:
    try:
        import zz_source as ZZ
        return bool(ZZ.ZZ_AVAILABLE)
    except Exception:
        return False


def _bs_ok() -> bool:
    try:
        import baostock_source as BS
        return bool(BS.BS_AVAILABLE)
    except Exception:
        return False


def sources_status() -> dict:
    """数据源健康状况（给看板/排查用）。"""
    return {"version": __version__,
            "stats": _STATS,
            "breaker": BREAKER.snapshot(),
            "sources": {"stock_daily": [n for n, _ in SOURCE_STOCK],
                        "zzshare_available": _zz_ok(),
                        "baostock_available": _bs_ok(),
                        "fund_nav": ["fund_open", "fund_etf"]}}


SOURCES = {"stock_daily": SOURCE_STOCK, "fund_nav": SOURCE_FUND}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 归一化 ===")
    raw = pd.DataFrame({"日期": ["2025-01-02", "2025-01-03"],
                        "开盘": [1, 2], "最高": [2, 3], "最低": [0.5, 1.5],
                        "收盘": [1.5, 2.5], "成交量": [100, 200]})
    n = normalize_daily(raw)
    assert list(n.columns) == STD_COLS, list(n.columns)
    assert len(n) == 2
    assert abs(n["amount"].iloc[0] - 1.5 * 100) < 1e-9
    print("  中文列名 → 标准列 ✓")

    print("\n=== 熔断器 ===")
    b = Breaker(fail_threshold=3, cooldown=1.0)
    assert b.allow("x")
    for _ in range(3):
        b.record("x", False, "boom")
    assert not b.allow("x"), "连续 3 次失败应熔断"
    b.record("x", True)
    assert b.allow("x"), "成功后应恢复"
    print("  连续失败熔断 / 成功恢复 ✓")

    print("\n=== 多源降级 ===")
    reset()
    calls = []

    def bad(*a, **k):
        calls.append("bad")
        raise ValueError("主源挂了")

    def good(*a, **k):
        calls.append("good")
        return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=50),
                             "open": 1.0, "high": 1.1, "low": 0.9,
                             "close": 1.0, "volume": 1e6, "amount": 1e6})

    src = [("primary", bad), ("backup", good)]
    res, name, errs = _try_sources("test", src)
    assert name == "backup" and len(res) == 50
    assert _STATS["test"]["fallbacks"] == 1
    print(f"  主源失败 → 降级到 {name}，记录 {len(errs)} 条错误 ✓")

    print("\n=== 全挂要报错，不能编数据 ===")
    reset()
    try:
        _try_sources("test2", [("a", bad), ("b", bad)])
        raise AssertionError("应当抛错")
    except RuntimeError as ex:
        assert "所有数据源均失败" in str(ex)
        print(f"  正确抛错：{str(ex)[:60]}…")
    print(f"\nOK v{__version__}")
