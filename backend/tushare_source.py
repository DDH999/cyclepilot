# -*- coding: utf-8 -*-
"""
Tushare 适配层（M59）
====================================================
为什么需要
----------
免费源在本机给不了「全市场 + 全历史 + 时点正确」：
* efinance/adata 限流，zzshare 429，baostock 慢且只 800 行
* 更关键：**历史指数权重、ST 变更的 ann_date、每日涨跌停价**免费源都没有

Tushare 接口对照
----------------
| 需求 | 接口 | 常见积分门槛 |
|---|---|---|
| 全历史行情（不复权） | `daily` | 120 |
| 复权因子（自己算后复权） | `adj_factor` | 120 |
| 股票清单（含退市） | `stock_basic` | 120 |
| 交易日历 | `trade_cal` | 120 |
| 每日涨跌停价 | `stk_limit` | 120 |
| 估值/市值 | `daily_basic` | 120 |
| ST 变更（含 **ann_date**） | `namechange` | 2000 |
| 历史指数权重 | `index_weight` | 2000 |
| 停复牌 | `suspend_d` | 2000 |

（门槛以官网为准；本模块会把「无权限」与「网络错误」严格区分开）

配置
----
token 从环境变量 `TUSHARE_TOKEN` 读；也可显式传入。
本项目中已有的 `~/.dsh/dsh-skills-stock/secrets.env` 里有该变量。

⚠ 复权
------
`daily` 返回**不复权**价，`adj_factor` 给复权因子。
本模块提供 `to_hfq()` —— 用因子合成**后复权**，可复现、不会出现负价。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"
__all__ = ["TOKEN_ENV", "load_token", "available", "pro", "stock_basic",
           "trade_cal", "daily_raw", "to_hfq", "daily_hfq", "stk_limit",
           "namechange", "index_weight", "permission_report"]

TOKEN_ENV = "TUSHARE_TOKEN"
_CACHE: dict = {}
_PERM: dict = {}          # 接口 → True/False/None(未知)


def load_token() -> str:
    """从环境变量或凭据文件读取 token。"""
    tok = os.environ.get(TOKEN_ENV, "").strip()
    if tok:
        return tok
    for p in (Path.home() / ".dsh" / "dsh-skills-stock" / "secrets.env",
              Path.home() / ".dsh" / "secrets.env"):
        try:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == TOKEN_ENV:
                    return v.strip().strip('"').strip("'")
        except Exception:
            continue
    return ""


def pro():
    """取 pro_api 单例。"""
    if "pro" in _CACHE:
        return _CACHE["pro"]
    tok = load_token()
    if not tok:
        raise RuntimeError("未配置 TUSHARE_TOKEN")
    import tushare as ts
    _CACHE["pro"] = ts.pro_api(tok)
    return _CACHE["pro"]


def _call(name: str, fn):
    """
    统一调用：区分「无权限」与「其他错误」。

    返回 (ok, df_or_none, reason)
    """
    if _PERM.get(name) is False:
        return False, None, f"{name}: 无权限（积分不足）"
    try:
        d = fn()
        _PERM[name] = True
        return True, d, None
    except Exception as ex:
        msg = str(ex)
        if "没有接口" in msg or ("权限" in msg and "频率" not in msg):
            _PERM[name] = False
            return False, None, f"{name}: 无权限（积分不足）"
        if "频率" in msg or "超限" in msg:
            # 限流 ≠ 无权限：不要缓存 False，否则整个进程都当它不可用
            return False, None, f"{name}: 频率超限（稍后重试）"
        return False, None, f"{name}: {type(ex).__name__}: {msg[:90]}"


def available() -> dict:
    """探测 token 与基础接口权限（结果缓存）。"""
    tok = load_token()
    if not tok:
        return {"ok": False, "reason": "未配置 TUSHARE_TOKEN", "token": False}
    ok, d, why = _call("trade_cal", lambda: pro().trade_cal(
        exchange="SSE", start_date="20240102", end_date="20240110"))
    return {"ok": bool(ok), "token": True,
            "token_masked": tok[:4] + "****",
            "reason": why,
            "perm": dict(_PERM)}


def permission_report() -> pd.DataFrame:
    """逐接口探测权限，返回表格。"""
    probes = [
        ("trade_cal", lambda: pro().trade_cal(exchange="SSE",
                                              start_date="20240102",
                                              end_date="20240110")),
        ("stock_basic", lambda: pro().stock_basic(exchange="SSE",
                                                  list_status="L")),
        ("daily", lambda: pro().daily(ts_code="600519.SH",
                                      start_date="20240102",
                                      end_date="20240110")),
        ("adj_factor", lambda: pro().adj_factor(ts_code="600519.SH",
                                                start_date="20240102",
                                                end_date="20240110")),
        ("stk_limit", lambda: pro().stk_limit(ts_code="600519.SH",
                                              start_date="20240102",
                                              end_date="20240110")),
        ("daily_basic", lambda: pro().daily_basic(ts_code="600519.SH",
                                                  start_date="20240102",
                                                  end_date="20240110")),
        ("namechange", lambda: pro().namechange(ts_code="600519.SH")),
        ("index_weight", lambda: pro().index_weight(index_code="000300.SH",
                                                     start_date="20240101",
                                                     end_date="20240131")),
        ("suspend_d", lambda: pro().suspend_d(ts_code="600519.SH",
                                              start_date="20240101",
                                              end_date="20240131")),
    ]
    rows = []
    for name, fn in probes:
        ok, d, why = _call(name, fn)
        rows.append({"接口": name, "可用": ok,
                     "行数": (len(d) if ok and d is not None else 0),
                     "说明": (why or "")[:50]})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
def stock_basic(list_status: str = "L") -> pd.DataFrame:
    """股票清单（L=上市 D=退市 P=暂停）。含 list_date / delist_date。"""
    ok, d, why = _call("stock_basic", lambda: pro().stock_basic(
        exchange="", list_status=list_status,
        fields="ts_code,symbol,name,area,industry,market,list_date,"
               "delist_date,is_hs"))
    if not ok:
        raise RuntimeError(why)
    return d


def trade_cal(start: str, end: str) -> pd.DatetimeIndex:
    """交易日历。"""
    ok, d, why = _call("trade_cal", lambda: pro().trade_cal(
        exchange="SSE", start_date=start, end_date=end))
    if not ok:
        raise RuntimeError(why)
    d = d[d["is_open"] == 1]
    return pd.DatetimeIndex(pd.to_datetime(d["cal_date"])).sort_values()


def daily_raw(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """不复权日线（含 pct_chg / vol / amount）。"""
    ok, d, why = _call("daily", lambda: pro().daily(
        ts_code=ts_code, start_date=start, end_date=end))
    if not ok:
        raise RuntimeError(why)
    return d.sort_values("trade_date")


def to_hfq(df: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """
    不复权 + 复权因子 → **后复权**。

    后复权不会出现负价，且不随未来分红变化（可复现）。
    """
    if df is None or len(df) == 0:
        return df
    a = adj[["trade_date", "adj_factor"]].copy()
    m = df.merge(a, on="trade_date", how="left")
    m["adj_factor"] = m["adj_factor"].ffill().bfill().fillna(1.0)
    base = float(m["adj_factor"].iloc[0]) or 1.0
    for c in ("open", "high", "low", "close"):
        m[c] = pd.to_numeric(m[c], errors="coerce") * m["adj_factor"] / base
    return m


def daily_hfq(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """后复权日线（标准列：date/open/high/low/close/volume/amount）。"""
    raw = daily_raw(ts_code, start, end)
    ok, adj, why = _call("adj_factor", lambda: pro().adj_factor(
        ts_code=ts_code, start_date=start, end_date=end))
    if not ok:
        adj = pd.DataFrame({"trade_date": raw["trade_date"],
                            "adj_factor": 1.0})
    m = to_hfq(raw, adj)
    out = pd.DataFrame({
        "date": pd.to_datetime(m["trade_date"], format="%Y%m%d",
                               errors="coerce"),
        "open": m["open"], "high": m["high"], "low": m["low"],
        "close": m["close"],
        "volume": pd.to_numeric(m["vol"], errors="coerce") * 100,  # 手→股
        "amount": pd.to_numeric(m["amount"], errors="coerce") * 1000,  # 千元→元
    }).dropna(subset=["date", "close"]).sort_values("date")
    return out.reset_index(drop=True)


def stk_limit(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """每日涨跌停价（真实值，不用猜 5%/10%/20%）。"""
    ok, d, why = _call("stk_limit", lambda: pro().stk_limit(
        ts_code=ts_code, start_date=start, end_date=end))
    if not ok:
        raise RuntimeError(why)
    return d


def namechange(ts_code: str = "") -> pd.DataFrame:
    """名称/ST 变更（含 ann_date 公告日）。"""
    fn = (lambda: pro().namechange(ts_code=ts_code)) if ts_code else \
        (lambda: pro().namechange())
    ok, d, why = _call("namechange", fn)
    if not ok:
        raise RuntimeError(why)
    return d


def index_weight(index_code: str, start: str, end: str) -> pd.DataFrame:
    """历史指数成分权重（按月）。"""
    ok, d, why = _call("index_weight", lambda: pro().index_weight(
        index_code=index_code, start_date=start, end_date=end))
    if not ok:
        raise RuntimeError(why)
    return d


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== token 与权限 ===")
    a = available()
    print(f"  token: {a.get('token_masked')}  可用: {a['ok']}")
    if not a["ok"]:
        print(f"  原因: {a['reason']}")
    print()
    print(permission_report().to_string(index=False))
