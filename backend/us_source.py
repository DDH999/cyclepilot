# -*- coding: utf-8 -*-
"""
美股数据源（M33）
====================================================
两个已授权的美股数据服务：

| 服务 | 包 | 能力 | 配额 |
|---|---|---|---|
| **ROIC.ai** | `roicai` v0.2.0 | 公司资料 / 最新价 / 历史价 / 三大报表 / 比率 / 估值倍数 | **免费档 5 次/分钟、2 年历史** |
| **xfinlink** | `xfinlink` v1.0.0 | 价格（含 adj_close、GICS 行业）/ 基本面 / 搜索 / 指数 / 持仓 / 内部人 | 实测 11s/次，无明确文档 |

关键约束
--------
* ROIC 免费档 **5 次/分钟** → 必须节流，否则 429
* 两家都是**海外服务**，国内网络可能不稳（实测本机通）
* 标识符格式不同：
  * ROIC 要 `NASDAQ:AAPL`（交易所:代码）
  * xfinlink 用裸 `AAPL`

设计
----
* 统一入口 `us_source`，屏蔽两家的差异
* ROIC 调用走**令牌桶节流**（默认 5 次/60 秒）
* 结果缓存 12 小时，避免重复烧配额
* 取不到就抛错，**绝不编数据**
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["available", "roicai_client", "roic_profile", "roic_latest",
           "roic_prices", "xfin_prices", "xfin_search", "status",
           "throttle_state", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "us"
CACHE_MAX_AGE_HOURS = 12

# ROIC 免费档 5 次/分钟 → 每次至少间隔 12 秒
ROIC_MIN_INTERVAL_S = 12.0
_LAST_CALL = {"roicai": 0.0, "xfinlink": 0.0}
_CALLS = {"roicai": 0, "xfinlink": 0}


def _creds():
    import credentials as CR
    return CR


def available() -> dict:
    """两个包 + Key 是否就绪。"""
    CR = _creds()
    out = {}
    for name, mod in (("roicai", "roicai"), ("xfinlink", "xfinlink")):
        try:
            import importlib
            m = importlib.import_module(mod)
            import importlib.metadata as md
            ver = md.version(mod)
            ok = True
        except Exception as ex:
            m, ver, ok = None, None, False
        out[name] = {"installed": ok, "version": ver,
                     "key": bool(CR.load_key(name)),
                     "ready": bool(ok and CR.load_key(name))}
    return out


def throttle_state() -> dict:
    now = time.time()
    return {k: {"last_call_ago_s": round(now - v, 1) if v else None,
                "calls_this_process": _CALLS[k],
                "min_interval_s": (ROIC_MIN_INTERVAL_S if k == "roicai" else 0)}
            for k, v in _LAST_CALL.items()}


def _throttle(name: str) -> None:
    """ROIC 免费档限速：两次调用之间至少隔 ROIC_MIN_INTERVAL_S 秒。"""
    if name != "roicai":
        return
    gap = time.time() - _LAST_CALL[name]
    if _LAST_CALL[name] and gap < ROIC_MIN_INTERVAL_S:
        wait = ROIC_MIN_INTERVAL_S - gap
        time.sleep(wait)
    _LAST_CALL[name] = time.time()
    _CALLS[name] += 1


def _cache_path(*parts: str) -> Path:
    safe = [str(p).replace(":", "_").replace("/", "_") for p in parts]
    return CACHE_DIR.joinpath(*safe[:-1]) / (safe[-1] + ".csv")


def _read_cache(key: str) -> pd.DataFrame | None:
    fp = CACHE_DIR / f"{key}.csv"
    if not fp.exists():
        return None
    try:
        if (time.time() - fp.stat().st_mtime) / 3600 > CACHE_MAX_AGE_HOURS:
            return None
        return pd.read_csv(fp)
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_DIR / f"{key}.csv", index=False)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# ROIC.ai
# ══════════════════════════════════════════════════════════
def roicai_client():
    """构造 ROIC 客户端（每次新建，SDK 内部会复用 httpx 连接池）。"""
    import roicai
    CR = _creds()
    key = CR.load_key("roicai")
    if not key:
        raise RuntimeError("未配置 ROICAI_API_KEY")
    return roicai.Roicai(api_key=key, timeout=30, max_retries=2)


def _roic_symbol(ticker: str) -> str:
    """AAPL → NASDAQ:AAPL；已带交易所前缀则原样返回。"""
    t = str(ticker).strip().upper()
    if ":" in t:
        return t
    # 粗略判断：纳斯达克常见首字母，其余按 NYSE
    return f"NASDAQ:{t}"


def roic_profile(ticker: str) -> dict:
    """公司资料（名称、行业、描述、多市场代码）。"""
    _key = f"roic_profile_{_roic_symbol(ticker)}"
    _throttle("roicai")
    cli = roicai_client()
    p = cli.companies.profiles.retrieve(_roic_symbol(ticker))
    d = p.model_dump() if hasattr(p, "model_dump") else dict(p)
    # SDK 的字段名和 REST 文档不完全一致：
    # REST 里叫 company_name / description，SDK 里是 short_description，
    # 公司名要靠 company_symbols / 名称字段兜底（实测 company_name 不存在）。
    name = (d.get("company_name") or d.get("name")
            or d.get("short_description"))
    desc = d.get("ai_description") or d.get("description")         or d.get("short_description") or ""
    return {"ok": True, "ticker": _roic_symbol(ticker),
            "company_name": name,
            "sector": d.get("sector"), "industry": d.get("industry"),
            "country": d.get("country") or d.get("fundamental_currency"),
            "currency": d.get("fundamental_currency") or d.get("currency"),
            "ceo": d.get("ceo"), "employees": d.get("number_of_employees"),
            "website": d.get("website"),
            "description": str(desc)[:400],
            "raw_keys": sorted(d.keys())[:20]}


def roic_latest(ticker: str) -> dict:
    """最新一个交易日行情。"""
    _throttle("roicai")
    cli = roicai_client()
    q = cli.stock_prices.latest.retrieve(_roic_symbol(ticker))
    d = q.model_dump() if hasattr(q, "model_dump") else dict(q)
    return {"ok": True, "symbol": d.get("symbol"),
            "date": str(d.get("date")),
            "open": d.get("open"), "high": d.get("high"),
            "low": d.get("low"), "close": d.get("close"),
            "volume": d.get("volume"), "currency": d.get("currency")}


def roic_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """历史日线（免费档只有近 2 年）。"""
    _throttle("roicai")
    cli = roicai_client()
    res = cli.stock_prices.list(_roic_symbol(ticker),
                                start_date=start, end_date=end)
    rows = []
    items = res.data if hasattr(res, "data") else res
    for it in items or []:
        d = it.model_dump() if hasattr(it, "model_dump") else dict(it)
        rows.append(d)
    if not rows:
        raise RuntimeError("ROIC 返回空")
    df = pd.DataFrame(rows)
    return df


# ══════════════════════════════════════════════════════════
# xfinlink
# ══════════════════════════════════════════════════════════
def _xfin_ready():
    import xfinlink
    CR = _creds()
    key = CR.load_key("xfinlink")
    if not key:
        raise RuntimeError("未配置 XFINLINK_API_KEY")
    xfinlink.set_api_key(key)
    xfinlink.set_timeout(40)
    return xfinlink


def xfin_prices(ticker: str, start: str, end: str,
                refresh: bool = False) -> dict:
    """
    价格历史（含 adj_close / GICS 行业 / 股本）。

    实测约 11 秒/次，所以默认缓存 12 小时。
    """
    ck = f"xfin_prices_{ticker}_{start}_{end}"
    if not refresh:
        cached = _read_cache(ck)
        if cached is not None and len(cached) > 0:
            return {"ok": True, "df": cached, "source": "cache",
                    "n": len(cached)}

    x = _xfin_ready()
    t0 = time.time()
    df = x.prices(ticker, start=start, end=end)
    _LAST_CALL["xfinlink"] = time.time()
    _CALLS["xfinlink"] += 1
    if df is None or len(df) == 0:
        raise RuntimeError("xfinlink 返回空")
    if isinstance(df, pd.DataFrame):
        _write_cache(ck, df)
    return {"ok": True, "df": df, "source": "live", "n": len(df),
            "elapsed_s": round(time.time() - t0, 1)}


def xfin_search(query: str) -> dict:
    """按关键词搜美股标的。"""
    x = _xfin_ready()
    t0 = time.time()
    df = x.search(query)
    _CALLS["xfinlink"] += 1
    if df is None or len(df) == 0:
        return {"ok": False, "reason": "无匹配"}
    cols = [c for c in ("ticker", "entity_name", "gics_sector", "country")
            if c in df.columns]
    return {"ok": True, "n": len(df), "elapsed_s": round(time.time() - t0, 1),
            "rows": df[cols].head(20).to_dict("records") if cols else []}


def status() -> dict:
    return {"available": available(), "throttle": throttle_state(),
            "cache_dir": str(CACHE_DIR)}


def format_report(st: dict) -> str:
    L = ["=" * 68, "美股数据源", "=" * 68]
    for name, info in st["available"].items():
        flag = "[就绪]" if info["ready"] else "[不可用]"
        L.append(f"  {name:<10}{flag:<10}v{info.get('version') or '?'}"
                 f"   Key {'已配' if info['key'] else '未配'}")
    L.append("-" * 68)
    th = st["throttle"]["roicai"]
    L.append(f"  ROIC 节流：最小间隔 {th['min_interval_s']:.0f}s"
             f"（免费档 5 次/分钟）   本进程已调用 {th['calls_this_process']} 次")
    L.append(f"  缓存：{st['cache_dir']}（12 小时有效）")
    L.append("=" * 68)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    st = status()
    print(format_report(st))

    print("\n=== ROIC.ai ===")
    if st["available"]["roicai"]["ready"]:
        try:
            p = roic_profile("AAPL")
            print(f"  资料: {p['company_name']}  [{p.get('sector')}]")
        except Exception as ex:
            print(f"  资料失败: {type(ex).__name__}: {str(ex)[:100]}")
        try:
            q = roic_latest("AAPL")
            print(f"  最新: {q['date']}  收盘 {q['close']}")
        except Exception as ex:
            print(f"  最新价失败: {type(ex).__name__}: {str(ex)[:100]}")
    else:
        print("  未就绪（缺包或 Key）")

    print("\n=== xfinlink ===")
    if st["available"]["xfinlink"]["ready"]:
        try:
            r = xfin_prices("AAPL", "2026-10-01", "2026-11-12")
            print(f"  价格: {r['n']} 行（来源 {r['source']}）")
        except Exception as ex:
            print(f"  价格失败: {type(ex).__name__}: {str(ex)[:120]}")
    else:
        print("  未就绪（缺包或 Key）")
    print(f"\nOK v{__version__}")
