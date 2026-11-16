# -*- coding: utf-8 -*-
"""
海外指数与 QDII 基准（M30）
====================================================
你有四个海外指数的定投：标普500、纳斯达克、日经225、富时100。
要给它们做体检，就必须有**正确的基准**——拿沪深300 去比标普基金毫无意义。

数据来源与故障链
----------------
每个指数按优先级尝试：

1. **yfinance**（Yahoo）——首选，字段干净，但**会被限流**
   实测连续请求 5 个符号全部返回 `YFRateLimitError: Too Many Requests`
2. **akshare 新浪接口**——兜底，实测稳定
   * 美股指：`index_us_stock_sina`（`.INX` / `.IXIC` / `.NDX` / `.DJI`）
   * 其他：`index_global_hist_sina`（`日经225指数` / `英国富时100指数`）
3. 全失败 → 抛错，不编数据

⚠ 实测结论（2026-11）：**yfinance 基本不可用（持续限流）**，
所以实际生效的是新浪接口。yfinance 仍保留在链首，等它恢复自动启用。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["INDEXES", "available", "index_history", "match_fund_index",
           "fund_vs_benchmark", "status", "diagnose", "proxy_of",
           "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "overseas"
CACHE_MAX_AGE_HOURS = 12

# key: (中文名, yfinance 符号, 新浪美股符号, 新浪全球名)
INDEXES: dict[str, dict] = {
    "sp500":  {"name": "标普500",      "yf": "^GSPC", "sina_us": ".INX",
               "em": "100.SPX"},
    "nasdaq": {"name": "纳斯达克综合", "yf": "^IXIC", "sina_us": ".IXIC",
               "em": "100.IXIC"},
    "ndx":    {"name": "纳斯达克100",  "yf": "^NDX",  "sina_us": ".NDX",
               "em": "100.NDX"},
    "dji":    {"name": "道琼斯工业",   "yf": "^DJI",  "sina_us": ".DJI",
               "em": "100.DJIA"},
    "nikkei": {"name": "日经225",      "yf": "^N225", "sina_global": "日经225指数",
               "em": "100.N225"},
    "ftse":   {"name": "英国富时100",  "yf": "^FTSE", "sina_global": "英国富时100指数",
               "em": "100.FTSE"},
    "hsi":    {"name": "恒生指数",     "yf": "^HSI",  "sina_us": ".HSI",
               "em": "100.HSI"},
}

# 基金名称关键词 → 指数 key（按优先级，长词在前）
NAME_RULES = [
    ("纳斯达克100", "ndx"), ("纳指100", "ndx"), ("纳斯达克", "nasdaq"),
    ("标普500", "sp500"), ("标普", "sp500"),
    ("日经225", "nikkei"), ("日经", "nikkei"),
    ("富时100", "ftse"), ("富时", "ftse"),
    ("恒生", "hsi"), ("道琼斯", "dji"),
]


def available() -> dict:
    """两个库装了没。"""
    out = {}
    for mod in ("yfinance", "akshare"):
        try:
            import importlib
            m = importlib.import_module(mod)
            out[mod] = {"ok": True, "version": getattr(m, "__version__", "?")}
        except Exception as ex:
            out[mod] = {"ok": False, "error": str(ex)[:80]}
    return out


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.csv"


def _read_cache(key: str) -> pd.Series | None:
    fp = _cache_path(key)
    if not fp.exists():
        return None
    try:
        age_h = (time.time() - fp.stat().st_mtime) / 3600
        if age_h > CACHE_MAX_AGE_HOURS:
            return None
        df = pd.read_csv(fp, parse_dates=["date"])
        s = df.set_index("date")["close"].astype("float64").dropna()
        return s if len(s) > 20 else None
    except Exception:
        return None


def _write_cache(key: str, s: pd.Series) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        s.rename("close").rename_axis("date").to_csv(
            _cache_path(key), float_format="%.6f")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# 各提供者
# ══════════════════════════════════════════════════════════
def _from_yfinance(key: str, days: int) -> pd.Series:
    import yfinance as yf
    sym = INDEXES[key]["yf"]
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.7) + 30)
             ).strftime("%Y-%m-%d")
    df = yf.download(sym, start=start, progress=False,
                     auto_adjust=True, timeout=15)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance 返回空（可能被限流）")
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = pd.Series(np.asarray(df[col]).ravel(),
                  index=pd.to_datetime(df.index)).dropna()
    if len(s) < 20:
        raise RuntimeError(f"yfinance 数据过少（{len(s)}）")
    return s.tail(days)


def _from_sina(key: str, days: int) -> pd.Series:
    import akshare as ak
    meta = INDEXES[key]
    if meta.get("sina_us"):
        df = ak.index_us_stock_sina(symbol=meta["sina_us"])
    elif meta.get("sina_global"):
        df = ak.index_global_hist_sina(symbol=meta["sina_global"])
    else:
        raise RuntimeError("该指数没有新浪接口")

    if df is None or len(df) == 0:
        raise RuntimeError("新浪返回空")
    dcol = "date" if "date" in df.columns else df.columns[0]
    ccol = "close" if "close" in df.columns else \
        ("收盘" if "收盘" in df.columns else df.columns[-1])
    s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[dcol], errors="coerce")).dropna()
    s = s.sort_index()
    if len(s) < 20:
        raise RuntimeError(f"新浪数据过少（{len(s)}）")
    return s.tail(days)


def proxy_of() -> str | None:
    """读取代理设置（支持 QP_PROXY 或系统代理变量）。"""
    import os
    for k in ("QP_PROXY", "HTTPS_PROXY", "https_proxy",
              "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return None


def _http():
    """优先用 curl_cffi（带浏览器指纹），退回 requests。"""
    try:
        from curl_cffi import requests as _r
        return _r
    except Exception:
        import requests as _r
        return _r


def _from_eastmoney(key: str, days: int) -> pd.Series:
    """
    东方财富国际指数（直连 HTTP）。

    国内网络下**实测 0.0~0.4 秒**，468+ 行，无需 key。
    这是本项目在墙内最可靠的海外指数源。
    """
    meta = INDEXES[key]
    secid = meta.get("em")
    if not secid:
        raise RuntimeError("该指数没有东财 secid")

    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid}"
           "&fields1=f1,f2,f3,f4,f5"
           "&fields2=f51,f52,f53,f54,f55,f56,f57"
           "&klt=101&fqt=1&beg=0&end=20500101")

    rq = _http()
    kw = {"timeout": 15,
          "headers": {"Referer": "https://quote.eastmoney.com/",
                      "User-Agent": "Mozilla/5.0"}}
    px = proxy_of()
    if px:
        kw["proxies"] = {"http": px, "https": px}
    try:
        resp = rq.get(url, **kw)
    except TypeError:
        # requests 与 curl_cffi 的 proxies 参数形态略有差异
        kw.pop("proxies", None)
        resp = rq.get(url, **kw)

    data = resp.json()
    d = data.get("data") or {}
    klines = d.get("klines") or []
    if not klines:
        raise RuntimeError(f"东财返回空（secid={secid}，rc={data.get('rc')}）")

    rows = [k.split(",") for k in klines]
    df = pd.DataFrame(rows)
    if df.shape[1] < 3:
        raise RuntimeError("东财返回列数不足")
    out = pd.Series(pd.to_numeric(df[2], errors="coerce").to_numpy(),
                    index=pd.to_datetime(df[0], errors="coerce"))
    out = out.dropna().sort_index()
    if len(out) < 20:
        raise RuntimeError(f"东财数据过少（{len(out)}）")
    return out.tail(days)


# 顺序说明：国内网络下东财最快最稳，放第一位；
# yfinance 排在最后 —— 它需要代理才通，没代理时只会白白多等 2 秒。
PROVIDERS = [
    ("eastmoney", _from_eastmoney),
    ("sina", _from_sina),
    ("yfinance", _from_yfinance),
]


def index_history(key: str, days: int = 1500, refresh: bool = False) -> dict:
    """
    取海外指数历史收盘价（带回退链 + 缓存）。

    返回 {ok, key, name, series, source, errors}
    """
    if key not in INDEXES:
        raise ValueError(f"未知指数 {key!r}，可选：{list(INDEXES)}")

    if not refresh:
        s = _read_cache(key)
        if s is not None and len(s) >= 20:
            return {"ok": True, "key": key, "name": INDEXES[key]["name"],
                    "series": s.tail(days), "source": "cache", "errors": []}

    errors = []
    for name, fn in PROVIDERS:
        try:
            s = fn(key, days)
            _write_cache(key, s)
            return {"ok": True, "key": key, "name": INDEXES[key]["name"],
                    "series": s, "source": name, "errors": errors}
        except Exception as ex:
            errors.append(f"{name}: {str(ex)[:100]}")
    return {"ok": False, "key": key, "name": INDEXES[key]["name"],
            "series": None, "source": None, "errors": errors,
            "reason": "所有海外指数数据源均失败"}


# ══════════════════════════════════════════════════════════
# 基金 → 基准
# ══════════════════════════════════════════════════════════
def match_fund_index(fund_name: str) -> dict:
    """
    按基金名称猜它跟哪个指数。

    只做关键词匹配——猜不出来就如实说猜不出来，不瞎配。
    """
    name = str(fund_name or "")
    for kw, key in NAME_RULES:
        if kw in name:
            return {"ok": True, "key": key, "index_name": INDEXES[key]["name"],
                    "matched": kw}
    return {"ok": False, "reason": f"名称「{name}」里没有可识别的指数关键词"}


def fund_vs_benchmark(nav, bench, rf_annual: float = 0.0) -> dict:
    """
    基金 vs 基准：跟踪误差、beta、相关性、超额。

    这是给 QDII 基金做体检的核心——不跟正确基准比，等于没比。
    """
    f = pd.Series(nav).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    b = pd.Series(bench).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    # 净值序列转收益
    for holder in (f, b):
        pass
    rf = pd.Series(f).copy()
    mean_abs = abs(float(rf.mean())) if len(rf) else 0.0
    cv = (float(rf.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    if len(rf) > 20 and float((rf > 0).mean()) > 0.95 and cv < 0.5:
        f = f.pct_change(fill_method=None).dropna()
    rb = b.copy()
    mean_abs = abs(float(rb.mean())) if len(rb) else 0.0
    cv = (float(rb.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    if len(rb) > 20 and float((rb > 0).mean()) > 0.95 and cv < 0.5:
        b = b.pct_change(fill_method=None).dropna()

    idx = f.index.intersection(b.index)
    if len(idx) < 60:
        return {"ok": False, "reason": f"重叠样本仅 {len(idx)} 天，至少需要 60 天"}
    x = b.loc[idx].to_numpy()
    y = f.loc[idx].to_numpy()

    var_b = float(np.var(x, ddof=1))
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    diff = y - x
    te = float(np.std(diff, ddof=0) * np.sqrt(252))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.var(x) > 0 and np.var(y) > 0 else float("nan")
    f_cagr = float(np.prod(1 + y) ** (252 / len(y)) - 1)
    b_cagr = float(np.prod(1 + x) ** (252 / len(x)) - 1)

    return {
        "ok": True, "n": int(len(idx)),
        "fund_annual": round(f_cagr, 6), "bench_annual": round(b_cagr, 6),
        "excess_annual": round(f_cagr - b_cagr, 6),
        "beta": round(beta, 4) if np.isfinite(beta) else None,
        "correlation": round(corr, 4) if np.isfinite(corr) else None,
        "tracking_error": round(te, 6),
        "info_ratio": round((f_cagr - b_cagr) / te, 4) if te > 0 else None,
        "fund_vol": round(float(np.std(y, ddof=0) * np.sqrt(252)), 6),
        "bench_vol": round(float(np.std(x, ddof=0) * np.sqrt(252)), 6),
        "start": str(pd.Timestamp(idx[0]).date()),
        "end": str(pd.Timestamp(idx[-1]).date()),
    }


def diagnose() -> dict:
    """
    诊断海外指数数据源连通性。

    区分三种失败：
      * IP 被封锁（HTTP 403）→ **代码解决不了，需要代理**
      * 限流（429 / YFRateLimitError）→ 等一会儿或换 IP
      * 正常
    """
    out = {"proxy": proxy_of(), "checks": [], "verdict": ""}
    rq = _http()

    def probe(name: str, url: str, expect_json: bool = False) -> dict:
        t0 = time.time()
        try:
            kw = {"timeout": 12,
                  "headers": {"User-Agent": "Mozilla/5.0",
                              "Referer": "https://quote.eastmoney.com/"}}
            px = proxy_of()
            if px:
                kw["proxies"] = {"http": px, "https": px}
            r = rq.get(url, **kw)
            dt = time.time() - t0
            code = getattr(r, "status_code", 0)
            ok = code == 200
            detail = ""
            if ok and expect_json:
                try:
                    ok = bool((r.json().get("data") or {}).get("klines"))
                    detail = "有 klines" if ok else "无 klines"
                except Exception:
                    ok = False
                    detail = "非 JSON"
            elif code == 403:
                detail = "403 被封锁"
            elif code == 429:
                detail = "429 限流"
            return {"name": name, "ok": ok, "status": code,
                    "elapsed_s": round(dt, 2), "detail": detail,
                    "url": url}
        except Exception as ex:
            return {"name": name, "ok": False, "status": 0,
                    "elapsed_s": round(time.time() - t0, 2),
                    "detail": f"{type(ex).__name__}: {str(ex)[:70]}",
                    "url": url}

    # 东财这条不能用「自己拼 URL」探测 —— 参数差一个就会返回空 klines，
    # 导致诊断说"不通"而实际能取（踩过）。直接调真实提供者函数。
    _t0 = time.time()
    try:
        _ser = _from_eastmoney("sp500", 30)
        out["checks"].append({"name": "东财国际指数", "ok": True, "status": 200,
                              "elapsed_s": round(time.time() - _t0, 2),
                              "detail": f"{len(_ser)} 点", "url": "provider"})
    except Exception as _ex:
        out["checks"].append({"name": "东财国际指数", "ok": False, "status": 0,
                              "elapsed_s": round(time.time() - _t0, 2),
                              "detail": f"{type(_ex).__name__}: {str(_ex)[:70]}",
                              "url": "provider"})
    out["checks"].append(probe(
        "新浪美股指数",
        "https://finance.sina.com.cn/realstock/company/sh000001/nc.shtml"))
    out["checks"].append(probe(
        "Yahoo Finance",
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=5d"))

    em = out["checks"][0]["ok"]
    yh = out["checks"][2]["ok"]
    if em and not yh:
        out["verdict"] = ("东财可用、Yahoo 被封锁 → 已自动走东财，"
                          "无需处理；要用 yfinance 请配代理（QP_PROXY）")
    elif em and yh:
        out["verdict"] = "两个都通，按优先级使用东财（更快）"
    elif not em and yh:
        out["verdict"] = "东财不通但 Yahoo 通，会自动走 yfinance"
    else:
        out["verdict"] = "两个都不通：检查网络，或配置代理 QP_PROXY"
    return out


def status() -> dict:
    """两个库 + 各指数缓存情况。"""
    out = {"available": available(), "indexes": {}}
    for k, meta in INDEXES.items():
        fp = _cache_path(k)
        out["indexes"][k] = {
            "name": meta["name"],
            "cached": fp.exists(),
            "cached_size": fp.stat().st_size if fp.exists() else 0,
        }
    return out


def format_report(res: dict, cmp: dict | None = None) -> str:
    L = ["=" * 68, "海外指数 / QDII 基准", "=" * 68]
    if not res.get("ok"):
        L.append(f"  {res.get('name')} 取数失败：{res.get('reason')}")
        for e in res.get("errors") or []:
            L.append(f"    - {e}")
        L.append("=" * 68)
        return "\n".join(L)

    s = res["series"]
    L.append(f"  {res['name']}（{res['key']}）  来源 {res['source']}")
    L.append(f"  区间 {s.index[0].date()} ~ {s.index[-1].date()}   {len(s)} 个交易日")
    L.append(f"  最新 {s.iloc[-1]:,.2f}   "
             f"近 1 年 {s.iloc[-1]/s.iloc[-min(252, len(s))]-1:+.2%}")
    if res.get("errors"):
        L.append("  降级记录：")
        for e in res["errors"]:
            L.append(f"    - {e}")
    if cmp:
        L.append("-" * 68)
        if not cmp.get("ok"):
            L.append(f"  对比失败：{cmp.get('reason')}")
        else:
            L.append(f"  基金 vs 基准（{cmp['start']} ~ {cmp['end']}，{cmp['n']} 天）")
            L.append(f"    基金年化 {cmp['fund_annual']*100:+.2f}%   "
                     f"基准年化 {cmp['bench_annual']*100:+.2f}%   "
                     f"超额 {cmp['excess_annual']*100:+.2f}%")
            L.append(f"    beta {cmp['beta']}   相关性 {cmp['correlation']}   "
                     f"跟踪误差 {cmp['tracking_error']*100:.2f}%   "
                     f"信息比率 {cmp['info_ratio']}")
    L.append("=" * 68)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 依赖 ===")
    print(available())

    print("\n=== 逐个指数取数 ===")
    for k in ("sp500", "ndx", "nikkei", "ftse"):
        r = index_history(k, days=300)
        if r["ok"]:
            s = r["series"]
            print(f"  {r['name']:<12} 来源={r['source']:<9} "
                  f"{len(s)} 行  最新 {s.iloc[-1]:,.2f}")
        else:
            print(f"  {k} 失败: {r.get('errors')}")

    print("\n=== 基金名 → 指数 ===")
    for nm, want in [("易方达标普500指数", "sp500"), ("广发纳斯达克100ETF联接", "ndx"),
                     ("华安三菱日联日经225ETF联接", "nikkei"),
                     ("建信富时100指数", "ftse"), ("华夏成长混合", None)]:
        m = match_fund_index(nm)
        ok = (m.get("key") == want) if want else (m["ok"] is False)
        print(f"  {nm:<24} → {m.get('index_name') or '无法识别'}  {'✓' if ok else '✗'}")
        assert ok, (nm, m)
    print(f"\nOK v{__version__}")
