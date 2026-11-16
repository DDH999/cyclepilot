# -*- coding: utf-8 -*-
"""
zzshare 数据源适配（M29）
====================================================
本机装了 `zzshare`（zzquant 出的股量化数据 SDK，v0.4.13）。
它是**独立的第三方数据源**，不经过 akshare，所以能让系统的取数更稳，
还带来两个 akshare 那边不好拿的东西：

1. **全市场估值快照**：指定交易日，一次拿 5000+ 只的 PE / PB / PCF / 市值
2. **行业板块成分**：104 个行业板块的归属

能力清单
--------
* `available()`      → 装了没、版本号
* `stock_daily()`    → 日线（已归一成系统的标准列）
* `market_valuation()` → 全市场估值快照
* `industry_map()`   → 股票 → 行业（有缓存，首次约 1 分钟）
* `plates()`         → 板块列表

⚠ 已知限制（实测）
------------------
* `stock_basic()` 的 `industry` 字段**全为空**（5603 行填充率 0%），不能用
* `finance_valuation` **当天取不到**（返回 0 行），要 T-1 或更早
* 行业映射需要逐个板块拉，首次慢，必须缓存
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["ZZ_AVAILABLE", "ZZ_VERSION", "available", "to_ts_code",
           "to_symbol", "stock_daily", "market_valuation", "industry_map",
           "plates", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "zzshare"
INDUSTRY_CACHE = CACHE_DIR / "industry.csv"
INDUSTRY_MAX_AGE_DAYS = 30


def _probe() -> tuple[bool, str | None]:
    try:
        import zzshare  # noqa: F401
        import importlib.metadata as md
        return True, md.version("zzshare")
    except Exception:
        return False, None


ZZ_AVAILABLE, ZZ_VERSION = _probe()


def available() -> dict:
    """zzshare 是否可用。"""
    return {"available": ZZ_AVAILABLE, "version": ZZ_VERSION,
            "module": "zzshare"}


# ══════════════════════════════════════════════════════════
# 代码格式
# ══════════════════════════════════════════════════════════
def to_ts_code(code: str) -> str:
    """600519 → 600519.SH；000001 → 000001.SZ；830799 → 830799.BJ"""
    c = "".join(ch for ch in str(code) if ch.isdigit()).zfill(6)
    if c.startswith(("60", "68", "51", "58", "11", "50", "90")):
        return f"{c}.SH"
    if c.startswith(("00", "30", "12", "15", "16", "18", "20")):
        return f"{c}.SZ"
    if c.startswith(("8", "4", "9")):
        return f"{c}.BJ"
    return f"{c}.SH"


def to_symbol(ts_code: str) -> str:
    """600519.SH → 600519"""
    return "".join(ch for ch in str(ts_code).split(".")[0] if ch.isdigit()).zfill(6)


# ══════════════════════════════════════════════════════════
# 日线
# ══════════════════════════════════════════════════════════
def stock_daily(code: str, days: int = 800,
                start: str | None = None,
                end: str | None = None) -> pd.DataFrame:
    """
    日线行情，归一成系统标准列（date/open/high/low/close/volume/amount）。

    zzshare 返回 ts_code/trade_date/open/high/low/close/vol/amount 等，
    这里统一成 date 索引前的一张表。
    """
    if not ZZ_AVAILABLE:
        raise RuntimeError("zzshare 未安装")
    import zzshare as zz

    ts = to_ts_code(code)
    # 快捷函数的参数名是 ts_code / start_date / end_date（YYYYMMDD）
    if start is None:
        start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 30)
                 ).strftime("%Y%m%d")
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")

    df = zz.daily(ts_code=ts, start_date=str(start).replace("-", ""),
                  end_date=str(end).replace("-", ""))
    if df is None or len(df) == 0:
        raise ValueError(f"zzshare 返回空（{ts}）")

    map_ = {"trade_date": "date", "ts_code": "ts_code",
            "open": "open", "high": "high", "low": "low", "close": "close",
            "vol": "volume", "volume": "volume", "amount": "amount",
            "pre_close": "pre_close", "pct_chg": "pct_chg"}
    d = df.rename(columns={c: map_.get(c, c) for c in df.columns})

    if "date" not in d.columns:
        raise ValueError("zzshare 返回缺 trade_date")
    d["date"] = pd.to_datetime(d["date"], format="%Y%m%d", errors="coerce")
    if d["date"].isna().all():
        d["date"] = pd.to_datetime(df.get("trade_date"), errors="coerce")

    for c in ("open", "high", "low", "close", "volume", "amount"):
        d[c] = pd.to_numeric(d.get(c, np.nan), errors="coerce")
    # amount 缺失时用 close×volume 估算（zzshare 的 vol 单位是手）
    if d["amount"].isna().all():
        d["amount"] = d["close"] * d["volume"] * 100

    out = d[["date", "open", "high", "low", "close", "volume", "amount"]] \
        .dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("zzshare 数据归一后为空")
    return out.tail(int(days)) if days else out


# ══════════════════════════════════════════════════════════
# 全市场估值
# ══════════════════════════════════════════════════════════
def market_valuation(date: str | None = None,
                     max_lookback_days: int = 7) -> dict:
    """
    全市场估值快照（指定交易日）。

    实测当天口径拿不到（返回空），会自动往前回退，最多 max_lookback_days 天。
    返回 {ok, date, df, n, tried}
    """
    if not ZZ_AVAILABLE:
        return {"ok": False, "reason": "zzshare 未安装"}

    import zzshare as zz

    base = pd.Timestamp(date) if date else pd.Timestamp.today().normalize()
    tried = []
    for k in range(int(max_lookback_days) + 1):
        d = (base - pd.Timedelta(days=k)).normalize()
        if d.weekday() >= 5:          # 周末直接跳过
            continue
        ds = d.strftime("%Y-%m-%d")
        tried.append(ds)
        try:
            df = zz.finance_valuation(date_value=ds)
        except Exception as ex:
            tried[-1] = f"{ds}({str(ex)[:40]})"
            continue
        if df is not None and len(df) > 100:
            df = df.copy()
            for c in ("pe_ratio", "pb_ratio", "pcf_ratio", "market_cap",
                      "circulating_market_cap"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "code" in df.columns:
                df["symbol"] = df["code"].map(to_symbol)
            return {"ok": True, "date": ds, "df": df, "n": int(len(df)),
                    "tried": tried}

    return {"ok": False, "reason": "最近交易日都取不到估值数据", "tried": tried}


def valuation_snapshot(date: str | None = None) -> dict:
    """全市场估值分布摘要（中位数、分位）。"""
    got = market_valuation(date)
    if not got.get("ok"):
        return got
    df = got["df"]
    out = {"ok": True, "date": got["date"], "n": got["n"]}
    for col, label in (("pe_ratio", "PE"), ("pb_ratio", "PB"),
                       ("pcf_ratio", "PCF")):
        if col not in df.columns:
            continue
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        pos = s[s > 0]
        out[label] = {
            "median": round(float(pos.median()), 3) if len(pos) else None,
            "p25": round(float(pos.quantile(0.25)), 3) if len(pos) else None,
            "p75": round(float(pos.quantile(0.75)), 3) if len(pos) else None,
            "n_positive": int(len(pos)),
            "n_negative": int((s <= 0).sum()),
        }
    return out


# ══════════════════════════════════════════════════════════
# 行业映射
# ══════════════════════════════════════════════════════════
def plates(plate_type: int = 14) -> pd.DataFrame:
    """板块列表。14=行业 15=概念 17=题材"""
    if not ZZ_AVAILABLE:
        raise RuntimeError("zzshare 未安装")
    import zzshare as zz
    df = zz.plates_list(plate_type=int(plate_type))
    # 实测：zzshare 的 plates_list 返回的是 list[dict]，不是 DataFrame。
    # 不统一成 DataFrame 的话，后面按列取就会报
    # "list indices must be integers or slices, not list"。
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if df is None or len(df) == 0:
        raise ValueError(f"板块列表为空（type={plate_type}）")
    return df


def industry_map(date: str | None = None, plate_type: int = 14,
                 cache: Path | str | None = None,
                 refresh: bool = False, max_plates: int | None = None) -> dict:
    """
    股票 → 行业板块 映射（首次约 1 分钟，之后走缓存）。

    返回 {ok, mapping: {symbol: plate_name}, n, source}
    """
    # 默认值必须在调用时解析：写成参数默认值会在定义时绑定，
    # 之后改模块级 INDUSTRY_CACHE 不生效（测试无法重定向，实测踩过）。
    cache = Path(cache) if cache is not None else Path(INDUSTRY_CACHE)
    if cache.exists() and not refresh:
        try:
            age_d = (time.time() - cache.stat().st_mtime) / 86400
            if age_d <= INDUSTRY_MAX_AGE_DAYS:
                df = pd.read_csv(cache, dtype={"symbol": str, "plate_name": str})
                # 守卫不能用 >100：小样本（或测试用小缓存）会永远读不到，
                # 每次都退回实时拉取，缓存形同虚设。
                if len(df) > 0:
                    return {"ok": True, "n": int(len(df)), "source": "cache",
                            "mapping": dict(zip(df["symbol"], df["plate_name"]))}
        except Exception:
            pass

    if not ZZ_AVAILABLE:
        return {"ok": False, "reason": "zzshare 未安装"}

    import zzshare as zz
    d = pd.Timestamp(date).strftime("%Y-%m-%d") if date else \
        pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        pl = plates(plate_type)
    except Exception as ex:
        return {"ok": False, "reason": f"取板块列表失败：{ex}"}

    rows = []
    items = list(pl[["plate_code", "plate_name"]].itertuples(index=False))
    if max_plates:
        items = items[:int(max_plates)]

    for code, name in items:
        try:
            st = zz.plates_stocks(plate_type=int(plate_type),
                                  plate_code=str(code), date=d)
            if isinstance(st, list):
                st = pd.DataFrame(st)
            if st is None or len(st) == 0:
                continue
            # 实测 plates_stocks 的代码列叫 stock_code，
            # 只找 code/ts_code 会全部被跳过（踩过）。
            col = next((c for c in ("code", "ts_code", "stock_code", "symbol")
                        if c in st.columns), None)
            if col is None:
                continue
            for v in st[col].astype(str):
                rows.append({"symbol": to_symbol(v), "plate_name": str(name)})
        except Exception:
            continue

    if not rows:
        return {"ok": False, "reason": "所有板块都没取到成分股"}

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"])
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, encoding="utf-8")
    except Exception:
        pass
    return {"ok": True, "n": int(len(df)), "source": "live",
            "mapping": dict(zip(df["symbol"], df["plate_name"]))}


def format_report(info: dict, val: dict | None = None) -> str:
    L = ["=" * 66, "zzshare 数据源", "=" * 66]
    L.append(f"  状态: {'可用' if info.get('available') else '不可用'}"
             + (f"   版本 {info.get('version')}" if info.get("version") else ""))
    if val:
        L.append("-" * 66)
        if not val.get("ok"):
            L.append(f"  估值快照: {val.get('reason')}")
            if val.get("tried"):
                L.append(f"    尝试过: {val['tried'][:5]}")
        else:
            L.append(f"  估值快照 {val['date']}   覆盖 {val['n']} 只")
            for k in ("PE", "PB", "PCF"):
                if k in val:
                    v = val[k]
                    L.append(f"    {k:<4} 中位 {v['median']}   "
                             f"P25 {v['p25']}   P75 {v['p75']}   "
                             f"(正数 {v['n_positive']} / 非正 {v['n_negative']})")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    info = available()
    print(f"zzshare 可用: {info['available']}  版本 {info['version']}")
    if not info["available"]:
        raise SystemExit("未安装 zzshare，跳过自测")

    print("\n=== 代码格式 ===")
    for c, want in [("600519", "600519.SH"), ("000001", "000001.SZ"),
                    ("300750", "300750.SZ"), ("830799", "830799.BJ")]:
        got = to_ts_code(c)
        assert got == want, (c, got)
        assert to_symbol(got) == c
    print("  6 位 ↔ ts_code 互转 ✓")

    print("\n=== 日线 ===")
    df = stock_daily("600519", days=60)
    assert len(df) > 20, len(df)
    assert {"date", "open", "high", "low", "close", "volume", "amount"}.issubset(df.columns)
    print(f"  {len(df)} 行，最新 {df['date'].iloc[-1].date()} 收盘 {df['close'].iloc[-1]:.2f}")

    print("\n=== 全市场估值 ===")
    snap = valuation_snapshot()
    print(format_report(info, snap))
    assert snap.get("ok"), snap
    assert snap["n"] > 1000, snap["n"]
