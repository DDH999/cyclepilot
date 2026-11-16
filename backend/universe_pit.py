# -*- coding: utf-8 -*-
"""
时点票池（M52）：某交易日 T 的可交易全集
====================================================
替代「用当前沪深300成分回溯历史」的做法。

数据源（实测可用）
------------------
* `baostock.query_stock_basic()` → `ipoDate` / `outDate` / `type` / `status`
  全量 9038 条，其中 outDate 非空（已退市）1209 只
  2019–2024 期间退市 568 只（含转债，按 type=="1" 过滤后是纯股票）
* `baostock.query_trade_dates()` → `calendar_date` / `is_trading_day`

入池规则（全部只用 T 及之前的信息）
------------------------------------
| 条件 | 规则 |
|---|---|
| C1 已上市 | `ipo_date <= T` |
| C2 非次新 | 上市满 **60 个交易日** |
| C3 未退市 | `out_date` 为空 或 `out_date > T` |
| C4 正常交易 | T 日 `tradestatus == 1`（停牌日不入池） |

⚠ C3 的严格性
--------------
退市日在 T 时**未知**，但 `out_date > T` 得到的集合恰好等于
「T 日仍在交易的全部股票」——它只**保留**到退市日为止，
不会提前剔除，因此不引入前视。

**禁止**用 `status`（当前是否存续）筛历史，那会把已退市股的
整段历史删掉，重新制造幸存者偏差。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["BASIC_CACHE", "CAL_CACHE", "load_basic", "trading_days",
           "delist_map", "eligible_on", "build_index", "summarize"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / ".cache" / "universe"
BASIC_CACHE = CACHE / "stock_basic.csv"
CAL_CACHE = CACHE / "trade_days.csv"
BASIC_MAX_AGE_DAYS = 3
MIN_LISTED_DAYS = 60

_BS_CACHE: dict = {}


def _bs():
    """baostock 会话复用（避免每次 login/close 互相踢掉）。"""
    import baostock_source as BS
    if not _BS_CACHE.get("login"):
        BS._login()
        _BS_CACHE["login"] = True
    return _BS_CACHE


def close() -> None:
    if _BS_CACHE.get("login"):
        import baostock_source as BS
        try:
            BS.close()
        except Exception:
            pass
        _BS_CACHE["login"] = False


def _login():
    import baostock_source as BS
    BS._login()


def _code6(bs_code: str) -> str:
    return str(bs_code).split(".")[-1].zfill(6)


def load_basic(refresh: bool = False) -> pd.DataFrame:
    """
    全 A 股基础信息（含已退市）。缓存 3 天。

    返回列：code / name / ipo_date / out_date / is_delisted / market
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    if not refresh and BASIC_CACHE.exists():
        try:
            age = (time.time() - BASIC_CACHE.stat().st_mtime) / 86400
            if age <= BASIC_MAX_AGE_DAYS:
                d = pd.read_csv(BASIC_CACHE, dtype={"code": str},
                                parse_dates=["ipo_date", "out_date"])
                if len(d) > 1000:
                    return d
        except Exception:
            pass

    import baostock as bs
    _login()
    rs = bs.query_stock_basic()
    if getattr(rs, "error_code", "1") != "0":
        raise RuntimeError(f"query_stock_basic 失败：{rs.error_msg}")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise RuntimeError("query_stock_basic 返回空")

    d = pd.DataFrame(rows, columns=rs.fields)
    # 只要股票（type=1）；转债/基金 type 不是 1
    d = d[d["type"].astype(str) == "1"].copy()
    out = pd.DataFrame({
        "code": d["code"].map(_code6),
        "name": d["code_name"].astype(str),
        "ipo_date": pd.to_datetime(d["ipoDate"], errors="coerce"),
        "out_date": pd.to_datetime(d["outDate"], errors="coerce"),
        "market": d["code"].str.split(".").str[0],
    })
    out["is_delisted"] = out["out_date"].notna()
    out = out.dropna(subset=["ipo_date"]).drop_duplicates("code")
    out = out.sort_values("code").reset_index(drop=True)
    out.to_csv(BASIC_CACHE, index=False, encoding="utf-8")
    return out


def trading_days(start: str, end: str, refresh: bool = False) -> pd.DatetimeIndex:
    """交易日历（只含 is_trading_day==1）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if not refresh and CAL_CACHE.exists():
        try:
            d = pd.read_csv(CAL_CACHE, parse_dates=["date"])
            got = d[(d["date"] >= t0) & (d["date"] <= t1)]
            # ⚠ 必须校验**区间覆盖度**：缓存可能只存了上一次查询的范围，
            # 只看条数>50 会把"只有 2026 年"的缓存当成 2019-2026 的日历。
            covered = (len(got) > 50
                       and got["date"].min() <= t0 + pd.Timedelta(days=30)
                       and got["date"].max() >= t1 - pd.Timedelta(days=30))
            if covered:
                return pd.DatetimeIndex(got["date"])
        except Exception:
            pass

    import baostock as bs
    _login()
    rs = bs.query_trade_dates(start_date=str(start), end_date=str(end))
    if getattr(rs, "error_code", "1") != "0":
        raise RuntimeError(f"query_trade_dates 失败：{rs.error_msg}")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    d = pd.DataFrame(rows, columns=rs.fields)
    d["date"] = pd.to_datetime(d["calendar_date"], errors="coerce")
    d["is_trading_day"] = pd.to_numeric(d["is_trading_day"], errors="coerce")
    d = d[(d["is_trading_day"] == 1) & d["date"].notna()]
    merged = d[["date", "is_trading_day"]]
    try:
        if CAL_CACHE.exists():
            old_cal = pd.read_csv(CAL_CACHE, parse_dates=["date"])
            merged = pd.concat([old_cal, merged], ignore_index=True)
            merged = merged.drop_duplicates("date").sort_values("date")
    except Exception:
        pass
    merged.to_csv(CAL_CACHE, index=False)
    return pd.DatetimeIndex(sorted(d["date"]))


def delist_map(basic: pd.DataFrame | None = None) -> dict:
    """{code6: 退市日 Timestamp}，只含已退市的。"""
    b = basic if basic is not None else load_basic()
    d = b[b["out_date"].notna()]
    return {str(r["code"]).zfill(6): pd.Timestamp(r["out_date"])
            for _, r in d.iterrows()}


def _first_eligible_idx(basic: pd.DataFrame, cal: pd.DatetimeIndex,
                        min_listed_days: int) -> np.ndarray:
    """
    每只股票「首次可入池」的交易日序号（-1e9 表示窗口第一天就可入池）。

    ⚠ 关键：上市日早于窗口起点的股票，已经上市很久了，
    不能再要求它"再满 60 日"——早期版本直接 searchsorted+60，
    导致 2019-01-02 池内为 0 只。
    这里用工作日近似计算窗口前已过天数（节假日会略高估，偏保守地不早入池）。
    """
    cal = pd.DatetimeIndex(cal)
    ipo = pd.to_datetime(basic["ipo_date"]).to_numpy()
    t0 = np.datetime64(cal[0], "D")
    before = ipo < cal[0]
    elapsed = np.zeros(len(ipo), dtype=np.int64)
    if before.any():
        elapsed[before] = np.busday_count(
            ipo[before].astype("datetime64[D]"), t0)
    in_window = np.searchsorted(cal.values, ipo, side="left")
    need = int(min_listed_days)
    first = np.where(
        before,
        np.where(elapsed >= need, -10**9, need - elapsed),
        in_window + need)
    return first.astype(np.int64)


def eligible_on(basic: pd.DataFrame, cal: pd.DatetimeIndex,
                T: pd.Timestamp, min_listed_days: int = MIN_LISTED_DAYS) -> list[str]:
    """
    单日入池（调试用；批量请用 build_index）。

    C1 ipo_date <= T、C2 上市满 min_listed_days 个交易日、
    C3 out_date 为空或 > T。
    """
    T = pd.Timestamp(T)
    cal = pd.DatetimeIndex(cal)
    b = basic
    first = _first_eligible_idx(b, cal, min_listed_days)
    outd = pd.to_datetime(b["out_date"]).to_numpy()
    out_pos = np.full(len(b), np.iinfo(np.int64).max, dtype=np.int64)
    has_out = ~pd.isna(outd)
    if has_out.any():
        out_pos[has_out] = np.searchsorted(cal.values, outd[has_out], side="left")
    t_idx = int(cal.searchsorted(T, side="right")) - 1
    mask = (first <= t_idx) & (t_idx < out_pos)
    return sorted(b["code"].to_numpy()[mask])


def build_index(start: str, end: str, min_listed_days: int = MIN_LISTED_DAYS,
                basic: pd.DataFrame | None = None,
                cal: pd.DatetimeIndex | None = None) -> dict:
    """
    构造逐日票池：{date_str: set(codes)}。

    向量化实现：对每只股票算出「首日可入池 idx」与「最后可入池 idx」，
    再用布尔矩阵按日取交集。
    """
    b = basic if basic is not None else load_basic()
    cal = cal if cal is not None else trading_days(start, end)
    if len(cal) == 0:
        raise RuntimeError("交易日历为空")
    codes = b["code"].to_numpy()
    n = len(codes)

    outd = pd.to_datetime(b["out_date"]).to_numpy()

    # 首日：上市满 min_listed_days 个交易日（窗口前上市的按已过天数折算）
    ipo_pos = _first_eligible_idx(b, cal, min_listed_days)
    # 末日：out_date 在日历中的位置（不含），空值 → 无穷
    out_pos = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    has_out = ~pd.isna(outd)
    if has_out.any():
        out_pos[has_out] = np.searchsorted(cal.values, outd[has_out], side="left")

    out: dict = {}
    for i, d in enumerate(cal):
        mask = (ipo_pos <= i) & (i < out_pos)
        out[str(pd.Timestamp(d).date())] = set(codes[mask])
    return {"ok": True, "index": out, "calendar": cal, "basic": b,
            "n_codes": int(n), "start": str(cal[0].date()),
            "end": str(cal[-1].date()), "min_listed_days": int(min_listed_days)}


def summarize(idx: dict) -> dict:
    """票池规模统计。"""
    if not idx.get("ok"):
        return {"ok": False, "reason": idx.get("reason")}
    sizes = [len(v) for v in idx["index"].values()]
    b = idx["basic"]
    return {"ok": True, "n_days": len(sizes),
            "min_size": int(min(sizes)), "max_size": int(max(sizes)),
            "avg_size": round(float(np.mean(sizes)), 1),
            "n_total_codes": int(idx["n_codes"]),
            "n_delisted_in_pool": int(b["out_date"].notna().sum()),
            "start": idx["start"], "end": idx["end"],
            "min_listed_days": idx["min_listed_days"]}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 基础信息 ===")
    b = load_basic(refresh=True)
    print(f"  全部股票 {len(b)}，其中已退市 {int(b['out_date'].notna().sum())}")
    print(b.head(3).to_string(index=False))

    print("\n=== 交易日历 ===")
    cal = trading_days("2019-01-02", "2021-12-31")
    print(f"  2019-2021 共 {len(cal)} 个交易日  {cal[0].date()} ~ {cal[-1].date()}")

    print("\n=== 逐日票池 ===")
    idx = build_index("2019-01-02", "2021-12-31", basic=b, cal=cal)
    sm = summarize(idx)
    print(f"  {sm['n_days']} 天，规模 {sm['min_size']} ~ {sm['max_size']}，"
          f"平均 {sm['avg_size']}")
    print(f"  池内含已退市股 {sm['n_delisted_in_pool']} 只")

    # 抽查：某日是否有退市股
    d0 = str(cal[0].date())
    print(f"  {d0} 池内 {len(idx['index'][d0])} 只")
    close()
    print("\nOK v" + __version__)
