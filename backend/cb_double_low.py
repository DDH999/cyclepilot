# -*- coding: utf-8 -*-
"""
可转债双低轮动（M43）
====================================================
「双低」= 价格低 + 转股溢价率低。散户可做、有实证超额的方向。

为什么双低可能有效
------------------
* **价格低** → 债性保护，下行有限（只要不违约）
* **溢价率低** → 跟涨能力强，正股上涨时转债能跟上
* 两者同时低 = 「下有底、上有弹性」，是转债的凸性来源

本模块提供三件事
----------------
1. **实时候选池**：当前双低值最低的 N 只（可今天就用）
2. **强赎规避**：转股价值接近 130 的转债会被强赎，**强赎 = 亏 30%+**
3. **历史回测**：2021~2025 全市场重建横截面，月度调仓

⚠ 必须说清楚的三个风险
----------------------
1. **幸存者偏差**：历史接口只返回**当前仍存续**的转债，
   已退市/已强赎的不在里面 → 回测结果**不可信地偏乐观**。
   本模块会把这个警告打在报告最前面。
2. **信用风险**：2024 年 8 月低价债批量崩盘（搜特转债违约），
   纯按"价格低"选会踩雷。所以要叠加评级和正股质量过滤。
3. **数据陈旧**：历史接口实测止于 2025-07，实时池子才是当下的。
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["CACHE_DIR", "load_current", "double_low_value", "screener",
           "redeem_risk", "fetch_history", "build_panel",
           "backtest_double_low", "format_screen", "format_backtest"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "cb"
HIST_DIR = CACHE_DIR / "hist"
HIST_MAX_AGE_HOURS = 24

# 强赎触发线：转股价值 ≥130 触发（多数条款 15/30 日）
REDEEM_LINE = 130.0
# 双低筛选的硬性排除（防踩信用雷）
MAX_PRICE = 135.0          # 价格太高失去债性保护
MIN_RATING_OK = ("AAA", "AA+", "AA", "AA-")   # 低于 AA- 直接排除


def _to_num(s) -> pd.Series:
    return pd.to_numeric(pd.Series(s).astype(str)
                         .str.replace("%", "", regex=False)
                         .str.replace(",", "", regex=False),
                         errors="coerce")


def load_current(refresh: bool = False) -> pd.DataFrame:
    """当前全市场转债横截面（含债现价、转股价、正股价、评级）。"""
    import cb_new as CB
    return CB.load_bonds(refresh=refresh)


def double_low_value(price: float, premium_pct: float) -> float:
    """
    双低值 = 价格 + 转股溢价率(%)。

    越小越好。常见做法是 price + premium×100，
    这样 100 元 + 10% 溢价 = 110，与"价格"同量纲，便于理解。
    """
    return float(price) + float(premium_pct)


def screener(top: int = 15, refresh: bool = False) -> dict:
    """
    实时双低候选池。

    过滤：已上市、价格 ≤MAX_PRICE、评级 ≥AA-、转股价值 <130（避开强赎）
    """
    df = load_current(refresh=refresh)
    if df is None or len(df) == 0:
        return {"ok": False, "reason": "转债列表为空"}

    d = df.copy()
    price = _to_num(d.get("债现价"))
    conv = _to_num(d.get("转股价"))
    stock = _to_num(d.get("正股价"))
    cv_field = _to_num(d.get("转股价值"))
    prem_field = _to_num(d.get("转股溢价率"))

    # 转股价值：优先字段，缺失则现算
    cv = cv_field.where(cv_field > 0, 100.0 / conv * stock)
    # 溢价率：优先字段，缺失则现算
    prem = prem_field.where(prem_field.notna(),
                            (price / cv - 1.0) * 100.0)

    d["_price"] = price
    d["_cv"] = cv
    d["_prem"] = prem
    d["_rating"] = d.get("信用评级", "").astype(str).str.strip()
    d["_size"] = _to_num(d.get("发行规模"))
    d["_dl"] = price + prem

    listed = d[(d["_price"] > 0) & (d["_cv"] > 0) & d["_prem"].notna()].copy()

    excluded: list[str] = []
    n0 = len(listed)
    # 价格过高
    hi = listed[listed["_price"] > MAX_PRICE]
    excluded.append(f"价格 >{MAX_PRICE:.0f}：{len(hi)} 只")
    listed = listed[listed["_price"] <= MAX_PRICE]
    # 评级过低
    def rating_ok(r: str) -> bool:
        if not r or r == "nan":
            return False
        return any(g in r for g in MIN_RATING_OK)
    bad_rating = listed[~listed["_rating"].map(rating_ok)]
    excluded.append(f"评级低于 AA- 或缺失：{len(bad_rating)} 只")
    listed = listed[listed["_rating"].map(rating_ok)]
    # 强赎风险
    near = listed[listed["_cv"] >= REDEEM_LINE]
    excluded.append(f"转股价值 ≥{REDEEM_LINE:.0f}（强赎区）：{len(near)} 只")
    listed = listed[listed["_cv"] < REDEEM_LINE]

    if len(listed) == 0:
        return {"ok": False, "reason": "过滤后没有候选",
                "excluded": excluded, "n_before": n0}

    top_df = listed.nsmallest(int(top), "_dl")
    items = []
    for _, row in top_df.iterrows():
        items.append({
            "code": str(row.get("债券代码", "")),
            "name": str(row.get("债券简称", "")),
            "stock": str(row.get("正股简称", "")),
            "price": round(float(row["_price"]), 2),
            "premium": round(float(row["_prem"]), 2),
            "convert_value": round(float(row["_cv"]), 2),
            "double_low": round(float(row["_dl"]), 2),
            "rating": row["_rating"],
            "size": (round(float(row["_size"]), 2)
                     if pd.notna(row["_size"]) else None),
        })
    return {"ok": True, "n_total": int(len(df)), "n_listed": n0,
            "n_after_filter": int(len(listed)), "excluded": excluded,
            "items": items, "top": int(top),
            "rule": (f"双低值 = 价格 + 溢价率(%)，"
                     f"价格≤{MAX_PRICE:.0f}，评级≥AA-，"
                     f"转股价值<{REDEEM_LINE:.0f}")}


def redeem_risk() -> dict:
    """
    强赎风险名单：转股价值接近/超过 130 的转债。

    被强赎 = 只能按面值+利息拿回（约 100~102），
    如果你在 130 买入，直接亏 25%+。
    """
    df = load_current()
    d = df.copy()
    cv_field = _to_num(d.get("转股价值"))
    conv = _to_num(d.get("转股价"))
    stock = _to_num(d.get("正股价"))
    cv = cv_field.where(cv_field > 0, 100.0 / conv * stock)
    price = _to_num(d.get("债现价"))

    d["_cv"] = cv
    d["_price"] = price
    hit = d[(d["_cv"] >= REDEEM_LINE) & (d["_price"] > 0)].copy()
    hit = hit.sort_values("_cv", ascending=False)
    items = []
    for _, row in hit.head(40).iterrows():
        items.append({"code": str(row.get("债券代码", "")),
                      "name": str(row.get("债券简称", "")),
                      "price": round(float(row["_price"]), 2),
                      "convert_value": round(float(row["_cv"]), 2)})
    return {"ok": True, "n": int(len(hit)), "items": items,
            "note": (f"转股价值 ≥{REDEEM_LINE:.0f} 已进入强赎触发区。"
                     f"持有者应尽快转股或卖出，不要等公告。")}


# ══════════════════════════════════════════════════════════
# 历史与回测
# ══════════════════════════════════════════════════════════
def _hist_path(code: str) -> Path:
    return HIST_DIR / f"{str(code).zfill(6)}.csv"


def fetch_history(codes: list[str], max_bonds: int | None = None,
                  refresh: bool = False) -> dict:
    """
    逐只拉转债历史（收盘价 + 转股溢价率），24 小时缓存。

    一次调用含网络请求，建议配合缓存使用。
    """
    import akshare as ak
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    todo = list(dict.fromkeys([str(c).zfill(6) for c in codes]))
    if max_bonds:
        todo = todo[:int(max_bonds)]

    got, failed = {}, []
    for i, code in enumerate(todo, 1):
        fp = _hist_path(code)
        if not refresh and fp.exists():
            try:
                if (time.time() - fp.stat().st_mtime) / 3600 <= HIST_MAX_AGE_HOURS:
                    df = pd.read_csv(fp, parse_dates=["date"])
                    if len(df) > 30:
                        got[code] = df
                        continue
            except Exception:
                pass
        pref = "sh" if code.startswith("11") else "sz"
        try:
            raw = ak.bond_zh_cov_value_analysis(symbol=code)
            if raw is None or len(raw) == 0:
                failed.append(code)
                continue
            df = pd.DataFrame({
                "date": pd.to_datetime(raw["日期"], errors="coerce"),
                "close": _to_num(raw["收盘价"]),
                "convert_value": _to_num(raw["转股价值"]),
                "premium": _to_num(raw["转股溢价率"]),
            }).dropna(subset=["date"])
            df = df[df["close"] > 0]
            if len(df) < 30:
                failed.append(code)
                continue
            df.to_csv(fp, index=False, encoding="utf-8")
            got[code] = df
        except Exception:
            failed.append(code)
        if i % 20 == 0:
            time.sleep(0.2)
    return {"ok": len(got) > 0, "n_ok": len(got), "n_fail": len(failed),
            "history": got, "failed": failed[:20]}


def build_panel(hist: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    拼成 date × code 的价格矩阵与溢价率矩阵。
    """
    px, pm = {}, {}
    for code, df in hist.items():
        s = df.dropna(subset=["date"]).set_index("date")
        px[code] = s["close"]
        pm[code] = s["premium"]
    price = pd.DataFrame(px).sort_index()
    prem = pd.DataFrame(pm).sort_index()
    return price, prem


def backtest_double_low(price: pd.DataFrame, premium: pd.DataFrame,
                       top_n: int = 10, rebalance_days: int = 20,
                       cost_per_turn: float = 0.0010) -> dict:
    """
    双低轮动回测：每 rebalance_days 个交易日，买双低值最低的 top_n 只，等权持有。

    ⚠ 只在「当天有价格」的转债里选，避免用未来数据。
    """
    if price.shape[1] < 3:
        return {"ok": False, "reason": f"可用转债仅 {price.shape[1]} 只"}
    dates = price.index
    if len(dates) < rebalance_days * 3:
        return {"ok": False, "reason": f"历史仅 {len(dates)} 天，太短"}

    records, nav, nav_dates, holdings_log = [], [1.0], [dates[0]], []
    prev_hold: set[str] = set()
    i = 0
    while i < len(dates) - rebalance_days:
        d0 = dates[i]
        p0 = price.iloc[i]
        pr0 = premium.iloc[i]
        ok = p0.notna() & pr0.notna() & (p0 > 0)
        if ok.sum() < top_n:
            i += rebalance_days
            continue
        dl = (p0 + pr0)[ok].sort_values()
        pick = list(dl.index[:top_n])
        holdings_log.append({"date": str(d0.date()), "n": len(pick),
                             "picks": pick[:5]})
        # 持有区间：从 d0 到 d0+rebalance（用下一期价格）
        seg = price.iloc[i:i + rebalance_days + 1][pick]
        rets = seg.pct_change(fill_method=None).iloc[1:]
        # 等权日收益
        daily = rets.mean(axis=1, skipna=True)
        # 换手成本
        turnover = len(set(pick) - prev_hold) / max(len(pick), 1)
        if daily.empty:
            i += rebalance_days
            continue
        daily.iloc[0] -= turnover * cost_per_turn
        for k, v in daily.items():
            if np.isfinite(v):
                nav.append(nav[-1] * (1.0 + float(v)))
                nav_dates.append(k)
        prev_hold = set(pick)
        i += rebalance_days

    if len(nav) < 50:
        return {"ok": False, "reason": f"有效净值点仅 {len(nav)}"}

    s = pd.Series(nav, index=pd.DatetimeIndex(nav_dates))
    s = s[~s.index.duplicated(keep="last")]
    r = s.pct_change(fill_method=None).dropna()
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    ann = float(s.iloc[-1] ** (1 / yrs) - 1)
    vol = float(r.std(ddof=0) * np.sqrt(252))
    dd = float((s / s.cummax() - 1).min())
    sharpe = float((r.mean() * 252) / vol) if vol > 0 else None

    return {"ok": True, "nav": s, "start": str(s.index[0].date()),
            "end": str(s.index[-1].date()), "years": round(yrs, 2),
            "annual": round(ann, 4), "vol": round(vol, 4),
            "sharpe": (round(sharpe, 3) if sharpe is not None else None),
            "max_drawdown": round(dd, 4),
            "final_nav": round(float(s.iloc[-1]), 4),
            "n_bonds_available": int(price.shape[1]),
            "top_n": int(top_n), "rebalance_days": int(rebalance_days),
            "rebalances": len(holdings_log),
            "sample_holdings": holdings_log[-3:],
            "warnings": [
                "**幸存者偏差**：历史接口只含当前仍存续的转债，"
                "已退市/已强赎的不在样本里 → 结果不可信地偏乐观",
                "未包含分红、利息，未考虑流动性冲击",
                "2024 年低价债信用事件期间，纯低价策略会出现大幅回撤",
            ]}


def format_screen(res: dict) -> str:
    L = ["=" * 72, "双低候选池（可转债）", "=" * 72]
    if not res.get("ok"):
        L.append(f"  无法生成：{res.get('reason')}")
        L.append("=" * 72)
        return "\n".join(L)
    L.append(f"  全市场 {res['n_total']} 只，已上市 {res['n_listed']} 只，"
             f"过滤后 {res['n_after_filter']} 只")
    L.append(f"  规则：{res['rule']}")
    for e in res.get("excluded") or []:
        L.append(f"    排除 {e}")
    L.append("-" * 72)
    L.append("  " + f"{'代码':<8}{'名称':<10}{'正股':<8}{'价格':>7}"
             f"{'溢价%':>8}{'转股价值':>9}{'双低':>8}{'评级':>6}")
    L.append("  " + "-" * 68)
    for x in res["items"]:
        L.append("  " + f"{x['code']:<8}{x['name'][:8]:<10}"
                 f"{x['stock'][:6]:<8}{x['price']:>7.2f}"
                 f"{x['premium']:>8.1f}{x['convert_value']:>9.1f}"
                 f"{x['double_low']:>8.1f}{x['rating']:>6}")
    L.append("=" * 72)
    L.append("  ⚠ 候选池 ≠ 买入信号。双低只筛「下有底、上有弹性」，")
    L.append("     不排除违约与强赎，务必分散（≥10 只）并检查强赎名单。")
    L.append("=" * 72)
    return "\n".join(L)


def format_backtest(res: dict) -> str:
    L = ["=" * 72, "双低轮动回测", "=" * 72]
    if not res.get("ok"):
        L.append(f"  无法回测：{res.get('reason')}")
        L.append("=" * 72)
        return "\n".join(L)
    L.append(f"  区间 {res['start']} ~ {res['end']}（{res['years']} 年）")
    L.append(f"  参数：持仓 {res['top_n']} 只，每 {res['rebalance_days']} 日调仓"
             f"，共调仓 {res['rebalances']} 次")
    L.append(f"  样本：{res['n_bonds_available']} 只转债")
    L.append("-" * 72)
    L.append(f"  总收益     {(res['final_nav']-1)*100:>8.2f}%")
    L.append(f"  年化       {res['annual']*100:>8.2f}%")
    L.append(f"  波动       {res['vol']*100:>8.2f}%")
    L.append(f"  夏普       {(res['sharpe'] or 0):>8.2f}")
    L.append(f"  最大回撤   {res['max_drawdown']*100:>8.2f}%")
    L.append("-" * 72)
    L.append("  ⚠ 必读警告：")
    for w in res["warnings"]:
        L.append(f"    · {w}")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    assert abs(double_low_value(105.0, 12.0) - 117.0) < 1e-9
    assert double_low_value(100.0, 5.0) < double_low_value(120.0, 5.0)
    print("双低值计算 ✓")

    # 回测逻辑离线自测：构造 4 只转债
    idx = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(7)
    names = ["A", "B", "C", "D"]
    px = pd.DataFrame({n: 100 * np.cumprod(
        1 + rng.normal(0.0003 + 0.0002 * i, 0.012, 400))
        for i, n in enumerate(names)}, index=idx)
    pm = pd.DataFrame({n: 10 + 3 * i + rng.normal(0, 1, 400)
                       for i, n in enumerate(names)}, index=idx)
    bt = backtest_double_low(px, pm, top_n=2, rebalance_days=20)
    print(f"回测 ✓ {bt.get('ok')} 年化 {bt.get('annual')} "
          f"回撤 {bt.get('max_drawdown')} 调仓 {bt.get('rebalances')}")
    assert bt["ok"]
    print(format_backtest(bt))
    print("\nOK v" + __version__)
