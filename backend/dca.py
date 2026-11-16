# -*- coding: utf-8 -*-
"""
定投台账与分析（M20）
====================================================
**定投不能用普通收益率。**

三种收益率的区别（这是最容易错的地方）：

| 口径 | 回答什么问题 | 定投该用哪个 |
|---|---|---|
| 简单收益率 = 市值/成本-1 | 我现在赚了百分之几 | 参考，但会高估 |
| 时间加权 TWR | 这个基金表现如何 | 评估基金用 |
| **资金加权 XIRR** | **我的钱实际年化多少** | ✅ **定投必须用这个** |

为什么：定投持续有新钱进来，早投的钱和晚投的钱"在场时间"不同。
用市值/成本算，等于假设所有钱第一天就投了。

台账格式（dca_ledger.csv）
--------------------------
    date,code,amount,nav
    2025-01-02,050025,1000,2.1500
    2025-02-03,050025,1000,2.2000

* date   申购确认日
* code   基金代码
* amount 投入金额（元）
* nav    当日单位净值（可留空，系统会按净值序列补）
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["LEDGER_DEFAULT", "load_ledger", "xirr", "fill_missing_nav",
           "position_summary", "dca_analysis", "compare_lump_sum",
           "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
LEDGER_DEFAULT = _ROOT / "dca_ledger.csv"


# ══════════════════════════════════════════════════════════
def load_ledger(path: Path | str = LEDGER_DEFAULT) -> pd.DataFrame:
    """
    读取定投台账。返回 date/code/amount/nav 四列的 DataFrame。

    缺 code 或 amount 的行直接丢弃并给出警告；date 解析失败也丢弃。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"定投台账不存在：{p}")

    raw = p.read_text(encoding="utf-8").lstrip(chr(0xFEFF))
    raw = chr(10).join(ln for ln in raw.splitlines()
                       if not ln.strip().startswith("#"))
    if not raw.strip():
        raise ValueError("台账为空（只有注释行）")

    df = pd.read_csv(pd.io.common.StringIO(raw), dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    d_col = col("date", "日期", "申购日", "确认日")
    c_col = col("code", "代码", "fund")
    a_col = col("amount", "金额", "投入")
    n_col = col("nav", "净值", "价格")
    if d_col is None or a_col is None:
        raise ValueError("台账至少需要 date 与 amount 两列")

    rows = []
    for _, r in df.iterrows():
        try:
            d = pd.to_datetime(str(r[d_col]).strip()).date()
            amt = float(str(r[a_col]).replace(",", "").strip())
        except Exception:
            continue
        if amt <= 0:
            continue
        code = "".join(ch for ch in str(r[c_col]).strip() if ch.isdigit()) \
            if c_col else ""
        nav = None
        if n_col is not None:
            try:
                v = str(r[n_col]).strip()
                if v and v.lower() != "nan":
                    nav = float(v)
            except Exception:
                nav = None
        rows.append({"date": d, "code": code.zfill(6) if code else "",
                     "amount": amt, "nav": nav})

    if not rows:
        raise ValueError("台账没有任何有效记录")
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def fill_missing_nav(ledger: pd.DataFrame,
                     navs: dict[str, pd.Series]) -> pd.DataFrame:
    """
    用净值序列补齐缺失的成交净值（取当日或之前最近一个交易日）。
    """
    out = ledger.copy()
    for i, row in out.iterrows():
        if row["nav"] is not None and np.isfinite(row["nav"]) and row["nav"] > 0:
            continue
        s = navs.get(row["code"])
        if s is None or len(s) == 0:
            continue
        ts = pd.Timestamp(row["date"])
        upto = s.loc[:ts]
        if len(upto) == 0:
            continue
        out.at[i, "nav"] = float(upto.iloc[-1])
    return out


# ══════════════════════════════════════════════════════════
# XIRR（资金加权年化收益率）
# ══════════════════════════════════════════════════════════
def xirr(cashflows, tol: float = 1e-8, max_iter: int = 300) -> dict:
    """
    XIRR：使 Σ cf_i / (1+r)^(t_i/365) = 0 的 r。

    cashflows: [(date 或 Timestamp, 金额)]，投入为负、期末价值为正。
    返回 {ok, rate, npv_at_rate, iterations, bracket}
    """
    if not cashflows or len(cashflows) < 2:
        return {"ok": False, "rate": None, "reason": "至少需要两笔现金流"}

    pts = sorted(((pd.Timestamp(d), float(a)) for d, a in cashflows),
                 key=lambda x: x[0])
    t0 = pts[0][0]
    flows = [((d - t0).days / 365.0, a) for d, a in pts]
    if not any(a < 0 for _, a in flows) or not any(a > 0 for _, a in flows):
        return {"ok": False, "rate": None,
                "reason": "现金流必须同时包含投入（负）与期末价值（正）"}

    def npv(r: float) -> float:
        if r <= -1:
            return float("inf")
        return float(sum(a / (1.0 + r) ** t for t, a in flows))

    lo, hi = -0.9999, 1.0
    # 先找符号变化区间
    f_lo, f_hi = npv(lo), npv(hi)
    tries = 0
    while np.sign(f_lo) == np.sign(f_hi) and tries < 60:
        hi *= 2.0
        f_hi = npv(hi)
        tries += 1
    if np.sign(f_lo) == np.sign(f_hi):
        return {"ok": False, "rate": None, "reason": "在合理区间内无解"}

    for it in range(1, int(max_iter) + 1):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return {"ok": True, "rate": float(mid), "npv_at_rate": f_mid,
                    "iterations": it, "bracket": (round(lo, 6), round(hi, 6))}
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return {"ok": True, "rate": float((lo + hi) / 2.0), "npv_at_rate": npv(mid),
            "iterations": int(max_iter), "bracket": (round(lo, 6), round(hi, 6))}


# ══════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════
def position_summary(ledger: pd.DataFrame,
                     current_navs: dict[str, float]) -> dict:
    """
    按基金汇总：投入、份额、平均成本、当前市值。

    缺现价的标的**不计入市值**，单独列出。
    """
    if len(ledger) == 0:
        return {"ok": False, "reason": "台账为空"}
    df = ledger.copy()
    df = df[df["nav"].notna() & (df["nav"] > 0)]
    if len(df) == 0:
        return {"ok": False, "reason": "没有任何有效净值，无法计算份额"}

    df = df.assign(shares=df["amount"] / df["nav"])
    per, missing = [], []
    for code, g in df.groupby("code"):
        shares = float(g["shares"].sum())
        invested = float(g["amount"].sum())
        avg_cost = invested / shares if shares > 0 else None
        cur = current_navs.get(code)
        if cur is None or not np.isfinite(cur) or cur <= 0:
            missing.append(code)
            mv = None
        else:
            mv = shares * float(cur)
        per.append({
            "code": code,
            "n_buys": int(len(g)),
            "invested": round(invested, 2),
            "shares": round(shares, 4),
            "avg_cost": round(avg_cost, 4) if avg_cost else None,
            "current_nav": round(float(cur), 4) if mv is not None else None,
            "market_value": round(mv, 2) if mv is not None else None,
            "pnl": round(mv - invested, 2) if mv is not None else None,
            "pnl_pct": round(mv / invested - 1, 6) if mv is not None and invested else None,
            "first_date": str(g["date"].min()),
            "last_date": str(g["date"].max()),
        })

    invested_total = float(sum(x["invested"] for x in per))
    mv_total = float(sum(x["market_value"] for x in per
                         if x["market_value"] is not None))
    return {
        "ok": True,
        "n_funds": len(per),
        "n_missing": len(missing),
        "missing": missing,
        "invested": round(invested_total, 2),
        "market_value": round(mv_total, 2),
        "pnl": round(mv_total - invested_total, 2) if mv_total else None,
        "pnl_pct": (round(mv_total / invested_total - 1, 6)
                    if invested_total > 0 and mv_total else None),
        "positions": sorted(per, key=lambda x: -(x["market_value"] or 0)),
        "warnings": ([f"{len(missing)} 只基金缺现价，未计入市值："
                      + "、".join(missing[:8])] if missing else []),
    }


def dca_analysis(ledger: pd.DataFrame, current_navs: dict[str, float],
                 as_of: date | str | None = None) -> dict:
    """
    定投全分析：汇总 + XIRR + 与一次性投入对比。

    XIRR 的期末价值按 as_of（默认今天）计算。
    """
    summ = position_summary(ledger, current_navs)
    if not summ.get("ok"):
        return summ

    end = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    flows = [(pd.Timestamp(r["date"]), -float(r["amount"])) for _, r in ledger.iterrows()]
    if summ["market_value"] > 0:
        flows.append((end, float(summ["market_value"])))

    x = xirr(flows)
    span_days = (end - pd.Timestamp(ledger["date"].min())).days
    out = dict(summ)
    out.update({
        "as_of": str(end.date()),
        "span_days": int(span_days),
        "span_years": round(span_days / 365.0, 2),
        "xirr": round(x["rate"], 6) if x.get("ok") else None,
        "xirr_ok": bool(x.get("ok")),
        "xirr_reason": None if x.get("ok") else x.get("reason"),
        "n_buys_total": int(len(ledger)),
    })

    # 定投 vs 一次性：把同样的总投入在第一天一次性投入
    first = ledger.sort_values("date").iloc[0]
    first_nav = float(first["nav"]) if first.get("nav") else None
    first_code = first["code"]
    cur_first = current_navs.get(first_code)
    if first_nav and cur_first and first_nav > 0:
        shares_lump = summ["invested"] / first_nav
        mv_lump = shares_lump * float(cur_first)
        out["lump_sum"] = {
            "assumption": f"全部 {summ['invested']:,.0f} 元在 "
                          f"{first['date']} 按 {first_code} 净值 {first_nav} 一次性买入",
            "market_value": round(mv_lump, 2),
            "pnl": round(mv_lump - summ["invested"], 2),
            "pnl_pct": round(mv_lump / summ["invested"] - 1, 6),
            "dca_minus_lump": round((summ["market_value"] or 0) - mv_lump, 2),
            "note": "若定投跑输一次性，说明这段时间是单边上行；"}
    return out


def compare_lump_sum(ledger: pd.DataFrame, nav_series: pd.Series,
                     code: str | None = None) -> dict:
    """
    精确对比：同一只基金、同样总投入，定投 vs 第一天一次性买入。
    """
    df = ledger if code is None else ledger[ledger["code"] == code]
    if len(df) == 0:
        return {"ok": False, "reason": "无该基金记录"}
    df = df.dropna(subset=["nav"])
    if len(df) == 0:
        return {"ok": False, "reason": "缺净值"}
    total = float(df["amount"].sum())
    shares_dca = float((df["amount"] / df["nav"]).sum())
    first = df.sort_values("date").iloc[0]
    shares_lump = total / float(first["nav"])
    last_nav = float(nav_series.iloc[-1]) if nav_series is not None and len(nav_series) else None
    if last_nav is None:
        return {"ok": False, "reason": "无最新净值"}
    mv_dca = shares_dca * last_nav
    mv_lump = shares_lump * last_nav
    return {
        "ok": True, "code": code or "合计", "total_invested": round(total, 2),
        "dca_shares": round(shares_dca, 4), "lump_shares": round(shares_lump, 4),
        "dca_value": round(mv_dca, 2), "lump_value": round(mv_lump, 2),
        "diff": round(mv_dca - mv_lump, 2),
        "verdict": ("定投更优（这段时间有下跌，摊低了成本）" if mv_dca > mv_lump
                    else "一次性更优（这段时间单边上行，越早买越好）"),
    }


def format_report(res: dict) -> str:
    L = ["=" * 68, "定投分析", "=" * 68]
    if not res.get("ok"):
        L.append(f"  无法分析：{res.get('reason')}")
        L.append("=" * 68)
        return chr(10).join(L)

    L.append(f"  截至 {res['as_of']}   跨度 {res['span_years']} 年   "
             f"定投 {res['n_buys_total']} 次")
    L.append(f"  累计投入 {res['invested']:>13,.2f}")
    L.append(f"  当前市值 {res['market_value']:>13,.2f}")
    if res.get("pnl") is not None:
        L.append(f"  浮动盈亏 {res['pnl']:>13,.2f}（{res['pnl_pct']*100:+.2f}%）")
    L.append("-" * 68)
    if res.get("xirr") is not None:
        L.append(f"  年化（XIRR，资金加权）{res['xirr']*100:+.2f}%   ← 这才是你的真实年化")
    else:
        L.append(f"  XIRR 无法计算：{res.get('xirr_reason')}")
    L.append("-" * 68)
    L.append(f"  {'代码':<8}{'次数':>5}{'投入':>12}{'平均成本':>10}"
             f"{'现净值':>9}{'市值':>12}{'盈亏%':>9}")
    L.append("  " + "-" * 62)
    for p in res["positions"]:
        pnl = "—" if p.get("pnl_pct") is None else f"{p['pnl_pct']*100:+.2f}"
        ac = "—" if p.get("avg_cost") is None else f"{p['avg_cost']:.4f}"
        cn = "—" if p.get("current_nav") is None else f"{p['current_nav']:.4f}"
        mv = "—" if p.get("market_value") is None else f"{p['market_value']:,.0f}"
        L.append(f"  {p['code']:<8}{p['n_buys']:>5}{p['invested']:>12,.0f}"
                 f"{ac:>10}{cn:>9}{mv:>12}{pnl:>9}")
    if res.get("lump_sum"):
        ls = res["lump_sum"]
        L.append("-" * 68)
        L.append(f"  对照（一次性）：{ls['assumption']}")
        L.append(f"    市值 {ls['market_value']:,.0f}   "
                 f"定投多赚/少赚 {ls['dca_minus_lump']:+,.0f} 元")
    if res.get("warnings"):
        L.append("-" * 68)
        for w in res["warnings"]:
            L.append(f"  ⚠ {w}")
    L.append("=" * 68)
    return chr(10).join(L)


if __name__ == "__main__":
    from datetime import timedelta

    # 1) XIRR 已知值：投 1000，一年后值 1100 → 年化 10%
    d0 = date(2024, 1, 2)
    x = xirr([(d0, -1000.0), (d0 + timedelta(days=365), 1100.0)])
    assert x["ok"], x
    assert abs(x["rate"] - 0.10) < 1e-4, x["rate"]
    print(f"XIRR 校验：-1000 → 1100（365 天）= {x['rate']*100:.4f}%  ✓")

    # 2) 两年翻倍 → 约 41.4%
    x2 = xirr([(d0, -1000.0), (d0 + timedelta(days=730), 2000.0)])
    assert abs(x2["rate"] - (2 ** 0.5 - 1)) < 1e-3, x2["rate"]
    print(f"XIRR 校验：-1000 → 2000（730 天）= {x2['rate']*100:.2f}%  ✓")

    # 3) 亏损 → 负 XIRR
    x3 = xirr([(d0, -1000.0), (d0 + timedelta(days=365), 800.0)])
    assert x3["ok"] and x3["rate"] < 0
    print(f"XIRR 校验：-1000 → 800（365 天）= {x3['rate']*100:.2f}%  ✓")

    # 4) 无效输入
    assert xirr([(d0, -1000.0)])["ok"] is False
    assert xirr([(d0, 100.0), (d0 + timedelta(days=30), 200.0)])["ok"] is False

    # 5) 台账 + 汇总
    (_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    fp = _ROOT / ".tmp" / "_dca_selftest.csv"
    fp.write_text(
        "date,code,amount,nav" + chr(10)
        + "2024-01-02,050025,1000,2.00" + chr(10)
        + "2024-07-01,050025,1000,1.80" + chr(10)
        + "2025-01-02,050025,1000,2.20" + chr(10),
        encoding="utf-8")
    led = load_ledger(fp)
    assert len(led) == 3, len(led)
    assert led["date"].is_monotonic_increasing

    navs = {"050025": 2.50}
    res = dca_analysis(led, navs, as_of="2025-07-01")
    assert res["ok"], res
    # 份额 = 500 + 555.56 + 454.55 = 1510.10；市值 = 3775.25
    assert abs(res["market_value"] - 1510.101 * 2.50) < 1.0, res["market_value"]
    assert res["xirr"] is not None and res["xirr"] > 0
    print()
    print(format_report(res))

    # 6) 定投 vs 一次性
    cmp_ = compare_lump_sum(led, pd.Series([2.0, 2.5]),
                            code="050025")
    assert cmp_["ok"], cmp_
    print(f"定投 vs 一次性：{cmp_['verdict']}（差 {cmp_['diff']:+,.0f} 元）")
    print(f"OK v{__version__}")
