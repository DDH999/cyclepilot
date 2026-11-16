# -*- coding: utf-8 -*-
"""
真实持仓台账（M18）
====================================================
这是"自己投资用"那条路线的起点：**先把你的真实持仓变成系统能算的东西。**

能力
----
1. 读取持仓 CSV（代码 / 股数 / 成本价，可选现金）
2. 拉最新价 → 市值、成本、盈亏、权重
3. 输出可直接喂给风控与归因的权重序列
4. 缺价的标的**不估值、不猜**，单独列出来告警

文件格式（holdings.csv）
------------------------
    code,shares,cost
    600519,100,1400.00
    000001,2000,11.20
    CASH,50000,          # 现金行：code 写 CASH，shares 为金额，cost 留空

* `cost` 可省略（则不算成本与盈亏，只算市值与权重）
* 除权除息后成本会被摊薄，这里按你填的成本算，不自动调整

依赖：numpy / pandas
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["Holding", "load_holdings", "mark_to_market", "weights_of",
           "fetch_last_prices", "fetch_prices_for", "groups_for",
           "format_report", "HOLDINGS_DEFAULT", "PRICE_CACHE"]

_ROOT = Path(__file__).resolve().parent.parent
HOLDINGS_DEFAULT = _ROOT / "holdings.csv"
PRICE_CACHE = _ROOT / ".cache" / "last_price.json"


class Holding:
    """
    一条持仓。

    kind: "stock"（A股）/ "fund"（场外基金）/ "etf"（场内ETF）/ "cash"
    基金必须显式声明 kind —— 因为基金代码和股票代码会撞号
    （000001 既是平安银行，也是华夏成长混合）。
    """

    __slots__ = ("code", "shares", "cost", "kind")

    def __init__(self, code: str, shares: float, cost: float | None = None,
                 kind: str = "stock"):
        self.code = code
        self.shares = float(shares)
        self.cost = None if cost is None else float(cost)
        self.kind = kind if kind in ("stock", "fund", "etf", "cash") else "stock"

    def __repr__(self) -> str:
        return f"Holding({self.code}, {self.shares}, {self.cost}, {self.kind})"

    def to_dict(self) -> dict:
        return {"code": self.code, "shares": self.shares,
                "cost": self.cost, "kind": self.kind}


# ══════════════════════════════════════════════════════════
def load_holdings(path: Path | str = HOLDINGS_DEFAULT) -> dict:
    """
    读取持仓 CSV。

    返回 {holdings: [Holding], cash: float, warnings: [str]}
    文件不存在 → 抛 FileNotFoundError（由调用方决定是否降级）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"持仓文件不存在：{p}")

    raw = p.read_text(encoding="utf-8")
    raw = raw.lstrip(chr(0xFEFF))
    # 支持 # 注释行（用户会在模板里写说明）
    raw = chr(10).join(ln for ln in raw.splitlines()
                      if not ln.strip().startswith("#"))
    if not raw.strip():
        raise ValueError("持仓文件为空（只有注释行）")
    df = pd.read_csv(pd.io.common.StringIO(raw), dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_col = col("code", "代码", "symbol")
    s_col = col("shares", "股数", "qty", "quantity", "数量")
    p_col = col("cost", "成本", "cost_price", "成本价")
    t_col = col("type", "类型", "kind", "品种")
    if c_col is None or s_col is None:
        raise ValueError("持仓文件需要至少两列：code（代码）与 shares（股数）")

    holdings: list[Holding] = []
    cash = 0.0
    warns: list[str] = []
    for _, row in df.iterrows():
        code_raw = str(row[c_col]).strip()
        if not code_raw or code_raw.lower() == "nan":
            continue
        digits = "".join(ch for ch in code_raw if ch.isdigit())

        try:
            shares = float(str(row[s_col]).replace(",", "").strip())
        except Exception:
            warns.append(f"{code_raw}: 股数无法解析，已跳过")
            continue
        if shares == 0:
            continue

        if code_raw.upper().startswith("CASH") or code_raw in ("现金", "cash"):
            cash += shares
            continue

        if len(digits) != 6:
            warns.append(f"{code_raw}: 不是 6 位 A 股代码，已跳过")
            continue

        cost = None
        if p_col is not None:
            try:
                v = str(row[p_col]).replace(",", "").strip()
                if v and v.lower() != "nan":
                    cost = float(v)
            except Exception:
                cost = None

        kind = "stock"
        if t_col is not None:
            tv = str(row[t_col]).strip().lower()
            if tv in ("fund", "基金", "of", "场外"):
                kind = "fund"
            elif tv in ("etf", "场内", "lof"):
                kind = "etf"
            elif tv in ("cash", "现金"):
                kind = "cash"
        holdings.append(Holding(digits, shares, cost, kind))

    if not holdings and cash <= 0:
        raise ValueError("持仓为空：没有任何有效标的或现金")

    return {"holdings": holdings, "cash": cash, "warnings": warns}


def fetch_last_prices(codes: list[str], loader=None,
                      cache: Path | str = PRICE_CACHE,
                      max_age_hours: float = 6.0,
                      refresh: bool = False) -> dict:
    """
    批量取最新价（带本地缓存）。

    loader(code) -> float，缺省用 daily_job.fetch_daily 的收盘价。
    取不到的标的**不写入缓存**，返回结果里也没有它（由调用方告警）。
    """
    import json
    import time

    cache = Path(cache)
    prices: dict[str, float] = {}
    if cache.exists() and not refresh:
        try:
            age_h = (time.time() - cache.stat().st_mtime) / 3600
            if age_h <= max_age_hours:
                raw = json.loads(cache.read_text(encoding="utf-8"))
                for c in codes:
                    v = raw.get(c)
                    if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                        prices[c] = float(v)
                if len(prices) == len(codes):
                    return prices
        except Exception:
            pass

    if loader is None:
        def loader(code):  # type: ignore
            import daily_job as DJ
            df = DJ.fetch_daily(code, days=10)
            if df is None or len(df) == 0:
                raise ValueError("no data")
            return float(df["close"].iloc[-1])

    for c in codes:
        if c in prices:
            continue
        try:
            v = float(loader(c))
            if np.isfinite(v) and v > 0:
                prices[c] = v
        except Exception:
            continue

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(prices, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    except Exception:
        pass
    return prices


def fetch_prices_for(data: dict, refresh: bool = False,
                     max_age_hours: float = 6.0) -> dict:
    """
    按持仓类型分别取价。

    * stock → akshare 日线收盘价
    * fund / etf → 基金单位净值（场外有 T+1 延迟）
    * 取不到的标的不返回（由 mark_to_market 记为 missing）
    """
    holdings = data.get("holdings", [])
    stock_codes = [h.code for h in holdings if h.kind == "stock"]
    fund_codes = [h.code for h in holdings if h.kind in ("fund", "etf")]

    prices: dict[str, float] = {}
    if stock_codes:
        prices.update(fetch_last_prices(stock_codes, refresh=refresh,
                                        max_age_hours=max_age_hours))
    if fund_codes:
        import fund as FD
        for h in holdings:
            if h.kind not in ("fund", "etf"):
                continue
            try:
                prices[h.code] = FD.latest_nav(h.code, kind=h.kind)
            except Exception:
                continue
    return prices


def groups_for(data: dict) -> dict:
    """
    分组映射（供风格暴露使用）。

    * 基金 → 按基金大类（股票型/混合型/债券型/货币型/指数型/QDII/FOF）
    * 股票 → 行业（若 .cache/industries.csv 存在），否则「未知」
    """
    groups: dict[str, str] = {}
    holdings = data.get("holdings", [])
    fund_codes = [h.code for h in holdings if h.kind in ("fund", "etf")]
    if fund_codes:
        try:
            import fund as FD
            cat = FD.load_catalog()
            for c in fund_codes:
                info = FD.lookup_fund(c, cat)
                groups[c] = info["bucket"] if info else "未知"
        except Exception:
            for c in fund_codes:
                groups[c] = "未知"

    stock_codes = [h.code for h in holdings if h.kind == "stock"]
    if stock_codes:
        ind_fp = _ROOT / ".cache" / "industries.csv"
        ind = {}
        if ind_fp.exists():
            try:
                df = pd.read_csv(ind_fp, dtype=str)
                cc = next((c for c in df.columns
                           if "code" in c.lower() or "代码" in c), df.columns[0])
                ic = next((c for c in df.columns
                           if "industry" in c.lower() or "行业" in c), df.columns[-1])
                ind = {str(r[cc]).zfill(6): str(r[ic]) for _, r in df.iterrows()}
            except Exception:
                ind = {}
        # 行业映射第二来源：zzshare 板块成分（首次约 1 分钟）
        if not ind:
            bs_fp = _ROOT / ".cache" / "baostock" / "industry.csv"
            if bs_fp.exists():
                try:
                    bdf = pd.read_csv(bs_fp, dtype=str)
                    ind = {str(r["symbol"]).zfill(6): str(r["industry"])
                           for _, r in bdf.iterrows()}
                except Exception:
                    ind = {}
        if not ind:
            zz_fp = _ROOT / ".cache" / "zzshare" / "industry.csv"
            if zz_fp.exists():
                try:
                    zdf = pd.read_csv(zz_fp, dtype=str)
                    ind = {str(r["symbol"]).zfill(6): str(r["plate_name"])
                           for _, r in zdf.iterrows()}
                except Exception:
                    ind = {}
        for c in stock_codes:
            groups[c] = ind.get(c, "未知")
    return groups


def mark_to_market(data: dict, prices: dict[str, float]) -> dict:
    """
    按最新价计算市值与盈亏。

    prices: {code: 最新价}；缺价的标的**不计入市值**，单独列出。
    """
    holdings: list[Holding] = data.get("holdings", [])
    cash = float(data.get("cash", 0.0))
    warns: list[str] = list(data.get("warnings", []))

    rows, missing = [], []
    mv_total = 0.0
    cost_total = 0.0
    have_cost_any = False

    for h in holdings:
        px = prices.get(h.code)
        if px is None or not np.isfinite(px) or px <= 0:
            missing.append(h.code)
            continue
        mv = h.shares * float(px)
        mv_total += mv
        row = {"code": h.code, "shares": h.shares, "price": round(float(px), 4),
               "market_value": round(mv, 2)}
        if h.cost is not None and h.cost > 0:
            cv = h.shares * h.cost
            cost_total += cv
            have_cost_any = True
            row.update({"cost_price": round(h.cost, 4),
                        "cost_value": round(cv, 2),
                        "pnl": round(mv - cv, 2),
                        "pnl_pct": round(mv / cv - 1, 6)})
        else:
            row.update({"cost_price": None, "cost_value": None,
                        "pnl": None, "pnl_pct": None})
        rows.append(row)

    total = mv_total + cash
    if total <= 0:
        return {"ok": False, "reason": "总资产为 0（缺价或空仓）",
                "missing": missing, "warnings": warns}

    for r in rows:
        r["weight"] = round(r["market_value"] / total, 6)

    if missing:
        warns.append(f"{len(missing)} 只标的取不到价格，未计入估值："
                     + "、".join(missing[:8]))

    total_pnl = (mv_total - cost_total) if have_cost_any else None
    return {
        "ok": True,
        "n_holdings": len(rows),
        "n_missing": len(missing),
        "missing": missing,
        "cash": round(cash, 2),
        "market_value": round(mv_total, 2),
        "total_value": round(total, 2),
        "cash_ratio": round(cash / total, 6),
        "cost_value": round(cost_total, 2) if have_cost_any else None,
        "total_pnl": round(total_pnl, 2) if total_pnl is not None else None,
        "total_pnl_pct": (round(mv_total / cost_total - 1, 6)
                          if have_cost_any and cost_total > 0 else None),
        "holdings": sorted(rows, key=lambda x: -x["market_value"]),
        "weights": {r["code"]: r["weight"] for r in rows},
        "warnings": warns,
    }


def weights_of(mtm: dict, with_cash: bool = False) -> pd.Series:
    """
    持仓权重序列（供 riskcontrol / 拥挤度 / 风格暴露使用）。

    with_cash=True 时把现金也作为一个"标的"（CASH）。
    """
    if not mtm.get("ok"):
        return pd.Series(dtype="float64")
    w = dict(mtm.get("weights", {}))
    if with_cash and mtm.get("cash", 0) > 0:
        w["CASH"] = mtm.get("cash_ratio", 0.0)
    s = pd.Series(w, dtype="float64")
    return s / s.sum() if s.sum() > 0 else s


def format_report(mtm: dict, top: int = 10) -> str:
    L = ["=" * 66, "持仓概览", "=" * 66]
    if not mtm.get("ok"):
        L.append(f"  无法估值：{mtm.get('reason')}")
        L.append("=" * 66)
        return "\n".join(L)

    L.append(f"  总资产 {mtm['total_value']:>14,.2f}")
    L.append(f"  持仓市值 {mtm['market_value']:>12,.2f}"
             f"   现金 {mtm['cash']:>12,.2f}"
             f"（{mtm['cash_ratio']*100:.1f}%）")
    if mtm.get("total_pnl") is not None:
        L.append(f"  浮动盈亏 {mtm['total_pnl']:>12,.2f}"
                 f"（{mtm['total_pnl_pct']*100:+.2f}%）")
    L.append("-" * 66)
    L.append(f"  {'代码':<8}{'股数':>10}{'现价':>10}{'市值':>13}{'权重':>8}{'盈亏%':>9}")
    L.append("  " + "-" * 60)
    for r in mtm["holdings"][:top]:
        pnl = (f"{r['pnl_pct']*100:+.2f}" if r.get("pnl_pct") is not None else "—")
        L.append(f"  {r['code']:<8}{r['shares']:>10,.0f}{r['price']:>10.2f}"
                 f"{r['market_value']:>13,.0f}{r['weight']*100:>7.2f}%{pnl:>9}")
    if len(mtm["holdings"]) > top:
        L.append(f"  … 另有 {len(mtm['holdings']) - top} 只")
    if mtm.get("warnings"):
        L.append("-" * 66)
        for w in mtm["warnings"]:
            L.append(f"  ⚠ {w}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    # 注意：系统临时目录在本机沙箱下不可写，用项目内 .tmp
    (_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    tmp = _ROOT / ".tmp" / "_portfolio_selftest.csv"
    tmp.write_text("code,shares,cost\n600519,100,1400\n000001,2000,11.2\n"
                   "CASH,50000,\n", encoding="utf-8")
    d = load_holdings(tmp)
    assert len(d["holdings"]) == 2 and d["cash"] == 50000

    prices = {"600519": 1453.0, "000001": 11.81}
    mtm = mark_to_market(d, prices)
    assert mtm["ok"]
    assert abs(mtm["market_value"] - (100 * 1453 + 2000 * 11.81)) < 1e-6
    assert abs(mtm["total_value"] - (mtm["market_value"] + 50000)) < 1e-6
    assert abs(sum(mtm["weights"].values())
               + mtm["cash_ratio"] - 1.0) < 1e-9
    print(format_report(mtm))

    mtm2 = mark_to_market(d, {"600519": 1453.0})       # 缺一只
    assert mtm2["n_missing"] == 1
    assert any("取不到价格" in w for w in mtm2["warnings"])
    print(f"缺价处理：missing={mtm2['missing']} 警告条数={len(mtm2['warnings'])} ✓")

    w = weights_of(mtm, with_cash=True)
    assert "CASH" in w.index and abs(w.sum() - 1) < 1e-9
    print(f"OK v{__version__}")
