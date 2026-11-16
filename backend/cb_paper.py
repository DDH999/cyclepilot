# -*- coding: utf-8 -*-
"""
可转债模拟盘：含滑点与冲击成本（M44）
====================================================
回答一个必须回答的问题：**回测里的成本假设够不够真实？**

之前的做法（不够）
------------------
`backtest_double_low(cost_per_turn=0.0010)` —— 一个**扁平**假设：
每单位换手扣 10bp。它不区分：
* 财通转债（日成交 2.08 亿）和一只日成交 300 万的冷门券
* 下 1 万 和 下 100 万
* 半价差、冲击、佣金是三种不同的成本

本模块的做法（三种成本分开算）
------------------------------
1. **佣金**：万二（0.02%），最低 0.1 元/笔。**转债无印花税。**
2. **半价差**：买入吃卖一、卖出吃买一 → 固定损失约 5~10bp（转债盘口较薄）
3. **冲击成本**：按参与率（下单额 ÷ 当日成交额）平方根缩放

   `impact_bp = k × sqrt(参与率) × 100`  （k 默认 1.0）

   例：下单 10 万 / 日成交 2 亿 = 参与率 0.05% → 冲击 ≈ 2.2bp
       下单 100 万 / 日成交 300 万 = 参与率 33% → 冲击 ≈ 57bp（并标记"过大"）

4. **可执行性检查**：参与率 >10% 的订单标记为「需要拆单」；
   >30% 直接标记「不可执行」（散户实际会被滑点吃掉）。

⚠ 诚实边界
-----------
* 冲击模型是**经验公式**，不是从成交明细拟合出来的
* 没有建模「下不完怎么办」，只做标记
* 转债 T+0，无 T+1 限制；但流动性差的券挂着不一定成交
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["COMMISSION_RATE", "COMMISSION_MIN", "HALF_SPREAD_BP",
           "impact_bp", "order_cost", "simulate_double_low",
           "format_report", "load_state", "save_state",
           "init_state", "advance", "format_state"]

_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = _ROOT / ".state"
PAPER_FILE = STATE_DIR / "cb_paper.json"
AMT_CACHE = _ROOT / ".cache" / "cb" / "amount"

# 转债交易成本实测参数
COMMISSION_RATE = 0.0002      # 万二
COMMISSION_MIN = 0.1          # 最低 0.1 元
HALF_SPREAD_BP = 5.0          # 半价差 5bp（转债盘口薄）
IMPACT_K = 1.0                # 冲击系数

PARTICIPATION_WARN = 0.10     # 超过 10% 需要拆单
PARTICIPATION_MAX = 0.30      # 超过 30% 视为不可执行


def impact_bp(participation: float, k: float = IMPACT_K) -> float:
    """
    冲击成本（bp）：k × sqrt(参与率) × 100

    参与率 0.05% → 2.2bp；1% → 10bp；10% → 31.6bp；30% → 54.8bp

    ⚠ 2026 修复：系数原写成 10，导致冲击被低估 10 倍
       （0.05% 参与率只算 0.22bp，实际应在 2bp 量级）。
    """
    p = max(float(participation), 0.0)
    return float(k * np.sqrt(p) * 100.0)


def order_cost(order_value: float, daily_amount: float | None,
               side: str = "buy", k: float = IMPACT_K) -> dict:
    """
    单笔下单的总成本拆解。

    order_value  : 下单金额（元）
    daily_amount : 该券当日成交额（元）；None 表示未知 → 用保守假设
    """
    if order_value <= 0:
        return {"ok": False, "reason": "下单金额为 0"}

    commission = max(float(order_value) * COMMISSION_RATE, COMMISSION_MIN)
    half_spread = float(order_value) * HALF_SPREAD_BP / 10000.0

    assumed = False
    if daily_amount is None or not np.isfinite(daily_amount) or daily_amount <= 0:
        # 未知成交额：用 3000 万做保守假设（转债中位数偏下）
        daily_amount = 3.0e7
        assumed = True

    part = float(order_value) / float(daily_amount)
    imp_bp = impact_bp(part, k)
    impact = float(order_value) * imp_bp / 10000.0

    total = commission + half_spread + impact
    return {"ok": True, "order_value": round(float(order_value), 2),
            "daily_amount": round(float(daily_amount), 2),
            "participation": round(part, 6),
            "commission": round(commission, 2),
            "half_spread": round(half_spread, 2),
            "impact": round(impact, 2),
            "impact_bp": round(imp_bp, 2),
            "total": round(total, 2),
            "total_bp": round(total / float(order_value) * 10000, 2),
            "amount_assumed": assumed,
            "too_large": bool(part > PARTICIPATION_WARN),
            "unexecutable": bool(part > PARTICIPATION_MAX)}


def _amount_panel(hist: dict, refresh: bool = False) -> pd.DataFrame:
    """
    成交额面板（date × code）。从日线成交量×收盘价得到。

    volume 单位实测为「张」，成交额 = volume × close（元）。
    """
    import akshare as ak
    AMT_CACHE.mkdir(parents=True, exist_ok=True)
    cols = {}
    for code, df in hist.items():
        fp = AMT_CACHE / f"{code}.csv"
        if not refresh and fp.exists():
            try:
                d = pd.read_csv(fp, parse_dates=["date"])
                if len(d) > 20:
                    cols[code] = d.set_index("date")["amount"]
                    continue
            except Exception:
                pass
        pref = "sh" if str(code).startswith("11") else "sz"
        try:
            raw = ak.bond_zh_hs_cov_daily(symbol=pref + str(code))
            if raw is None or len(raw) == 0:
                continue
            d = pd.DataFrame({
                "date": pd.to_datetime(raw["date"], errors="coerce"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
                "volume": pd.to_numeric(raw["volume"], errors="coerce"),
            }).dropna()
            d["amount"] = d["close"] * d["volume"]
            d = d[d["amount"] > 0]
            if len(d) < 20:
                continue
            d[["date", "amount"]].to_csv(fp, index=False, encoding="utf-8")
            cols[code] = d.set_index("date")["amount"]
        except Exception:
            continue
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def simulate_double_low(price: pd.DataFrame, premium: pd.DataFrame,
                        amount: pd.DataFrame | None = None,
                        top_n: int = 10, rebalance_days: int = 20,
                        capital: float = 100_000.0,
                        k: float = IMPACT_K) -> dict:
    """
    模拟盘回放：按双低选券、逐笔算真实成本、记录净值。

    与 `backtest_double_low` 的区别：这里**每笔交易都按参与率算成本**，
    并统计被"标记为过大"的订单。
    """
    if price.shape[1] < 3:
        return {"ok": False, "reason": f"可用转债仅 {price.shape[1]} 只"}
    dates = price.index
    if len(dates) < rebalance_days * 3:
        return {"ok": False, "reason": f"历史仅 {len(dates)} 天，太短"}

    nav, nav_dates = [capital], [dates[0]]
    prev: set[str] = set()
    trades: list[dict] = []
    total_fee = total_spread = total_impact = 0.0
    too_large = 0
    unexec = 0
    gross_nav = [capital]

    i = 0
    while i < len(dates) - rebalance_days:
        p0, pr0 = price.iloc[i], premium.iloc[i]
        ok = p0.notna() & pr0.notna() & (p0 > 0)
        if ok.sum() < top_n:
            i += rebalance_days
            continue
        dl = (p0 + pr0)[ok].sort_values()
        pick = list(dl.index[:top_n])

        cur_nav = nav[-1]
        per = cur_nav / len(pick)
        seg_px = price.iloc[i:i + rebalance_days + 1][pick]

        # —— 建仓成本（每只按参与率算）
        entry_cost = 0.0
        for code in pick:
            amt = None
            if amount is not None and code in amount.columns:
                row = amount.iloc[i:i + 1]
                if len(row):
                    v = row[code].iloc[0]
                    amt = float(v) if pd.notna(v) else None
            c = order_cost(per, amt, "buy", k)
            if not c.get("ok"):
                continue
            entry_cost += c["total"]
            total_fee += c["commission"]
            total_spread += c["half_spread"]
            total_impact += c["impact"]
            if c["too_large"]:
                too_large += 1
            if c["unexecutable"]:
                unexec += 1
            trades.append({"date": str(dates[i].date()), "code": code,
                           "value": c["order_value"],
                           "participation": c["participation"],
                           "cost_bp": c["total_bp"]})

        # —— 持有期收益
        rets = seg_px.pct_change(fill_method=None).iloc[1:]
        daily = rets.mean(axis=1, skipna=True)
        if daily.empty:
            i += rebalance_days
            continue

        # 建仓成本按净值比例扣一次
        cost_ratio = entry_cost / max(cur_nav, 1.0)
        # 换手成本（卖出旧仓 + 买入新仓）：按参与率估，近似为建仓成本×换手率
        turnover = len(set(pick) - prev) / max(len(pick), 1)
        exit_cost = entry_cost * turnover
        total_fee += exit_cost * 0.4     # 卖出佣金占比近似
        total_spread += exit_cost * 0.3
        total_impact += exit_cost * 0.3

        for idx_k, v in enumerate(daily.items()):
            if not np.isfinite(v[1]):
                continue
            nav[-1] = nav[-1]
            nav.append(nav[-1] * (1.0 + float(v[1])))
            nav_dates.append(v[0])
            gross_nav.append(gross_nav[-1] * (1.0 + float(v[1])))
        # 扣建仓+换手成本
        if len(nav) > 0:
            nav[-1] *= (1.0 - cost_ratio - cost_ratio * turnover)

        prev = set(pick)
        i += rebalance_days

    if len(nav) < 50:
        return {"ok": False, "reason": f"有效净值点仅 {len(nav)}"}

    s = pd.Series(nav, index=pd.DatetimeIndex(nav_dates))
    s = s[~s.index.duplicated(keep="last")]
    g = pd.Series(gross_nav, index=pd.DatetimeIndex(nav_dates))
    g = g[~g.index.duplicated(keep="last")]
    g = g.reindex(s.index).ffill()

    def _stats(x: pd.Series) -> dict:
        r = x.pct_change(fill_method=None).dropna()
        yrs = max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
        ann = float(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1
        vol = float(r.std(ddof=0) * np.sqrt(252)) if len(r) else 0.0
        return {"annual": round(ann, 4), "vol": round(vol, 4),
                "sharpe": (round(float(r.mean() * 252 / vol), 3)
                           if vol > 0 else None),
                "max_drawdown": round(float((x / x.cummax() - 1).min()), 4)}

    net = _stats(s)
    gross = _stats(g)
    total_cost = total_fee + total_spread + total_impact

    return {"ok": True, "capital": capital,
            "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
            "n_bonds": int(price.shape[1]), "top_n": int(top_n),
            "rebalance_days": int(rebalance_days),
            "nav_net": s, "nav_gross": g,
            "net": net, "gross": gross,
            "final_net": round(float(s.iloc[-1]), 2),
            "final_gross": round(float(g.iloc[-1]), 2),
            "cost_drag_annual": round(gross["annual"] - net["annual"], 4),
            "fee_total": round(total_fee, 2),
            "spread_total": round(total_spread, 2),
            "impact_total": round(total_impact, 2),
            "cost_total": round(total_cost, 2),
            "n_trades": len(trades),
            "too_large": too_large, "unexecutable": unexec,
            "avg_cost_bp": (round(float(np.mean([t["cost_bp"] for t in trades])), 2)
                            if trades else None),
            "trades_sample": trades[:8],
            "warnings": [
                "幸存者偏差：只有当前存续的转债，已退市/强赎的缺失",
                "成交额未知的券按 3000 万/日保守假设",
                "冲击系数 IMPACT_K=1.0 为经验值，未从成交明细拟合",
            ]}


def format_report(res: dict) -> str:
    L = ["=" * 72, "可转债模拟盘（含滑点与冲击成本）", "=" * 72]
    if not res.get("ok"):
        L.append(f"  无法模拟：{res.get('reason')}")
        L.append("=" * 72)
        return "\n".join(L)

    L.append(f"  本金 {res['capital']:,.0f} 元   "
             f"{res['start']} ~ {res['end']}")
    L.append(f"  持仓 {res['top_n']} 只，每 {res['rebalance_days']} 日调仓，"
             f"样本 {res['n_bonds']} 只，成交 {res['n_trades']} 笔")
    L.append("-" * 72)
    L.append("  " + f"{'':<12}{'毛收益':>12}{'净收益':>12}")
    L.append("  " + f"{'年化':<12}{res['gross']['annual']*100:>11.2f}%"
             f"{res['net']['annual']*100:>11.2f}%")
    L.append("  " + f"{'夏普':<12}{(res['gross']['sharpe'] or 0):>12.2f}"
             f"{(res['net']['sharpe'] or 0):>12.2f}")
    L.append("  " + f"{'最大回撤':<12}{res['gross']['max_drawdown']*100:>11.2f}%"
             f"{res['net']['max_drawdown']*100:>11.2f}%")
    L.append("  " + f"{'终值':<12}{res['final_gross']:>12,.0f}"
             f"{res['final_net']:>12,.0f}")
    L.append("-" * 72)
    L.append(f"  成本拆解（合计 {res['cost_total']:,.0f} 元）")
    L.append(f"    佣金     {res['fee_total']:>12,.0f}")
    L.append(f"    半价差   {res['spread_total']:>12,.0f}")
    L.append(f"    冲击成本 {res['impact_total']:>12,.0f}")
    L.append(f"    单笔平均成本 {res['avg_cost_bp']} bp")
    L.append(f"    年化拖累 {res['cost_drag_annual']*100:.2f} 个百分点")
    L.append("-" * 72)
    L.append(f"  可执行性：参与率>10% 的订单 {res['too_large']} 笔，"
             f">30%（不可执行）{res['unexecutable']} 笔")
    L.append("=" * 72)
    L.append("  ⚠ 必读：")
    for w in res["warnings"]:
        L.append(f"    · {w}")
    L.append("=" * 72)
    return "\n".join(L)


# ══════════════════════════════════════════════════════════
def _current_prices() -> dict:
    """当前转债价格表（债现价）。"""
    import cb_double_low as DL
    df = DL.load_current(refresh=True)
    out = {}
    for _, r in df.iterrows():
        try:
            px = float(r.get("债现价"))
        except Exception:
            continue
        if np.isfinite(px) and px > 0:
            out[str(r.get("债券代码", "")).zfill(6)] = {
                "price": px, "name": str(r.get("债券简称", ""))}
    return out


def init_state(capital: float = 100_000.0) -> dict:
    """初始化模拟盘（默认 10 万，空仓）。"""
    st = {"ok": True, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "capital": float(capital), "cash": float(capital),
          "holdings": [], "nav_history": [], "orders": [],
          "bars_since_rebalance": 0, "rebalance_days": 20,
          "top_n": 10, "last_step": None}
    save_state(st)
    return st


def _nav_of(st: dict, prices: dict) -> float:
    nav = float(st.get("cash") or 0.0)
    for h in st.get("holdings") or []:
        px = prices.get(h["code"], {}).get("price")
        if px is not None:
            nav += float(h["qty"]) * float(px)
        else:
            nav += float(h["qty"]) * float(h.get("last_price") or h["cost_basis"])
    return nav


def advance(state: dict | None = None, capital: float = 100_000.0,
            execute: bool = True, force_rebalance: bool = False) -> dict:
    """
    推进一步模拟盘（每天调用一次）。

    execute=True 时按当前价成交（含滑点/冲击/佣金）；
    False 时只标记净值、不动仓。
    """
    st = state or load_state()
    if not st.get("ok") and not st.get("capital"):
        st = init_state(capital)

    import cb_double_low as DL
    prices = _current_prices()
    today = datetime.now().strftime("%Y-%m-%d")

    before = _nav_of(st, prices)
    st["bars_since_rebalance"] = int(st.get("bars_since_rebalance") or 0) + 1
    rebalance_days = int(st.get("rebalance_days") or 20)
    top_n = int(st.get("top_n") or 10)
    due = (st["bars_since_rebalance"] >= rebalance_days) or force_rebalance

    executed: list[dict] = []
    skipped: list[str] = []
    cost_sum = 0.0

    if execute and due:
        sc = DL.screener(top=top_n)
        if not sc.get("ok"):
            skipped.append(f"候选池不可用：{sc.get('reason')}")
            due = False
        else:
            target = {x["code"]: x for x in sc["items"]}
            # 卖出不在目标的
            keep = []
            for h in st["holdings"]:
                if h["code"] in target:
                    keep.append(h)
                    continue
                px = prices.get(h["code"], {}).get("price")
                if px is None:
                    keep.append(h)
                    skipped.append(f"{h['code']} 无现价，保留")
                    continue
                val = float(h["qty"]) * float(px)
                amt = None
                c = order_cost(val, amt, "sell")
                proceeds = val - c["total"]
                st["cash"] = float(st["cash"]) + proceeds
                cost_sum += c["total"]
                executed.append({"date": today, "code": h["code"],
                                 "side": "sell", "qty": h["qty"],
                                 "price": round(px, 3),
                                 "cost": c["total"],
                                 "participation": c["participation"]})
            st["holdings"] = keep
            # 买入目标（等权）
            have = {h["code"] for h in st["holdings"]}
            todo = [c for c in target if c not in have]
            if todo:
                per = float(st["cash"]) / len(todo)
                for code in todo:
                    info = target[code]
                    px = prices.get(code, {}).get("price") or info["price"]
                    if not px or px <= 0 or per < 1000:   # 至少 1000 元
                        continue
                    c = order_cost(per, None, "buy")
                    spend = per - c["total"]
                    qty = int(spend / px / 10) * 10      # 10 张一手
                    if qty <= 0:
                        continue
                    real = qty * px
                    c = order_cost(real, None, "buy")
                    st["cash"] = float(st["cash"]) - real - c["total"]
                    cost_sum += c["total"]
                    st["holdings"].append({
                        "code": code, "name": info["name"],
                        "qty": qty, "cost_basis": round(px, 3),
                        "last_price": round(px, 3)})
                    executed.append({"date": today, "code": code,
                                     "side": "buy", "qty": qty,
                                     "price": round(px, 3),
                                     "cost": c["total"],
                                     "participation": c["participation"]})
            st["bars_since_rebalance"] = 0

    # 更新持仓最新价 + 净值
    for h in st["holdings"]:
        px = prices.get(h["code"], {}).get("price")
        if px:
            h["last_price"] = round(float(px), 3)
    after = _nav_of(st, prices)
    st["nav_history"].append({"date": today, "nav": round(after, 2),
                              "rebalanced": bool(executed)})
    st["last_step"] = today
    st["orders"] = (st.get("orders") or []) + executed
    save_state(st)

    return {"ok": True, "date": today, "nav_before": round(before, 2),
            "nav_after": round(after, 2), "rebalanced": bool(executed),
            "n_executed": len(executed), "cost": round(cost_sum, 2),
            "holdings": st["holdings"], "skipped": skipped,
            "state": st}


def format_state(st: dict) -> str:
    L = ["=" * 72, "可转债模拟盘状态", "=" * 72]
    if not st.get("ok") and "capital" not in st:
        L.append(f"  还没有模拟盘：{st.get('reason', '无状态文件')}")
        L.append("  先跑：cbpaper --init --capital 100000")
        L.append("=" * 72)
        return "\n".join(L)

    navh = st.get("nav_history") or []
    cap = float(st.get("capital") or 0)
    nav = navh[-1]["nav"] if navh else cap
    ret = (nav / cap - 1) if cap > 0 else 0.0
    L.append(f"  本金 {cap:,.0f}   当前净值 {nav:,.0f}   收益 {ret*100:+.2f}%")
    L.append(f"  现金 {float(st.get('cash') or 0):,.0f}   "
             f"持仓 {len(st.get('holdings') or [])} 只   "
             f"距下次调仓 {int(st.get('rebalance_days') or 20) - int(st.get('bars_since_rebalance') or 0)} 天")
    L.append(f"  记录 {len(navh)} 次  成交 {len(st.get('orders') or [])} 笔")
    L.append("-" * 72)
    if st.get("holdings"):
        L.append("  " + f"{'代码':<8}{'名称':<10}{'数量':>8}{'成本':>8}"
                 f"{'现价':>8}{'浮盈':>8}")
        L.append("  " + "-" * 52)
        for h in st["holdings"]:
            px = float(h.get("last_price") or h["cost_basis"])
            cb = float(h["cost_basis"])
            pnl = (px / cb - 1) * 100 if cb > 0 else 0
            L.append("  " + f"{h['code']:<8}{str(h.get('name', ''))[:8]:<10}"
                     f"{h['qty']:>8}{cb:>8.2f}{px:>8.2f}{pnl:>7.1f}%")
    else:
        L.append("  （空仓）")
    L.append("=" * 72)
    return "\n".join(L)


def load_state() -> dict:
    if not PAPER_FILE.exists():
        return {"ok": False, "reason": "还没有模拟盘状态"}
    try:
        return json.loads(PAPER_FILE.read_text(encoding="utf-8"))
    except Exception as ex:
        return {"ok": False, "reason": str(ex)}


def save_state(state: dict) -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    rec = dict(state)
    rec["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    PAPER_FILE.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return {"ok": True, "path": str(PAPER_FILE)}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 冲击成本表 ===")
    for part in (0.0005, 0.001, 0.01, 0.05, 0.10, 0.30):
        print(f"  参与率 {part*100:>6.2f}%  → 冲击 {impact_bp(part):>6.2f} bp")
    assert impact_bp(0) == 0.0
    assert impact_bp(0.01) > impact_bp(0.001)
    print("  单调递增 ✓")

    print("\n=== 单笔成本拆解 ===")
    for val, amt in ((10000, 2.0e8), (100000, 2.0e8),
                     (1000000, 3.0e6), (5000000, 3.0e6)):
        c = order_cost(val, amt)
        print(f"  下单 {val:>9,} / 日成交 {amt:>12,.0f} → "
              f"参与 {c['participation']*100:>5.2f}%  成本 {c['total_bp']:>6.1f}bp "
              f"{'【需拆单】' if c['too_large'] else ''}"
              f"{'【不可执行】' if c['unexecutable'] else ''}")
    c = order_cost(100000, None)
    assert c["amount_assumed"] is True
    print("  未知成交额 → 标记为假设 ✓")

    print("\n=== 模拟盘回放（合成数据）===")
    idx = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(7)
    px = pd.DataFrame({f"C{i}": 100 * np.cumprod(
        1 + rng.normal(0.0005, 0.012, 400)) for i in range(8)}, index=idx)
    pm = pd.DataFrame({f"C{i}": 10 + 3 * i +
                       rng.normal(0, 1, 400) for i in range(8)}, index=idx)
    amt = pd.DataFrame({c: rng.uniform(1e7, 3e8, 400) for c in px.columns},
                       index=idx)
    res = simulate_double_low(px, pm, amt, top_n=5, rebalance_days=20,
                              capital=100000)
    print(format_report(res))
    assert res["ok"]
    assert res["net"]["annual"] <= res["gross"]["annual"] + 1e-9
    print("\nOK v" + __version__)
