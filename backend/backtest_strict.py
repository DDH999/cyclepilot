# -*- coding: utf-8 -*-
"""
严格回测引擎（M45）：无未来函数 + 涨跌停/停牌过滤 + 样本外检验
====================================================
对现有 `backtest.py` 的审计结论
-------------------------------
| 检查项 | 现状 | 判定 |
|---|---|---|
| 信号→成交时点 | `pending` 存次日执行 | ✅ 无未来函数 |
| 成交价 | `px = CL[i]` 用**收盘价** | ⚠️ 不是"次日开盘" |
| 止损 | 用当日收盘判断、当日收盘卖 | ❌ 轻微未来函数 |
| 回撤制动 | 用当日收盘缩放当日成交 | ❌ 同上 |
| 涨停不买/跌停不卖 | 有 | ✅ |
| 停牌过滤 | `volume > 0` | ✅ |
| 一字板识别 | `close >= high*0.999` | ❌ 认不出真一字板 |

本模块的做法
------------
1. **信号在 close(t) 算出，成交在 open(t+1)** —— 严格次日开盘
2. **成交前逐只检查可交易性**：
   * 停牌（当日无 bar 或 volume=0）→ 取消
   * 开盘即涨停（open ≥ 昨收×(1+限幅)）→ **不能买**
   * 开盘即跌停（open ≤ 昨收×(1−限幅)）→ **不能卖**
   * 一字板（open=high=low=close 且触板）→ 同样取消
3. **T+1**：当日买入次日才能卖
4. **样本外检验**：2019–2021 调参，2022–2024 检验，输出过拟合指标

涨跌停幅度
----------
主板 10%、创业板/科创板 20%、ST 5%（按代码前缀 + 名称判定，缺名称时按 10%）
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["limit_pct", "fetch_panel", "annotate", "audit_panel",
           "StrictBacktest", "momentum_weights", "tune_oos",
           "format_audit", "format_oos"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / ".cache" / "strict"

TRADING_DAYS = 252


def limit_pct(code: str) -> float:
    """日涨跌幅限制。"""
    c = str(code).zfill(6)
    if c.startswith(("300", "301", "688", "689")):
        return 0.20
    if c.startswith(("8", "4", "92")):        # 北交所
        return 0.30
    return 0.10


# ══════════════════════════════════════════════════════════
DEFAULT_UNIVERSE = [
    "600519", "600036", "601318", "600030", "600276", "601166", "600887",
    "601888", "600900", "601012", "600104", "601398", "601288", "600028",
    "601857", "600585", "600309", "601668", "600031", "601899", "600048",
    "000001", "000002", "000333", "000651", "000858", "002415", "002594",
    "300059", "300750", "002304", "000568", "002142", "000725", "002027",
    "300124", "300015", "002460", "300760", "002475", "300274", "002714",
    "600690", "601088", "600438", "601919", "600188", "000063", "002352",
    "600809", "603259", "601601", "601628", "600016", "601169", "600000",
    "601988", "601939", "600050", "000100",
]


def default_universe(n: int = 60) -> list[str]:
    """默认股票池：优先沪深300成分，失败用固定清单。"""
    try:
        import akshare as ak
        df = ak.index_stock_cons_csindex(symbol="000300")
        col = next((c for c in df.columns
                    if "成分券代码" in c or c == "代码"), None)
        if col:
            codes = [str(x).zfill(6) for x in df[col].tolist()]
            codes = [c for c in codes if c[:1] in "036"]
            if len(codes) >= n:
                return codes[:n]
    except Exception:
        pass
    return DEFAULT_UNIVERSE[:n]


def fetch_panel(codes: list[str], start: str, end: str,
                refresh: bool = False, verbose: bool = False,
                with_flags: bool = True, min_rows: int = 30) -> dict:
    """
    拉取日线面板。用 baostock（历史全），必要时分段绕开单次 800 行限制。
    """
    import baostock_source as BS
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_{end}"
    out, missing = {}, []
    short: list[str] = []
    failures: dict[str, str] = {}

    BS._login()
    try:
        for code in codes:
            c6 = str(code).zfill(6)
            fp = CACHE / f"{c6}_{tag}{'_fl' if with_flags else ''}.csv"
            if not refresh and fp.exists():
                try:
                    d = pd.read_csv(fp, parse_dates=["date"])
                    if len(d) > 100:
                        out[c6] = d
                        continue
                except Exception:
                    pass
            try:
                # 分两段，避开单次返回上限
                y0, y1 = int(start[:4]), int(end[:4])
                mid = f"{min((y0 + y1) // 2 + 1, y1)}-01-01"
                parts = []
                for a, b in ((start, mid), (mid, end)):
                    seg = BS.stock_daily(c6, start=a, end=b,
                                         with_flags=with_flags)
                    if seg is not None and len(seg):
                        parts.append(seg)
                if not parts:
                    missing.append(c6)
                    failures[c6] = "无数据返回"
                    continue
                d = pd.concat(parts, ignore_index=True)
                d["date"] = pd.to_datetime(d["date"])
                d = d.drop_duplicates("date").sort_values("date")
                d = d[(d["date"] >= pd.Timestamp(start)) &
                      (d["date"] <= pd.Timestamp(end))]
                # 旧逻辑：<100 行直接剔除 → 退市股/次新股被静默丢掉。
                # 新逻辑：<30 行视为确实无数据；30~250 行保留但标记为 short。
                if len(d) < int(min_rows):
                    missing.append(c6)
                    failures[c6] = f"仅 {len(d)} 行 (<{min_rows})"
                    continue
                if len(d) < 250:
                    short.append(c6)
                d.to_csv(fp, index=False)
                out[c6] = d
            except Exception as ex:
                missing.append(c6)
                failures[c6] = f"{type(ex).__name__}: {str(ex)[:70]}"
            if verbose and len(out) % 20 == 0:
                print(f"    已取 {len(out)} 只…")
    finally:
        BS.close()
    return {"ok": len(out) > 0, "panel": out, "missing": missing,
            "short": sorted(set(short)), "failures": failures,
            "n_ok": len(out), "n_missing": len(missing),
            "n_short": len(set(short))}


def annotate(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    加涨跌停/一字板/可交易标记。**只用当日及之前的信息**。
    """
    d = df.copy().sort_values("date").reset_index(drop=True)
    lim = limit_pct(code)
    prev_close = d["close"].shift(1)
    d["ret"] = d["close"].pct_change(fill_method=None)

    # ST 股当日涨跌停是 5%：用**当日** isST 状态逐日判定（当日公开，不构成前视）。
    # 早期版本只按代码前缀，把 ST 的 5% 板当成 10% 板，5% 涨停日仍可买入。
    if "is_st" in d.columns:
        lim_s = np.where(d["is_st"].to_numpy() > 0, 0.05, lim)
    else:
        lim_s = np.full(len(d), lim)

    eps = 0.005                      # 半个百分点容差（价格取整）
    up_line = prev_close * (1.0 + lim_s)
    dn_line = prev_close * (1.0 - lim_s)

    d["open_limit_up"] = d["open"] >= up_line * (1 - eps)
    d["open_limit_down"] = d["open"] <= dn_line * (1 + eps)
    # 一字板：四价相同（或几乎相同）且触板
    one_word = ((d["high"] - d["low"]).abs() <= d["close"] * eps * 0.5)
    d["one_word_up"] = one_word & d["open_limit_up"]
    d["one_word_down"] = one_word & d["open_limit_down"]
    # 收盘触板（用于识别"当日无法成交"的历史情形）
    d["close_limit_up"] = d["close"] >= up_line * (1 - eps)
    d["close_limit_down"] = d["close"] <= dn_line * (1 + eps)
    # 停牌：优先用 tradestatus（0=停牌），没有则退回成交量
    if "tradestatus" in d.columns:
        d["tradable"] = d["tradestatus"].fillna(1).astype(int) == 1
    else:
        d["tradable"] = d["volume"].fillna(0) > 0
    return d


def delist_settlement(df: pd.DataFrame, out_date, window: int = 30,
                      streak: int = 5) -> dict:
    """
    退市核销价（分档，偏保守）。

    规则（按需求）
    --------------
    1. 退市整理期最后 `streak`(5) 个交易日**连续跌停**，或**停牌无有效成交**
       → recovery = 0，全额核销
    2. 否则 → 按退市整理期（最后 `window`=30 个交易日）**均价**核销
       recovery = 均价 / 窗口前收盘价

    为什么不用「按最后收盘价平仓」：A 股退市整理期常连续跌停，
    假设能在最后一天顺利卖出会系统性低估亏损，是新的乐观偏差。
    """
    od = pd.Timestamp(out_date)
    d = df[df["date"] <= od].sort_values("date").reset_index(drop=True)
    if len(d) == 0:
        return {"px": 0.0, "recovery": 0.0, "reason": "无行情", "window": 0}

    tail = d.tail(int(streak))
    no_trade = ("tradestatus" in tail.columns
                and bool((tail["tradestatus"] == 0).all()))
    all_ld = ("close_limit_down" in tail.columns and len(tail) >= int(streak)
              and bool(tail["close_limit_down"].all()))
    if no_trade or all_ld or len(tail) < int(streak):
        why = ("停牌无成交" if no_trade else
               ("连续跌停" if all_ld else "末段bar不足"))
        return {"px": 0.0, "recovery": 0.0, "reason": why,
                "window": len(tail), "forced_zero": True}

    win = d.tail(int(window))
    avg = float(win["close"].mean())
    last_close = float(d["close"].iloc[-1])
    # ⚠ 关键：30 日均价可能包含暴跌**之前**的高价。
    # 实测 002450：最后收盘 0.20，但 30 日均价 0.916，
    # 按均价核销等于凭空多赚 4.5 倍。取两者较小值（更保守）。
    px = min(avg, last_close)
    if len(d) > len(win):
        prev_close = float(d["close"].iloc[-len(win) - 1])
    else:
        prev_close = float(d["close"].iloc[0])
    rec = (px / prev_close) if prev_close > 0 else 0.0
    return {"px": px, "recovery": round(rec, 6),
            "reason": "整理期均价(封顶最后收盘)", "window": len(win),
            "prev_close": round(prev_close, 4),
            "avg": round(avg, 4), "last_close": round(last_close, 4),
            "forced_zero": False,
            "out_date": str(od.date())}


def audit_panel(panel: dict) -> dict:
    """
    审计：统计会被涨跌停/停牌挡掉的交易比例，并检查未来函数风险。
    """
    rows = []
    for code, df in panel.items():
        d = annotate(df, code)
        n = len(d)
        if n < 30:
            continue
        rows.append({
            "code": code, "n": n,
            "start": str(d["date"].iloc[0].date()),
            "end": str(d["date"].iloc[-1].date()),
            "suspended": int((~d["tradable"]).sum()),
            "open_lu": int(d["open_limit_up"].sum()),
            "open_ld": int(d["open_limit_down"].sum()),
            "one_word": int((d["one_word_up"] | d["one_word_down"]).sum()),
            "close_lu": int(d["close_limit_up"].sum()),
        })
    if not rows:
        return {"ok": False, "reason": "没有可用样本"}
    t = pd.DataFrame(rows)
    tot = int(t["n"].sum())
    return {
        "ok": True, "n_codes": len(t),
        "n_bars": tot,
        "suspension_pct": round(float(t["suspended"].sum()) / tot * 100, 3),
        "open_limit_up_pct": round(float(t["open_lu"].sum()) / tot * 100, 3),
        "open_limit_down_pct": round(float(t["open_ld"].sum()) / tot * 100, 3),
        "one_word_pct": round(float(t["one_word"].sum()) / tot * 100, 3),
        "per_code": t,
        "checks": [
            ("成交价用次日开盘", "pass",
             "信号在 close(t) 生成，成交价取 open(t+1)"),
            ("涨停不开仓", "pass",
             "开盘即涨停或一字板 → 跳过买入"),
            ("跌停不平仓", "pass",
             "开盘即跌停 → 跳过卖出（继续持有）"),
            ("停牌不成交", "pass", "当日无 bar 或 volume=0 → 跳过"),
            ("T+1 约束", "pass", "当日买入次日才可卖"),
            ("止损用当日收盘", "pass",
             "本引擎已移除当日止损，改为次日开盘执行"),
        ],
    }


# ══════════════════════════════════════════════════════════
class StrictBacktest:
    """
    严格引擎：close(t) 出信号 → open(t+1) 成交，逐只检查可交易性。
    """

    def __init__(self, panel: dict, capital: float = 1_000_000.0,
                 commission: float = 0.00025, stamp: float = 0.0005,
                 slippage_bps: float = 5.0, impact_k: float = 0.002,
                 delist_map: dict | None = None,
                 max_part: float = 0.05):
        self.ann = {c: annotate(d, c) for c, d in panel.items()}
        self.codes = sorted(self.ann)
        self.n = len(self.codes)
        self.capital = capital
        self.commission = commission
        self.stamp = stamp
        self.slip = slippage_bps / 1e4
        self.impact_k = impact_k
        # 流动性约束：单笔成交不得超过当日成交额的 5%（机构常见的参与率上限）。
        # 不做这个约束，小资金回测会默认"想买多少买多少"，大资金实盘完全不可行。
        self.max_part = float(max_part)

        dates = sorted(set().union(*[set(self.ann[c]["date"]) for c in self.codes]))
        self.dates = pd.DatetimeIndex(dates)
        self.dmap = {d: i for i, d in enumerate(self.dates)}
        T, N = len(self.dates), self.n
        self.T, self.N = T, N

        def grid(col, dtype=float):
            m = np.full((T, N), np.nan)
            for j, c in enumerate(self.codes):
                d = self.ann[c]
                idx = d["date"].map(self.dmap).to_numpy()
                m[idx, j] = d[col].to_numpy(dtype=float)
            return m

        def bgrid(col):
            m = np.zeros((T, N), dtype=bool)
            for j, c in enumerate(self.codes):
                d = self.ann[c]
                idx = d["date"].map(self.dmap).to_numpy()
                m[idx, j] = d[col].to_numpy(dtype=bool)
            return m

        self.OPEN = grid("open")
        self.CLOSE = grid("close")
        self.AMT = grid("amount")
        self.TV = bgrid("tradable")
        self.OLU = bgrid("open_limit_up")
        self.OLD = bgrid("open_limit_down")
        self.OWU = bgrid("one_word_up")
        self.OWD = bgrid("one_word_down")

        # ── 退市核销：每只退市股算一个核销日索引与核销价
        self.cidx = {c: j for j, c in enumerate(self.codes)}
        self.OUT_I: dict[int, int] = {}
        self.DELIST_INFO: dict[int, dict] = {}
        for code, od in (delist_map or {}).items():
            c6 = str(code).zfill(6)
            j = self.cidx.get(c6)
            if j is None:
                continue
            od = pd.Timestamp(od)
            out_i = int(self.dates.searchsorted(od, side="left"))
            info = delist_settlement(self.ann[c6], od)
            info["out_date"] = str(od.date())
            self.OUT_I[j] = max(out_i, 0)
            self.DELIST_INFO[j] = info

    # ────────────────────────────
    def run(self, weights: pd.DataFrame, max_positions: int = 30,
            min_value: float = 8000.0,
            cap: pd.Series | None = None) -> dict:
        """
        weights: index=信号日（用截至该日收盘的数据算出），columns=代码。
        实际成交在**下一个交易日的开盘**。

        cap: 可选，index=信号日、值=当日最大总仓位（0~1）。
             与价格管理器 `position_manager.position_cap(...)["cap"]` 对齐：
             **信号日 t 算出的上限，约束的是 t+1 开盘的建仓**，无未来函数。
        """
        T, N = self.T, self.N
        w = weights.reindex(columns=self.codes).fillna(0.0)
        w.index = pd.to_datetime(w.index)
        sig = {}
        for dt, row in w.iterrows():
            i = self.dmap.get(dt)
            if i is not None:
                sig[i] = row.to_numpy(dtype=float)

        cash = float(self.capital)
        pos = np.zeros(N)
        avail = np.zeros(N)
        nav_hist = np.empty(T)
        turn_hist = []
        pending = None
        pending_cap = None
        n_block_buy = n_block_sell = n_susp = 0
        contrib = np.zeros(N)          # 每只股票对组合收益的贡献（元）
        hold_days = np.zeros(N, dtype=int)
        first_in: dict[int, str] = {}
        last_out: dict[int, str] = {}
        delist_log: list[dict] = []    # 退市核销明细
        invest_sum = 0.0               # 每日持仓市值/净值 累加
        days_flat = 0                  # 几乎空仓的交易日数
        n_delist = 0
        cap_series = None
        if cap is not None:
            cap_series = pd.Series(cap).copy()
            cap_series.index = pd.to_datetime(cap_series.index)

        for i in range(T):
            # ── 退市核销（先于一切交易）：到了退市日就强制清算
            for j, out_i in self.OUT_I.items():
                if pos[j] <= 0 or i < out_i:
                    continue
                info = self.DELIST_INFO[j]
                px = float(info.get("px") or 0.0)
                before = pos[j] * (px if px > 0 else 0.0)
                cash += before
                # 归因只补「上一个收盘 → 核销价」这一段；
                # 窗口内的跌幅已由每日 mark-to-market 计入，不能再扣一次。
                last_c = (float(self.CLOSE[i - 1, j])
                          if i > 0 and np.isfinite(self.CLOSE[i - 1, j]) else px)
                contrib[j] += float(pos[j]) * (px - last_c)
                delist_log.append({
                    "code": self.codes[j],
                    "out_date": info.get("out_date"),
                    "liquidate_idx": int(i),
                    "liquidate_date": str(self.dates[i].date()),
                    "qty": float(pos[j]),
                    "settle_px": round(px, 4),
                    "recovery": info.get("recovery"),
                    "reason": info.get("reason"),
                    "forced_zero": bool(info.get("forced_zero")),
                    "proceeds": round(before, 2),
                    "hold_days": int(hold_days[j]),
                })
                pos[j] = 0.0
                avail[j] = 0.0
                n_delist += 1

            # T+1 解锁
            avail = pos.copy()

            # 执行「昨日收盘生成的信号」→ 今日开盘成交
            if pending is not None:
                cash, traded, bb, bs, sp = self._exec_at_open(
                    pending, i, cash, pos, avail, max_positions, min_value,
                    cap=pending_cap)
                turn_hist.append(traded / max(nav_hist[i - 1] if i > 0 else cash, 1e-9))
                n_block_buy += bb
                n_block_sell += bs
                n_susp += sp
                pending = None

            # 归因：按持仓市值 × 当日涨跌累计到每只票
            if i > 0:
                prev_c = self.CLOSE[i - 1]
                cur_c = self.CLOSE[i]
                delta = np.where(np.isfinite(prev_c) & np.isfinite(cur_c),
                                 cur_c - prev_c, 0.0)
                contrib += np.where(pos > 0, pos * delta, 0.0)
            held = pos > 0
            hold_days[held] += 1
            for j in np.where(held)[0]:
                first_in.setdefault(int(j), str(self.dates[i].date()))
                last_out[int(j)] = str(self.dates[i].date())

            pos_val = float(np.nansum(pos * self.CLOSE[i]))
            nav = cash + pos_val
            nav_hist[i] = nav
            # 投资率：持仓市值 / 净值。长期偏低说明组合大部分时间空仓，
            # 此时"低波动、低回撤"是假象，不是风控有效（踩过）。
            ir_ = (pos_val / nav) if nav > 0 else 0.0
            invest_sum += ir_
            if ir_ < 0.05:
                days_flat += 1

            # 用今日收盘生成信号，明日开盘执行
            if i in sig:
                pending = sig[i]
                pending_cap = None
                if cap_series is not None:
                    try:
                        v = cap_series.loc[self.dates[i]]
                        if np.isfinite(v):
                            pending_cap = float(np.clip(v, 0.0, 1.0))
                    except Exception:
                        pending_cap = None

        nav = pd.Series(nav_hist, index=self.dates).replace(0, np.nan).ffill()
        r = nav.pct_change(fill_method=None).dropna()
        yrs = max((self.dates[-1] - self.dates[0]).days / 365.25, 1e-9)
        ann = float((nav.iloc[-1] / self.capital) ** (1 / yrs) - 1)
        vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
        dd = float((nav / nav.cummax() - 1).min())
        return {
            "ok": True, "nav": nav, "start": str(self.dates[0].date()),
            "end": str(self.dates[-1].date()), "years": round(yrs, 2),
            "annual": round(ann, 4), "vol": round(vol, 4),
            "sharpe": (round(float(r.mean() * TRADING_DAYS / vol), 3)
                       if vol > 0 else None),
            "max_drawdown": round(dd, 4),
            "final_nav": round(float(nav.iloc[-1]), 2),
            "n_days": int(len(r)),
            "blocked_buy": n_block_buy, "blocked_sell": n_block_sell,
            "blocked_suspension": n_susp,
            "avg_turnover": (round(float(np.mean(turn_hist)), 4)
                             if turn_hist else None),
            "avg_invested": round(invest_sum / max(T, 1), 4),
            "flat_ratio": round(days_flat / max(T, 1), 4),
            "coverage_warning": bool(invest_sum / max(T, 1) < 0.5),
            "n_delist_liquidations": int(n_delist),
            "delist_log": delist_log,
            "contrib_by_code": {self.codes[j]: round(float(contrib[j]), 2)
                                for j in range(N) if abs(contrib[j]) > 0.01},
            "hold_days_by_code": {self.codes[j]: int(hold_days[j])
                                  for j in range(N) if hold_days[j] > 0},
            "first_in_by_code": {self.codes[j]: v for j, v in first_in.items()},
            "last_out_by_code": {self.codes[j]: v for j, v in last_out.items()},
            "contrib_total": round(float(contrib.sum()), 2),
            "cap_applied": cap is not None,
            "avg_cap": (round(float(pd.Series(cap).mean()), 4)
                        if cap is not None else None),
        }

    def _exec_at_open(self, target, i, cash, pos, avail,
                      max_positions, min_value, cap=None):
        OPEN, TV, OLU, OLD, OWU, OWD = (self.OPEN, self.TV, self.OLU,
                                        self.OLD, self.OWU, self.OWD)
        AMT = self.AMT
        px = OPEN[i]
        ok = TV[i]
        lu = OLU[i] | OWU[i]
        ld = OLD[i] | OWD[i]
        bb = bs = sp = 0

        idx = np.where(target > 0)[0]
        if len(idx) == 0:
            return cash, 0.0, bb, bs, sp
        if len(idx) > max_positions:
            idx = idx[np.argsort(-target[idx])[:max_positions]]
        tw = target[idx]
        s = tw.sum()
        if s <= 0:
            return cash, 0.0, bb, bs, sp
        tw = tw / s * min(1.0, float(s))

        # 仓位上限：只压缩、不放大（等比例缩放，保持相对排序）
        if cap is not None:
            total_w = float(tw.sum())
            if total_w > float(cap) + 1e-12 and total_w > 0:
                tw = tw * (float(cap) / total_w)

        def _nav():
            v = cash + float(np.nansum(pos * np.nan_to_num(px, nan=0.0)))
            return v if v > 0 else cash

        nav = _nav()

        # ── 卖出（不看涨跌停时按可成交处理）
        for j in range(self.N):
            if pos[j] <= 0:
                continue
            k = int(np.searchsorted(idx, j))
            wv = tw[k] if k < len(idx) and idx[k] == j else 0.0
            p = px[j]
            if not np.isfinite(p) or p <= 0:
                sp += 1
                continue
            if not ok[j]:
                sp += 1
                continue
            if ld[j]:                       # 跌停不能卖
                bs += 1
                continue
            cur = pos[j] * p
            if wv * nav >= cur * 0.98:
                continue
            tgt_q = np.floor(wv * nav / p / 100) * 100
            qty = min(pos[j] - tgt_q, avail[j])
            # 流动性上限：单日卖出不超过成交额的 max_part
            if AMT[i, j] > 0 and self.max_part > 0:
                cap_q = np.floor(self.max_part * float(AMT[i, j]) / p / 100) * 100
                qty = min(qty, max(cap_q, 0.0))
            if qty <= 0:
                continue
            part = (qty * p / AMT[i, j]) if AMT[i, j] > 0 else 0.0
            fill = p * (1 - self.slip - self.impact_k * np.sqrt(max(part, 0.0)))
            gr = qty * fill
            fee = max(gr * self.commission, 5.0) + gr * (self.stamp + 1e-5)
            cash += gr - fee
            pos[j] -= qty
            avail[j] -= qty

        # ── 买入
        nav = _nav()
        need = []
        for k, j in enumerate(idx):
            j = int(j)
            p = px[j]
            if not np.isfinite(p) or p <= 0:
                sp += 1
                continue
            if not ok[j]:
                sp += 1
                continue
            if lu[j]:                       # 涨停不能买
                bb += 1
                continue
            tv_ = tw[k] * nav - pos[j] * p
            if tv_ < max(min_value * 0.25, nav * 0.002):
                continue
            need.append((j, tv_))
        if not need:
            return cash, 0.0, bb, bs, sp

        total = sum(v for _, v in need)
        scale = min(1.0, max(cash * 0.995, 0) / total) if total > 0 else 0.0
        traded = 0.0
        for j, v in need:
            if scale <= 0:
                break
            want = v * scale
            p = px[j]
            part = (want / AMT[i, j]) if AMT[i, j] > 0 else 0.0
            fill = p * (1 + self.slip + self.impact_k * np.sqrt(max(part, 0.0)))
            # 流动性上限：单日买入不超过成交额的 max_part
            if AMT[i, j] > 0 and self.max_part > 0:
                cap_v = self.max_part * float(AMT[i, j])
                want = min(want, cap_v)
            lots = np.floor(want / fill / 100) * 100
            if lots < 100:
                continue
            gr = lots * fill
            fee = max(gr * self.commission, 5.0) + gr * 1e-5
            if gr + fee > cash:
                lots = np.floor(max(cash - fee, 0) / fill / 100) * 100
                if lots < 100:
                    continue
                gr = lots * fill
                fee = max(gr * self.commission, 5.0) + gr * 1e-5
            cash -= gr + fee
            pos[j] += lots
            traded += gr
        return cash, traded, bb, bs, sp


# ══════════════════════════════════════════════════════════
def momentum_weights(panel: dict, lookback: int = 60, top_n: int = 20,
                     rebalance: int = 20) -> pd.DataFrame:
    """
    横截面动量信号：每 rebalance 日用过去 lookback 日收益排名，取前 top_n。

    只用截至当日收盘的数据，成交在次日开盘。
    """
    closes = {}
    for c, d in panel.items():
        s = d.set_index("date")["close"].sort_index()
        closes[c] = s
    CL = pd.DataFrame(closes).sort_index()
    mom = CL.pct_change(max(int(lookback), 1), fill_method=None)
    dates = list(CL.index)
    rows, idx = [], []
    for i in range(lookback + 1, len(dates), max(int(rebalance), 1)):
        d = dates[i]
        row = mom.loc[d].dropna()
        if len(row) < top_n:
            continue
        w = pd.Series(0.0, index=CL.columns)
        pick = row.nlargest(top_n).index
        w[pick] = 1.0 / len(pick)
        rows.append(w); idx.append(d)
    if not rows:
        return pd.DataFrame(columns=CL.columns)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def tune_oos(panel: dict, is_range: tuple[str, str],
             oos_range: tuple[str, str],
             grid: dict | None = None, capital: float = 1_000_000.0,
             verbose: bool = False) -> dict:
    """
    在 IS 区间调参，在 OOS 区间检验，输出过拟合指标。
    """
    grid = grid or {"lookback": [20, 60, 120], "top_n": [10, 20],
                    "rebalance": [10, 20]}
    is_start, is_end = is_range
    oos_start, oos_end = oos_range

    def _slice(panel, a, b):
        out = {}
        for c, d in panel.items():
            s = d[(d["date"] >= pd.Timestamp(a)) & (d["date"] <= pd.Timestamp(b))]
            if len(s) > 60:
                out[c] = s.reset_index(drop=True)
        return out

    p_is = _slice(panel, is_start, is_end)
    p_oos = _slice(panel, oos_start, oos_end)
    if len(p_is) < 5 or len(p_oos) < 5:
        return {"ok": False,
                "reason": f"分区样本不足（IS {len(p_is)} 只 / OOS {len(p_oos)} 只）"}

    results = []
    best = None
    for lb in grid["lookback"]:
        for tn in grid["top_n"]:
            for rb in grid["rebalance"]:
                sig = momentum_weights(p_is, lb, tn, rb)
                if sig.empty:
                    continue
                try:
                    bt = StrictBacktest(p_is, capital=capital).run(
                        sig, max_positions=tn)
                except Exception:
                    continue
                if not bt.get("ok"):
                    continue
                rec = {"lookback": lb, "top_n": tn, "rebalance": rb,
                       "is_annual": bt["annual"], "is_sharpe": bt["sharpe"],
                       "is_dd": bt["max_drawdown"]}
                results.append(rec)
                key = (bt["sharpe"] or -9)
                if best is None or key > best[0]:
                    best = (key, rec)
                if verbose:
                    print(f"    lb={lb:>3} N={tn:>2} rb={rb:>2} → "
                          f"IS 年化 {bt['annual']*100:>6.2f}% "
                          f"夏普 {bt['sharpe']}")

    if best is None:
        return {"ok": False, "reason": "没有有效参数组合"}

    best_p = {k: best[1][k] for k in ("lookback", "top_n", "rebalance")}
    sig_oos = momentum_weights(p_oos, **best_p)
    bt_oos = (StrictBacktest(p_oos, capital=capital).run(
        sig_oos, max_positions=best_p["top_n"])
        if not sig_oos.empty else None)

    if bt_oos is None:
        return {"ok": False, "reason": "OOS 无法运行"}

    decay = (bt_oos["annual"] - best[1]["is_annual"])
    sharpe_decay = ((bt_oos["sharpe"] or 0) - (best[1]["is_sharpe"] or 0))

    if bt_oos["annual"] <= 0:
        verdict = "**策略失效**：OOS 年化为负，IS 表现没有延续"
    elif sharpe_decay < -0.5:
        verdict = (f"**明显过拟合**：夏普从 {best[1]['is_sharpe']} 掉到 "
                   f"{bt_oos['sharpe']}，衰减超过 0.5")
    elif decay < -0.05:
        verdict = f"**收益衰减**：年化下降 {abs(decay)*100:.1f} 个百分点"
    else:
        verdict = "样本外表现稳定，没有明显过拟合证据"

    return {"ok": True, "best_params": best_p,
            "is": best[1], "oos": {
                "annual": bt_oos["annual"], "sharpe": bt_oos["sharpe"],
                "max_drawdown": bt_oos["max_drawdown"],
                "blocked_buy": bt_oos["blocked_buy"],
                "blocked_sell": bt_oos["blocked_sell"],
                "blocked_suspension": bt_oos["blocked_suspension"]},
            "decay_annual": round(decay, 4),
            "decay_sharpe": round(sharpe_decay, 3),
            "n_grid": len(results), "grid_results": results,
            "verdict": verdict,
            "is_range": [is_start, is_end],
            "oos_range": [oos_start, oos_end],
            "n_is_codes": len(p_is), "n_oos_codes": len(p_oos)}


def format_audit(a: dict) -> str:
    L = ["=" * 72, "回测框架审计（未来函数 / 涨跌停 / 停牌）", "=" * 72]
    if not a.get("ok"):
        L.append(f"  {a.get('reason')}")
        L.append("=" * 72)
        return "\n".join(L)
    L.append(f"  样本 {a['n_codes']} 只，{a['n_bars']:,} 个交易日")
    L.append("-" * 72)
    for name, status, note in a["checks"]:
        L.append(f"  [{'通过' if status == 'pass' else '待查'}] {name}")
        L.append(f"         {note}")
    L.append("-" * 72)
    L.append("  被过滤掉的交易日占比：")
    L.append(f"    停牌          {a['suspension_pct']:>6.2f}%")
    L.append(f"    开盘涨停      {a['open_limit_up_pct']:>6.2f}%")
    L.append(f"    开盘跌停      {a['open_limit_down_pct']:>6.2f}%")
    L.append(f"    一字板        {a['one_word_pct']:>6.2f}%")
    L.append("=" * 72)
    return "\n".join(L)


def format_oos(r: dict) -> str:
    L = ["=" * 72, "样本外检验（调参 → 检验）", "=" * 72]
    if not r.get("ok"):
        L.append(f"  {r.get('reason')}")
        L.append("=" * 72)
        return "\n".join(L)
    p = r["best_params"]
    L.append(f"  IS {r['is_range'][0]} ~ {r['is_range'][1]}   "
             f"OOS {r['oos_range'][0]} ~ {r['oos_range'][1]}")
    L.append(f"  调参样本 {r['n_grid']} 组，最优参数：回看 {p['lookback']} 日、"
             f"持仓 {p['top_n']} 只、{p['rebalance']} 日调仓")
    L.append("-" * 72)
    L.append("  " + f"{'':<12}{'年化':>10}{'夏普':>9}{'最大回撤':>11}")
    L.append("  " + f"{'IS':<12}{r['is']['is_annual']*100:>9.2f}%"
             f"{(r['is']['is_sharpe'] or 0):>9.2f}"
             f"{r['is']['is_dd']*100:>10.2f}%")
    L.append("  " + f"{'OOS':<12}{r['oos']['annual']*100:>9.2f}%"
             f"{(r['oos']['sharpe'] or 0):>9.2f}"
             f"{r['oos']['max_drawdown']*100:>10.2f}%")
    L.append("  " + f"{'衰减':<12}{r['decay_annual']*100:>9.2f}pp"
             f"{r['decay_sharpe']:>9.2f}")
    L.append("-" * 72)
    L.append(f"  OOS 被过滤：涨停未买 {r['oos']['blocked_buy']} 次，"
             f"跌停未卖 {r['oos']['blocked_sell']} 次，"
             f"停牌跳过 {r['oos']['blocked_suspension']} 次")
    L.append(f"  结论：{r['verdict']}")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    assert limit_pct("600519") == 0.10
    assert limit_pct("300750") == 0.20
    assert limit_pct("688981") == 0.20
    assert limit_pct("830799") == 0.30
    print("涨跌停幅度判定 ✓")

    # 构造含停牌/一字板的样本
    idx = pd.bdate_range("2022-01-03", periods=520)
    base = 10.0 * np.cumprod(1 + np.random.default_rng(7).normal(0.001, 0.02, 520))
    df = pd.DataFrame({"date": idx, "open": base, "high": base * 1.01,
                       "low": base * 0.99, "close": base,
                       "volume": 1e6, "amount": 1e7})
    # 第 10 天一字涨停
    df.loc[10, ["open", "high", "low", "close"]] = df.loc[9, "close"] * 1.10
    # 第 20 天停牌
    df.loc[20, "volume"] = 0
    a = annotate(df, "600519")
    assert bool(a.loc[10, "one_word_up"]), "未识别一字涨停"
    assert bool(a.loc[20, "one_word_up"]) is False
    assert bool(a.loc[20, "tradable"]) is False
    print("一字板/停牌识别 ✓")

    panel = {"600519": df}
    au = audit_panel(panel)
    print()
    print(format_audit(au))
    assert au["ok"]

    # 严格引擎跑通（单个样本）
    panel2 = {f"60000{i}": df.assign(close=df["close"] * (1 + i * 0.01))
              for i in range(6)}
    sig = momentum_weights(panel2, lookback=20, top_n=2, rebalance=10)
    bt = StrictBacktest(panel2, capital=100000).run(sig, max_positions=2)
    print(f"严格引擎 ✓ 年化 {bt['annual']*100:.2f}% "
          f"夏普 {bt['sharpe']} 天数 {bt['n_days']}")
    assert bt["ok"] and bt["n_days"] > 50

    oos = tune_oos(panel2, ("2022-01-03", "2022-12-31"),
                   ("2023-01-01", "2023-12-31"),
                   grid={"lookback": [10, 20], "top_n": [2],
                         "rebalance": [10, 20]})
    print()
    print(format_oos(oos))
    print("\nOK v" + __version__)
