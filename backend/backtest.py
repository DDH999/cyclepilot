# -*- coding: utf-8 -*-
"""
QuantPilot · A股真实规则回测引擎 (v2 · numpy 矩阵化)
=================================================================
规则完备性：
  T+1 / 涨跌停(10%,20%) / 印花税0.05% / 佣金0.025%(最低5元) / 过户费0.001%
  / 100股手数 / 停牌跳过 / 现金约束 / 个股止损 / 组合回撤制动
  / 冲击成本(ADV平方根) / 退市核销
性能：日循环全部走 numpy 整型索引，避免 pandas 标量查找。
=================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


class Costs:
    def __init__(self, commission=0.00025, min_commission=5.0,
                 stamp_duty=0.0005, transfer_fee=0.00001, slippage_bps=5.0):
        self.commission = commission
        self.min_commission = min_commission
        self.stamp_duty = stamp_duty
        self.transfer_fee = transfer_fee
        self.slippage = slippage_bps / 1e4

    def buy(self, value: float) -> float:
        return max(value * self.commission, self.min_commission) + value * self.transfer_fee

    def sell(self, value: float) -> float:
        return (max(value * self.commission, self.min_commission)
                + value * (self.stamp_duty + self.transfer_fee))

    def describe(self) -> dict:
        return {"佣金率": self.commission, "最低佣金": self.min_commission,
                "印花税(卖)": self.stamp_duty, "过户费": self.transfer_fee,
                "滑点bps": self.slippage * 1e4}


def limit_pct(code: str) -> float:
    c = str(code)
    return 0.20 if c.startswith(("300", "301", "688")) else 0.10


def build_market(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for code, d in panel.items():
        d = d.drop_duplicates("date").set_index("date").sort_index()
        if "pct_chg" not in d.columns:
            d["pct_chg"] = d["close"].pct_change(fill_method=None) * 100
        lim = limit_pct(code)
        d["ret"] = d["close"].pct_change(fill_method=None)
        d["limit_up"] = (d["ret"] >= lim * 0.975) & (d["close"] >= d["high"] * 0.999)
        d["limit_down"] = (d["ret"] <= -lim * 0.975) & (d["close"] <= d["low"] * 1.001)
        d["tradable"] = d["volume"].fillna(0) > 0
        keep = ["open", "close", "high", "low", "volume", "ret",
                "limit_up", "limit_down", "tradable"]
        if "amount" in d.columns:
            d["amount"] = d["amount"].fillna(d["close"] * d["volume"])
            keep.append("amount")
        out[code] = d[keep]
    return out


class Backtest:
    def __init__(self, panel: dict[str, pd.DataFrame], capital: float = 1_000_000.0,
                 costs: Costs | None = None, verbose: bool = False,
                 stop_loss: float | None = None,
                 delist_recovery: float = 0.30,
                 dd_brake: float | None = 0.15,
                 dd_brake_scale: float = 0.30,
                 impact_k: float = 0.002):
        self.mkt = build_market(panel)
        self.codes = sorted(self.mkt)
        self.n = len(self.codes)
        self.cidx = {c: i for i, c in enumerate(self.codes)}
        self.capital0 = capital
        self.costs = costs or Costs()
        self.verbose = verbose
        self.stop_loss = stop_loss
        self.delist_recovery = delist_recovery
        self.dd_brake = dd_brake
        self.dd_brake_scale = dd_brake_scale
        self.impact_k = impact_k

        # ── 对齐到统一日期轴，全部转 numpy
        frames = {c: self.mkt[c]["close"] for c in self.codes}
        self.close = pd.DataFrame(frames).sort_index()
        self.dates = self.close.index
        self.dmap = {d: i for i, d in enumerate(self.dates)}
        CL = self.close.to_numpy(dtype=float)              # (T, N)，含 NaN
        self.CL = np.nan_to_num(CL, nan=0.0)
        self.VALID = np.isfinite(CL)                        # 当日是否有真实价

        def _bool(col):
            df = pd.DataFrame({c: self.mkt[c][col] for c in self.codes}).sort_index()
            return df.reindex(self.dates).fillna(False).to_numpy(dtype=bool)

        self.TV = _bool("tradable")
        self.LU = _bool("limit_up")
        self.LD = _bool("limit_down")
        if all("amount" in self.mkt[c].columns for c in self.codes):
            AM = pd.DataFrame({c: self.mkt[c]["amount"] for c in self.codes}).sort_index()
            self.AMT = np.nan_to_num(AM.reindex(self.dates).to_numpy(dtype=float), nan=0.0)
        else:
            self.AMT = None

        # ── 退市识别：最后真实交易日 + 21 个交易日核销
        self.last_real = {}
        self.writedown = {}
        for j, c in enumerate(self.codes):
            v = self.VALID[:, j]
            if not v.any():
                continue
            last_i = int(np.max(np.where(v)[0]))
            self.last_real[j] = last_i
            wd = last_i + 21
            if wd < len(self.dates):
                self.writedown[j] = wd

    # ───────────────────────────── 主循环
    def run(self, weights: pd.DataFrame, rebalance: str = "M",
            max_positions: int = 30, min_value: float = 8000.0) -> dict:
        CL, TV, LU, LD, VALID, AMT = self.CL, self.TV, self.LU, self.LD, self.VALID, self.AMT
        T = len(self.dates)

        w = weights.reindex(columns=self.codes).fillna(0.0)
        w.index = pd.to_datetime(w.index)
        sig = {}
        for dt in w.index:
            i = self.dmap.get(dt)
            if i is not None:
                sig[i] = w.loc[dt].to_numpy(dtype=float)

        cash = float(self.capital0)
        pos: dict[int, float] = {}
        avail: dict[int, float] = {}
        cbnp: dict[int, float] = {}
        nav_hist = np.empty(T); turn_hist = []
        pending = None
        n_stops = n_wo = n_brake = 0
        peak = cash
        slips = self.costs.slippage

        for i in range(T):
            # T+1 解锁
            for j in avail:
                avail[j] = pos.get(j, 0.0)

            # 退市核销
            for j in list(pos):
                wd = self.writedown.get(j)
                if wd is not None and i >= wd and pos[j] > 0:
                    p = CL[self.last_real[j], j]
                    cash += pos[j] * p * self.delist_recovery
                    pos.pop(j, None); avail.pop(j, None); cbnp.pop(j, None)
                    n_wo += 1

            # 个股止损
            if self.stop_loss:
                for j in list(pos):
                    q = pos[j]
                    if q <= 0:
                        continue
                    p = CL[i, j]
                    cb = cbnp.get(j, 0.0)
                    if p > 0 and cb > 0 and (p / cb - 1) < -self.stop_loss:
                        if not TV[i, j] or LD[i, j]:
                            continue
                        qty = min(q, avail.get(j, 0.0))
                        if qty <= 0:
                            continue
                        fill = p * (1 - slips)
                        gr = qty * fill
                        cash += gr - self.costs.sell(gr)
                        pos[j] = q - qty
                        avail[j] = avail.get(j, 0.0) - qty
                        if pos[j] <= 0:
                            cbnp.pop(j, None)
                        n_stops += 1

            # 组合回撤制动
            if pending is not None and self.dd_brake:
                cur = cash + sum(q * CL[i, j] for j, q in pos.items() if q > 0)
                if peak > 0 and cur / peak - 1 < -self.dd_brake:
                    pending = pending * self.dd_brake_scale
                    n_brake += 1

            # 执行昨日信号
            if pending is not None:
                cash, traded = self._exec(pending, i, cash, pos, avail, cbnp,
                                          max_positions, min_value)
                turn_hist.append(traded / max(peak, 1e-9))
                pending = None

            nav = cash + sum(q * CL[i, j] for j, q in pos.items() if q > 0)
            nav_hist[i] = nav
            peak = max(peak, nav)

            if i in sig:
                pending = sig[i]

        navdf = pd.Series(nav_hist, index=self.dates)
        turn = pd.Series(turn_hist) if turn_hist else pd.Series(dtype=float)
        res = self._metrics(navdf, turn, pos)
        res["n_stops"] = n_stops
        res["n_writeoffs"] = n_wo
        res["n_brakes"] = n_brake
        return res

    # ───────────────────────────── 执行
    def _exec(self, target: np.ndarray, i: int, cash, pos, avail, cbnp,
              max_positions, min_value):
        CL, TV, LU, LD, AMT = self.CL, self.TV, self.LU, self.LD, self.AMT
        px = CL[i]
        ok = TV[i]; lu = LU[i]; ld = LD[i]
        inv = 1 - self.costs.slippage; inp = 1 + self.costs.slippage

        idx = np.where(target > 0)[0]
        if len(idx) == 0:
            return cash, 0.0
        if len(idx) > max_positions:
            idx = idx[np.argsort(-target[idx])[:max_positions]]
        tw = target[idx]
        s = tw.sum()
        if s <= 0:
            return cash, 0.0
        tw = tw / s * min(1.0, float(s))          # gross 上限 1

        nav = cash + sum(q * px[j] for j, q in pos.items() if q > 0)

        # 1) 卖出
        for j in list(pos):
            if pos[j] <= 0:
                continue
            k = int(np.searchsorted(idx, j))
            wv = tw[k] if k < len(idx) and idx[k] == j else 0.0
            p = px[j]
            if p <= 0 or not ok[j] or ld[j]:
                continue
            cur_val = pos[j] * p
            if wv * nav >= cur_val * 0.98:
                continue
            tgt_q = np.floor(wv * nav / p / 100) * 100
            qty = min(pos[j] - tgt_q, avail.get(j, pos[j]))
            if qty <= 0:
                continue
            part = (qty * p / AMT[i, j]) if (AMT is not None and AMT[i, j] > 0) else 0.0
            fill = p * (1 - self.costs.slippage - self.impact_k * np.sqrt(max(part, 0.0)))
            gr = qty * fill
            cash += gr - self.costs.sell(gr)
            pos[j] -= qty
            avail[j] = avail.get(j, 0.0) - qty
            if pos[j] <= 0:
                cbnp.pop(j, None)

        # 2) 买入
        need = []
        for k, j in enumerate(idx):
            j = int(j)
            if not ok[j] or lu[j] or px[j] <= 0:
                continue
            tv_ = tw[k] * nav - pos.get(j, 0.0) * px[j]
            if tv_ < max(min_value * 0.25, nav * 0.002):
                continue
            need.append((j, tv_))
        if not need:
            return cash, 0.0
        total = sum(v for _, v in need)
        scale = min(1.0, max(cash * 0.995, 0) / total)
        traded = 0.0
        for j, v in need:
            if scale <= 0:
                break
            want = v * scale
            p = px[j]
            part = (want / AMT[i, j]) if (AMT is not None and AMT[i, j] > 0) else 0.0
            fill = p * (1 + self.costs.slippage + self.impact_k * np.sqrt(max(part, 0.0)))
            lots = np.floor(want / fill / 100) * 100
            if lots < 100:
                continue
            gr = lots * fill
            fee = self.costs.buy(gr)
            if gr + fee > cash:
                lots = np.floor(max(cash - fee, 0) / fill / 100) * 100
                if lots < 100:
                    continue
                gr = lots * fill
                fee = self.costs.buy(gr)
            cash -= gr + fee
            old = pos.get(j, 0.0)
            pos[j] = old + lots
            oc = cbnp.get(j, 0.0)
            cbnp[j] = (old * oc + lots * fill) / pos[j] if pos[j] > 0 else fill
            avail[j] = avail.get(j, 0.0)          # T+1
            traded += gr
        return cash, traded

    # ───────────────────────────── 指标
    def _metrics(self, nav: pd.Series, turnover: pd.Series, pos: dict) -> dict:
        r = nav.pct_change(fill_method=None).dropna()
        if len(r) < 30:
            return {"error": "样本不足"}
        years = len(r) / 252
        tot = nav.iloc[-1] / nav.iloc[0] - 1
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
        vol = r.std() * np.sqrt(252)
        sharpe = (r.mean() * 252) / vol if vol > 0 else np.nan
        dn = r[r < 0].std() * np.sqrt(252)
        sortino = (r.mean() * 252) / dn if dn > 0 else np.nan
        dd = nav / nav.cummax() - 1
        mdd = dd.min()
        calmar = cagr / abs(mdd) if mdd < 0 else np.nan
        return {"total_return": float(tot), "cagr": float(cagr),
                "ann_vol": float(vol), "sharpe": float(sharpe), "sortino": float(sortino),
                "max_drawdown": float(mdd), "calmar": float(calmar),
                "win_rate": float((r > 0).mean()),
                "ann_turnover": float(turnover.mean() * 252) if len(turnover) else np.nan,
                "final_nav": float(nav.iloc[-1]), "days": int(len(r)),
                "n_positions": int(len([q for q in pos.values() if q > 0])),
                "nav": nav, "returns": r, "drawdown": dd, "turnover": turnover}


def benchmark_ew(panel: dict[str, pd.DataFrame]) -> pd.Series:
    px = pd.DataFrame({c: d.set_index("date")["close"] for c, d in panel.items()}).sort_index().ffill()
    return (1 + px.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()


def attribution(nav: pd.Series, bench: pd.Series) -> dict:
    df = pd.DataFrame({"s": nav.pct_change(fill_method=None), "b": bench.pct_change(fill_method=None)}).dropna()
    if len(df) < 30:
        return {}
    beta = float(df["s"].cov(df["b"]) / df["b"].var())
    alpha_d = float(df["s"].mean() - beta * df["b"].mean())
    te = float((df["s"] - df["b"]).std() * np.sqrt(252))
    return {"beta": beta, "alpha_ann": alpha_d * 252, "tracking_error": te,
            "information_ratio": float((df["s"] - df["b"]).mean() * 252 / te) if te > 0 else np.nan,
            "bench_total": float((1 + df["b"]).prod() - 1)}
