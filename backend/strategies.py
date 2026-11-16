# -*- coding: utf-8 -*-
"""
QuantPilot · 信号与策略库 (截面 / 时序)
所有信号在 T 日收盘可知，回测在 T+1 执行 —— 无未来函数。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def price_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    px = pd.DataFrame({c: d.set_index("date")["close"] for c, d in panel.items()})
    return px.sort_index().ffill()


def vol_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    v = pd.DataFrame({c: d.set_index("date")["volume"] for c, d in panel.items()})
    return v.sort_index()


def to_weights_buffered(score: pd.DataFrame, dates: pd.DatetimeIndex, top_n: int = 30,
                        buffer: float = 0.5, weighting: str = "equal",
                        px: pd.DataFrame | None = None,
                        industry: dict[str, str] | None = None,
                        max_per_ind: int | None = None) -> pd.DataFrame:
    """
    带缓冲带的换手控制 + 可选行业中性化。
    - buffer: 持仓跌出 top_n*(1+buffer) 才卖出
    - industry/max_per_ind: 限制单一行业最多持有 max_per_ind 只（防止押赛道）
    """
    hold: list[str] = []
    rows = {}
    exit_rank = max(int(top_n * (1 + buffer)), top_n + 1)
    ind_of = lambda c: (industry or {}).get(c, "未知")

    for dt in dates:
        if dt not in score.index:
            continue
        s = score.loc[dt].dropna()
        if len(s) < max(10, top_n // 2):
            continue
        # 注意：必须用 method="first" 打破并列。
        # 因子在截尾/离散化后常有大量并列值，用默认 average 排名会让
        # "前 top_n 名" 变成 "名次 124.5"，导致一只都选不出来。
        rank = s.rank(ascending=False, method="first")

        # 保留仍在前 exit_rank 的持仓（同时受行业上限约束）
        keep = []
        used: dict[str, int] = {}
        for c in hold:
            if c not in rank.index or rank[c] > exit_rank:
                continue
            k = ind_of(c)
            if max_per_ind and k != "未知" and used.get(k, 0) >= max_per_ind:
                continue
            keep.append(c)
            used[k] = used.get(k, 0) + 1
        hold = keep

        # 补足（放宽到 exit_rank；"未知"行业不受上限约束，避免数据缺口饿死组合）
        if len(hold) < top_n:
            for c in rank.sort_values().index:
                if len(hold) >= top_n:
                    break
                if c in hold or rank[c] > exit_rank:
                    continue
                k = ind_of(c)
                if max_per_ind and k != "未知" and used.get(k, 0) >= max_per_ind:
                    continue
                hold.append(c)
                used[k] = used.get(k, 0) + 1

        # 兜底：行业上限把组合饿死时，放宽约束二次补足（保证不会因数据缺口清空持仓）
        if len(hold) < top_n:
            for c in rank.sort_values().index:
                if len(hold) >= top_n:
                    break
                if c in hold or rank[c] > max(exit_rank, top_n * 4):
                    continue
                hold.append(c)

        hold = hold[:top_n]
        if len(hold) < max(5, top_n // 3):
            continue
        if weighting == "invol" and px is not None:
            vol = px.pct_change(fill_method=None).rolling(60).std().loc[:dt].iloc[-1]
            inv = (1 / vol.reindex(hold).replace(0, np.nan)).fillna(0)
            w = inv / inv.sum() if inv.sum() > 0 else pd.Series(1 / len(hold), index=hold)
        else:
            w = pd.Series(1 / len(hold), index=hold)
        rows[dt] = w
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.fillna(0.0)


def apply_vol_target(weights: pd.DataFrame, px: pd.DataFrame,
                     target_vol: float = 0.15, lookback: int = 40,
                     cap: float = 1.0) -> pd.DataFrame:
    """组合层波动率目标：整体仓位 = target/实现波动率，上限 cap。
    权重之和即为目标股票仓位（剩余为现金），回测引擎按 gross 处理。"""
    r = px.pct_change(fill_method=None).mean(axis=1)
    rv = r.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / rv.replace(0, np.nan)).clip(0.0, cap).fillna(0.0)
    out = weights.copy()
    for dt in out.index:
        s = scale.asof(dt) if dt in scale.index else scale.loc[:dt].iloc[-1]
        out.loc[dt] = out.loc[dt] * float(s)
    return out


def regime_overlay(weights: pd.DataFrame, bench_close: pd.Series,
                   ma: int = 200, risk_off: float = 0.30) -> pd.DataFrame:
    """
    市场状态过滤：基准指数在 200 日均线之上满仓，之下降至 risk_off。
    这是控制最大回撤最有效的单一手段，也是竞赛风控分的核心。
    """
    on = (bench_close > bench_close.rolling(ma).mean())
    scale = on.astype(float).replace(0.0, risk_off)
    scale = scale.reindex(weights.index.union(scale.index)).ffill().reindex(weights.index).fillna(risk_off)
    out = weights.mul(scale, axis=0)
    return out


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    if freq == "M":
        g = s.groupby([index.year, index.month]).max()
    elif freq == "W":
        g = s.groupby([index.isocalendar().year, index.isocalendar().week]).max()
    else:
        g = s.groupby([index.year, index.quarter]).max()
    return pd.DatetimeIndex(sorted(g.values))


# ═══════════════════════════ 截面因子
def f_momentum(px: pd.DataFrame, lookback: int = 120, skip: int = 20) -> pd.DataFrame:
    """经典 12-1 动量：过去 lookback 日收益，跳过最近 skip 日（避开短期反转）"""
    return px.shift(skip) / px.shift(lookback + skip) - 1


def f_reversal(px: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """短期反转：过去 5 日收益，取负"""
    return -(px / px.shift(lookback) - 1)


def f_lowvol(px: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    r = px.pct_change(fill_method=None)
    return -r.rolling(lookback).std()


def f_volratio(px: pd.DataFrame, short: int = 5, long: int = 60) -> pd.DataFrame:
    """短期波动 / 长期波动 —— 波动率抬升是风险信号"""
    r = px.pct_change(fill_method=None)
    return -(r.rolling(short).std() / r.rolling(long).std())


def f_trend(px: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    ma_f, ma_s = px.rolling(fast).mean(), px.rolling(slow).mean()
    return ma_f / ma_s - 1


def f_liquidity(px: pd.DataFrame, vol: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """成交额对数 —— 偏好流动性（避免买不动也卖不掉）"""
    amt = (px * vol).rolling(lookback).mean()
    return np.log1p(amt)


# ═══════════════════════════ 标准化
def xs_zscore(sig: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    m = sig.sub(sig.mean(axis=1), axis=0)
    s = sig.std(axis=1).replace(0, np.nan)
    z = m.div(s, axis=0)
    return z.clip(-clip, clip)


def winsorize_rows(sig: pd.DataFrame, p: float = 0.02) -> pd.DataFrame:
    lo = sig.quantile(p, axis=1)
    hi = sig.quantile(1 - p, axis=1)
    return sig.clip(lo, hi, axis=0)


# ═══════════════════════════ 组合构建
def to_weights(score: pd.DataFrame, dates: pd.DatetimeIndex, top_n: int = 30,
               min_score: float | None = None, weighting: str = "equal",
               px: pd.DataFrame | None = None) -> pd.DataFrame:
    """截面打分 -> 目标权重（行=调仓日，列=票，值=权重，和为1）"""
    rows = {}
    for dt in dates:
        if dt not in score.index:
            continue
        s = score.loc[dt].dropna()
        if len(s) < max(10, top_n // 2):
            continue
        if min_score is not None:
            s = s[s >= min_score]
        if len(s) < max(10, top_n // 2):
            continue
        sel = s.nlargest(min(top_n, len(s)))
        if weighting == "invol" and px is not None:
            vol = px.pct_change(fill_method=None).rolling(60).std().loc[:dt].iloc[-1]
            inv = (1 / vol.reindex(sel.index).replace(0, np.nan)).fillna(0)
            w = inv / inv.sum() if inv.sum() > 0 else pd.Series(1 / len(sel), index=sel.index)
        else:
            w = pd.Series(1 / len(sel), index=sel.index)
        rows[dt] = w
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.fillna(0.0)


def composite(factors: list[pd.DataFrame], weights: list[float] | None = None,
              min_frac: float = 0.5) -> pd.DataFrame:
    """
    多因子加权合成（各因子先截面 z-score）。

    关键：因子全缺的股票必须保持 NaN，绝不能被当成 0 分。
    否则已退市/长期停牌股（所有因子 NaN → 0 分）会在活股票普遍负分时排到第一，
    回测会反复选中"死股票"，导致仓位为空、净值不动。
    """
    zs = [xs_zscore(winsorize_rows(f)) for f in factors]
    if weights is None:
        weights = [1.0] * len(zs)
    arr = np.stack([z.values * w for z, w in zip(zs, weights)])
    valid = np.isfinite(arr)
    cnt = valid.sum(axis=0)
    tot = np.nansum(arr, axis=0)
    need = max(1, int(np.ceil(len(zs) * min_frac)))
    tot[cnt < need] = np.nan          # 有效因子不足一半 → 视为无信号
    return pd.DataFrame(tot, index=zs[0].index, columns=zs[0].columns)


# ═══════════════════════════ 时序择时（用波动率状态模型）
def vol_target_scale(px: pd.DataFrame, target_vol: float = 0.15,
                     lookback: int = 60, cap: float = 1.0) -> pd.Series:
    """组合层波动率目标：按实现波动率缩放仓位（0~cap）"""
    r = px.pct_change(fill_method=None).mean(axis=1)
    rv = r.rolling(lookback).std() * np.sqrt(252)
    return (target_vol / rv).clip(0.0, cap).fillna(0.0)
