# -*- coding: utf-8 -*-
"""
QuantPilot · 因子库
=========================================
四大族：价量 / 基本面 / 行业相对 / 时序
所有因子在调仓日 T 只使用 T 及之前的信息（价量自然满足；基本面用公告日对齐）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ══════════════════════════════════════════ 价量因子
def _px(panel):
    return pd.DataFrame({c: d.set_index("date")["close"] for c, d in panel.items()}).sort_index()


def _amt(panel):
    return pd.DataFrame({c: d.set_index("date")["amount"] for c, d in panel.items()}).sort_index()


def _vol(panel):
    return pd.DataFrame({c: d.set_index("date")["volume"] for c, d in panel.items()}).sort_index()


PRICE_FACTORS = {}


def price_factors(panel: dict, reb: pd.DatetimeIndex | None = None) -> dict[str, pd.DataFrame]:
    px, amt, vol = _px(panel), _amt(panel), _vol(panel)
    r = px.pct_change(fill_method=None)
    out = {}
    _reb = reb

    def reg(name, series, desc, family="价量"):
        out[name] = series.reindex(_reb) if _reb is not None else series
        PRICE_FACTORS[name] = (desc, family)

    # ── 动量族
    reg("mom_120_20", px.shift(20) / px.shift(140) - 1, "动量 120-20")
    reg("mom_250_20", px.shift(20) / px.shift(270) - 1, "动量 250-20")
    reg("rev_5",  -(px / px.shift(5) - 1), "短期反转 5日")
    reg("rev_20", -(px / px.shift(20) - 1), "中期反转 20日")

    # ── 波动族
    reg("lowvol_60", -r.rolling(60).std(), "低波动 60日")
    reg("vol_ratio", -(r.rolling(5).std() / r.rolling(60).std()), "波动率压缩")
    reg("skew_20", -r.rolling(20).skew(), "收益偏度 20日")
    reg("kurt_20", -r.rolling(20).kurt(), "收益峰度 20日")
    reg("down_vol_ratio", -(r.where(r < 0).rolling(60).std() /
                            r.where(r > 0).rolling(60).std()), "下行/上行波动比")

    # ── 趋势族
    reg("trend_20_60", px.rolling(20).mean() / px.rolling(60).mean() - 1, "趋势 20/60")
    reg("ma200_gap", px / px.rolling(200).mean() - 1, "离 200 日均线距离")
    reg("high_52w", px / px.rolling(250).max(), "距 52 周高点")

    # ── 流动性 / 微观结构
    reg("liquidity", np.log1p(amt.rolling(20).mean()), "流动性（成交额对数）")
    reg("amihud", -(r.abs() / amt.replace(0, np.nan)).rolling(20).mean() * 1e10, "Amihud 非流动性")
    reg("amt_z", (amt - amt.rolling(20).mean()) / amt.rolling(20).std(), "成交额异动")
    reg("vol_z", (vol - vol.rolling(20).mean()) / vol.rolling(20).std(), "成交量异动")
    reg("turnover_stab", -vol.rolling(20).std() / vol.rolling(60).mean(), "换手稳定性")

    # ── 彩票型 / MAX 效应
    reg("max_ret_20", -r.rolling(20).max(), "彩票偏好（MAX 效应）")
    reg("min_ret_20", r.rolling(20).min(), "尾部最小收益")
    reg("up_days", (r > 0).rolling(20).mean(), "上涨天数占比")

    # ── 自相关 / 状态
    reg("autocorr_20", r.rolling(60).apply(
        lambda x: pd.Series(x).autocorr(20) if len(x) > 25 else np.nan, raw=False), "收益自相关")
    return out


# ══════════════════════════════════════════ 时序因子（自身标准化）
def ts_factors(panel: dict, reb: pd.DatetimeIndex | None = None) -> dict[str, pd.DataFrame]:
    px = _px(panel)
    r = px.pct_change(fill_method=None)
    out = {}
    out["ts_mom_20"] = (px / px.shift(20) - 1) / (r.rolling(20).std() * np.sqrt(20))
    out["ts_mom_60"] = (px / px.shift(60) - 1) / (r.rolling(60).std() * np.sqrt(60))
    out["ts_vol_raw"] = -r.rolling(60).std()
    PRICE_FACTORS.update({"ts_mom_20": ("时序动量 20日（标准化）", "时序"),
                          "ts_mom_60": ("时序动量 60日（标准化）", "时序"),
                          "ts_vol_raw": ("原始波动率（负向）", "时序")})
    return {k: (v.replace([np.inf, -np.inf], np.nan).reindex(reb) if reb is not None else v)
            for k, v in out.items()}


# ══════════════════════════════════════════ 行业相对因子
def industry_factors(panel: dict, industry: dict[str, str],
                     reb: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    px = _px(panel)
    r = px.pct_change(fill_method=None)
    mom = px.shift(20) / px.shift(140) - 1
    vol = r.rolling(60).std()
    ind = pd.Series({c: industry.get(c, "未知") for c in px.columns})

    out = {}
    for name, raw, sign in [("ind_mom", mom, 1), ("ind_lowvol", vol, -1),
                            ("ind_rel_mom", mom, 1), ("ind_rel_lowvol", vol, -1)]:
        rows = {}
        for d in reb:
            if d not in raw.index:
                continue
            s = raw.loc[d]
            grp = ind.reindex(s.index)
            med = s.groupby(grp).transform("median")
            if name.startswith("ind_rel"):
                rows[d] = sign * (s - med)
            else:
                rows[d] = sign * med
        if rows:
            out[name] = pd.DataFrame(rows).T
    PRICE_FACTORS.update({
        "ind_mom": ("行业动量（行业中位）", "行业相对"),
        "ind_lowvol": ("行业低波（负向中位）", "行业相对"),
        "ind_rel_mom": ("个股相对行业动量", "行业相对"),
        "ind_rel_lowvol": ("个股相对行业低波", "行业相对"),
    })
    return out


# ══════════════════════════════════════════ 基本面因子（时点对齐）
FUND_META = {
    "f_roe": ("ROE", "基本面"), "f_np_yoy": ("净利润同比", "基本面"),
    "f_rev_yoy": ("营收同比", "基本面"), "f_gm": ("销售毛利率", "基本面"),
    "f_gm_chg": ("毛利率同比变化", "基本面"), "f_accel": ("盈利加速", "基本面"),
    "f_cash_q": ("现金流质量", "基本面"), "f_ey": ("盈利收益率", "基本面"),
    "f_by": ("账面收益率", "基本面"),
}


def fundamental_factors(panel, reb, px) -> dict[str, pd.DataFrame]:
    import fundamentals as F
    fp = F.build_factor_panel()
    if fp.empty:
        return {}
    cols = [c for c in F.FACTORS if c in fp.columns]
    right = fp.dropna(subset=["announce"])[["code", "announce", "bps", "eps_ttm"] + cols]
    right = right.sort_values("announce").reset_index(drop=True)
    codes = list(panel)
    grid = pd.DataFrame([(d, c) for d in reb for c in codes], columns=["date", "code"])
    grid = grid.sort_values("date").reset_index(drop=True)
    m = pd.merge_asof(grid, right, left_on="date", right_on="announce", by="code",
                      direction="backward", tolerance=pd.Timedelta(days=400))
    plong = px.loc[[d for d in reb if d in px.index]].stack().rename("price").reset_index()
    plong.columns = ["date", "code", "price"]
    m = m.merge(plong, on=["date", "code"], how="left")
    m["f_ey"] = m["eps_ttm"] / m["price"]
    m["f_by"] = m["bps"] / m["price"]
    use = cols + ["f_ey", "f_by"]
    out = {}
    for f in use:
        mat = m.pivot(index="date", columns="code", values=f).reindex(index=reb, columns=codes)
        out[f] = mat.replace([np.inf, -np.inf], np.nan)
    return out


def all_factors(panel, industry, reb) -> dict[str, pd.DataFrame]:
    px = _px(panel)
    fx = {}
    fx.update(price_factors(panel, reb))
    fx.update(ts_factors(panel, reb))
    try:
        fx.update(industry_factors(panel, industry, reb))
    except Exception as e:
        print(f"[factors] 行业因子失败: {e}")
    try:
        fx.update(fundamental_factors(panel, reb, px))
    except Exception as e:
        print(f"[factors] 基本面因子失败: {e}")
    return fx
