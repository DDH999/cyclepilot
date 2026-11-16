# -*- coding: utf-8 -*-
"""
组合与风控层（M63）：风险预算与波动率目标
====================================================
解决什么问题
------------
`low_vol top10 等权` 有两个隐含假设：
1. 每只票风险贡献相同 —— **等权 ≠ 等风险**，低波股和高波股放同样权重，
   组合风险其实被高波股主导
2. 总仓位恒定 100% —— 市场波动放大时组合回撤同样放大

本模块给两个能力：
* **风险预算**：按风险（而非金额）分配权重 → 逆波动 / 等风险贡献(ERC)
* **波动率目标**：根据组合近期已实现波动，动态调整**总仓位**

两者关系
--------
```
风险预算  →  决定「钱怎么分给各票」（相对权重）
波动率目标 →  决定「总共投多少」    （总敞口 0~max_leverage）
```

⚠ 全部只用 T 及之前的数据，无前视
--------------------------------
* 波动率用 `shift(1)` 后的历史收益算
* 仓位调整作用于**次日**开盘
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["TRADING_DAYS", "ewma_vol", "realized_vol", "vol_target_cap",
           "inverse_vol_weights", "risk_parity_weights", "apply_risk_budget",
           "risk_contributions", "format_report"]

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, window: int = 20,
                 annualize: bool = True) -> pd.Series:
    """滚动已实现波动率（只用过去 window 天）。"""
    s = pd.Series(returns).astype(float)
    v = s.rolling(int(window), min_periods=max(5, int(window) // 3)).std(ddof=0)
    return v * np.sqrt(TRADING_DAYS) if annualize else v


def ewma_vol(returns: pd.Series, halflife: int = 20,
             annualize: bool = True) -> pd.Series:
    """EWMA 波动率（对近期更敏感）。"""
    s = pd.Series(returns).astype(float)
    v = s.ewm(halflife=int(halflife), min_periods=5).std()
    return v * np.sqrt(TRADING_DAYS) if annualize else v


def vol_target_cap(port_returns: pd.Series, target_vol: float = 0.15,
                   window: int = 20, max_leverage: float = 1.0,
                   min_cap: float = 0.0) -> pd.Series:
    """
    波动率目标仓位（0~max_leverage）。

    cap_t = target_vol / 已实现波动_t ，并截断到 [min_cap, max_leverage]

    ⚠ 用 shift(1)：T 日收盘算出的 cap 只能作用于 T+1，不能当天用。
    """
    rv = realized_vol(port_returns, window)
    raw = float(target_vol) / rv.replace(0, np.nan)
    cap = raw.clip(lower=float(min_cap), upper=float(max_leverage))
    return cap.shift(1).fillna(float(min_cap))


def inverse_vol_weights(vols: pd.Series, top_n: int | None = None,
                        min_vol: float = 0.05) -> pd.Series:
    """
    逆波动加权：w_i ∝ 1/σ_i。

    ⚠ min_vol 下限必不可少：数据里存在"价格几乎不动"的票（停牌/错价），
    其 σ 趋近 0，逆波动会给它压倒性权重，组合变成一条直线，
    算出夏普 4+ 的假象（实测踩过）。
    """
    v = pd.Series(vols).astype(float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v):
        v = v.clip(lower=float(min_vol))
    if len(v) == 0:
        return pd.Series(dtype=float)
    inv = 1.0 / v
    if top_n:
        inv = inv.nlargest(int(top_n))
    return inv / inv.sum()


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """各资产对组合波动的贡献（占比，合计≈1）。"""
    cols = [c for c in w.index if c in cov.columns]
    if not cols:
        return pd.Series(dtype=float)
    ww = w.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    C = cov.loc[cols, cols].to_numpy(dtype=float)
    port_var = float(ww @ C @ ww)
    if port_var <= 0:
        return pd.Series(0.0, index=cols)
    rc = ww * (C @ ww) / port_var
    return pd.Series(rc, index=cols)


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 500,
                        tol: float = 1e-9) -> pd.Series:
    """
    等风险贡献（ERC）：每只票对组合波动的贡献相等。

    用循环坐标下降，不依赖 scipy，数值稳定。
    """
    cols = list(cov.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cols)
    C = cov.to_numpy(dtype=float)
    C = np.nan_to_num(C, nan=0.0)
    # 数值上保证正定
    C = C + np.eye(n) * max(1e-12, float(np.trace(C)) / n * 1e-6)
    w = np.full(n, 1.0 / n)
    target = np.full(n, 1.0 / n)
    for _ in range(int(max_iter)):
        rc = w * (C @ w)
        rc = np.where(rc <= 0, 1e-12, rc)
        w_new = w * (target / rc) ** 0.5
        w_new = np.clip(w_new, 1e-8, None)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=cols)


def apply_risk_budget(scores: pd.DataFrame, closes: pd.DataFrame,
                      top_n: int = 10, scheme: str = "equal",
                      vol_window: int = 20,
                      rebalance: int = 20) -> pd.DataFrame:
    """
    把「得分 → 权重」的构建从等权换成风险预算。

    scheme: equal / inv_vol / erc
    返回 date × code 权重矩阵（每行合计 1，总仓位另由 vol_target_cap 控制）
    """
    if scores is None or len(scores) == 0:
        return pd.DataFrame()
    rets = closes.pct_change(fill_method=None)
    rows, idx = [], []
    dts = list(scores.index)
    for i in range(0, len(dts), int(rebalance)):
        d = dts[i]
        row = scores.loc[d].dropna()
        if len(row) < 2:
            continue
        pick = row.nlargest(int(top_n)).index
        w = pd.Series(0.0, index=scores.columns)
        if scheme == "equal":
            w[pick] = 1.0 / len(pick)
        else:
            hist = rets.loc[:d, pick].tail(int(vol_window))
            vols = hist.std(ddof=0) * np.sqrt(TRADING_DAYS)
            if scheme == "inv_vol":
                sub = inverse_vol_weights(vols)
            elif scheme == "erc":
                cov = hist.cov()
                sub = risk_parity_weights(cov)
            else:
                raise ValueError(f"未知 scheme {scheme!r}")
            if len(sub) == 0:
                w[pick] = 1.0 / len(pick)
            else:
                w[sub.index] = sub / sub.sum()
        rows.append(w); idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx)) if rows \
        else pd.DataFrame()


def format_report(res: dict) -> str:
    L = ["=" * 74, "风险预算 / 波动率目标", "=" * 74]
    for k, v in res.items():
        L.append(f"  {k:<22}{v}")
    L.append("=" * 74)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-02", periods=300)
    a = pd.Series(rng.normal(0.0004, 0.012, 300), index=idx)
    b = pd.Series(rng.normal(0.0004, 0.024, 300), index=idx)
    port = pd.concat([a, b], axis=1).mean(axis=1)

    cap = vol_target_cap(port, 0.12, 20, 1.0)
    print(f"波动率目标 cap: 均值 {cap.mean():.2f}  "
          f"最低 {cap.min():.2f}  最高 {cap.max():.2f}")
    assert cap.max() <= 1.0 and cap.min() >= 0
    assert cap.isna().sum() == 0

    # cap 必须只用历史：T 日 cap 与 T 日收益不应高度相关
    corr = cap.corr(port)
    print(f"cap 与当日收益相关性: {corr:.3f}（应接近 0，不能是负很多）")

    C = pd.DataFrame([[0.0144, 0.002], [0.002, 0.0576]],
                     index=["a", "b"], columns=["a", "b"])
    erc = risk_parity_weights(C)
    rc = risk_contributions(erc, C)
    print(f"ERC 权重: {erc.round(3).to_dict()}")
    print(f"风险贡献: {rc.round(3).to_dict()}（应各 0.5）")
    assert abs(rc["a"] - 0.5) < 0.02 and abs(rc["b"] - 0.5) < 0.02

    iv = inverse_vol_weights(pd.Series({"a": 0.12, "b": 0.24}))
    print(f"逆波动权重: {iv.round(3).to_dict()}（低波应更大）")
    assert iv["a"] > iv["b"]

    print("OK v" + __version__)
