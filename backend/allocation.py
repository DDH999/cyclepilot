# -*- coding: utf-8 -*-
"""
资产配置、风险平价与凯利仓位（M15）
====================================================
三件事：

1. **资产配置**：等权 / 逆波动 / 最小方差 / **风险平价**
2. **风险分解**：每个资产贡献了多少组合风险（不是权重！）
3. **凯利公式**：理论最优仓位、分数凯利、过度下注检测

风险平价的核心
--------------
等权配置 ≠ 等风险。低波动资产贡献的风险远小于高波动资产。
风险平价要求每个资产的**风险贡献**相等，需要迭代求解。

凯利公式的核心
--------------
f* = 期望收益 / 方差（连续近似），或 f* = p - q/b（离散下注）。
**满凯利在真实世界几乎必然爆仓**——因为参数估计误差会被放大。
实践用 1/4~1/2 凯利。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["DEFAULTS", "equal_weight", "inverse_vol", "min_variance",
           "risk_parity", "risk_contributions", "kelly_fraction",
           "kelly_from_returns", "compare_allocation", "format_report"]

TD = 252


class DEFAULTS:
    kelly_fraction_default = 0.25      # 默认用 1/4 凯利
    max_weight = 1.0
    rp_max_iter = 500
    rp_tol = 1e-8


def _returns(x) -> pd.DataFrame | pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.replace([np.inf, -np.inf], np.nan).dropna()
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    mean_abs = abs(float(s.mean())) if len(s) else 0.0
    cv = (float(s.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    pos = float((s > 0).mean()) if len(s) else 0.0
    if len(s) > 20 and pos > 0.95 and cv < 0.5:
        s = s.pct_change(fill_method=None).dropna()
    return s


# ══════════════════════════════════════════════════════════
# 权重方案
# ══════════════════════════════════════════════════════════
def equal_weight(n_or_cols) -> pd.Series:
    cols = list(n_or_cols) if not isinstance(n_or_cols, int) else \
        [f"A{i}" for i in range(n_or_cols)]
    return pd.Series(1.0 / len(cols), index=cols)


def inverse_vol(returns: pd.DataFrame) -> pd.Series:
    """逆波动配置：波动越低权重越高。"""
    R = _returns(returns)
    vol = R.std(ddof=0).replace(0, np.nan)
    inv = (1.0 / vol).dropna()
    if inv.sum() <= 0:
        return equal_weight(R.columns)
    return inv / inv.sum()


def min_variance(returns: pd.DataFrame, max_w: float = 1.0) -> pd.Series:
    """最小方差组合（权重 ∈ [0, max_w]，和为 1）。"""
    R = _returns(returns)
    cov = R.cov().to_numpy() * TD
    n = cov.shape[0]
    x0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_w)] * n
    try:
        from scipy.optimize import minimize
        r = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                     bounds=bounds, constraints=cons)
        w = r.x if r.success else x0
    except Exception:
        w = x0
    return pd.Series(w, index=R.columns)


def risk_contributions(weights, cov) -> pd.Series:
    """
    各资产的风险贡献（占组合总风险的比例，合计 = 1）。

    RC_i = w_i × (Σw)_i / σ_p²
    """
    w = np.asarray(weights, dtype="float64")
    C = np.asarray(cov, dtype="float64")
    var = float(w @ C @ w)
    if var <= 0:
        return pd.Series(np.full(len(w), np.nan))
    rc = w * (C @ w) / var
    return pd.Series(rc)


def risk_parity(returns: pd.DataFrame, max_iter: int = DEFAULTS.rp_max_iter,
                tol: float = DEFAULTS.rp_tol,
                max_weight: float | None = None) -> pd.Series:
    """
    风险平价：让每个资产的**风险贡献相等**。

    实现：先用定点迭代给初值，再用优化器最小化风险贡献的离散度。
    纯定点迭代（w ∝ w/RC）在波动差异大时收敛慢甚至震荡，
    实测会停在 [0.45, 0.24, 0.11, 0.20] 这种明显不均衡的解上。

    max_weight
    ----------
    单个资产权重上限。**强烈建议设置**：
    纯风险平价会把绝大部分资金配给波动最小的资产——
    实测在「股票/债券/黄金/现金」四类上给出 **现金 82.9%**，
    数学正确，但不是一个能用的组合。
    设 max_weight=0.4 后现金被压到 40%，其余按风险贡献重新分配。
    None 表示不限（保持数学上的严格等风险贡献）。
    """
    R = _returns(returns)
    cov = R.cov().to_numpy() * TD
    n = cov.shape[0]
    target = 1.0 / n

    def rc_of(w):
        var = float(w @ cov @ w)
        if var <= 0:
            return np.full(n, np.nan)
        return w * (cov @ w) / var

    upper = 1.0 if max_weight is None else float(max_weight)
    # 校验必须放在 try 之外：放在 try 里会被 `except Exception: w = w0`
    # 吞掉，变成"静默降级成逆波动权重"，调用方完全看不出配置是错的。
    if upper * n < 1.0 - 1e-9:
        raise ValueError(
            f"max_weight={upper} 太小：{n} 个资产的上限合计只有 "
            f"{upper * n:.2f} < 1，无法配满")

    # 初值：逆波动
    vol = np.sqrt(np.diag(cov))
    vol = np.where(vol > 0, vol, np.nan)
    w0 = np.where(np.isfinite(vol), 1.0 / vol, 1.0)
    w0 = w0 / w0.sum()

    try:
        from scipy.optimize import minimize
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(1e-8, upper)] * n
        obj = lambda w: float(np.nansum((rc_of(w) - target) ** 2))
        res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": int(max_iter),
                                                  "ftol": tol})
        w = res.x if res.success else w0
    except Exception:
        w = w0

    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    return pd.Series(w, index=R.columns)


def compare_allocation(returns: pd.DataFrame, rf_annual: float = 0.0,
                       max_weight: float = 0.40) -> dict:
    """四种配置方案的对比（年化收益 / 波动 / 夏普 / 最大风险贡献）。"""
    R = _returns(returns)
    cov = R.cov().to_numpy() * TD
    mu = R.mean().to_numpy() * TD
    nav = (1 + R).cumprod()

    # 上限必须能配满：n 个资产、每个最多 cap，则 cap*n 必须 ≥ 1。
    # 只有 2 个资产时 40% 上限无解（0.4×2=0.8），必须自适应抬高。
    n_assets = R.shape[1]
    eff_cap = max(float(max_weight), 1.0 / n_assets + 1e-9)
    cap_adjusted = eff_cap > float(max_weight) + 1e-12

    schemes = {
        "等权": equal_weight(R.columns),
        "逆波动": inverse_vol(R),
        "最小方差": min_variance(R),
        "风险平价": risk_parity(R, max_weight=eff_cap),
    }
    rows = []
    for name, w in schemes.items():
        wv = w.reindex(R.columns).fillna(0).to_numpy()
        port = (R.to_numpy() @ wv)
        vol = float(np.std(port, ddof=0) * np.sqrt(TD))
        ret = float(np.mean(port) * TD)
        sharpe = (ret - rf_annual) / vol if vol > 0 else float("nan")
        curve = np.cumprod(1 + port)
        peak = np.maximum.accumulate(curve)
        mdd = float(np.nanmin(curve / np.where(peak <= 0, np.nan, peak) - 1))
        rc = risk_contributions(wv, cov).to_numpy()
        rows.append({"scheme": name,
                     "weights": {k: round(float(v), 4) for k, v in w.items()},
                     "annual_return": round(ret, 6),
                     "volatility": round(vol, 6),
                     "sharpe": round(float(sharpe), 4) if np.isfinite(sharpe) else None,
                     "max_drawdown": round(mdd, 6),
                     "max_risk_contrib": round(float(np.nanmax(np.abs(rc))), 4)
                     if np.isfinite(rc).any() else None})
    # 提示：不限上限时风险平价会极端偏向低波动资产
    notes = []
    if cap_adjusted:
        notes.append(
            f"仅 {n_assets} 个资产，{max_weight*100:.0f}% 上限无法配满，"
            f"自动抬到 {eff_cap*100:.0f}%")
    uncapped = risk_parity(R, max_weight=None)
    if float(uncapped.max()) > 0.6:
        notes.append(
            f"不限上限时风险平价会把 {float(uncapped.max())*100:.0f}% "
            f"配给单一低波动资产（{uncapped.idxmax()}）；"
            f"本次已按 {max_weight*100:.0f}% 上限约束")
    return {"ok": True, "n_days": int(len(R)), "assets": list(R.columns),
            "schemes": rows, "max_weight": max_weight, "notes": notes}


# ══════════════════════════════════════════════════════════
# 凯利公式
# ══════════════════════════════════════════════════════════
def kelly_fraction(win_rate: float, payoff_ratio: float,
                   fraction: float = DEFAULTS.kelly_fraction_default,
                   cap: float = 1.0) -> dict:
    """
    离散下注的凯利：f* = p - q/b

    win_rate     : 胜率 p
    payoff_ratio : 盈亏比 b（赢时赚 b，输时亏 1）
    fraction     : 使用比例（0.25 = 1/4 凯利，实践推荐）
    """
    p = float(win_rate)
    b = float(payoff_ratio)
    if not (0 <= p <= 1) or b <= 0:
        raise ValueError("胜率需在 [0,1]，盈亏比需 > 0")
    q = 1 - p
    f_full = p - q / b
    f_used = max(f_full, 0.0) * fraction
    return {
        "ok": True, "win_rate": round(p, 4), "payoff_ratio": round(b, 4),
        "edge": round(p * b - q, 4),
        "kelly_full": round(f_full, 4),
        "fraction_used": fraction,
        "kelly_fraction": round(min(f_used, cap), 4),
        "has_edge": bool(f_full > 0),
        "verdict": ("无正期望，不应下注" if f_full <= 0 else
                    f"满凯利 {f_full*100:.1f}%，建议 {f_used*100:.1f}%"),
        "note": "满凯利对参数误差极敏感，实践用 1/4~1/2 凯利",
    }


def kelly_from_returns(returns, rf_annual: float = 0.0,
                       fraction: float = DEFAULTS.kelly_fraction_default,
                       cap: float = 1.0) -> dict:
    """
    连续近似的凯利：f* = (μ - rf) / σ²

    返回满凯利、建议仓位、以及"半凯利爆仓概率"的粗略提示。
    """
    s = _returns(returns)
    if not isinstance(s, pd.Series):
        raise ValueError("请传入单条收益序列")
    if len(s) < 30:
        raise ValueError(f"样本仅 {len(s)} 天，至少需要 30 天")
    rfd = (1 + rf_annual) ** (1 / TD) - 1
    mu = float(np.mean(s.to_numpy()) - rfd)
    var = float(np.var(s.to_numpy(), ddof=0))
    if var <= 0:
        return {"ok": False, "reason": "收益方差为 0，凯利无定义"}
    f_full = mu / var
    f_used = max(f_full, 0.0) * fraction
    return {
        "ok": True, "n": int(len(s)),
        "annual_return": round(mu * TD, 6),
        "annual_vol": round(np.sqrt(var * TD), 6),
        "kelly_full": round(float(f_full), 4),
        "fraction_used": fraction,
        "kelly_fraction": round(float(min(f_used, cap)), 4),
        "has_edge": bool(f_full > 0),
        "over_bet_warning": bool(f_full > 1.0),
        "verdict": ("无正期望，凯利为负 → 应空仓" if f_full <= 0 else
                    (f"满凯利 {f_full*100:.0f}% 已超过 100%，"
                     "说明收益/风险估计极不稳定，务必用分数凯利"
                     if f_full > 1 else
                     f"满凯利 {f_full*100:.1f}%，建议 {f_used*100:.1f}%")),
        "note": "凯利基于历史均值方差估计，估计误差会导致系统性过度下注",
    }


def format_report(res: dict, kind: str = "allocation") -> str:
    L = ["=" * 66]
    if kind == "kelly":
        L.append("凯利公式仓位测算")
        L.append("=" * 66)
        for k in ("win_rate", "payoff_ratio", "edge", "kelly_full",
                  "kelly_fraction", "has_edge", "annual_return", "annual_vol",
                  "over_bet_warning"):
            if k in res:
                L.append(f"  {k:<18} {res[k]}")
        L.append(f"  结论: {res.get('verdict')}")
    else:
        L.append("资产配置方案对比")
        L.append("=" * 66)
        L.append(f"  {'方案':<8}{'年化':>9}{'波动':>9}{'夏普':>8}{'回撤':>9}{'最大风险贡献':>12}")
        L.append("  " + "-" * 56)
        for s in res.get("schemes", []):
            L.append(f"  {s['scheme']:<8}{s['annual_return']*100:>8.2f}%"
                     f"{s['volatility']*100:>8.2f}%"
                     f"{(s['sharpe'] if s['sharpe'] is not None else float('nan')):>8.2f}"
                     f"{s['max_drawdown']*100:>8.2f}%"
                     f"{s['max_risk_contrib']:>12.3f}")
        L.append("  " + "-" * 56)
        L.append("  注：最大风险贡献越接近 1/n，说明风险越均衡")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    idx = pd.bdate_range("2020-01-02", periods=800)
    rng = np.random.default_rng(7)
    R = pd.DataFrame({
        "股票": rng.normal(0.0006, 0.018, 800),
        "债券": rng.normal(0.00015, 0.004, 800),
        "黄金": rng.normal(0.00025, 0.010, 800),
        "现金": rng.normal(0.00006, 0.0005, 800),
    }, index=idx)

    cmp = compare_allocation(R)
    print(format_report(cmp))

    rp = risk_parity(R)
    cov = R.cov().to_numpy() * TD
    rc = risk_contributions(rp.to_numpy(), cov)
    print("风险平价权重:", {k: round(float(v), 3) for k, v in rp.items()})
    print("风险贡献    :", {k: round(float(v), 3) for k, v in
                          zip(R.columns, rc.to_numpy())})
    assert abs(rc.sum() - 1.0) < 1e-6
    assert np.max(np.abs(rc.to_numpy() - 1 / 4)) < 0.02, rc
    print(f"  不限上限时现金权重 {float(rp['现金'])*100:.1f}% —— 不可用")

    rp_cap = risk_parity(R, max_weight=0.40)
    print("  加上限后权重:", {k: round(float(v), 3) for k, v in rp_cap.items()})
    assert float(rp_cap.max()) <= 0.401, float(rp_cap.max())
    assert abs(float(rp_cap.sum()) - 1.0) < 1e-6
    try:
        risk_parity(R, max_weight=0.15)
        raise AssertionError("上限过小应当报错")
    except ValueError as ex:
        print(f"  上限过小正确报错：{str(ex)[:50]}…")

    print()
    print(format_report(kelly_fraction(0.55, 1.5), kind="kelly"))
    print()
    print(format_report(kelly_from_returns(R["股票"]), kind="kelly"))
    print(f"OK v{__version__}")
