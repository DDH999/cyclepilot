# -*- coding: utf-8 -*-
"""
组合理论检验（M14）
====================================================
把五个经典理论做成**能跑出数字的检验**，而不是抄公式：

1. **EMH 弱式检验**：收益有没有自相关？随机游走检验（方差比）
2. **MPT 有效前沿**：最小方差组合、切点组合、有效前沿曲线
3. **CAPM**：beta、Jensen alpha、alpha 的 t 值、R²
4. **APT / 多因子**：对多因子的暴露、各因子 t 值、残差
5. **主动管理基本法则**：IR = IC × √宽度，三个量任知两个可推第三个

依赖：numpy / pandas / scipy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats, optimize

__version__ = "0.1.0"
__all__ = ["emh_test", "variance_ratio", "efficient_frontier", "capm_regression",
           "apt_regression", "fundamental_law", "format_report"]

TD = 252


def _ret(x) -> pd.Series:
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    mean_abs = abs(float(s.mean())) if len(s) else 0.0
    cv = (float(s.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    pos = float((s > 0).mean()) if len(s) else 0.0
    if len(s) > 20 and pos > 0.95 and cv < 0.5:      # 净值 → 收益
        s = s.pct_change(fill_method=None).dropna()
    return s


# ══════════════════════════════════════════════════════════
# 1. EMH 弱式有效性
# ══════════════════════════════════════════════════════════
def _acf(x: np.ndarray, k: int) -> float:
    n = len(x)
    if n <= k + 2:
        return float("nan")
    m = x.mean()
    denom = float(np.sum((x - m) ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum((x[k:] - m) * (x[:-k] - m)) / denom)


def variance_ratio(x, q: int = 5) -> dict:
    """
    方差比检验（Lo-MacKinlay 简化版）。

    随机游走下 Var(q 期收益) = q × Var(1 期收益)，即 VR=1。
    VR 显著偏离 1 → 存在可预测性（动量 VR>1 / 均值回归 VR<1）。
    """
    a = np.asarray(_ret(x), dtype="float64")
    n = len(a)
    if n < q * 5:
        return {"ok": False, "reason": f"样本 {n} 天，不足以做 q={q} 检验"}
    var1 = float(np.var(a, ddof=1))
    if var1 <= 0:
        return {"ok": False, "reason": "收益方差为 0"}
    # q 期重叠收益
    agg = np.convolve(a, np.ones(q), mode="valid")
    varq = float(np.var(agg, ddof=1))
    vr = varq / (q * var1)
    # 同方差下的近似标准误
    se = np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n))
    z = (vr - 1.0) / se if se > 0 else float("nan")
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
    return {"ok": True, "q": q, "vr": round(float(vr), 4), "z": round(float(z), 4),
            "p_value": round(p, 4) if np.isfinite(p) else None}


def emh_test(returns, lags: int = 5) -> dict:
    """
    弱式有效市场检验。

    三项证据：
      * 各阶自相关系数（|ρ| 小 → 无线性可预测性）
      * Ljung-Box Q 统计量（联合检验，p 大 → 不可预测）
      * 方差比（VR≈1 → 随机游走）

    结论只针对**这条序列**，不代表"市场有效/无效"的普遍断言。
    """
    a = np.asarray(_ret(returns), dtype="float64")
    n = len(a)
    if n < 60:
        return {"ok": False, "reason": f"样本仅 {n} 天，至少需要 60 天"}

    acfs = {f"AC{i}": round(_acf(a, i), 4) for i in range(1, lags + 1)}
    # Ljung-Box
    q_stat = 0.0
    for i in range(1, lags + 1):
        rho = _acf(a, i)
        if np.isfinite(rho):
            q_stat += rho ** 2 / (n - i)
    q_stat *= n * (n + 2)
    p_lb = float(1 - stats.chi2.cdf(q_stat, df=lags))

    vrs = {q: variance_ratio(a, q) for q in (2, 5, 10)}
    vr_ok = [v for v in vrs.values() if v.get("ok")]

    max_abs_acf = max(abs(v) for v in acfs.values() if np.isfinite(v))
    if p_lb < 0.05:
        verdict = ("存在显著自相关（Ljung-Box p<0.05）："
                   "该序列不是随机游走，弱式有效性不成立")
        efficient = False
    elif max_abs_acf > 0.1:
        verdict = "自相关偏大但未达显著，可能存在弱可预测性"
        efficient = False
    else:
        verdict = "未发现显著线性可预测性，与该序列的弱式有效性一致"
        efficient = True

    return {"ok": True, "n": n, "acf": acfs, "ljung_box_q": round(q_stat, 3),
            "ljung_box_p": round(p_lb, 4), "variance_ratio": vrs,
            "max_abs_acf": round(max_abs_acf, 4),
            "efficient": efficient, "verdict": verdict,
            "note": "结论仅针对本序列；市场整体有效性需多资产联合检验"}


# ══════════════════════════════════════════════════════════
# 2. MPT 有效前沿
# ══════════════════════════════════════════════════════════
def efficient_frontier(returns: pd.DataFrame, rf_annual: float = 0.0,
                       n_points: int = 15) -> dict:
    """
    均值-方差有效前沿（允许做空限制：权重 ∈ [0,1]，和为 1）。

    returns: 行=日期，列=资产
    返回：最小方差组合、最大夏普（切点）组合、前沿曲线
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        raise ValueError("需要至少 2 个资产的收益矩阵")
    R = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(R) < 30:
        raise ValueError(f"样本仅 {len(R)} 天，至少需要 30 天")

    mu = R.mean().to_numpy() * TD
    cov = R.cov().to_numpy() * TD
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def port_vol(w):
        return float(np.sqrt(max(w @ cov @ w, 0.0)))

    # 最小方差
    res_mv = optimize.minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                               bounds=bounds, constraints=cons)
    w_mv = res_mv.x if res_mv.success else x0

    # 最大夏普
    def neg_sharpe(w):
        vol = port_vol(w)
        return -(float(w @ mu) - rf_annual) / vol if vol > 0 else 1e9

    res_ms = optimize.minimize(neg_sharpe, x0, method="SLSQP",
                               bounds=bounds, constraints=cons)
    w_ms = res_ms.x if res_ms.success else x0

    frontier = []
    lo, hi = float(mu.min()), float(mu.max())
    for tgt in np.linspace(lo, hi, n_points):
        c = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
             {"type": "eq", "fun": lambda w, t=tgt: float(w @ mu) - t})
        r = optimize.minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                              bounds=bounds, constraints=c)
        if r.success:
            frontier.append({"return": round(float(tgt), 6),
                             "vol": round(port_vol(r.x), 6)})

    return {
        "ok": True, "assets": list(R.columns), "n_days": int(len(R)),
        "min_variance": {"weights": dict(zip(R.columns, np.round(w_mv, 4))),
                         "return": round(float(w_mv @ mu), 6),
                         "vol": round(port_vol(w_mv), 6),
                         "sharpe": round((float(w_mv @ mu) - rf_annual)
                                         / max(port_vol(w_mv), 1e-12), 4)},
        "max_sharpe": {"weights": dict(zip(R.columns, np.round(w_ms, 4))),
                       "return": round(float(w_ms @ mu), 6),
                       "vol": round(port_vol(w_ms), 6),
                       "sharpe": round((float(w_ms @ mu) - rf_annual)
                                       / max(port_vol(w_ms), 1e-12), 4)},
        "frontier": frontier,
    }


# ══════════════════════════════════════════════════════════
# 3. CAPM
# ══════════════════════════════════════════════════════════
def _ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """含截距的 OLS，返回系数、t 值、R²。"""
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = max(n - Xd.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        cov = np.full((Xd.shape[1], Xd.shape[1]), np.nan)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    labels = ["alpha"] + names
    return {"coef": dict(zip(labels, beta.tolist())),
            "t": dict(zip(labels, tvals.tolist())),
            "r2": float(r2), "resid": resid, "n": n}


def capm_regression(returns, bench, rf_annual: float = 0.0) -> dict:
    """CAPM：r_p - rf = alpha + beta·(r_m - rf) + e。"""
    rp, rm = _ret(returns), _ret(bench)
    idx = rp.index.intersection(rm.index)
    if len(idx) < 30:
        raise ValueError(f"重叠样本仅 {len(idx)} 天，至少需要 30 天")
    rfd = (1 + rf_annual) ** (1 / TD) - 1
    y = (rp.loc[idx] - rfd).to_numpy()
    x = (rm.loc[idx] - rfd).to_numpy()
    fit = _ols(y, x.reshape(-1, 1), ["beta"])
    alpha_d, beta = fit["coef"]["alpha"], fit["coef"]["beta"]
    t_alpha = fit["t"]["alpha"]

    expected = rf_annual + beta * (float(rm.mean() * TD) - rf_annual)
    actual = float(rp.mean() * TD)
    return {
        "ok": True, "n": fit["n"], "beta": round(beta, 4),
        "alpha_annual": round(alpha_d * TD, 6),
        "alpha_t": round(float(t_alpha), 3),
        "alpha_significant": bool(abs(t_alpha) > 1.96),
        "r2": round(fit["r2"], 4),
        "actual_return": round(actual, 6),
        "capm_expected": round(expected, 6),
        "mispricing": round(actual - expected, 6),
        "verdict": ("alpha 显著（|t|>1.96），CAPM 未能解释其收益"
                    if abs(t_alpha) > 1.96 else
                    "alpha 不显著，收益基本可由市场 beta 解释"),
    }


# ══════════════════════════════════════════════════════════
# 4. APT / 多因子
# ══════════════════════════════════════════════════════════
def apt_regression(returns, factors: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """
    多因子回归（APT 实证形式）：r = a + Σ β_i·f_i + e。

    factors: 行=日期，列=因子收益
    因子间高度共线会导致 t 值不可信，结果里会给出条件数提示。
    """
    rp = _ret(returns)
    F = factors.replace([np.inf, -np.inf], np.nan).dropna()
    idx = rp.index.intersection(F.index)
    if len(idx) < 30:
        raise ValueError(f"重叠样本仅 {len(idx)} 天，至少需要 30 天")
    rfd = (1 + rf_annual) ** (1 / TD) - 1
    y = (rp.loc[idx] - rfd).to_numpy()
    X = F.loc[idx].to_numpy()
    names = [str(c) for c in F.columns]
    fit = _ols(y, X, names)

    cond = float(np.linalg.cond(np.column_stack([np.ones(len(X)), X])))
    loadings = {k: round(v, 4) for k, v in fit["coef"].items() if k != "alpha"}
    tvals = {k: round(float(v), 2) for k, v in fit["t"].items() if k != "alpha"}
    sig = [k for k, v in tvals.items() if abs(v) > 1.96]

    return {
        "ok": True, "n": fit["n"], "factor_names": names,
        "alpha_annual": round(float(fit["coef"]["alpha"]) * TD, 6),
        "alpha_t": round(float(fit["t"]["alpha"]), 3),
        "loadings": loadings, "t_values": tvals,
        "significant_factors": sig, "r2": round(fit["r2"], 4),
        "condition_number": round(cond, 1),
        "multicollinearity_warning": cond > 1e4,
        "verdict": (f"对 {len(sig)}/{len(names)} 个因子有显著暴露"
                    + ("；但这基本是因子组合而非独立 alpha"
                       if len(sig) >= max(1, len(names) // 2) else "")),
    }


# ══════════════════════════════════════════════════════════
# 5. 主动管理基本法则
# ══════════════════════════════════════════════════════════
def fundamental_law(ir: float | None = None, ic: float | None = None,
                    breadth: float | None = None) -> dict:
    """
    IR = IC × √宽度（Grinold）。

    * 已知 IC、宽度 → 预期 IR
    * 已知 IR、宽度 → 反推所需 IC
    * 已知 IR、IC  → 反推所需独立决策次数

    宽度必须是**独立**决策次数（每天选 100 只不等于宽度 100）。
    """
    known = sum(x is not None for x in (ir, ic, breadth))
    if known < 2:
        raise ValueError("三者中至少给两个")

    if ir is None:
        out_ir = ic * np.sqrt(max(breadth, 0))
        mode = "求 IR"
    elif ic is None:
        out_ir = ir
        ic = ir / np.sqrt(max(breadth, 0)) if breadth else float("nan")
        mode = "反推所需 IC"
    else:
        out_ir = ir
        breadth = (ir / ic) ** 2 if ic else float("nan")
        mode = "反推所需独立决策次数"

    feasible = bool(abs(ic) <= 0.10) if (ic is not None and np.isfinite(ic)) else None
    return {
        "ok": True, "mode": mode,
        "ir": None if not np.isfinite(out_ir) else round(float(out_ir), 4),
        "ic": None if ic is None or not np.isfinite(ic) else round(float(ic), 5),
        "breadth": None if breadth is None or not np.isfinite(breadth)
                   else round(float(breadth), 2),
        "ic_feasible": feasible,
        "verdict": (
            "所需 IC 超出合理范围（实测 IC 很少超过 0.05），该目标不现实"
            if feasible is False else
            "参数在合理范围内" if feasible else "无法判断"),
        "note": "宽度按独立决策计；相关性高的决策要打折",
    }


# ══════════════════════════════════════════════════════════
def format_report(res: dict) -> str:
    L = ["=" * 66, f"组合理论检验：{res.get('title', '')}", "=" * 66]
    for k, v in res.items():
        if k in ("title", "ok"):
            continue
        if isinstance(v, dict):
            L.append(f"  {k}:")
            for kk, vv in v.items():
                L.append(f"      {kk:<18} {vv}")
        elif isinstance(v, list):
            L.append(f"  {k}: {len(v)} 项")
        else:
            L.append(f"  {k:<18} {v}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    idx = pd.bdate_range("2020-01-02", periods=800)
    rng = np.random.default_rng(7)

    # EMH：随机收益应不可预测
    rand = pd.Series(rng.normal(0.0004, 0.011, 800), index=idx)
    e = emh_test(rand)
    print(f"EMH: LB p={e['ljung_box_p']}  VR5={e['variance_ratio'][5]['vr']}")
    print(f"     {e['verdict']}")

    # 有自相关 → 应判无效
    ar = np.zeros(800)
    for i in range(1, 800):
        ar[i] = 0.3 * ar[i - 1] + rng.normal(0, 0.01)
    e2 = emh_test(pd.Series(ar, index=idx))
    assert not e2["efficient"], e2
    print(f"AR(1) 序列: LB p={e2['ljung_box_p']} → 判无效 ✓")

    # MPT
    R = pd.DataFrame({f"A{i}": rng.normal(0.0005 * (i + 1), 0.01 + 0.002 * i, 800)
                      for i in range(4)}, index=idx)
    ef = efficient_frontier(R)
    print(f"\nMPT: 最小方差 vol={ef['min_variance']['vol']:.4f} "
          f"切点 Sharpe={ef['max_sharpe']['sharpe']:.3f} "
          f"前沿 {len(ef['frontier'])} 点")

    # CAPM
    bench = pd.Series(rng.normal(0.0004, 0.01, 800), index=idx)
    port = 1.2 * bench + rng.normal(0.0002, 0.004, 800)
    c = capm_regression(pd.Series(np.cumprod(1 + port), index=idx),
                        pd.Series(np.cumprod(1 + bench), index=idx))
    print(f"CAPM: beta={c['beta']} alpha={c['alpha_annual']*100:.2f}%/年 "
          f"t={c['alpha_t']} R²={c['r2']}")

    # APT
    F = pd.DataFrame({"mkt": bench.values,
                      "size": rng.normal(0, 0.008, 800),
                      "value": rng.normal(0, 0.007, 800)}, index=idx)
    a = apt_regression(pd.Series(np.cumprod(1 + port), index=idx), F)
    print(f"APT: 显著因子 {a['significant_factors']} R²={a['r2']} "
          f"条件数={a['condition_number']}")

    # 主动管理法则
    fl = fundamental_law(ic=0.03, breadth=252)
    print(f"\n法则是: IC=0.03 宽度=252 → IR={fl['ir']}")
    fl2 = fundamental_law(ir=1.0, breadth=252)
    print(f"        想 IR=1.0 宽度=252 → 需要 IC={fl2['ic']} ({fl2['verdict']})")

    print(f"\nOK v{__version__}")
