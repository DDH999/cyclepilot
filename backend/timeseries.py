# -*- coding: utf-8 -*-
"""
时间序列分析（M16）
====================================================
六个能直接用于判断"这串收益是不是有结构"的工具：

1. **自相关 ACF / 偏自相关 PACF**：有没有线性记忆
2. **AR(1) 系数**：昨天涨，今天还涨吗
3. **单位根检验（近似 ADF）**：序列是均值回归还是随机游走
4. **半衰期**：偏离均值后多久回归一半
5. **Hurst 指数**：>0.5 趋势性，<0.5 均值回归，≈0.5 随机
6. **方差比**：多周期波动是否按 √t 放大

⚠️ 诚实说明：本机没有 statsmodels，ADF 用「AR(1) 的 t 统计量 + 近似临界值」实现，
结论方向上可用，**精确 p 值请用专业库复核**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["acf", "pacf", "ar1_coef", "adf_proxy", "half_life", "hurst",
           "variance_ratio", "ts_report", "format_report"]

# AR(1) 单位根检验的近似临界值（含截距、大样本）
ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def _arr(x) -> np.ndarray:
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    mean_abs = abs(float(s.mean())) if len(s) else 0.0
    cv = (float(s.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    pos = float((s > 0).mean()) if len(s) else 0.0
    if len(s) > 20 and pos > 0.95 and cv < 0.5:
        s = s.pct_change(fill_method=None).dropna()
    return s.to_numpy()


def acf(x, nlags: int = 10) -> list[float]:
    a = _arr(x)
    n = len(a)
    if n < nlags + 5:
        return [float("nan")] * nlags
    m = a.mean()
    denom = float(np.sum((a - m) ** 2))
    if denom <= 0:
        return [float("nan")] * nlags
    return [float(np.sum((a[k:] - m) * (a[:-k] - m)) / denom)
            for k in range(1, nlags + 1)]


def pacf(x, nlags: int = 10) -> list[float]:
    """偏自相关（Durbin-Levinson 递推）。"""
    r = acf(x, nlags)
    if any(not np.isfinite(v) for v in r):
        return [float("nan")] * nlags
    phi = [[0.0] * nlags for _ in range(nlags)]
    out: list[float] = []
    for k in range(nlags):
        if k == 0:
            phi[0][0] = r[0]
        else:
            num = r[k] - sum(phi[k - 1][j] * r[k - 1 - j] for j in range(k))
            den = 1.0 - sum(phi[k - 1][j] * r[j] for j in range(k))
            phi[k][k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(k):
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - 1 - j]
        out.append(float(phi[k][k]))
    return out


def ar1_coef(x) -> dict:
    """AR(1)：x_t = c + φ·x_{t-1} + e，返回 φ、t 值、R²。"""
    a = _arr(x)
    if len(a) < 30:
        return {"ok": False, "reason": "样本不足"}
    y, X = a[1:], np.column_stack([np.ones(len(a) - 1), a[:-1]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - 2, 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    t = beta[1] / se if se and np.isfinite(se) and se > 0 else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {"ok": True, "phi": round(float(beta[1]), 4),
            "const": round(float(beta[0]), 8),
            "t": round(float(t), 3) if np.isfinite(t) else None,
            "r2": round(float(r2), 4),
            "significant": bool(np.isfinite(t) and abs(t) > 1.96)}


def adf_proxy(x) -> dict:
    """
    近似 ADF：检验 φ=1（单位根）。

    t 统计量越负越倾向"平稳（均值回归）"。
    临界值为大样本近似值，非精确 p 值。
    """
    a = _arr(x)
    if len(a) < 50:
        return {"ok": False, "reason": "样本不足（至少 50）"}
    y, X = a[1:], np.column_stack([np.ones(len(a) - 1), a[:-1]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi = float(beta[1])
    resid = y - X @ beta
    dof = max(len(y) - 2, 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "矩阵奇异"}
    if not np.isfinite(se) or se <= 0:
        return {"ok": False, "reason": "标准误无效"}
    t = (phi - 1.0) / se

    if t < ADF_CRIT["1%"]:
        concl, stationary = "强平稳（1% 显著），强烈均值回归", True
    elif t < ADF_CRIT["5%"]:
        concl, stationary = "平稳（5% 显著），存在均值回归", True
    elif t < ADF_CRIT["10%"]:
        concl, stationary = "弱平稳（10% 显著）", True
    else:
        concl, stationary = "无法拒绝单位根，接近随机游走", False

    return {"ok": True, "phi": round(phi, 4), "t": round(float(t), 3),
            "crit": ADF_CRIT, "stationary": stationary, "verdict": concl,
            "note": "近似实现（AR(1) t 统计量），精确检验请用 statsmodels"}


def half_life(x) -> dict:
    """
    均值回归半衰期：偏离均值后回归一半所需天数。

    由 AR(1) 得：half_life = -ln(2)/ln(1+φ)（φ 为自回归系数，通常为负）
    """
    r = ar1_coef(x)
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason")}
    phi = r["phi"]
    if phi >= 0:
        return {"ok": True, "phi": phi, "half_life": None,
                "verdict": "无均值回归特征（φ≥0），半衰期不适用"}
    hl = -np.log(2) / np.log(1 + phi) if (1 + phi) > 0 else float("nan")
    return {"ok": True, "phi": phi,
            "half_life": round(float(hl), 2) if np.isfinite(hl) else None,
            "verdict": f"约 {hl:.1f} 天回归一半" if np.isfinite(hl) else "无法计算"}


def hurst(x, max_lag: int = 40) -> dict:
    """
    Hurst 指数（**方差比标度法** / aggregated variance）。

    原理：VR(q) ≈ q^(2H-1)，因此

        H = 0.5 × (log VR(q) / log q + 1)

    * H≈0.5 随机游走；H>0.55 趋势；H<0.45 均值回归

    为什么不用经典 R/S：在 800 点、最小尺度 5 的情况下，
    R/S 的小尺度偏差会把均值回归序列误判成趋势（实测踩过）。
    方差比法尺度和 VR 检验自洽，稳定性更好。
    """
    a = _arr(x)
    n = len(a)
    if n < 200:
        return {"ok": False, "reason": "样本不足（至少 200）"}
    if float(np.var(a, ddof=1)) <= 0:
        return {"ok": False, "reason": "方差为 0"}

    qs = [q for q in (2, 4, 8, 16, 32) if q * 5 <= n and q <= max_lag]
    ests, weights = [], []
    for q in qs:
        v1 = float(np.var(a, ddof=1))
        agg = np.convolve(a, np.ones(q), mode="valid")
        vr = float(np.var(agg, ddof=1)) / (q * v1)
        if vr <= 0:
            continue
        h = 0.5 * (np.log(vr) / np.log(q) + 1.0)
        if np.isfinite(h):
            ests.append(h)
            weights.append(np.log(q))
    if len(ests) < 2:
        return {"ok": False, "reason": "有效尺度不足"}

    h = float(np.average(ests, weights=weights))
    if h > 0.55:
        verdict = "趋势性为主（H>0.55）：动量类策略可能有效"
    elif h < 0.45:
        verdict = "均值回归为主（H<0.45）：反转类策略可能有效"
    else:
        verdict = "接近随机游走（H≈0.5）：无稳定可预测结构"
    return {"ok": True, "hurst": round(h, 4), "n": n,
            "scales": qs[:len(ests)], "verdict": verdict,
            "method": "方差比标度（aggregated variance）",
            "note": "样本越短估计越不稳；<200 天仅供参考"}


def variance_ratio(x, q: int = 5) -> dict:
    """方差比：VR>1 趋势、VR<1 反转、VR≈1 随机。"""
    a = _arr(x)
    n = len(a)
    if n < q * 5:
        return {"ok": False, "reason": "样本不足"}
    v1 = float(np.var(a, ddof=1))
    if v1 <= 0:
        return {"ok": False, "reason": "方差为 0"}
    agg = np.convolve(a, np.ones(q), mode="valid")
    vr = float(np.var(agg, ddof=1)) / (q * v1)
    return {"ok": True, "q": q, "vr": round(vr, 4),
            "verdict": ("趋势延续" if vr > 1.15
                        else "均值回归" if vr < 0.85 else "接近随机游走")}


def ts_report(returns) -> dict:
    """一次性给出全部时间序列诊断。"""
    a = _arr(returns)
    if len(a) < 60:
        return {"ok": False, "reason": f"样本仅 {len(a)} 天，至少需要 60 天"}
    r = {
        "ok": True, "n": int(len(a)),
        "acf": [round(v, 4) for v in acf(a, 6)],
        "pacf": [round(v, 4) for v in pacf(a, 6)],
        "ar1": ar1_coef(a),
        "adf": adf_proxy(a),
        "half_life": half_life(a),
        "hurst": hurst(a),
        "variance_ratio": {q: variance_ratio(a, q) for q in (2, 5, 10)},
    }
    h = r["hurst"].get("hurst")
    adf_ok = r["adf"].get("stationary")
    if adf_ok and h is not None and h < 0.45:
        r["summary"] = "序列呈均值回归特征：反转类策略有理论基础"
    elif h is not None and h > 0.55:
        r["summary"] = "序列呈趋势特征：动量类策略有理论基础"
    else:
        r["summary"] = "序列接近随机游走：没有稳定的可预测结构"
    return r


def format_report(res: dict) -> str:
    L = ["=" * 66, "时间序列诊断", "=" * 66]
    if not res.get("ok"):
        L.append(f"  无法评估：{res.get('reason')}")
        L.append("=" * 66)
        return "\n".join(L)
    L.append(f"  样本 {res['n']} 天")
    L.append(f"  ACF(1-6) : {res['acf']}")
    L.append(f"  PACF(1-6): {res['pacf']}")
    ar = res.get("ar1") or {}
    L.append(f"  AR(1) φ={ar.get('phi')}  t={ar.get('t')}  "
             f"{'显著' if ar.get('significant') else '不显著'}")
    adf = res.get("adf") or {}
    L.append(f"  单位根   t={adf.get('t')}  {adf.get('verdict')}")
    hl = res.get("half_life") or {}
    L.append(f"  半衰期   {hl.get('half_life')} 天  {hl.get('verdict')}")
    hu = res.get("hurst") or {}
    L.append(f"  Hurst    {hu.get('hurst')}  {hu.get('verdict')}")
    L.append("-" * 66)
    vr = res.get("variance_ratio") or {}
    for q, v in vr.items():
        if v.get("ok"):
            L.append(f"  方差比 q={q:<3} VR={v['vr']:.3f}  {v['verdict']}")
    L.append("-" * 66)
    L.append(f"  结论: {res.get('summary')}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    idx = pd.bdate_range("2020-01-02", periods=800)
    rng = np.random.default_rng(7)

    rand = pd.Series(rng.normal(0, 0.01, 800), index=idx)
    r1 = ts_report(rand)
    print(format_report(r1))
    assert r1["ok"]

    # 构造均值回归
    mr = np.zeros(800)
    for i in range(1, 800):
        mr[i] = -0.4 * mr[i - 1] + rng.normal(0, 0.01)
    r2 = ts_report(pd.Series(mr, index=idx))
    print(format_report(r2))
    assert r2["hurst"]["hurst"] < 0.55

    # 构造趋势
    tr = np.zeros(800)
    for i in range(1, 800):
        tr[i] = 0.5 * tr[i - 1] + rng.normal(0, 0.01)
    r3 = ts_report(pd.Series(tr, index=idx))
    assert r3["hurst"]["hurst"] > r1["hurst"]["hurst"]
    print(f"Hurst: 随机 {r1['hurst']['hurst']} < 趋势 {r3['hurst']['hurst']} ✓")
    print(f"OK v{__version__}")
