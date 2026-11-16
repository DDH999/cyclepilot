# -*- coding: utf-8 -*-
"""
马科维茨与凯利：能算，但要用得对（M38）
====================================================
这两个是教科书经典，但**直接照做会亏钱**。本模块的职责是：
算出来、测出它的毛病、给出**能实际执行**的结论。

实测（四个海外指数，共同区间 40 年）
------------------------------------
**马科维茨（最大夏普）**

全样本权重：标普 16.5% / 纳指 24.6% / 日经 20.8% / 富时 38.1%

滚动 2 年窗口的权重摆幅：

| 指数 | 最高-最低摆幅 |
|---|---|
| 纳指100 | **72.0 个百分点**（0% ↔ 72%） |
| 富时100 | 50.8 |
| 标普500 | 39.9 |
| 日经225 | 17.4 |
| **平均** | **45.0** |

→ **每季度重算，权重能翻一倍。这不是可执行的配置。**

**凯利（连续近似 f* = μ/σ²）**

| 指数 | 年化 | 波动 | 满凯利 | 1/4 凯利 |
|---|---|---|---|---|
| 标普500 | 16.2% | 15.1% | **707%** | 177% |
| 纳指100 | 21.3% | 20.5% | **509%** | 127% |
| 日经225 | 20.6% | 24.8% | 335% | 84% |
| 富时100 | 9.7% | 11.4% | **747%** | 187% |

→ **凯利在说"上 3~7 倍杠杆"。对无杠杆定投的人，这等于没用。**

结论（本模块会打印）
--------------------
1. 马科维茨：输入是估计出来的期望收益，误差会放大成极端权重 → **不建议直接用**
2. 凯利：隐含杠杆 → **只能当上限，不能当目标**
3. 实测四种配置的夏普：等权 1.36 / 逆波动 1.36 / 风险平价 1.37 / 最小方差 1.24
   → **等权几乎不输任何优化方案**，但简单得多

⚠ 这不是说两个理论错，而是**它们的输入（期望收益）无法可靠估计**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["STABILITY_LIMIT", "to_returns", "markowitz_weights",
           "markowitz_stability", "kelly_sizing", "optimize_report",
           "format_report"]

TD = 252
# 平均摆幅超过这个值，就判定「不可执行」
STABILITY_LIMIT = 20.0     # 百分点


def to_returns(navs: dict[str, pd.Series] | pd.DataFrame) -> pd.DataFrame:
    """净值字典/价格 → 日收益对齐矩阵。"""
    if isinstance(navs, pd.DataFrame):
        df = navs.replace([np.inf, -np.inf], np.nan)
        # 若像净值（全正且平滑）则转收益
        if df.shape[0] > 20 and (df.min().min() > 0):
            return df.pct_change(fill_method=None).dropna()
        return df.dropna()
    cols = {}
    for k, s in navs.items():
        ser = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
        cols[k] = ser.pct_change(fill_method=None)
    return pd.DataFrame(cols).dropna()


def markowitz_weights(R: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """最大夏普（切点）组合，无做空。"""
    from scipy.optimize import minimize
    mu = R.mean().to_numpy() * TD
    cov = R.cov().to_numpy() * TD
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n

    def neg_sharpe(w):
        v = float(np.sqrt(max(w @ cov @ w, 0.0)))
        return -(float(w @ mu) - rf_annual) / v if v > 0 else 1e9

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                   constraints=cons)
    w = res.x if res.success else x0
    return {c: round(float(v), 4) for c, v in zip(R.columns, w)}


def markowitz_stability(R: pd.DataFrame, window: int = 504,
                        step: int = 63) -> dict:
    """
    滚动窗口重算最优权重，量化「摆幅」。

    window 大约 2 年，step 约一个季度。
    """
    if len(R) < window + step:
        return {"ok": False,
                "reason": f"样本 {len(R)} 天不足以做 {window} 天滚动"}
    rows = []
    for end in range(window, len(R) + 1, step):
        sub = R.iloc[end - window:end]
        try:
            w = markowitz_weights(sub)
            w["_date"] = R.index[end - 1]
            rows.append(w)
        except Exception:
            continue
    if len(rows) < 3:
        return {"ok": False, "reason": "有效窗口不足 3 个"}

    dw = pd.DataFrame(rows).set_index("_date")
    swing = (dw.max() - dw.min())
    std = dw.std(ddof=0)
    avg_swing = float(swing.mean())
    # 注意：swing 是小数（0.45 = 45 个百分点），STABILITY_LIMIT 是百分点。
    # 直接比较会把 77pp 的摆幅判成"稳定"（踩过）。
    avg_swing_pp = avg_swing * 100.0
    unstable = avg_swing_pp > STABILITY_LIMIT

    worst = swing.idxmax()
    return {
        "ok": True, "n_windows": len(rows), "window": window,
        "weights": {c: [round(float(x), 4) for x in dw[c]]
                    for c in dw.columns},
        "dates": [str(d.date()) for d in dw.index],
        "latest": {c: round(float(dw[c].iloc[-1]), 4) for c in dw.columns},
        "swing": {c: round(float(v), 4) for c, v in swing.items()},
        "std": {c: round(float(v), 4) for c, v in std.items()},
        "avg_swing": round(avg_swing, 4),
        "avg_swing_pp": round(avg_swing_pp, 1),
        "worst_asset": worst,
        "unstable": bool(unstable),
        "limit": STABILITY_LIMIT,
        "verdict": (
            f"不稳定：权重平均摆幅 {avg_swing_pp:.0f} 个百分点"
            f"（阈值 {STABILITY_LIMIT:.0f}pp），最不稳的是 {worst}"
            if unstable else
            f"相对稳定：平均摆幅 {avg_swing_pp:.0f} 个百分点"
            f"（阈值 {STABILITY_LIMIT:.0f}pp）"),
    }


def kelly_sizing(R: pd.DataFrame, rf_annual: float = 0.0,
                 max_leverage: float = 1.0) -> dict:
    """
    连续凯利 f* = (μ - rf) / σ²，逐资产 + 组合。

    `max_leverage=1.0` 表示「不接受杠杆」，此时凯利只能当**上限**用。
    """
    out = {"ok": True, "assets": {}, "rf_annual": rf_annual}
    for c in R.columns:
        s = R[c].dropna()
        mu = float(s.mean()) * TD - rf_annual
        var = float(s.var(ddof=0)) * TD
        f = mu / var if var > 0 else float("nan")
        out["assets"][c] = {
            "annual_return": round(mu + rf_annual, 6),
            "annual_vol": round(float(np.sqrt(var)), 6),
            "kelly_full": round(float(f), 4) if np.isfinite(f) else None,
            "kelly_quarter": round(float(f * 0.25), 4) if np.isfinite(f) else None,
        }

    # 组合层面：等权组合的凯利
    port = R.mean(axis=1)
    mu_p = float(port.mean()) * TD - rf_annual
    var_p = float(port.var(ddof=0)) * TD
    f_p = mu_p / var_p if var_p > 0 else float("nan")

    over = [c for c, v in out["assets"].items()
            if v["kelly_full"] is not None and v["kelly_full"] > max_leverage]
    out.update({
        "portfolio_kelly_full": round(float(f_p), 4) if np.isfinite(f_p) else None,
        "portfolio_kelly_quarter": (round(float(f_p * 0.25), 4)
                                    if np.isfinite(f_p) else None),
        "max_leverage": max_leverage,
        "assets_implying_leverage": over,
        "verdict": (
            f"{len(over)}/{len(R.columns)} 个资产的满凯利超过 "
            f"{max_leverage*100:.0f}%（隐含杠杆）→ 凯利只能当上限，不能当目标"
            if over else
            "各资产满凯利均在 100% 以内，可按 1/4 凯利配置"),
    })
    return out


def optimize_report(R: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """
    一次给出三个结论：马科维茨稳不稳、凯利暗示什么、实际该选哪个方案。
    """
    import allocation as AL

    ms = markowitz_stability(R)
    ks = kelly_sizing(R, rf_annual=rf_annual)
    if not ms.get("ok"):
        return {"ok": False, "reason": ms.get("reason")}

    # 四种方案实测对比
    cmp_ = AL.compare_allocation(R)
    rows = cmp_["schemes"]
    best = max(rows, key=lambda s: (s["sharpe"] or -9))
    eq = next((s for s in rows if s["scheme"] == "等权"), None)
    rp = next((s for s in rows if s["scheme"] == "风险平价"), None)

    # 判断：等权是否「几乎不输」
    eq_gap = (best["sharpe"] - eq["sharpe"]
              if (eq and best["sharpe"] is not None
                  and eq["sharpe"] is not None) else 0.0)
    rep_gap = eq_gap
    eq_close = eq_gap <= 0.05

    if ms["unstable"]:
        rec, why = "等权", (f"马科维茨权重摆幅 {ms['avg_swing_pp']:.0f} 个"
                            f"百分点，不可执行；等权最稳且几乎不输最优")
    elif eq_close:
        rec, why = "等权", ("优化带来的夏普提升 ≤0.05，不值得多一层复杂度")
    else:
        # 不稳定已排除，这里就取实测夏普最高的那个
        rec = best["scheme"]
        why = (f"实测夏普最高（{best['sharpe']:.2f}），"
               f"比等权高 {rep_gap:.2f}")

    kelly_cap = None
    if ks.get("portfolio_kelly_full") is not None:
        kelly_cap = min(1.0, max(0.0, float(ks["portfolio_kelly_full"])))

    return {"ok": True, "n_days": int(len(R)),
            "markowitz": ms, "kelly": ks,
            "schemes": rows, "best_scheme": best["scheme"],
            "equal_weight_gap": (round(best["sharpe"] - eq["sharpe"], 4)
                                 if eq and best["sharpe"] and eq["sharpe"]
                                 else None),
            "recommend": rec, "reason": why,
            "kelly_position_cap": (round(kelly_cap, 4)
                                   if kelly_cap is not None else None)}


def format_report(rep: dict) -> str:
    L = ["=" * 70, "马科维茨 / 凯利：能用吗", "=" * 70]
    if not rep.get("ok"):
        L.append(f"  无法评估：{rep.get('reason')}")
        L.append("=" * 70)
        return "\n".join(L)

    ms = rep["markowitz"]
    L.append(f"  样本 {rep['n_days']} 天\n")
    L.append("  【马科维茨（最大夏普）】")
    L.append(f"    滚动 {ms['window']} 天、{ms['n_windows']} 个窗口重算")
    L.append("    " + f"{'指数':<10}{'最新权重':>10}{'摆幅':>10}")
    L.append("    " + "-" * 32)
    for c in ms["swing"]:
        L.append("    " + f"{c:<10}{ms['latest'][c]*100:>9.1f}%"
                 f"{ms['swing'][c]*100:>9.1f}pp")
    L.append(f"    平均摆幅 {ms['avg_swing']*100:.1f}pp"
             f"（阈值 {ms['limit']:.0f}pp）")
    L.append(f"    → {ms['verdict']}")

    ks = rep["kelly"]
    L.append("\n  【凯利（连续近似 f* = μ/σ²）】")
    L.append("    " + f"{'指数':<10}{'年化':>9}{'波动':>9}{'满凯利':>10}{'1/4凯利':>10}")
    L.append("    " + "-" * 50)
    for c, v in ks["assets"].items():
        full = v["kelly_full"]
        q = v["kelly_quarter"]
        L.append("    " + f"{c:<10}{v['annual_return']*100:>8.2f}%"
                 f"{v['annual_vol']*100:>8.2f}%"
                 f"{(full*100 if full is not None else float('nan')):>9.0f}%"
                 f"{(q*100 if q is not None else float('nan')):>9.0f}%")
    L.append(f"    组合满凯利 "
             f"{(ks['portfolio_kelly_full'] or 0)*100:.0f}%")
    L.append(f"    → {ks['verdict']}")

    L.append("\n  【四种配置实测】")
    L.append("    " + f"{'方案':<10}{'年化':>9}{'波动':>9}{'夏普':>8}{'回撤':>9}")
    L.append("    " + "-" * 45)
    for s in rep["schemes"]:
        L.append("    " + f"{s['scheme']:<10}{s['annual_return']*100:>8.2f}%"
                 f"{s['volatility']*100:>8.2f}%"
                 f"{(s['sharpe'] if s['sharpe'] is not None else float('nan')):>8.2f}"
                 f"{s['max_drawdown']*100:>8.2f}%")

    L.append("-" * 70)
    L.append(f"  建议：用【{rep['recommend']}】")
    L.append(f"  理由：{rep['reason']}")
    if rep.get("kelly_position_cap") is not None:
        L.append(f"  凯利仓位上限：{rep['kelly_position_cap']*100:.0f}%"
                 f"（不超过此值，且不加杠杆）")
    L.append("=" * 70)
    L.append("  ⚠ 两个理论都没错，错的是它们的输入：期望收益无法可靠估计。")
    L.append("     本文的结论只适用于「长期持有宽基、不加杠杆」这种场景。")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-02", periods=1500)
    A = np.cumprod(1 + rng.normal(0.0005, 0.012, 1500))
    B = np.cumprod(1 + rng.normal(0.0004, 0.016, 1500))
    C = np.cumprod(1 + rng.normal(0.0003, 0.009, 1500))
    navs = {"A": pd.Series(A, index=idx), "B": pd.Series(B, index=idx),
            "C": pd.Series(C, index=idx)}

    R = to_returns(navs)
    print(f"收益矩阵 {R.shape}")
    # 净值 → 收益：行数少 1，且数值是小数（不是价格量级）
    assert R.shape[0] == 1499, R.shape
    assert float(R.abs().max().max()) < 0.5, float(R.abs().max().max())

    st = markowitz_stability(R)
    print(f"稳定性：{st.get('verdict')}  （{st.get('n_windows')} 窗口）")
    assert st["ok"]

    ks = kelly_sizing(R)
    print(f"凯利：{ks['verdict']}")
    assert ks["ok"]

    rep = optimize_report(R)
    print()
    print(format_report(rep))
    assert rep["ok"] and rep["recommend"] in ("等权", "风险平价", "最小方差")
    print("\nOK v" + __version__)
