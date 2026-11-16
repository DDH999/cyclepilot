# -*- coding: utf-8 -*-
"""
滚动样本外检验（Purged Walk-Forward，M11）
====================================================
一次 IS/OOS 切分只能证明"那一段时间"的表现。
真正该问的是：**换一段样本外，还成立吗？**

本模块把收益序列切成多段，逐段独立评估，并做两件事：

1. **Purge（清洗）**：在训练段与测试段之间留出隔离带，
   避免重叠期信息泄漏（对收益序列而言主要是边界污染）
2. **Embargo（禁运）**：测试段开始后再空出若干天不参与任何统计

判读
----
* 各段 OOS 夏普全为正 → 稳定
* 一半为正一半为负 → 运气，不成立
* 分段与整体结论严重背离 → 结论依赖特定时期

注意：对纯收益序列，walk-forward 的语义是**分段稳定性检验**，
不涉及模型重训（因为没有模型）。这一点必须如实说明，不能包装成"滚动训练"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["WFCfg", "DEFAULTS", "sharpe_of", "purged_walk_forward",
           "stability_from_folds", "format_report"]


class WFCfg:
    def __init__(self, n_folds: int = 5, embargo: int = 5, purge: int = 5,
                 min_fold_days: int = 40):
        self.n_folds = n_folds
        self.embargo = embargo
        self.purge = purge
        self.min_fold_days = min_fold_days


DEFAULTS = WFCfg()


def _to_returns(data) -> pd.Series:
    s = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    mean_abs = abs(float(s.mean())) if len(s) else 0.0
    cv = (float(s.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    pos_ratio = float((s > 0).mean()) if len(s) else 0.0
    if len(s) > 20 and pos_ratio > 0.95 and cv < 0.5:  # 净值 → 收益
        s = s.pct_change(fill_method=None).dropna()
    return s


def sharpe_of(returns, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    """年化夏普（超额、总体标准差）。零方差返回 0。"""
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    rfd = (1 + rf_annual) ** (1 / trading_days) - 1
    ex = r.to_numpy() - rfd
    sd = float(np.std(ex, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(np.mean(ex) / sd * np.sqrt(trading_days))


def purged_walk_forward(data, cfg: WFCfg = DEFAULTS,
                        rf_annual: float = 0.0) -> dict:
    """
    把序列切成 n_folds 段，逐段评估（段内先 purge 再 embargo）。

    返回
    ----
    ok          是否可评估
    folds       [{fold, start, end, days, sharpe, annual_return, max_drawdown}]
    oos_sharpe  各段夏普均值
    positive    夏普为正的段数
    hit_rate    positive / n_folds
    std         段间夏普标准差
    verdict     判读结论
    """
    r = _to_returns(data)
    n = len(r)
    if n < cfg.min_fold_days * 2:
        return {"ok": False, "reason": f"样本仅 {n} 天，至少需要 {cfg.min_fold_days * 2} 天"}

    k = max(2, int(cfg.n_folds))
    bounds = np.linspace(0, n, k + 1).astype(int)

    folds = []
    for i in range(k):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        # purge：段首空出若干天，避免与前一段尾巴重叠
        t_lo = lo + cfg.purge if i > 0 else lo
        # embargo：段尾留出隔离带
        t_hi = hi - cfg.embargo if i < k - 1 else hi
        if t_hi - t_lo < cfg.min_fold_days:
            t_lo, t_hi = lo, hi
        seg = r.iloc[t_lo:t_hi]
        if len(seg) < 10:
            continue
        nav = np.cumprod(1 + seg.to_numpy())
        peak = np.maximum.accumulate(nav)
        mdd = float(np.nanmin(nav / np.where(peak <= 0, np.nan, peak) - 1))
        folds.append({
            "fold": i + 1,
            "start": str(seg.index[0].date()) if hasattr(seg.index[0], "date") else str(seg.index[0]),
            "end": str(seg.index[-1].date()) if hasattr(seg.index[-1], "date") else str(seg.index[-1]),
            "days": int(len(seg)),
            "sharpe": round(sharpe_of(seg, rf_annual), 4),
            "annual_return": round(float(np.mean(seg) * 252), 6),
            "max_drawdown": round(mdd, 6),
        })

    if len(folds) < 2:
        return {"ok": False, "reason": "有效分段不足 2 段"}

    sh = np.array([f["sharpe"] for f in folds], dtype="float64")
    positive = int((sh > 0).sum())
    res = {
        "ok": True, "n": int(n), "n_folds": len(folds), "folds": folds,
        "oos_sharpe": round(float(np.mean(sh)), 4),
        "positive": positive,
        "hit_rate": round(positive / len(sh), 4),
        "std": round(float(np.std(sh, ddof=0)), 4),
        "best": round(float(sh.max()), 4),
        "worst": round(float(sh.min()), 4),
        "purge": cfg.purge, "embargo": cfg.embargo,
    }
    res["verdict"], res["stable"] = stability_from_folds(res)
    return res


def stability_from_folds(res: dict) -> tuple[str, bool]:
    """根据分段结果给出判读。"""
    hit = res.get("hit_rate", 0.0)
    mean = res.get("oos_sharpe", 0.0)
    spread = res.get("std", 0.0)

    if hit >= 0.8 and mean > 0.3 and spread < 1.0:
        return "稳定：多数样本外分段为正，且段间波动不大", True
    if hit >= 0.6 and mean > 0:
        return "基本可用：多数分段为正，但稳定性一般", True
    if hit <= 0.2:
        return "不成立：样本外几乎全部为负，结论不可信", False
    if mean <= 0:
        return "不成立：样本外均值为负", False
    return "存疑：分段正负参半，更像运气而非规律", False


def format_report(res: dict) -> str:
    L = ["=" * 66, "滚动样本外检验（Purged Walk-Forward）", "=" * 66]
    if not res.get("ok"):
        L.append(f"  无法评估：{res.get('reason', '')}")
        L.append("=" * 66)
        return "\n".join(L)

    L.append(f"  样本 {res['n']} 天，切 {res['n_folds']} 段"
             f"（purge={res['purge']} 天，embargo={res['embargo']} 天）")
    L.append(f"  {'段':>3}  {'区间':<24}{'天数':>6}{'夏普':>9}{'年化':>10}{'回撤':>9}")
    L.append("  " + "-" * 62)
    for f in res["folds"]:
        L.append(f"  {f['fold']:>3}  {f['start']} ~ {f['end']:<10}"
                 f"{f['days']:>6}{f['sharpe']:>9.2f}"
                 f"{f['annual_return']*100:>9.2f}%{f['max_drawdown']*100:>8.2f}%")
    L.append("  " + "-" * 62)
    L.append(f"  分段夏普均值 {res['oos_sharpe']:.2f}   "
             f"正收益段 {res['positive']}/{res['n_folds']}"
             f"（命中率 {res['hit_rate']*100:.0f}%）   段间标准差 {res['std']:.2f}")
    L.append(f"  最好 {res['best']:.2f} / 最差 {res['worst']:.2f}")
    L.append(f"  结论: {res['verdict']}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    idx = pd.bdate_range("2020-01-02", periods=1000)
    rng = np.random.default_rng(7)

    stable = pd.Series(np.cumprod(1 + rng.normal(0.0008, 0.010, 1000)), index=idx)
    r1 = purged_walk_forward(stable)
    assert r1["ok"], r1
    print(format_report(r1))

    # 前 600 天有效、后 400 天归零 → 应被判不稳定
    mixed = rng.normal(0.0012, 0.010, 1000)
    mixed[600:] = rng.normal(-0.0012, 0.010, 400)
    s2 = pd.Series(np.cumprod(1 + mixed), index=idx)
    r2 = purged_walk_forward(s2)
    print(format_report(r2))
    assert r2["hit_rate"] < 1.0
    print(f"OK v{__version__}")
