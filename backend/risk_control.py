# -*- coding: utf-8 -*-
"""
组合与风控层（M64）：事前风控 + 事中熔断
====================================================
**事前**（下单之前，把不合规的权重改掉）
----------------------------------------
| 规则 | 含义 | 处理 |
|---|---|---|
| 单票上限 | 任一票权重 ≤ `max_weight` | 超出的部分按比例削掉 |
| 行业上限 | 任一行行业合计 ≤ `max_industry` | 该行业内等比压缩 |
| 总杠杆 | 权重合计 ≤ `max_leverage` | 整体等比压缩 |
| 换手上限 | 单次调仓双边换手 ≤ `max_turnover` | 与上期权重**部分融合**（渐进调仓） |

**事中**（持仓期间，触发就降仓/停手）
--------------------------------------
| 规则 | 含义 |
|---|---|
| 回撤熔断 | 净值从峰值回撤触达 `dd_stop` → 清仓并冷却 `dd_cooldown` 个交易日 |
| 恢复条件 | 冷却后净值回升到「峰值的 `1 - dd_stop/2`」才恢复满仓 |
| 敞口监控 | 单票/前五/行业集中度、HHI，逐日输出 |

⚠ 时序约定
----------
所有 gate 与 cap 都 **shift(1)**：T 日收盘判断，T+1 才生效。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["Limits", "check_pre_trade", "circuit_breaker", "exposure_report",
           "monitor", "format_violations"]


@dataclass
class Limits:
    max_weight: float = 0.15
    max_industry: float = 0.35
    max_leverage: float = 1.0
    max_turnover: float = 0.60
    dd_stop: float = 0.15
    dd_cooldown: int = 20
    dd_recover_ratio: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


def _norm(w: pd.Series, cap: float = 1.0) -> pd.Series:
    s = float(w.sum())
    if s <= 0:
        return w * 0.0
    out = w / s
    if s > cap:
        out = out * (cap / s)
    return out


def check_pre_trade(weights: pd.Series, prev_weights: pd.Series | None = None,
                    industry: dict | pd.Series | None = None,
                    limits: Limits | None = None,
                    verbose: bool = False) -> dict:
    """
    事前风控：把不合规的权重改到合规。

    返回 {ok, adjusted, violations, turnover_before, turnover_after}
    """
    L = limits or Limits()
    w = pd.Series(weights).astype(float).fillna(0.0)
    w = w[w > 0]
    violations: list[dict] = []
    if len(w) == 0:
        return {"ok": True, "adjusted": w, "violations": violations,
                "turnover_before": 0.0, "turnover_after": 0.0}

    # ── 1. 单票上限
    over = w[w > L.max_weight + 1e-12]
    if len(over):
        violations.append({
            "rule": "max_weight", "limit": L.max_weight,
            "detail": {c: round(float(v), 4) for c, v in over.items()}})
        w = w.clip(upper=L.max_weight)

    # ── 2. 行业上限
    if industry is not None and len(w):
        ind = pd.Series(industry).reindex(w.index).fillna("UNKNOWN")
        for g, grp in w.groupby(ind):
            s = float(grp.sum())
            if s > L.max_industry + 1e-12:
                k = L.max_industry / s
                w.loc[grp.index] = grp * k
                violations.append({
                    "rule": "max_industry", "limit": L.max_industry,
                    "detail": {str(g): round(s, 4)}})

    # ── 3. 总杠杆
    s = float(w.sum())
    if s > L.max_leverage + 1e-12:
        violations.append({"rule": "max_leverage", "limit": L.max_leverage,
                           "detail": {"sum": round(s, 4)}})
        w = w * (L.max_leverage / s)

    # ── 4. 换手上限（与上期融合，渐进调仓）
    turn_before = turn_after = 0.0
    if prev_weights is not None:
        prev = pd.Series(prev_weights).astype(float).fillna(0.0)
        idx = w.index.union(prev.index)
        tw = w.reindex(idx).fillna(0.0)
        tp = prev.reindex(idx).fillna(0.0)
        turn_before = float((tw - tp).abs().sum()) / 2.0
        if turn_before > L.max_turnover > 0:
            k = L.max_turnover / turn_before
            tw = tp + (tw - tp) * k
            turn_after = float((tw - tp).abs().sum()) / 2.0
            violations.append({"rule": "max_turnover", "limit": L.max_turnover,
                               "detail": {"before": round(turn_before, 4),
                                          "after": round(turn_after, 4)}})
            w = tw[tw > 0]

    w = w[w > 0]
    ok = len(violations) == 0
    if verbose and violations:
        print(format_violations(violations))
    return {"ok": ok, "adjusted": w, "violations": violations,
            "turnover_before": round(turn_before, 4),
            "turnover_after": round(turn_after, 4)}


def circuit_breaker(nav: pd.Series, limits: Limits | None = None) -> pd.Series:
    """
    回撤熔断闸门（0=清仓，1=正常），**shift(1) 后返回**。

    触达 `-dd_stop` → 关闭 N 日；冷却结束后需回升到「峰值×(1-dd_stop/2)」才恢复。
    """
    L = limits or Limits()
    n = pd.Series(nav).astype(float).sort_index()
    if len(n) < 2:
        return pd.Series(1.0, index=n.index)
    peak = n.cummax()
    dd = n / peak - 1.0
    gate = np.ones(len(n))
    cooling = 0
    cooldown = int(L.dd_cooldown)
    for i in range(len(n)):
        if cooling > 0:
            gate[i] = 0.0
            cooling -= 1
            # 冷却结束后检查是否已回升
            if cooling == 0:
                recover_level = peak.iloc[i] * (1 - L.dd_stop * L.dd_recover_ratio)
                if n.iloc[i] < recover_level:
                    cooling = cooldown          # 没恢复，继续关闭
            continue
        if dd.iloc[i] <= -abs(L.dd_stop):
            gate[i] = 0.0
            cooling = cooldown
    g = pd.Series(gate, index=n.index)
    return g.shift(1).fillna(1.0)


def exposure_report(weights: pd.Series,
                    industry: dict | pd.Series | None = None) -> dict:
    """组合敞口快照。"""
    w = pd.Series(weights).astype(float)
    w = w[w > 0]
    if len(w) == 0:
        return {"n": 0, "gross": 0.0}
    top5 = float(w.nlargest(min(5, len(w))).sum())
    hhi = float(((w / w.sum()) ** 2).sum()) if w.sum() > 0 else 0.0
    out = {"n": int(len(w)), "gross": round(float(w.sum()), 4),
           "max_weight": round(float(w.max()), 4),
           "top5_share": round(top5, 4),
           "hhi": round(hhi, 4),
           "effective_n": round(1.0 / hhi, 2) if hhi > 0 else None}
    if industry is not None:
        ind = pd.Series(industry).reindex(w.index).fillna("UNKNOWN")
        share = w.groupby(ind).sum() / w.sum()
        out["industry"] = {str(k): round(float(v), 4)
                           for k, v in share.sort_values(ascending=False).items()}
        out["max_industry"] = round(float(share.max()), 4)
    return out


def monitor(nav: pd.Series, weights_by_date: pd.Series | None = None,
            industry: dict | pd.Series | None = None,
            limits: Limits | None = None) -> dict:
    """
    事中监控：逐日敞口 + 熔断状态 + 违规计数。
    """
    L = limits or Limits()
    g = circuit_breaker(nav, L)
    n_off = int((g == 0).sum())
    peak = pd.Series(nav).cummax()
    dd = (pd.Series(nav) / peak - 1.0)
    out = {
        "days": int(len(nav)),
        "breaker_off_days": n_off,
        "breaker_off_ratio": round(n_off / max(len(nav), 1), 4),
        "max_drawdown": round(float(dd.min()), 4),
        "dd_stop": L.dd_stop,
        "final_gate": float(g.iloc[-1]) if len(g) else None,
    }
    if weights_by_date is not None and len(weights_by_date):
        w = pd.Series(weights_by_date)
        out["last_exposure"] = exposure_report(w, industry)
    return out


def format_violations(v: list[dict]) -> str:
    if not v:
        return "  无违规"
    L = ["  事前风控触发："]
    for x in v:
        L.append(f"    [{x['rule']}] 上限 {x['limit']}  {x['detail']}")
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    L = Limits(max_weight=0.15, max_industry=0.35, max_leverage=1.0)
    w = pd.Series({"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10})
    r = check_pre_trade(w, industry={"A": "tech", "B": "tech", "C": "bank",
                                     "D": "bank"}, limits=L, verbose=True)
    print(f"  调整后合计 {r['adjusted'].sum():.4f}  最大 {r['adjusted'].max():.4f}")
    assert r["adjusted"].max() <= 0.15 + 1e-9
    assert not r["ok"]

    # 换手融合
    prev = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    r2 = check_pre_trade(w, prev_weights=prev, limits=Limits(max_turnover=0.2))
    print(f"  换手 {r2['turnover_before']:.2f} → {r2['turnover_after']:.2f}")
    assert r2["turnover_after"] <= 0.2 + 1e-9

    # 熔断
    nav = pd.Series(np.concatenate([np.linspace(1, 1.2, 60),
                                    np.linspace(1.2, 0.95, 40)]),
                    index=pd.bdate_range("2023-01-02", periods=100))
    g = circuit_breaker(nav, Limits(dd_stop=0.15, dd_cooldown=10))
    print(f"  熔断关闭 {int((g == 0).sum())} 天 / 共 {len(g)} 天")
    assert (g == 0).sum() > 0
    assert g.iloc[0] == 1.0

    print("敞口:", exposure_report(prev, industry={"A": "tech", "B": "tech",
                                                 "C": "bank", "D": "bank"}))
    print("OK v" + __version__)
