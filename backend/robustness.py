# -*- coding: utf-8 -*-
"""
扰动测试与压力测试（M27）
====================================================
两件不同的事，很多人会混：

**扰动测试**：把输入「轻微改一下」，看结论会不会翻。
    → 测的是**结论的稳定性**（脆弱 = 稍微一动就变）
    → 改的是：加点噪声、换个起点、随机删几天、成本再高一点

**压力测试**：把市场「换成极端情形」，看策略活不活。
    → 测的是**抗打击能力**（不是稳定性）
    → 换的是：波动翻倍、尾部事件、流动性枯竭、相关性断裂

一句话：扰动测"准不准"，压力测"扛不扛"。

诚实边界
--------
* 所有情景都是**合成**的，不是预测。它们回答"如果那样会怎样"，不回答"会不会那样"。
* 随机扰动都固定种子，结果可复现。
* 扰动只改输入，不改策略逻辑——所以它测不出"策略本身写死某个参数"这类问题。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["PerturbCfg", "StressCfg", "metrics_of", "to_returns",
           "perturbation_suite", "stress_suite", "format_perturbation",
           "format_stress"]

TD = 252


# ══════════════════════════════════════════════════════════
# 基础
# ══════════════════════════════════════════════════════════
def to_returns(x) -> pd.Series:
    """净值 → 收益（用变异系数判别，和项目其他模块一致）。"""
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if len(s) < 3:
        return s
    mean_abs = abs(float(s.mean()))
    cv = (float(s.std(ddof=0)) / mean_abs) if mean_abs > 1e-12 else float("inf")
    pos = float((s > 0).mean())
    if len(s) > 20 and pos > 0.95 and cv < 0.5:
        s = s.pct_change(fill_method=None).dropna()
    return s


def metrics_of(returns) -> dict:
    """年化收益 / 波动 / 夏普 / 最大回撤 / 卡玛。"""
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 10:
        return {"ok": False, "reason": "样本不足"}
    a = r.to_numpy(dtype="float64")
    nav = np.cumprod(1 + a)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.where(peak <= 0, np.nan, peak) - 1
    sd = float(np.std(a, ddof=0))
    cagr = float(nav[-1] ** (TD / len(a)) - 1) if nav[-1] > 0 else float("nan")
    mdd = float(np.nanmin(dd))
    return {
        "ok": True,
        "cagr": cagr,
        "vol": sd * np.sqrt(TD),
        "sharpe": (float(np.mean(a)) / sd * np.sqrt(TD)) if sd > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": (cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "final": float(nav[-1]),
    }


class PerturbCfg:
    def __init__(self, noise_mult: float = 0.30, n_noise: int = 30,
                 block: int = 20, n_boot: int = 60,
                 shift_step: int = 20,
                 drop_pct: float = 0.05, n_drop: int = 30,
                 cost_extra_bps: tuple = (5, 10, 20),
                 turnover: float = 12.0,
                 seed: int = 7):
        self.noise_mult = noise_mult
        self.n_noise = n_noise
        self.block = block
        self.n_boot = n_boot
        self.shift_step = shift_step
        self.drop_pct = drop_pct
        self.n_drop = n_drop
        self.cost_extra_bps = cost_extra_bps
        self.turnover = turnover
        self.seed = seed


class StressCfg:
    def __init__(self, vol_mults: tuple = (1.5, 2.0, 3.0),
                 tail_prob: float = 0.01, tail_size: float = -0.08,
                 tail_rounds: int = 5,
                 drift_annual: float = -0.40,
                 liquidity_mult: float = 3.0,
                 survive_cagr: float = 0.0,
                 survive_mdd: float = -0.45,
                 seed: int = 11):
        self.vol_mults = vol_mults
        self.tail_prob = tail_prob
        self.tail_size = tail_size
        self.tail_rounds = tail_rounds
        self.drift_annual = drift_annual
        self.liquidity_mult = liquidity_mult
        self.survive_cagr = survive_cagr
        self.survive_mdd = survive_mdd
        self.seed = seed


# ══════════════════════════════════════════════════════════
# 扰动测试
# ══════════════════════════════════════════════════════════
def _p_noise(r: pd.Series, cfg: PerturbCfg) -> list[dict]:
    """加噪声：在原收益上叠加 N(0, mult×σ)。测"结论是否依赖精确数值"。"""
    rng = np.random.default_rng(cfg.seed)
    sd = float(r.std(ddof=0))
    out = []
    for _ in range(cfg.n_noise):
        pert = r + rng.normal(0, cfg.noise_mult * sd, len(r))
        m = metrics_of(pert)
        if m.get("ok"):
            out.append(m)
    return out


def _p_bootstrap(r: pd.Series, cfg: PerturbCfg) -> list[dict]:
    """块重抽样：保留短期自相关，打乱长期顺序。测"是否只在特定时段有效"。"""
    rng = np.random.default_rng(cfg.seed + 1)
    a = r.to_numpy()
    n = len(a)
    b = max(2, min(int(cfg.block), n // 5))
    n_blocks = int(np.ceil(n / b))
    out = []
    for _ in range(cfg.n_boot):
        starts = rng.integers(0, max(n - b, 1), n_blocks)
        idx = np.concatenate([np.arange(s, min(s + b, n)) for s in starts])[:n]
        m = metrics_of(pd.Series(a[idx]))
        if m.get("ok"):
            out.append(m)
    return out


def _p_shift(r: pd.Series, cfg: PerturbCfg) -> list[dict]:
    """起点平移：换一个开始日期再算。测"结论是否依赖某个起点"。"""
    out = []
    step = max(int(cfg.shift_step), 1)
    for k in range(0, max(len(r) // 3, 1), step):
        seg = r.iloc[k:]
        if len(seg) < 60:
            break
        m = metrics_of(seg)
        if m.get("ok"):
            out.append(m)
    return out


def _p_drop(r: pd.Series, cfg: PerturbCfg) -> list[dict]:
    """随机删掉一部分交易日。测"是否靠少数几天撑起收益"。"""
    rng = np.random.default_rng(cfg.seed + 2)
    a = r.to_numpy()
    n = len(a)
    k = max(int(n * cfg.drop_pct), 1)
    out = []
    for _ in range(cfg.n_drop):
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, k, replace=False)] = False
        m = metrics_of(pd.Series(a[keep]))
        if m.get("ok"):
            out.append(m)
    return out


def _p_cost(r: pd.Series, cfg: PerturbCfg) -> list[dict]:
    """
    追加成本：每年多付 extra_bps × turnover。
    测"策略的收益是不是全靠省下来的交易成本"。
    """
    out = []
    for extra in cfg.cost_extra_bps:
        # 年化成本 → 日成本
        daily = (extra / 1e4) * cfg.turnover / TD
        m = metrics_of(r - daily)
        if m.get("ok"):
            m["extra_bps"] = extra
            out.append(m)
    return out


def _robust(items: list[dict], cfg: PerturbCfg) -> dict:
    """把一个扰动族的分布压成一个稳健性分数（0~100）。"""
    if not items:
        return {"ok": False, "n": 0}
    sh = np.array([x["sharpe"] for x in items], dtype="float64")
    cg = np.array([x["cagr"] for x in items], dtype="float64")
    med = float(np.median(sh))
    cv = float(np.std(cg, ddof=0) / abs(np.mean(cg))) if abs(np.mean(cg)) > 1e-9 else float("inf")
    pos_rate = float((sh > 0).mean())
    # 一半看正收益比例，一半看收益离散度
    stab = 1.0 - min(cv / 2.0, 1.0) if np.isfinite(cv) else 0.0
    score = 100.0 * (0.5 * pos_rate + 0.5 * stab)
    return {"ok": True, "n": len(items),
            "sharpe_median": round(med, 4),
            "sharpe_p10": round(float(np.percentile(sh, 10)), 4),
            "sharpe_p90": round(float(np.percentile(sh, 90)), 4),
            "cagr_median": round(float(np.median(cg)), 4),
            "cagr_cv": round(cv, 4) if np.isfinite(cv) else None,
            "pos_rate": round(pos_rate, 4),
            "stability": round(float(stab), 4),
            "score": round(float(np.clip(score, 0, 100)), 2)}


def perturbation_suite(data, cfg: PerturbCfg | None = None) -> dict:
    """
    跑全部四类扰动，给出综合稳健性。

    返回 {ok, base, families:{...}, score, grade, verdict, worst}
    """
    cfg = cfg or PerturbCfg()
    r = to_returns(data)
    if len(r) < 120:
        return {"ok": False, "reason": f"样本仅 {len(r)} 天，至少需要 120 天"}

    base = metrics_of(r)
    if not base.get("ok"):
        return {"ok": False, "reason": "基准序列无法计算"}

    families = {
        "加噪声": _p_noise(r, cfg),
        "块重抽样": _p_bootstrap(r, cfg),
        "起点平移": _p_shift(r, cfg),
        "随机删日": _p_drop(r, cfg),
        "追加成本": _p_cost(r, cfg),
    }

    stats = {k: _robust(v, cfg) for k, v in families.items()}
    scores = [s["score"] for s in stats.values() if s.get("ok")]
    if not scores:
        return {"ok": False, "reason": "所有扰动族都无法计算"}

    total = float(np.mean(scores))
    if total >= 80:
        grade, verdict = "A", "稳健：轻微改变输入后结论基本不变"
    elif total >= 65:
        grade, verdict = "B", "较稳：有波动但不至于翻转"
    elif total >= 50:
        grade, verdict = "C", "偏脆：某些扰动下结论会明显变化"
    else:
        grade, verdict = "D", "脆弱：结论高度依赖特定输入，不可轻信"

    worst = min((s for s in stats.items() if s[1].get("ok")),
                key=lambda kv: kv[1]["score"]) if stats else None

    return {"ok": True, "n_days": int(len(r)), "base": base,
            "families": stats, "score": round(total, 2),
            "grade": grade, "verdict": verdict,
            "worst_family": worst[0] if worst else None,
            "cfg": {"noise_mult": cfg.noise_mult, "n_noise": cfg.n_noise,
                    "block": cfg.block, "n_boot": cfg.n_boot,
                    "drop_pct": cfg.drop_pct, "seed": cfg.seed}}


# ══════════════════════════════════════════════════════════
# 压力测试
# ══════════════════════════════════════════════════════════
def _s_vol(r: pd.Series, mult: float) -> pd.Series:
    """波动放大：收益围绕均值放大（保留方向与均值）。"""
    m = float(r.mean())
    return r * mult + m * (1 - mult)


def _s_tail(r: pd.Series, cfg: StressCfg) -> pd.Series:
    """尾部注入：随机若干天出现极端下跌。"""
    rng = np.random.default_rng(cfg.seed)
    a = r.to_numpy().copy()
    n = len(a)
    k = max(int(n * cfg.tail_prob), cfg.tail_rounds)
    k = min(k, max(n // 20, 1))
    idx = rng.choice(n, k, replace=False)
    a[idx] = cfg.tail_size
    return pd.Series(a, index=r.index)


def _s_drift(r: pd.Series, cfg: StressCfg) -> pd.Series:
    """单边下跌：把整段收益平移，使年化变成指定值（默认 -40%）。"""
    target_daily = (1 + cfg.drift_annual) ** (1 / TD) - 1
    old_daily = float(np.mean(r.to_numpy()))
    return r + (target_daily - old_daily)


def _s_liquidity(r: pd.Series, cfg: StressCfg) -> pd.Series:
    """流动性枯竭：滑点成本按倍数放大。"""
    base_daily_cost = 0.0005 * 12 / TD        # 5bp × 年换手12
    extra = base_daily_cost * (cfg.liquidity_mult - 1)
    return r - extra


def _s_corr_break(r: pd.Series, bench: pd.Series | None) -> pd.Series | None:
    """
    相关性断裂：假设所有资产与市场相关性→1，
    即组合收益完全跟随市场。没有基准则无法构造。
    """
    if bench is None:
        return None
    b = to_returns(bench)
    if len(b) < 60:
        return None
    idx = r.index.intersection(b.index)
    if len(idx) < 60:
        return None
    return b.loc[idx]


def stress_suite(data, bench=None, cfg: StressCfg | None = None) -> dict:
    """
    跑全部压力情景，判定每个情景下是否存活。

    存活判据（可调）：情景年化 > survive_cagr 且 最大回撤 > survive_mdd。
    """
    cfg = cfg or StressCfg()
    r = to_returns(data)
    if len(r) < 120:
        return {"ok": False, "reason": f"样本仅 {len(r)} 天，至少需要 120 天"}

    base = metrics_of(r)
    if not base.get("ok"):
        return {"ok": False, "reason": "基准序列无法计算"}

    scen: dict[str, pd.Series | None] = {}
    for mult in cfg.vol_mults:
        scen[f"波动 ×{mult:g}"] = _s_vol(r, mult)
    scen["尾部注入"] = _s_tail(r, cfg)
    scen["单边下跌 40%"] = _s_drift(r, cfg)
    scen["流动性枯竭"] = _s_liquidity(r, cfg)
    # 即使构造不出来也要占位——否则用户会以为"测过了没问题"，
    # 实际是压根没测。这一条和整体的"不编数据"原则一致。
    scen["相关性断裂"] = _s_corr_break(r, bench)

    rows = []
    for name, series in scen.items():
        if series is None:
            rows.append({"scenario": name, "ok": False,
                         "reason": "缺少基准序列，无法构造"})
            continue
        m = metrics_of(series)
        if not m.get("ok"):
            rows.append({"scenario": name, "ok": False, "reason": m.get("reason")})
            continue
        survived = (m["cagr"] > cfg.survive_cagr
                    and m["max_drawdown"] > cfg.survive_mdd)
        rows.append({
            "scenario": name, "ok": True,
            "cagr": round(m["cagr"], 6),
            "max_drawdown": round(m["max_drawdown"], 6),
            "sharpe": round(m["sharpe"], 4),
            "survived": bool(survived),
            "verdict": "存活" if survived else "击穿",
        })

    good = [x for x in rows if x.get("ok")]
    n_alive = sum(1 for x in good if x["survived"])
    worst = min(good, key=lambda x: x["cagr"]) if good else None

    if not good:
        return {"ok": False, "reason": "所有情景都无法评估"}

    rate = n_alive / len(good)
    if rate >= 0.85:
        grade, verdict = "A", "抗压良好：多数极端情景下仍存活"
    elif rate >= 0.6:
        grade, verdict = "B", "基本抗压：少数情景会被击穿"
    elif rate >= 0.4:
        grade, verdict = "C", "抗压偏弱：近半情景击穿"
    else:
        grade, verdict = "D", "抗压差：多数极端情景下无法存活"

    return {"ok": True, "n_days": int(len(r)), "base": base,
            "scenarios": rows, "n_alive": n_alive, "n_total": len(good),
            "survival_rate": round(rate, 4), "grade": grade, "verdict": verdict,
            "worst": worst,
            "criteria": {"survive_cagr": cfg.survive_cagr,
                         "survive_mdd": cfg.survive_mdd}}


# ══════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════
def format_perturbation(res: dict) -> str:
    L = ["=" * 68, "扰动测试（结论稳不稳）", "=" * 68]
    if not res.get("ok"):
        L.append(f"  无法评估：{res.get('reason')}")
        L.append("=" * 68)
        return "\n".join(L)

    b = res["base"]
    L.append(f"  基准：{res['n_days']} 天   年化 {b['cagr']*100:+.2f}%   "
             f"夏普 {b['sharpe']:.2f}   回撤 {b['max_drawdown']*100:.2f}%")
    L.append("-" * 68)
    L.append("  " + f"{'扰动族':<12}{'样本':>6}{'夏普中位':>10}"
             f"{'P10':>8}{'P90':>8}{'正收益比':>10}{'稳健分':>8}")
    L.append("  " + "-" * 62)
    for name, s in res["families"].items():
        if not s.get("ok"):
            L.append("  " + f"{name:<12}{'-':>6}{'无法计算':>30}")
            continue
        L.append("  " + f"{name:<12}{s['n']:>6}{s['sharpe_median']:>10.2f}"
                 f"{s['sharpe_p10']:>8.2f}{s['sharpe_p90']:>8.2f}"
                 f"{s['pos_rate']*100:>9.0f}%{s['score']:>8.1f}")
    L.append("-" * 68)
    L.append(f"  综合稳健性 {res['score']:.1f}/100   等级 {res['grade']}")
    L.append(f"  最脆弱的扰动：{res.get('worst_family') or '—'}")
    L.append(f"  结论: {res['verdict']}")
    L.append("=" * 68)
    return "\n".join(L)


def format_stress(res: dict) -> str:
    L = ["=" * 68, "压力测试（极端情景扛不扛）", "=" * 68]
    if not res.get("ok"):
        L.append(f"  无法评估：{res.get('reason')}")
        L.append("=" * 68)
        return "\n".join(L)

    b = res["base"]
    L.append(f"  基准：{res['n_days']} 天   年化 {b['cagr']*100:+.2f}%   "
             f"回撤 {b['max_drawdown']*100:.2f}%")
    L.append(f"  存活判据：情景年化 > {res['criteria']['survive_cagr']*100:.0f}% "
             f"且回撤 > {res['criteria']['survive_mdd']*100:.0f}%")
    L.append("-" * 68)
    L.append("  " + f"{'情景':<16}{'年化':>10}{'回撤':>10}{'夏普':>8}{'结果':>8}")
    L.append("  " + "-" * 58)
    for x in res["scenarios"]:
        if not x.get("ok"):
            L.append("  " + f"{x['scenario']:<16}" + ("无法构造" if "基准" in str(x.get("reason")) else "无法评估"))
            continue
        L.append("  " + f"{x['scenario']:<16}{x['cagr']*100:>9.2f}%"
                 f"{x['max_drawdown']*100:>9.2f}%{x['sharpe']:>8.2f}"
                 f"{x['verdict']:>8}")
    L.append("-" * 68)
    L.append(f"  存活 {res['n_alive']}/{res['n_total']}"
             f"（{res['survival_rate']*100:.0f}%）   等级 {res['grade']}")
    if res.get("worst"):
        w = res["worst"]
        L.append(f"  最差情景：{w['scenario']}  年化 {w['cagr']*100:.2f}%  "
                 f"回撤 {w['max_drawdown']*100:.2f}%")
    L.append(f"  结论: {res['verdict']}")
    L.append("=" * 68)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    idx = pd.bdate_range("2020-01-02", periods=1000)
    rng = np.random.default_rng(7)
    bench = rng.normal(0.0003, 0.010, 1000)

    # 稳健策略：稳定的正收益
    stable = pd.Series(0.0006 + rng.normal(0, 0.006, 1000), index=idx)
    ps = perturbation_suite(stable)
    assert ps["ok"], ps
    print(format_perturbation(ps))
    assert ps["grade"] in ("A", "B"), ps["grade"]

    # 脆弱策略：靠少数几天撑起来
    lucky = np.full(1000, -0.0004)
    lucky[[200, 500, 800]] = 0.12
    lucky_s = pd.Series(lucky, index=idx)
    ps2 = perturbation_suite(lucky_s)
    assert ps2["ok"]
    print(format_perturbation(ps2))
    assert ps2["score"] < ps["score"], (ps2["score"], ps["score"])
    print(f"稳健 {ps['score']:.1f} > 脆弱 {ps2['score']:.1f} ✓\n")

    ss = stress_suite(stable, bench=pd.Series(np.cumprod(1 + bench), index=idx))
    assert ss["ok"], ss
    print(format_stress(ss))
    assert ss["n_total"] >= 5, ss["n_total"]
    print(f"OK v{__version__}")
