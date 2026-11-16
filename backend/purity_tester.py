# -*- coding: utf-8 -*-
"""
策略纯度检测（M2）
====================================================
回答一个问题：**这个策略到底在干什么？**

一个"看起来能赚钱"的策略，可能在偷偷做别的事：
  * 其实是买指数（与基准高度相关 → 不是 alpha，是 beta）
  * 其实只押了 2~3 只票（等效持仓数极低 → 不是组合，是押注）
  * 其实大部分时间空仓（净值被"抹平" → 不是策略，是现金）
  * 收益全靠极少数几天（→ 不可复现）

四类纯度指标
------------
1. effective_n   等效持仓数 = 1 / Σw²（权重归一后）
2. exposure      实际有仓位的天数占比
3. bench_r2      策略日收益对基准回归的 R²（越高越像基准）
4. top5d_share   收益最高的 5% 交易日贡献了总收益的多少

评分：基准 100 分，按四项逾越阈值扣分，结果裁剪到 [0, 100]。

依赖：numpy / pandas
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = [
    "PurityCfg", "DEFAULTS", "effective_n", "top_share", "exposure_ratio",
    "bench_beta_r2", "return_concentration", "purity_report", "format_report",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PurityCfg:
    """纯度阈值与扣分权重。"""
    min_days: int = 30
    min_holdings: int = 3

    # 扣分权重（合计 100）
    w_r2: float = 35.0
    w_conc: float = 25.0
    w_exposure: float = 20.0
    w_return_conc: float = 20.0

    # 阈值：free 以下不扣分，bad 及以上扣满
    r2_free: float = 0.30
    r2_bad: float = 0.70
    hhi_free: float = 0.100      # 等效持仓 ≥10 只不扣分
    hhi_bad: float = 0.250       # 等效持仓 ≤4 只扣满
    exposure_ok: float = 0.80
    exposure_bad: float = 0.50
    top5d_free: float = 0.15
    top5d_bad: float = 0.35

    # 极低仓位判定：平均总仓位低于此值 → 标记 cash_dominated（不豁免、不扣分，仅标注）
    cash_dominated_gross: float = 0.20


DEFAULTS = PurityCfg()


# ============================================================
# 指标
# ============================================================
def _norm(w: np.ndarray) -> np.ndarray:
    s = float(np.sum(w))
    return (w / s) if s > 0 else np.zeros_like(w)


def effective_n(w: np.ndarray) -> float:
    """等效持仓数 = 1 / Σ(归一权重²)。空仓返回 0。"""
    a = np.asarray(w, dtype="float64")
    if not np.isfinite(a).all() or a.sum() <= 0:
        return 0.0
    p = _norm(a)
    s2 = float(np.sum(p ** 2))
    return float(1.0 / s2) if s2 > 0 else 0.0


def top_share(w: np.ndarray, k: int = 5) -> float:
    """前 k 大权重合计占比。"""
    a = np.asarray(w, dtype="float64")
    if not np.isfinite(a).all() or a.sum() <= 0:
        return 0.0
    p = np.sort(_norm(a))[::-1]
    return float(p[:k].sum())


def exposure_ratio(weights: pd.DataFrame) -> float:
    """有仓位的天数占比（当日总权重 > 1e-9）。"""
    if weights is None or len(weights) == 0:
        return 0.0
    gross = weights.sum(axis=1).to_numpy()
    return float(np.mean(np.isfinite(gross) & (gross > 1e-9)))


def bench_beta_r2(returns: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """策略日收益对基准做单变量回归，返回 (beta, R²)。"""
    if returns is None or bench is None:
        return float("nan"), float("nan")
    idx = returns.index.intersection(bench.index)
    if len(idx) < 10:
        return float("nan"), float("nan")
    y = returns.loc[idx].to_numpy(dtype="float64")
    x = bench.loc[idx].to_numpy(dtype="float64")
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 10 or np.var(x) <= 0:
        return float("nan"), float("nan")
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    yhat = beta * (x - x.mean())
    ss_res = float(np.sum((y - y.mean() - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, float(np.clip(r2, 0.0, 1.0))


def return_concentration(returns: pd.Series, q: float = 0.05) -> float:
    """
    收益集中度：最好的 q 比例交易日，贡献了「总绝对波动」的多少（有界 0~1）。

    分母用 |r| 之和而非总收入——总收入接近 0 时比值会爆炸
    （实测出现过 744% 这种无意义数值）。
    """
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 20:
        return float("nan")
    denom = float(np.abs(r).sum())
    if denom <= 1e-12:
        return float("nan")
    k = max(1, int(len(r) * q))
    top = float(r.nlargest(k).sum())
    return float(np.clip(top / denom, 0.0, 1.0))


# ============================================================
# 主接口
# ============================================================
def _ramp(x: float, free: float, bad: float) -> float:
    """把 x 映射到 [0,1]：<=free 为 0，>=bad 为 1，中间线性。"""
    if not np.isfinite(x):
        return 0.0
    if x <= free:
        return 0.0
    if x >= bad:
        return 1.0
    return (x - free) / (bad - free)


def purity_report(weights: pd.DataFrame | None = None,
                  returns: pd.Series | None = None,
                  bench: pd.Series | None = None,
                  cfg: PurityCfg = DEFAULTS,
                  verbose: bool = False) -> dict:
    """
    计算策略纯度。

    参数
    ----
    weights : 目标权重矩阵（index=日期, columns=代码）。缺失则无法评估持仓类指标
    returns : 策略日收益序列
    bench   : 基准日收益序列（可选；缺失则跳过 beta/R²）

    返回 dict
    --------
    ok           是否可给出评分
    score        0~100（不可评估时 None）
    reason_tag   None / "data_unavailable" / "math_abnormal"（供裁决层分级）
    metrics      各指标原始值
    deductions   各项扣分
    verdict      一句话结论
    """
    out = {"ok": False, "score": None, "reason_tag": None,
           "metrics": {}, "deductions": {}, "verdict": "", "flags": []}

    # 收益序列：既没有权重也没有收益 → 数据未提供
    if returns is None and weights is None:
        out.update(reason_tag="data_unavailable",
                   verdict="未提供收益序列与持仓数据，无法评估纯度")
        return out
    if returns is not None:
        r = pd.to_numeric(pd.Series(returns), errors="coerce")
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < cfg.min_days:
            out.update(reason_tag="math_abnormal",
                       verdict=f"收益样本仅 {len(r)} 天，少于 {cfg.min_days} 天，无法可靠评估纯度")
            return out
    else:
        r = None

    # 持仓类指标
    eff_n = hhi = expo = t1 = t5 = avg_gross = float("nan")
    if weights is not None and len(weights) > 0:
        w = weights.fillna(0.0)
        eff_list, t1_list, t5_list = [], [], []
        for _, row in w.iterrows():
            a = row.to_numpy(dtype="float64")
            if not np.isfinite(a).all() or a.sum() <= 0:
                continue
            n = effective_n(a)
            if n > 0:
                eff_list.append(n)
                t1_list.append(top_share(a, 1))
                t5_list.append(top_share(a, 5))
        if eff_list:
            eff_n = float(np.median(eff_list))
            hhi = float(1.0 / eff_n) if eff_n > 0 else float("nan")
            t1 = float(np.median(t1_list))
            t5 = float(np.median(t5_list))
        expo = exposure_ratio(w)
        gross_series = w.sum(axis=1).to_numpy()
        gross_series = gross_series[np.isfinite(gross_series)]
        avg_gross = float(np.mean(gross_series)) if len(gross_series) else float("nan")

    beta, r2 = (bench_beta_r2(r, bench) if (r is not None and bench is not None)
                else (float("nan"), float("nan")))
    top5d = return_concentration(r) if r is not None else float("nan")

    out["metrics"] = {
        "effective_n": None if not np.isfinite(eff_n) else round(eff_n, 4),
        "hhi": None if not np.isfinite(hhi) else round(hhi, 6),
        "top1_share": None if not np.isfinite(t1) else round(t1, 4),
        "top5_share": None if not np.isfinite(t5) else round(t5, 4),
        "exposure": None if not np.isfinite(expo) else round(expo, 4),
        "avg_gross": None if not np.isfinite(avg_gross) else round(avg_gross, 4),
        "bench_beta": None if not np.isfinite(beta) else round(beta, 4),
        "bench_r2": None if not np.isfinite(r2) else round(r2, 4),
        "top5day_share": None if not np.isfinite(top5d) else round(top5d, 4),
    }

    # 若完全没有持仓数据，只能靠回归与收益集中度，仍可给分但标注
    has_w = np.isfinite(eff_n) and np.isfinite(expo)
    if not has_w and not np.isfinite(r2) and not np.isfinite(top5d):
        out.update(reason_tag="data_unavailable",
                   verdict="持仓与基准数据均缺失，无法评估纯度")
        return out

    # 全程空仓：指标对该策略不适用 → 豁免，不判 0 分、不降级
    if np.isfinite(expo) and expo <= 1e-9 and weights is not None:
        out.update(
            ok=False, score=None, reason_tag="exempt", deductions={},
            verdict=("全程空仓：未产生任何交易，夏普比率等收益类指标不适用，本项豁免。"
                     "特别说明：该策略对参数与交易成本均不敏感（因为没有交易），"
                     "但这也意味着无法用收益类指标证明其有效性。"))
        return out

    # 极低仓位：仍可算分，但必须特别标注（不是"空仓"，也不是"失败"）
    flags = []
    extra_note = ""
    if np.isfinite(avg_gross) and avg_gross < cfg.cash_dominated_gross:
        flags.append("cash_dominated")
        extra_note = (f"极低仓位：平均总仓位仅 {avg_gross * 100:.0f}%，"
                      "接近现金管理而非选股策略，需人工确认其投资意图")

    ded = {
        "bench_beta": round(cfg.w_r2 * _ramp(r2, cfg.r2_free, cfg.r2_bad), 2),
        "concentration": round(cfg.w_conc * _ramp(hhi, cfg.hhi_free, cfg.hhi_bad), 2),
        "exposure": round(cfg.w_exposure *
                          _ramp(cfg.exposure_ok - expo if np.isfinite(expo) else 0.0,
                                0.0, cfg.exposure_ok - cfg.exposure_bad), 2),
        "return_concentration": round(cfg.w_return_conc *
                                      _ramp(top5d, cfg.top5d_free, cfg.top5d_bad), 2),
    }
    score = float(np.clip(100.0 - sum(ded.values()), 0.0, 100.0))

    notes = []
    if np.isfinite(r2) and r2 >= cfg.r2_bad:
        notes.append(f"与基准 R²={r2:.2f}，策略主要是 beta 而非 alpha")
    if np.isfinite(eff_n) and eff_n < cfg.min_holdings:
        notes.append(f"等效持仓仅 {eff_n:.1f} 只，实为押注而非组合")
    if np.isfinite(expo) and expo < cfg.exposure_bad:
        notes.append(f"仅 {expo * 100:.0f}% 的时间有仓位，净值被现金抹平")
    if np.isfinite(top5d) and top5d >= cfg.top5d_bad:
        notes.append(f"收益高度集中于少数交易日（最好 5% 的天贡献绝对波动的 {top5d * 100:.0f}%）")

    if extra_note:
        notes.append(extra_note)
    verdict = "纯度良好：无明显 beta 伪装/押注/空仓/收益集中问题" if not notes \
        else "；".join(notes)

    out.update(ok=True, score=round(score, 2), deductions=ded, verdict=verdict,
               flags=flags, reason_tag=None if has_w else "data_unavailable")
    if verbose:
        log.info(format_report(out))
    return out


def format_report(res: dict) -> str:
    L = ["=" * 66, "策略纯度检测", "=" * 66]
    m = res.get("metrics", {})
    L.append(f"  等效持仓数: {m.get('effective_n')}   实际仓位天数占比: {m.get('exposure')}")
    L.append(f"  基准 R²   : {m.get('bench_r2')}   Beta: {m.get('bench_beta')}")
    L.append(f"  收益集中度: {m.get('top5day_share')}   HHI: {m.get('hhi')}")
    L.append("-" * 66)
    L.append(f"  扣分: {res.get('deductions')}")
    L.append(f"  纯度得分: {res.get('score')}")
    L.append(f"  结论: {res.get('verdict')}")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=300)
    codes = [f"{i:06d}" for i in range(30)]
    w = pd.DataFrame(1 / 30, index=dates, columns=codes)
    r = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
    b = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
    res = purity_report(w, r, b, verbose=True)
    assert res["ok"] and res["score"] > 60, res["score"]

    w2 = pd.DataFrame(0.0, index=dates, columns=codes)
    w2.iloc[:, 0] = 1.0                       # 单票满仓
    bad = purity_report(w2, r, b)
    assert bad["score"] < res["score"]
    print(f"OK v{__version__}  分散={res['score']}  单票={bad['score']}")
