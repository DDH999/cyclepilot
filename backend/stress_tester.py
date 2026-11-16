# -*- coding: utf-8 -*-
"""
策略压力测试器（M1 v3）
====================================================
依赖：numpy / pandas

对外接口
--------
apply_slippage        扣滑点后的净值/年化/夏普/回撤
test_extreme_crash    单个极端行情区间的表现切片
run_crash_suite       遍历 CRASH_WINDOWS 批量跑（原字典不再是死代码）
format_report         把结果 dict 渲染成可读文本（不直接 print）
to_json               仅在序列化时做精度格式化

设计约定
--------
* 库函数 **默认不往 stdout 输出**（verbose=False），日志走 logging
* 返回值保持 float64 **全精度**；需要展示/存储时用 to_json / format_report 格式化
* 所有可调阈值见 StressCfg，可用 dataclasses.replace 覆盖

变更日志
--------
v0.1.0  初版
v0.2.0  夏普改超额口径 + 总体标准差；比率加下限与裁剪；阈值外置
v0.3.0  默认静默 + logging；去中间 round；__all__/__version__；
        输入校验；日期校验；cumprod 溢出保护；单日年化改 nan 并告警；
        run_crash_suite 消化 CRASH_WINDOWS
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__version__ = "0.3.0"
__all__ = [
    "StressCfg", "DEFAULTS", "CRASH_WINDOWS", "TRADING_DAYS",
    "to_series", "rf_daily", "max_drawdown", "sharpe_annual", "bounded_ratio",
    "compound_nav", "apply_slippage", "test_extreme_crash", "run_crash_suite",
    "format_report", "to_json",
]

log = logging.getLogger(__name__)
TRADING_DAYS = 252


# ============================================================
# 配置
# ============================================================
@dataclass(frozen=True)
class StressCfg:
    """全部业务阈值集中于此，可用 replace() 覆盖。"""
    trading_days: int = TRADING_DAYS
    min_obs: int = 2
    min_crash_days: int = 5          # 少于该天数的区间，年化结果仅参考并告警
    rf_annual: float = 0.0
    ddof: int = 0
    slippage_rate: float = 0.0005
    turnover: float = 0.10
    crash_start: str = "2024-01-01"
    crash_end: str = "2024-02-08"
    mdd_floor: float = 1e-4
    ratio_clip: float = 20.0
    survive_good: float = 0.10
    survive_ok: float = 0.05
    crash_ratio_alert: float = 2.0


DEFAULTS = StressCfg()

CRASH_WINDOWS = {
    "2024微盘股流动性危机": ("2024-01-01", "2024-02-08"),
    "2015股灾": ("2015-06-12", "2015-08-26"),
    "2018贸易摩擦": ("2018-01-24", "2018-12-28"),
    "2020新冠冲击": ("2020-01-20", "2020-03-23"),
}


# ============================================================
# 基础工具
# ============================================================
def to_series(x: Iterable | pd.Series | pd.DataFrame) -> pd.Series:
    """
    统一转成 float Series，保留原索引与索引名。

    注意：传入 list / ndarray 时只会生成 RangeIndex(0,1,2,...)，
    对 apply_slippage 无影响；但 test_extreme_crash 需要日期索引。
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"只接受单列收益，收到 {x.shape[1]} 列")
        s = x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series(list(x))
    idx_name = s.index.name
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    s.index.name = idx_name
    return s


def rf_daily(rf_annual: float, trading_days: int = TRADING_DAYS) -> float:
    """年化无风险利率 -> 日利率（几何折算）。"""
    if rf_annual <= -1:
        raise ValueError("rf_annual 必须 > -1")
    return (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0


def compound_nav(r: np.ndarray) -> np.ndarray:
    """
    复利净值。极端序列下 cumprod 可能溢出，溢出时退化到对数域累加。
    """
    a = np.asarray(r, dtype="float64")
    nav = np.cumprod(1.0 + a)
    if not np.all(np.isfinite(nav)):
        with np.errstate(invalid="ignore", divide="ignore"):
            nav = np.exp(np.cumsum(np.log1p(a)))
    return nav


def max_drawdown(nav) -> float:
    """
    最大回撤（负数）。峰值 <= 0 的时点不参与计算。
    边界：若所有净值都为 0 或输入为空，返回 0.0（而非 NaN）。
    """
    a = np.asarray(nav, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    peak = np.maximum.accumulate(a)
    dd = a / np.where(peak <= 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd)) if np.isfinite(dd).any() else 0.0


def sharpe_annual(returns, rf_annual: float = 0.0, ddof: int = 0,
                  trading_days: int = TRADING_DAYS) -> float:
    """
    年化**超额**夏普： (mean(r) - rf_daily) / std(r, ddof) * sqrt(252)

    ddof=0 为总体标准差（默认）。零方差（含浮点误差）返回 0.0。
    """
    r = to_series(returns)
    if len(r) < 2:
        return 0.0
    excess = r.to_numpy() - rf_daily(rf_annual, trading_days)
    sd = float(np.std(excess, ddof=ddof))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(trading_days))


def bounded_ratio(num: float, den: float, floor: float, clip: float) -> float:
    """num / max(|den|, floor)，结果裁剪到 [-clip, clip]；非有限输入返回 nan。"""
    if not (np.isfinite(num) and np.isfinite(den)):
        return float("nan")
    d = np.sign(den) * max(abs(den), floor) if den != 0 else floor
    return float(np.clip(num / d, -clip, clip))


def _check_nonneg(name: str, v: float) -> None:
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"{name} 必须是非负有限数，收到 {v}")


# ============================================================
# 滑点压力测试
# ============================================================
def apply_slippage(df_returns, slippage_rate: float | None = None,
                   turnover: float | None = None,
                   turnover_period: str = "day",
                   rf_annual: float | None = None,
                   verbose: bool = False,
                   cfg: StressCfg = DEFAULTS) -> dict:
    """
    强制扣除滑点后的净值与年化。

    参数
    ----
    df_returns     : 日收益序列
    slippage_rate  : 单边滑点，必须 >= 0。0.0005 = 5bps
    turnover       : 每日换手比例，必须 >= 0。
                     月频 ≈ 0.1；周频 ≈ 0.4；日频全换手 = 1.0
    verbose        : 是否输出报告（默认 False，走 logging）
    cfg            : 阈值配置

    日成本 = 2 * turnover * slippage_rate（买卖双边）

    返回 dict，数值为 float64 全精度；展示用 format_report，存取用 to_json。
    """
    sl = cfg.slippage_rate if slippage_rate is None else float(slippage_rate)
    tv = cfg.turnover if turnover is None else float(turnover)
    rf = cfg.rf_annual if rf_annual is None else float(rf_annual)
    _check_nonneg("slippage_rate", sl)
    _check_nonneg("turnover", tv)

    period = str(turnover_period or "day").strip().lower()
    if period in ("year", "annual", "y", "年"):
        tv = tv / float(cfg.trading_days)
        period = "year"
    elif period in ("day", "daily", "d", "日"):
        period = "day"
    else:
        raise ValueError(
            f"turnover_period 只能是 day 或 year，收到 {turnover_period!r}")

    td, dd = cfg.trading_days, cfg.ddof
    r = to_series(df_returns)
    if len(r) < cfg.min_obs:
        raise ValueError(f"收益序列过短（至少 {cfg.min_obs} 个点）")

    cost = 2.0 * tv * sl
    rg = r.to_numpy()
    rn = rg - cost
    nav_g, nav_n = compound_nav(rg), compound_nav(rn)
    n = len(r)

    ann_g = float(nav_g[-1] ** (td / n) - 1) if nav_g[-1] > 0 else float("nan")
    ann_n = float(nav_n[-1] ** (td / n) - 1) if nav_n[-1] > 0 else float("nan")

    if not np.isfinite(ann_n):
        verdict = "净值归零，策略不可用"
    elif ann_n >= cfg.survive_good:
        verdict = "扣费后仍显著为正，抗摩擦能力尚可"
    elif ann_n >= cfg.survive_ok:
        verdict = f"扣费后低于 {cfg.survive_good * 100:.0f}%，收益主要被交易成本吞噬"
    elif ann_n > 0:
        verdict = f"扣费后不足 {cfg.survive_ok * 100:.0f}%，实盘大概率亏损"
    else:
        verdict = "扣费后为负，策略不可用"

    res = {
        "n_days": int(n),
        "slippage_rate": sl,
        "turnover": tv,
        "cost_per_day": cost,
        "annual_return_gross": ann_g,
        "annual_return_net": ann_n,
        "turnover_period": period,
        "daily_cost_pct": cost,
        "sanity_warning": (
            f"每日成本 {cost*100:.2f}% 过高，几乎必然不可用："
            "请确认换手口径（年换手需传 turnover_period=\"year\"）"
            if cost > 0.01 else None),
        "annual_cost_drag": ann_g - ann_n,
        "sharpe_gross": sharpe_annual(rg, rf, dd, td),
        "sharpe_net": sharpe_annual(rn, rf, dd, td),
        "max_drawdown_gross": max_drawdown(nav_g),
        "max_drawdown_net": max_drawdown(nav_n),
        "nav_gross": nav_g,
        "nav_net": nav_n,
        "survive": bool(np.isfinite(ann_n) and ann_n >= cfg.survive_ok),
        "rf_annual": rf,
        "ddof": int(dd),
        "verdict": verdict,
        "version": __version__,
    }
    if verbose:
        log.info(format_report(res, kind="slippage"))
    return res


# ============================================================
# 极端行情切片
# ============================================================
def test_extreme_crash(df_returns, date_start: str | None = None,
                       date_end: str | None = None,
                       label: str = "2024微盘股灾难",
                       verbose: bool = False,
                       cfg: StressCfg = DEFAULTS) -> dict:
    """
    截取极端行情区间，计算该策略在该区间的最差表现。

    校验：收益序列索引必须是 DatetimeIndex；date_start 不得晚于 date_end。

    vs_full = 区间回撤 / 全样本回撤，分母取 max(|全样本回撤|, mdd_floor)，
    结果裁剪到 [0, ratio_clip]。

    区间交易日少于 cfg.min_crash_days 时，年化结果仅作参考并发出警告。
    """
    ds = str(date_start or cfg.crash_start)
    de = str(date_end or cfg.crash_end)
    if pd.Timestamp(ds) > pd.Timestamp(de):
        raise ValueError(f"date_start({ds}) 不能晚于 date_end({de})")

    s_full = to_series(df_returns)
    if not isinstance(s_full.index, pd.DatetimeIndex):
        raise ValueError(
            "收益序列的索引必须是日期（DatetimeIndex），否则无法按时间切片。\n"
            "正确做法: pd.Series(daily_returns, index=pd.to_datetime(dates))")

    win = s_full.loc[ds:de]
    base = {"label": label, "window": f"{ds} ~ {de}", "ok": bool(len(win)),
            "days": int(len(win))}
    if len(win) == 0:
        base["message"] = "该区间无数据，可能收益序列未覆盖此时间段"
        if verbose:
            log.info("[%s] 区间 %s~%s 无数据，跳过", label, ds, de)
        return base

    nav = compound_nav(win.to_numpy())
    mdd_win = max_drawdown(nav)
    mdd_full = max_drawdown(compound_nav(s_full.to_numpy()))
    period_ret = float(nav[-1] - 1.0)
    worst = float(win.min())

    if period_ret <= -1.0:
        ann = float("nan")          # 亏光，年化无意义（不再用 -1.0 魔法值）
    else:
        ann = float((1 + period_ret) ** (cfg.trading_days / max(len(win), 1)) - 1)
        if len(win) < cfg.min_crash_days:
            warnings.warn(
                f"区间仅 {len(win)} 个交易日，年化结果为外推值，仅供参考",
                RuntimeWarning, stacklevel=2)
            ann = float("nan")

    ratio = bounded_ratio(mdd_win, mdd_full, cfg.mdd_floor, cfg.ratio_clip)
    ratio = abs(ratio) if np.isfinite(ratio) else None

    if ratio is None:
        verdict = "无法比较"
    elif ratio > cfg.crash_ratio_alert:
        verdict = "极端行情下回撤远超全样本，风险控制失效"
    else:
        verdict = "极端行情下回撤可控"

    res = {**base,
           "period_return": period_ret,
           "max_drawdown": mdd_win,
           "worst_day": worst,
           "annual_return": ann,
           "full_sample_max_drawdown": mdd_full,
           "vs_full": ratio,
           "verdict": verdict,
           "version": __version__}
    if verbose:
        log.info(format_report(res, kind="crash"))
    return res


def run_crash_suite(df_returns, windows: dict[str, Sequence[str]] | None = None,
                    verbose: bool = False,
                    cfg: StressCfg = DEFAULTS) -> list[dict]:
    """
    批量跑 CRASH_WINDOWS 里的全部窗口（无数据的自动跳过）。

    返回按严重度排序的结果列表（回撤倍数从大到小）。
    """
    wins = windows or CRASH_WINDOWS
    out = []
    for label, (ds, de) in wins.items():
        r = test_extreme_crash(df_returns, ds, de, label=label,
                               verbose=False, cfg=cfg)
        if r["ok"]:
            out.append(r)
        elif verbose:
            log.info("[%s] 无数据，跳过", label)
    out.sort(key=lambda x: (x.get("vs_full") is None, -(x.get("vs_full") or 0)))
    if verbose:
        for r in out:
            log.info(format_report(r, kind="crash"))
    return out


# ============================================================
# 渲染与序列化
# ============================================================
def format_report(res: dict, kind: str = "slippage") -> str:
    """把结果 dict 渲染成多行文本（不直接打印）。"""
    L = ["=" * 66]
    if kind == "crash":
        L.append(f"极端行情切片测试 -- {res['label']}")
        L.append("=" * 66)
        if not res["ok"]:
            L.append(f"  {res.get('message', '无数据')}")
            return "\n".join(L)
        L.append(f"  区间: {res['window']}  共 {res['days']} 个交易日")
        L.append(f"  区间收益: {res['period_return'] * 100:>8.2f}%")
        L.append(f"  区间最大回撤: {res['max_drawdown'] * 100:>8.2f}%")
        L.append(f"  单日最差: {res['worst_day'] * 100:>8.2f}%")
        line = f"  全样本最大回撤: {res['full_sample_max_drawdown'] * 100:>8.2f}%"
        if res.get("vs_full") is not None:
            line += f"   区间回撤是全样本的 {res['vs_full']:.2f} 倍"
        L.append(line)
        L.append(f"  年化(参考): {res['annual_return'] * 100:>8.2f}%")
        L.append(f"  结论: {res['verdict']}")
    else:
        L.append("滑点压力测试")
        L.append("=" * 66)
        L.append(f"  样本 {res['n_days']} 天 | 单边滑点 {res['slippage_rate'] * 1e4:.1f}bps"
                 f" | 日换手 {res['turnover'] * 100:.0f}%"
                 f" | 日成本 {res['cost_per_day'] * 1e4:.2f}bps")
        L.append(f"  年化收益: {res['annual_return_gross'] * 100:>7.2f}%  ->  "
                 f"{res['annual_return_net'] * 100:>7.2f}%"
                 f"   (拖累 {res['annual_cost_drag'] * 100:.2f} 个点)")
        L.append(f"  夏普    : {res['sharpe_gross']:>7.2f}  ->  {res['sharpe_net']:>7.2f}"
                 f"   (超额口径 rf={res['rf_annual'] * 100:.2f}%, ddof={res['ddof']})")
        L.append(f"  最大回撤: {res['max_drawdown_gross'] * 100:>7.2f}%  ->  "
                 f"{res['max_drawdown_net'] * 100:>7.2f}%")
        L.append(f"  结论: {res['verdict']}")
    L.append("=" * 66)
    return "\n".join(L)


def to_json(res: dict, digits: int = 6) -> dict:
    """
    仅在序列化时做精度格式化；内部计算不 round。

    nav 数组按 digits 保留，标量同理；非数值字段原样通过。
    """
    out = {}
    for k, v in res.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else round(float(v), digits)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.ndarray):
            out[k] = [round(float(x), digits) for x in v if np.isfinite(x)]
        elif isinstance(v, list):
            out[k] = [round(float(x), digits) if isinstance(x, (np.floating, float))
                      else x for x in v]
        else:
            out[k] = v
    return out


# ============================================================
# 自测（只做断言，打印交给 logging）
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    s = sharpe_annual([0.01, 0.02, 0.03], ddof=0)
    assert abs(s - 38.884436) < 1e-3, s
    assert sharpe_annual([0.01] * 20) == 0.0
    assert np.isfinite(bounded_ratio(1.0, 0.0, 0.05, 2.0))
    assert bounded_ratio(1.0, 0.0, 0.05, 2.0) == 2.0
    assert max_drawdown(np.zeros(5)) == 0.0
    assert max_drawdown(np.array([1.0, 1.2, 0.9])) == 0.9 / 1.2 - 1

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=520)
    rets = pd.Series(rng.normal(0.0006, 0.013, len(dates)), index=dates)
    m = (dates >= "2024-01-20") & (dates <= "2024-02-08")
    rets[m] += rng.normal(-0.02, 0.02, m.sum())

    r1 = apply_slippage(rets, verbose=True)
    assert r1["annual_return_net"] < r1["annual_return_gross"]
    assert r1["annual_cost_drag"] > 0

    r2 = test_extreme_crash(rets, verbose=True)
    assert r2["ok"] and r2["max_drawdown"] < 0

    suite = run_crash_suite(rets)
    print(f"OK  version={__version__}  crash_suite={len(suite)} 个窗口有数据")
