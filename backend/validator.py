# -*- coding: utf-8 -*-
"""
量化策略真实性检测器（M1 v3）
====================================================
依赖：numpy / pandas / scipy

对外接口
--------
check_future_function  代码扫描：True=干净，False=有作弊嫌疑
scan_code              返回逐条命中明细
calculate_overfitting  DSR / PBO / 夏普衰退
format_report          渲染可读文本（不直接 print）
to_json                仅在序列化时格式化精度

设计约定
--------
* 库函数默认不往 stdout 输出（verbose=False），日志走 logging
* 夏普为**超额口径**：扣除 rf_annual，使用**总体标准差**（ddof=0）
* 比率类指标一律做分母下限 + 裁剪，避免 sr→0 时放大成异常值
* 返回值为 float64 全精度，展示用 format_report，存取用 to_json

变更日志
--------
v0.1.0  初版
v0.2.0  夏普口径明确化（ddof=0 + rf）；比率加保护；阈值外置
v0.3.0  默认静默 + logging；去中间 round；__all__/__version__；
        入参校验；零方差浮点容差；PBO 在 sr_is<=0 时返回中性值
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict, replace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

__version__ = "0.4.0"
__all__ = [
    "Thresholds", "DEFAULTS", "Finding", "LINE_RULES", "MISSING_POLICY",
    "to_returns", "rf_daily", "sharpe_annual", "max_drawdown",
    "bounded_ratio", "log_ratio", "deflated_sharpe", "pbo_from_retention",
    "grade_pbo", "scan_code", "check_future_function", "calculate_overfitting",
    "format_report", "to_json",
]

log = logging.getLogger(__name__)

EULER = 0.5772156649015329
TRADING_DAYS = 252


# ============================================================
# 配置
# ============================================================
@dataclass(frozen=True)
class Thresholds:
    """全部业务阈值集中于此，可用 replace(DEFAULTS, xxx=...) 覆盖。"""
    min_obs: int = 30
    trading_days: int = TRADING_DAYS
    rf_annual: float = 0.0
    ddof: int = 0
    dsr_significant: float = 0.95
    sr_floor: float = 0.05
    keep_clip: float = 2.0
    pbo_slope: float = 6.0
    pbo_mid: float = 0.6
    pbo_neutral: float = 0.5
    pbo_high: float = 0.70
    decay_warn: float = 0.50

    # PBO 分级（诊断结论）
    pbo_trust_max: float = 0.50      # <= 此值：可以肯定
    pbo_suspect_min: float = 0.01    # <  此值：异常偏低，需人工复核
    pbo_clamp_low: float = 0.0       # 负值强制截断到这里
    pbo_clamp_high: float = 1.0


DEFAULTS = Thresholds()

# 缺失值策略：fail-closed —— nan 一律视为"不通过 + 需人工复核"，
# 绝不静默跳过，绝不默认放行。
MISSING_POLICY = "fail_closed"


# ============================================================
# 基础工具
# ============================================================
def to_returns(x: Iterable | pd.Series | pd.DataFrame) -> pd.Series:
    """统一转成 float Series，保留索引与索引名；去掉 inf/nan。"""
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


def sharpe_annual(returns, rf_annual: float = 0.0, ddof: int = 0,
                  trading_days: int = TRADING_DAYS) -> float:
    """
    年化**超额**夏普： (mean(r) - rf_daily) / std(r, ddof) * sqrt(252)

    ddof=0 为总体标准差（默认）。零方差（含浮点误差）返回 0.0。
    """
    r = to_returns(returns)
    if len(r) < 2:
        return 0.0
    excess = r.to_numpy() - rf_daily(rf_annual, trading_days)
    sd = float(np.std(excess, ddof=ddof))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(trading_days))


def max_drawdown(nav) -> float:
    """最大回撤（负数）。全零或空输入返回 0.0，不返回 NaN。"""
    a = np.asarray(nav, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    peak = np.maximum.accumulate(a)
    dd = a / np.where(peak <= 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd)) if np.isfinite(dd).any() else 0.0


def bounded_ratio(num: float, den: float, floor: float, clip: float) -> float:
    """num / max(|den|, floor)，结果裁剪到 [-clip, clip]；非有限输入返回 nan。"""
    if not (np.isfinite(num) and np.isfinite(den)):
        return float("nan")
    d = np.sign(den) * max(abs(den), floor) if den != 0 else floor
    return float(np.clip(num / d, -clip, clip))


def log_ratio(a: float, b: float, floor: float) -> float:
    """对数比值 log((a+floor)/(b+floor))，仅同号正数有定义。"""
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return float("nan")
    return float(np.log((a + floor) / (b + floor)))


# ============================================================
# 代码扫描
# ============================================================
@dataclass
class Finding:
    rule: str
    name: str
    severity: str
    line: int
    snippet: str
    why: str
    fix: str

    def to_dict(self) -> dict:
        return asdict(self)


LINE_RULES = [
    ("F01", "shift 取未来数据", "high",
     r"\.shift\(\s*-\s*\d+",
     "shift(-N) 把未来的值搬到当天，是最典型的未来函数写法。",
     "改用 shift(+N)，或确认该序列不是价格/收益。"),
    ("F02", "疑似未来字段名", "high",
     r"\b(future|tomorrow|next_?ret|lookahead)\w*\b",
     "变量名本身暴露了它在使用未来信息。",
     "删除该字段，或改用当期及历史数据。"),
    ("F03", "涨跌幅计算未禁用前向填充", "high",
     r"\.pct_change\(\s*\)",
     "pct_change() 默认 fill_method='pad'，会把停牌/退市股票的 NaN 价格前向填充成常数，"
     "使其收益恒为 0、波动率为 0，在低波因子里被误选为最优。",
     "改成 pct_change(fill_method=None)。"),
    ("F04", "缺失值被当作中性分", "high",
     r"(fillna\(\s*0(\.0)?\s*\)|fill_value\s*=\s*0\b)",
     "把缺失因子填 0（z-score 后的均值）会让「没有任何数据」的股票拿到中性偏高分，"
     "在市场普遍负分时排到最前，等于系统性买入数据缺失股。",
     "改为「有效因子数不足则剔除」，不要填 0。"),
    ("F05", "排序未打破并列", "medium",
     r"\.rank\((?![^)]*method\s*=)",
     "rank() 默认 method='average'，大量并列时给出 124.5 这类平均名次，"
     "导致「取前 30 名」一只都取不到，选股静默失败。",
     "改成 rank(method=\"first\")。"),
    ("F06", "随机切分训练/测试集", "high",
     r"(train_test_split\((?![^)]*shuffle\s*=\s*False)"
     r"|\.sample\([^)]*frac\s*=\s*1[^)]*\))",
     "时序数据随机切分会把未来样本混进训练集，等价于用考卷答案复习。",
     "按时间切分（前段训练、后段测试），并留清洗间隔。"),
]

_CLOSE_RE = re.compile(r"\bclose\b|\b收盘\b")
_TRADE_RE = re.compile(r"\bopen\b(?!\s*\()|\bbuy\b|\bsell\b|\border\b|\b下单\b|\b买入\b|\b卖出\b")


def _strip_comment(line: str) -> str:
    i = line.find("#")
    return line if i < 0 else line[:i]


def scan_code(code_str: str) -> list[Finding]:
    """逐行扫描策略源码，返回全部命中项。"""
    out: list[Finding] = []
    if not code_str:
        return out
    for i, raw in enumerate(code_str.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        for rid, name, sev, pat, why, fix in LINE_RULES:
            if re.search(pat, line):
                out.append(Finding(rid, name, sev, i, raw.strip()[:120], why, fix))
        if _CLOSE_RE.search(line) and _TRADE_RE.search(line):
            out.append(Finding(
                "F06b", "用当日收盘价做当日交易决策", "high", i, raw.strip()[:120],
                "当天收盘价在收盘那一刻才知道，用它决定当天买卖价等于未来函数。",
                "用 T 日收盘生成信号、T+1 日成交；或改用 open / 前一日 close。"))
    return out


def check_future_function(code_str: str, verbose: bool = False) -> bool:
    """True=干净；False=存在 high 级命中（作弊嫌疑）。"""
    findings = scan_code(code_str)
    blocking = [f for f in findings if f.severity == "high"]
    ok = not blocking
    if verbose:
        log.info(format_report({"findings": findings, "ok": ok}, kind="code"))
    return ok


# ============================================================
# 过拟合评估
# ============================================================
def deflated_sharpe(sr_daily: float, n_obs: int, n_trials: int = 1,
                    skew: float = 0.0, kurt: float = 3.0,
                    sr_variance: float | None = None,
                    cfg: Thresholds = DEFAULTS) -> float:
    """
    紧缩夏普比率（Bailey & López de Prado, 2014）。
    sr_daily 为**日均**超额夏普；样本不足返回 nan。
    """
    if n_trials < 1:
        raise ValueError("n_trials 必须 >= 1")
    if n_obs < cfg.min_obs or not np.isfinite(sr_daily):
        return float("nan")
    var = sr_variance if (sr_variance and sr_variance > 0) else 1.0 / n_obs
    n = max(int(n_trials), 1)
    z1 = stats.norm.ppf(1 - 1.0 / n) if n > 1 else 0.0
    z2 = stats.norm.ppf(1 - 1.0 / (n * np.e))
    sr0 = np.sqrt(var) * ((1 - EULER) * z1 + EULER * z2)
    denom = np.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2, 1e-9))
    return float(stats.norm.cdf((sr_daily - sr0) * np.sqrt(n_obs - 1) / denom))


def pbo_from_retention(keep: float, cfg: Thresholds = DEFAULTS) -> float:
    """由夏普保持率映射 PBO（近似）；keep=cfg.pbo_mid 时 PBO=0.5。"""
    if not np.isfinite(keep):
        return cfg.pbo_neutral
    return float(np.clip(1.0 / (1.0 + np.exp(cfg.pbo_slope * (keep - cfg.pbo_mid))),
                         0.0, 1.0))


def grade_pbo(pbo_value, cfg: Thresholds = DEFAULTS, informative: bool = True) -> dict:
    """
    PBO 分级诊断（**永不抛异常**）。

    分级规则
    --------
    0.00 < PBO < 0.01  REVIEW   异常偏低，需人工复核（疑似泄漏/样本重叠/标签穿越）
    0.01 <= PBO <= 0.50  PASS   可以肯定
    PBO > 0.50         REBUILD  建议回炉重造，仔细检查底层计算公式
    PBO 为 nan/inf     UNKNOWN  无法判定，按 fail-closed 处理

    参数 informative=False 表示该 PBO 是**占位值**（例如样本内夏普 <= 0 时的中性 0.5），
    不代表任何证据，此时一律返回 UNKNOWN 且不放行——避免把"无法判断"误当成"可以肯定"。

    负值强制截断为 cfg.pbo_clamp_low（默认 0），超过上限截断为 1。
    任何异常输入都返回 UNKNOWN 而不是崩溃。

    返回 dict
    --------
    pbo            截断后的值（None 表示无法给出）
    pbo_raw        原始输入（保留以便审计）
    pbo_clamped    是否发生了截断
    level          PASS / REBUILD / REVIEW / UNKNOWN
    is_pass        是否通过该单项
    needs_review   是否需要人工复核
    diagnosis      诊断结论（中文，可直接进报告）
    advice         下一步建议
    """
    out = {"pbo": None, "pbo_raw": None, "pbo_clamped": False,
           "level": "UNKNOWN", "is_pass": False, "needs_review": True,
           "diagnosis": "", "advice": ""}
    try:
        raw = float(pbo_value)
    except (TypeError, ValueError):
        out["diagnosis"] = "无法判定：PBO 输入不是数值"
        out["advice"] = "按 fail-closed 策略视为不通过，必须人工复核"
        return out

    out["pbo_raw"] = raw
    if not np.isfinite(raw):
        out["diagnosis"] = "无法判定：PBO 含缺失值（nan/inf）"
        out["advice"] = "按 fail-closed 策略视为不通过，必须人工复核"
        return out

    clamped = float(np.clip(raw, cfg.pbo_clamp_low, cfg.pbo_clamp_high))
    out["pbo"] = clamped
    out["pbo_clamped"] = abs(clamped - raw) > 1e-12

    prefix = ""
    if raw < cfg.pbo_clamp_low:
        prefix = (f"原始值 {raw:.4f} 为负数，已强制截断为 {cfg.pbo_clamp_low:g}；")

    if not informative:
        out.update(
            level="UNKNOWN", is_pass=False, needs_review=True,
            diagnosis=prefix + f"无法判定（PBO 为占位值 {clamped:.4f}）："
                               "样本内夏普 ≤ 0，不存在「样本内最优选择」过程，此值不代表任何证据。",
            advice="不得据此放行。需先让样本内策略本身成立，或改为多策略 CSCV 严格计算 PBO。",
        )
        return out

    if clamped < cfg.pbo_suspect_min:
        out.update(
            level="REVIEW", is_pass=False, needs_review=True,
            diagnosis=prefix + f"异常偏低（PBO={clamped:.4f}）：过拟合概率接近 0 反而可疑，"
                               "通常意味着样本内/样本外高度一致，需排查数据泄漏、"
                               "样本重叠、标签穿越或样本外区段过短。",
            advice="人工复核数据切分方式与特征时点，确认无误后方可采用。",
        )
    elif clamped <= cfg.pbo_trust_max:
        out.update(
            level="PASS", is_pass=True, needs_review=False,
            diagnosis=prefix + f"可以肯定（PBO={clamped:.4f}，"
                               f"处于 {cfg.pbo_suspect_min}~{cfg.pbo_trust_max} 可信区间）。",
            advice="进入下一阶段验证：滑点压力测试与极端行情切片。",
        )
    else:
        out.update(
            level="REBUILD", is_pass=False, needs_review=True,
            diagnosis=prefix + f"建议回炉重造（PBO={clamped:.4f} > {cfg.pbo_trust_max}）："
                               "过拟合概率过高，样本外基本不可信。",
            advice="请仔细检查底层计算公式：数据对齐时点、因子合成方向、指标口径、"
                   "参数搜索次数、样本外切分方式。",
        )
    return out


def calculate_overfitting(returns_is, returns_oos, n_trials: int = 1,
                          rf_annual: float | None = None,
                          verbose: bool = False,
                          cfg: Thresholds = DEFAULTS) -> dict:
    """
    用样本内/样本外收益评估过拟合。

    口径：超额夏普（扣 rf_annual），总体标准差（ddof=0）。
    比率：keep = bounded_ratio(sr_oos, sr_is, sr_floor, keep_clip)，decay = 1 - keep。
    sr_is <= 0 时 PBO 返回中性值 0.5（不存在「样本内最优」的选择过程）。

    返回 dict，数值为全精度；展示用 format_report，存取用 to_json。
    """
    if n_trials < 1:
        raise ValueError("n_trials 必须 >= 1")
    cfg = cfg if rf_annual is None else replace(cfg, rf_annual=float(rf_annual))

    r_is, r_oos = to_returns(returns_is), to_returns(returns_oos)
    if len(r_is) < cfg.min_obs or len(r_oos) < cfg.min_obs:
        raise ValueError(
            f"样本太短：样本内 {len(r_is)} 天、样本外 {len(r_oos)} 天，"
            f"至少各需 {cfg.min_obs} 天")

    td, rf, dd = cfg.trading_days, cfg.rf_annual, cfg.ddof
    sr_is = sharpe_annual(r_is, rf, dd, td)
    sr_oos = sharpe_annual(r_oos, rf, dd, td)
    sr_is_d, sr_oos_d = sr_is / np.sqrt(td), sr_oos / np.sqrt(td)

    a_oos = r_oos.to_numpy()
    skew = float(stats.skew(a_oos, bias=False)) if len(a_oos) > 3 else 0.0
    kurt = float(stats.kurtosis(a_oos, fisher=False, bias=False)) if len(a_oos) > 3 else 3.0
    dsr = deflated_sharpe(sr_oos_d, len(r_oos), n_trials, skew, kurt, cfg=cfg)

    keep = bounded_ratio(sr_oos, sr_is, cfg.sr_floor, cfg.keep_clip)
    decay = (1.0 - keep) if np.isfinite(keep) else float("nan")
    lr = log_ratio(sr_oos, sr_is, cfg.sr_floor)
    pbo_raw = cfg.pbo_neutral if sr_is <= 0 else pbo_from_retention(keep, cfg)
    diag = grade_pbo(pbo_raw, cfg, informative=(sr_is > 0))  # 永不抛异常；负数截断

    # 未做参数搜索（n_trials<=1）时，PBO 偏低是正常的：
    # PBO 衡量的是"选择过程"的过拟合，没有选择就无从过拟合。
    # 买入持有、等权组合的低 PBO 不能被当成泄漏证据。
    if diag["level"] == "REVIEW" and int(n_trials) <= 1:
        diag = {**diag, "level": "PASS", "is_pass": True, "needs_review": False,
                "diagnosis": diag["diagnosis"].rstrip("。") +
                "（本次 n_trials=1，未做参数搜索，PBO 偏低属正常，不作为泄漏证据）",
                "advice": "无需处理。若后续做过参数搜索，请重新评估。"}
    pbo = diag["pbo"] if diag["pbo"] is not None else cfg.pbo_neutral

    nav_oos = np.cumprod(1.0 + a_oos)
    if not np.all(np.isfinite(nav_oos)):
        nav_oos = np.exp(np.cumsum(np.log1p(a_oos)))
    rf_d = rf_daily(rf, td)
    excess_is = r_is.to_numpy() - rf_d
    excess_oos = a_oos - rf_d

    if not np.isfinite(dsr):
        verdict = "样本不足，无法判定"
    elif pbo >= cfg.pbo_high:
        verdict = "高度过拟合：样本外基本失效，不应采信"
    elif pbo >= cfg.pbo_neutral or dsr < cfg.dsr_significant:
        verdict = "疑似过拟合：样本外表现不足以支持该策略"
    elif np.isfinite(decay) and decay > cfg.decay_warn:
        verdict = "衰减明显：虽仍为正，但需警惕可持续性"
    else:
        verdict = "暂未发现明显过拟合"

    res = {
        "dsr": dsr,
        "pbo": pbo,
        "pbo_raw": diag["pbo_raw"],
        "pbo_clamped": diag["pbo_clamped"],
        "pbo_method": "近似估计（由超额夏普保持率映射，非 CSCV 枚举）",
        # ── 诊断结论（新增字段）
        "diagnosis_level": diag["level"],
        "diagnosis": diag["diagnosis"],
        "advice": diag["advice"],
        "is_pass": diag["is_pass"],
        "needs_review": diag["needs_review"],
        "missing_policy": MISSING_POLICY,
        "sharpe_is": sr_is,
        "sharpe_oos": sr_oos,
        "sharpe_retention": keep,
        "sharpe_decay": decay,
        "sharpe_log_ratio": lr,
        "annual_return_is": float(np.mean(excess_is) * td),
        "annual_return_oos": float(np.mean(excess_oos) * td),
        "max_drawdown_oos": max_drawdown(nav_oos),
        "worst_day_oos": float(np.min(r_oos)),
        "n_is": int(len(r_is)), "n_oos": int(len(r_oos)),
        "n_trials": int(n_trials),
        "rf_annual": float(rf), "ddof": int(dd),
        "verdict": verdict,
        "note": "PBO 为近似值；严格计算需多个候选策略做 CSCV 组合枚举。"
                "比率类指标已做分母下限与裁剪保护。",
        "version": __version__,
    }
    if verbose:
        log.info(format_report(res, kind="overfit"))
    return res


# ============================================================
# 渲染与序列化
# ============================================================
def format_report(res: dict, kind: str = "overfit") -> str:
    """把结果渲染成多行文本（不直接打印）。"""
    L = ["=" * 66]
    if kind == "code":
        L.append("策略代码真实性检测")
        L.append("=" * 66)
        fs = res.get("findings", [])
        if not fs:
            L.append("  OK 未发现可疑写法")
        for f in fs:
            tag = {"high": "[严重]", "medium": "[中等]", "low": "[提示]"}[f.severity]
            L.append(f"  {tag} 第 {f.line} 行  {f.rule} {f.name}")
            L.append(f"          代码: {f.snippet}")
            L.append(f"          原因: {f.why}")
            L.append(f"          改法: {f.fix}")
        L.append("-" * 66)
        L.append(f"  结论: {'通过' if res.get('ok') else '不通过（存在未来函数嫌疑）'}")
    else:
        L.append("过拟合评估（DSR / PBO）")
        L.append("=" * 66)
        L.append(f"  口径: 超额夏普(rf={res['rf_annual'] * 100:.2f}%) / "
                 f"{'总体' if res['ddof'] == 0 else '样本'}标准差")
        L.append(f"  样本内 {res['n_is']:>5} 天  年化超额 {res['annual_return_is'] * 100:>7.2f}%"
                 f"  夏普 {res['sharpe_is']:>6.2f}")
        L.append(f"  样本外 {res['n_oos']:>5} 天  年化超额 {res['annual_return_oos'] * 100:>7.2f}%"
                 f"  夏普 {res['sharpe_oos']:>6.2f}")
        keep = res["sharpe_retention"]
        decay = res["sharpe_decay"]
        lr = res["sharpe_log_ratio"]
        L.append(f"  保持率 {keep:>6.2f}   衰减 {decay * 100:>6.1f}%   "
                 f"对数比值 {lr if np.isfinite(lr) else float('nan'):>6.2f}")
        L.append("-" * 66)
        L.append(f"  DSR = {res['dsr'] if np.isfinite(res['dsr']) else None}")
        L.append(f"  PBO = {res['pbo']}  （近似值）")
        if res.get("pbo_clamped"):
            L.append(f"  [截断] 原始 PBO = {res.get('pbo_raw')}，已按策略强制截断")
        L.append(f"  结论: {res['verdict']}")
        if res.get("diagnosis_level"):
            L.append("-" * 66)
            L.append(f"  诊断结论 [{res['diagnosis_level']}]: {res['diagnosis']}")
            L.append(f"  建议: {res['advice']}")
            L.append(f"  是否通过: {'是' if res.get('is_pass') else '否'}"
                     f"   需人工复核: {'是' if res.get('needs_review') else '否'}")
    L.append("=" * 66)
    return "\n".join(L)


def to_json(res: dict, digits: int = 6) -> dict:
    """仅在序列化时格式化精度；内部计算不 round。"""
    out = {}
    for k, v in res.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else round(float(v), digits)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.ndarray):
            out[k] = [round(float(x), digits) for x in v if np.isfinite(x)]
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    s = sharpe_annual([0.01, 0.02, 0.03], ddof=0)
    assert abs(s - 38.884436) < 1e-3, s
    assert sharpe_annual([0.01] * 20) == 0.0
    assert check_future_function('x = close.shift(-1)', verbose=False) is False
    assert check_future_function('x = close.shift(1)', verbose=False) is True

    rng = np.random.default_rng(7)
    r = calculate_overfitting(rng.normal(0.0015, 0.012, 750),
                              rng.normal(0.0, 0.013, 500), n_trials=30, verbose=True)
    assert r["pbo"] > 0.5
    print(f"OK  version={__version__}  DSR={r['dsr']:.4f}  PBO={r['pbo']:.4f}")
