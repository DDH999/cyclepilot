# -*- coding: utf-8 -*-
"""
数据质量体检（M10）
====================================================
"垃圾进、垃圾出"——所以在评估策略之前，先评估数据本身。

检查项
------
1. **日期完整性**：应有交易日 vs 实际有记录的日子，缺多少
2. **重复日期**：同一天出现多条
3. **非正价格 / 异常值**：≤0、极大极小
4. **冻结行情**：连续多日数值完全相同（停牌 / 前向填充残留）
5. **异常跳变**：单日 |收益| 超阈值（拆股未复权 / 错误价格）
6. **陈旧数据**：最后一个日期距今天数
7. **缺失率**：NaN 占比

输出：问题清单 + 0~100 分 + 等级 + 是否可用于评估。
**数据不合格时，后续结论一律不可信**——这是整条流程的第一道闸门。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["DataCfg", "DEFAULTS", "is_level_series", "check_series",
           "format_report"]


class DataCfg:
    """阈值（经验值，可覆盖）。"""

    def __init__(self, min_days: int = 60,
                 max_missing_pct: float = 5.0,
                 max_frozen_run: int = 20,
                 jump_threshold: float = 0.35,
                 max_stale_days: int = 10,
                 max_nan_pct: float = 2.0):
        self.min_days = min_days
        self.max_missing_pct = max_missing_pct
        self.max_frozen_run = max_frozen_run
        self.jump_threshold = jump_threshold
        self.max_stale_days = max_stale_days
        self.max_nan_pct = max_nan_pct


DEFAULTS = DataCfg()


def is_level_series(s: pd.Series) -> bool:
    """
    净值/价格 vs 收益率：用「正数占比 + 变异系数」判别。

    不能用 `min > 0`：序列里夹一个脏的负值就会让整条净值序列被误判成收益，
    从而跳过非正值检查——这正是漏检的来源。改用 95% 以上为正即可。
    """
    if len(s) == 0:
        return False
    mean_abs = abs(float(s.mean()))
    if mean_abs < 1e-12:
        return False
    cv = float(s.std(ddof=0)) / mean_abs
    pos_ratio = float((s > 0).mean())
    return bool(pos_ratio > 0.95 and cv < 0.5)


def _to_series(x) -> pd.Series:
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    return s.astype("float64").sort_index() if len(s) else s


def _expected_bdays(start, end) -> int:
    return int(len(pd.bdate_range(start, end)))


def check_series(data, cfg: DataCfg = DEFAULTS,
                 name: str = "序列") -> dict:
    """
    对一条净值/收益序列做数据质量体检。

    返回 dict
    --------
    ok           数据是否可用于评估
    score        0~100（越高越干净）
    level        干净 / 可用 / 存疑 / 不可用
    issues       [{kind, level, msg}]
    metrics      各项量化指标
    """
    s = _to_series(data)
    issues: list[dict] = []
    metrics: dict = {"n_raw": int(len(s))}

    if len(s) < 2:
        return {"ok": False, "score": 0.0, "level": "不可用", "name": name,
                "issues": [{"kind": "empty", "level": "high", "msg": "数据为空或仅 1 个点"}],
                "metrics": metrics}

    # ── 日期索引
    has_date = isinstance(s.index, pd.DatetimeIndex)
    if not has_date:
        issues.append({"kind": "no_date", "level": "high",
                       "msg": "索引不是日期，无法做时间完整性检查（后续时间序列检验全部受限）"})

    dup = int(s.index.duplicated().sum()) if has_date else 0
    if dup:
        issues.append({"kind": "duplicate", "level": "high",
                       "msg": f"存在 {dup} 个重复日期"})

    nan_cnt = int(s.isna().sum())
    nan_pct = nan_cnt / len(s) * 100 if len(s) else 0
    if nan_pct > cfg.max_nan_pct:
        issues.append({"kind": "nan", "level": "mid",
                       "msg": f"缺失值占 {nan_pct:.1f}%（阈值 {cfg.max_nan_pct}%）"})

    clean = s.dropna()
    if len(clean) < cfg.min_days:
        issues.append({"kind": "too_short", "level": "high",
                       "msg": f"有效样本仅 {len(clean)} 天，少于 {cfg.min_days}"})

    # ── 日期完整性
    missing_pct = 0.0
    if has_date and len(clean) >= 2:
        start, end = clean.index[0], clean.index[-1]
        expect = _expected_bdays(start, end)
        missing = max(expect - len(clean), 0)
        missing_pct = missing / expect * 100 if expect else 0.0
        metrics.update({"start": str(start.date()), "end": str(end.date()),
                        "expected_bdays": expect, "missing_bdays": missing})
        if missing_pct > cfg.max_missing_pct:
            issues.append({"kind": "missing_days", "level": "mid",
                           "msg": f"交易日缺失 {missing_pct:.1f}%（缺 {missing} 天）"})
        stale = (date.today() - end.date()).days
        metrics["stale_days"] = stale
        if stale > cfg.max_stale_days:
            issues.append({"kind": "stale", "level": "mid",
                           "msg": f"最后一条数据距今 {stale} 天，可能未更新"})

    # ── 数值合理性
    level_series = is_level_series(clean)
    metrics["kind"] = "净值/价格" if level_series else "收益"
    if level_series:
        non_pos = int((clean <= 0).sum())
        if non_pos:
            issues.append({"kind": "non_positive", "level": "high",
                           "msg": f"存在 {non_pos} 个非正数值"})
        rets = clean.pct_change(fill_method=None).dropna()
    else:
        rets = clean
        big = int((clean.abs() > 10).sum())
        if big:
            issues.append({"kind": "absurd_return", "level": "high",
                           "msg": f"存在 {big} 个 |值|>10 的收益（疑似百分数未折算）"})

    # ── 冻结行情
    if level_series:
        frozen_run, cur = 0, 1
        arr = clean.to_numpy()
        for i in range(1, len(arr)):
            cur = cur + 1 if arr[i] == arr[i - 1] else 1
            frozen_run = max(frozen_run, cur)
        metrics["max_frozen_run"] = frozen_run
        if frozen_run >= cfg.max_frozen_run:
            issues.append({"kind": "frozen", "level": "high",
                           "msg": f"最长连续 {frozen_run} 天数值不变（停牌或已前向填充）"})

    # ── 异常跳变
    jumps = rets[rets.abs() > cfg.jump_threshold]
    metrics["jump_count"] = int(len(jumps))
    if len(jumps):
        issues.append({"kind": "jump", "level": "mid",
                       "msg": f"存在 {len(jumps)} 次单日跳变 >{cfg.jump_threshold*100:.0f}%"
                              f"（拆股未复权？）"})

    # ── 零收益占比（收益序列）
    zero_pct = 0.0
    if not level_series and len(clean):
        zero_pct = float((clean == 0).sum()) / len(clean) * 100
        metrics["zero_return_pct"] = round(zero_pct, 2)
        if zero_pct > 30:
            issues.append({"kind": "zero_heavy", "level": "mid",
                           "msg": f"{zero_pct:.0f}% 的日收益为 0（大量停牌或未交易）"})

    metrics.update({"n_clean": int(len(clean)), "nan_pct": round(nan_pct, 2),
                    "missing_pct": round(missing_pct, 2),
                    "dup": dup})

    # ── 评分
    score = 100.0
    high = sum(1 for x in issues if x["level"] == "high")
    mid = sum(1 for x in issues if x["level"] == "mid")
    score -= high * 30 + mid * 10
    score = float(np.clip(score, 0, 100))

    if high:
        level, ok = "不可用", False
    elif score >= 85:
        level, ok = "干净", True
    elif score >= 70:
        level, ok = "可用", True
    else:
        level, ok = "存疑", False

    return {"ok": ok, "score": round(score, 2), "level": level, "name": name,
            "issues": issues, "metrics": metrics}


def format_report(res: dict) -> str:
    L = ["=" * 66, f"数据质量体检：{res.get('name', '')}", "=" * 66]
    m = res.get("metrics", {})
    L.append(f"  结论: {res.get('level')}   得分 {res.get('score')}/100"
             f"   可用于评估: {'是' if res.get('ok') else '否'}")
    L.append(f"  样本: {m.get('n_clean', 0)} 天"
             + (f"（{m.get('start')} ~ {m.get('end')}）" if m.get("start") else "")
             + f"   类型: {m.get('kind', '?')}")
    if m.get("expected_bdays"):
        L.append(f"  日期完整性: 应有 {m['expected_bdays']} 个交易日，"
                 f"缺 {m['missing_bdays']} 天（{m.get('missing_pct', 0):.1f}%）")
    if m.get("stale_days") is not None:
        L.append(f"  数据新鲜度: 距今 {m['stale_days']} 天")
    L.append("-" * 66)
    if res.get("issues"):
        L.append("  问题清单:")
        for x in res["issues"]:
            tag = {"high": "[严重]", "mid": "[中等]", "low": "[提示]"}[x["level"]]
            L.append(f"    {tag} {x['msg']}")
    else:
        L.append("  未发现问题")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    idx = pd.bdate_range("2024-01-02", periods=300)
    rng = np.random.default_rng(7)
    good = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 300)), index=idx)
    r1 = check_series(good, name="干净净值")
    assert r1["ok"] and r1["score"] >= 85, r1

    # 制造问题：冻结 30 天 + 跳变 + 缺 10 天
    bad = good.copy()
    bad.iloc[100:130] = bad.iloc[100]
    bad.iloc[200] = bad.iloc[199] * 1.8
    bad = bad.drop(bad.index[[10, 20, 30, 40, 50, 60, 70, 80, 90, 95]])
    r2 = check_series(bad, name="脏数据")
    assert not r2["ok"], r2
    kinds = {x["kind"] for x in r2["issues"]}
    assert "frozen" in kinds and "jump" in kinds, kinds
    assert r2["score"] < r1["score"]
    print(format_report(r1))
    print(format_report(r2))
    print(f"OK v{__version__}")
