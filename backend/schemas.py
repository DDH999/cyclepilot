# -*- coding: utf-8 -*-
"""
模块间数据契约（Pydantic v2）
====================================================
把 M1 → M2 → M3 → M4 之间传递的 dict 用 Pydantic Model 固定下来，
避免某一层改字段名把下游打崩。

设计原则
--------
* `extra="allow"`：允许下游新增字段，不因多字段而失败（向后兼容）
* 只强制约束**下游真正依赖**的字段，其余可选
* 所有模型带 `contract` 版本号，便于版本管理
* 校验失败返回错误清单，**不抛异常**，方便流水线继续跑并记录问题

用法
----
    from schemas import StressResult, validate_payload
    ok, errs = validate_payload(StressResult, res_dict)
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__version__ = "0.1.0"
CONTRACT_VERSION = "1.0"
__all__ = [
    "CONTRACT_VERSION", "StressResult", "CrashResult", "OverfitResult",
    "PurityResult", "ItemDict", "ReportModel", "validate_payload",
]


class _Base(BaseModel):
    """所有契约的基类：允许多余字段，兼容演化。"""
    model_config = ConfigDict(extra="allow")
    contract: str = CONTRACT_VERSION


class StressResult(_Base):
    """滑点压力测试输出（stress_tester.apply_slippage）。"""
    n_days: int = Field(ge=0)
    survive: bool
    verdict: str
    annual_return_gross: float | None = None
    annual_return_net: float | None = None
    annual_cost_drag: float | None = None
    sharpe_gross: float | None = None
    sharpe_net: float | None = None
    max_drawdown_net: float | None = None
    slippage_rate: float | None = None
    turnover: float | None = None
    version: str | None = None


class CrashResult(_Base):
    """极端行情切片输出（stress_tester.test_extreme_crash）。"""
    label: str
    ok: bool
    days: int = 0
    period_return: float | None = None
    max_drawdown: float | None = None
    worst_day: float | None = None
    annual_return: float | None = None
    vs_full: float | None = None
    verdict: str | None = None
    message: str | None = None


class OverfitResult(_Base):
    """过拟合评估输出（validator.calculate_overfitting）。"""
    diagnosis_level: str
    diagnosis: str
    is_pass: bool
    needs_review: bool
    pbo: float | None = None
    pbo_raw: float | None = None
    pbo_clamped: bool | None = None
    dsr: float | None = None
    sharpe_is: float | None = None
    sharpe_oos: float | None = None
    verdict: str | None = None
    missing_policy: str | None = None
    version: str | None = None


class PurityResult(_Base):
    """策略纯度输出（purity_tester.purity_report）。"""
    ok: bool
    verdict: str
    score: float | None = None
    reason_tag: str | None = None
    flags: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    deductions: dict = Field(default_factory=dict)


class ItemDict(_Base):
    """单个检测项的裁决结果。"""
    name: str
    status: str
    score: float
    reason: str | None = None
    note: str = ""
    value: float | None = None


class ReportModel(_Base):
    """六维体检总报告（scoring.full_report）。"""
    verdict: str
    grade: str
    score: float
    items: list[ItemDict] = Field(default_factory=list)
    radar: dict = Field(default_factory=dict)
    dimensions: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    counts: dict = Field(default_factory=dict)
    policy: str = ""
    summary: str = ""
    audit: list[dict] = Field(default_factory=list)
    override_applied: bool = False
    grade_capped: bool = False
    version: str | None = None


def validate_payload(model: type[BaseModel], payload: dict) -> tuple[bool, list[str]]:
    """
    校验一个 dict 是否符合契约。

    返回 (是否通过, 错误清单)。**不抛异常**，便于流水线记录并继续。
    """
    if not isinstance(payload, dict):
        return False, [f"期望 dict，收到 {type(payload).__name__}"]
    try:
        model.model_validate(payload)
        return True, []
    except Exception as ex:  # pydantic.ValidationError 及其它
        errs: list[str] = []
        for e in getattr(ex, "errors", lambda: [])():
            loc = ".".join(str(x) for x in e.get("loc", ()))
            errs.append(f"{loc}: {e.get('msg')}")
        return False, errs or [str(ex)]
