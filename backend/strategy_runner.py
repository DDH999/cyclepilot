# -*- coding: utf-8 -*-
"""
策略运行器（M24）
====================================================
打通「策略代码 → 真实回测 → 六维体检」这条链。

之前体检平台吃的是**已经算好的收益序列**，你得先自己回测。
现在可以直接交代码：

    from strategy_runner import full_checkup
    rep = full_checkup(MY_CODE, codes=["600519", "000001"])

用户代码契约（只写一个函数）
----------------------------
    def make_weights(panel):
        # panel: {code: DataFrame 列含 date/close/open/high/low/volume/amount}
        # 返回: DataFrame(index=日期, columns=代码) 目标权重，行和 ≤ 1
        ...

回测走的是真实 A 股规则（backtest.py）：
T+1、涨跌停（10%/20%）、印花税 0.05%、佣金万 2.5（最低 5 元）、过户费、滑点、冲击成本。

⚠ 安全边界（必须知道）
----------------------
本项目**不做沙箱隔离**。用户代码在同一个 Python 进程里 exec，
可以读写文件、访问网络。只适合「自己写、自己跑」的场景；
如果将来要做多用户在线服务，必须换到子进程 + 容器。
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["CONTRACT", "build_panel", "validate_weights", "run_from_code",
           "full_checkup", "format_report"]

CONTRACT = """
def make_weights(panel):
    '''返回目标权重矩阵。

    panel: dict[str, DataFrame]
        DataFrame 列: date / close / open / high / low / volume / amount
        行: 每个交易日一行
    返回: DataFrame(index=日期, columns=代码)
        权重 0~1，每行之和 ≤ 1（不足部分视为现金）

    ⚠ 索引陷阱：信号 Series 默认是 RangeIndex，
      如果直接 pd.DataFrame({code: series}, index=日期)，
      pandas 会按索引对齐，结果全是 NaN → 全 0 → 回测空转。
      正确写法：
          w = pd.DataFrame(out)          # 先用默认索引
          w.index = pd.to_datetime(df["date"])   # 再赋日期
    '''
    return None
"""


# ══════════════════════════════════════════════════════════
def build_panel(codes: list[str], start: str | None = None,
                end: str | None = None, loader=None) -> dict:
    """
    取行情构造回测面板。loader(code, days) -> DataFrame（缺省用 daily_job）。

    取不到数据的标的**直接跳过**并记录，不填充假数据。
    """
    panel: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    if loader is None:
        def loader(code, days=None):  # type: ignore
            import daily_job as DJ
            return DJ.fetch_daily(code, days=days or 800)

    for c in codes:
        try:
            df = loader(c)
            if df is None or len(df) < 60:
                missing.append(c)
                continue
            df = df.copy()
            if "date" not in df.columns:
                df = df.reset_index()
                if "date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"])
            need = {"open", "high", "low", "close"}
            if not need.issubset(df.columns):
                missing.append(c)
                continue
            if "volume" not in df.columns:
                df["volume"] = 1e6
            if "amount" not in df.columns:
                df["amount"] = df["close"] * df["volume"]
            df = df.sort_values("date").reset_index(drop=True)
            if start:
                df = df[df["date"] >= pd.Timestamp(start)]
            if end:
                df = df[df["date"] <= pd.Timestamp(end)]
            if len(df) >= 60:
                panel[c] = df.reset_index(drop=True)
            else:
                missing.append(c)
        except Exception:
            missing.append(c)

    if not panel:
        raise RuntimeError(f"没有可用行情（{len(codes)} 只全部失败）")
    return {"panel": panel, "missing": missing,
            "n_days": min(len(v) for v in panel.values())}


def validate_weights(w, panel: dict, max_sum: float = 1.0001) -> dict:
    """校验用户返回的权重矩阵，返回 {ok, cleaned, issues}。"""
    issues = []
    if w is None or not isinstance(w, pd.DataFrame):
        return {"ok": False, "cleaned": None,
                "issues": ["make_weights 必须返回 DataFrame"]}
    if len(w) == 0:
        return {"ok": False, "cleaned": None, "issues": ["权重矩阵为空"]}

    codes = list(panel.keys())
    cols = [c for c in w.columns if c in codes]
    if not cols:
        return {"ok": False, "cleaned": None,
                "issues": [f"权重列与行情代码无交集（行情 {codes[:4]}…）"]}

    extra = [c for c in w.columns if c not in codes]
    if extra:
        issues.append(f"忽略 {len(extra)} 个无行情代码的列：{extra[:5]}")

    out = w[cols].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    neg = int((out < 0).sum().sum())
    if neg:
        issues.append(f"{neg} 个负权重被截为 0（不支持做空）")
        out = out.clip(lower=0.0)

    rs = out.sum(axis=1)
    over = rs[rs > max_sum]
    if len(over):
        issues.append(f"{len(over)} 行权重和 > 1，已等比缩放")
        scale = np.where(rs > max_sum, 1.0 / rs, 1.0)
        out = out.mul(scale, axis=0)

    if float(np.abs(out.to_numpy()).sum()) <= 1e-12:
        # 最常见的静默失败：信号 Series 是 RangeIndex，
        # 构造 DataFrame 时传了 DatetimeIndex，pandas 重索引后全变 NaN → 全 0。
        # 回测会空转、收益 0%，却一路"成功"。必须明确拒绝。
        return {"ok": False, "cleaned": out,
                "issues": ["权重全为 0。常见原因：信号与日期的索引没对齐。"
                           "建议写法：w = pd.DataFrame(out); "
                           "w.index = pd.to_datetime(df['date'])"]}

    if len(out) < 20:
        issues.append(f"权重只有 {len(out)} 行，太短")
        return {"ok": False, "cleaned": out, "issues": issues}

    return {"ok": True, "cleaned": out, "issues": issues}


# ══════════════════════════════════════════════════════════
def run_from_code(code_str: str, panel: dict, capital: float = 1_000_000.0,
                  rebalance: str = "M", max_positions: int = 30,
                  costs=None, timeout_s: float = 60.0) -> dict:
    """
    执行用户代码并回测。

    不沙箱（见模块文档）。异常一律捕获并返回，不向外抛。
    """
    from backtest import Backtest, Costs, benchmark_ew

    t0 = time.perf_counter()
    ns = {"pd": pd, "np": np, "__builtins__": __builtins__}
    try:
        exec(compile(code_str, "<strategy>", "exec"), ns)  # noqa: S102
    except Exception as ex:
        return {"ok": False, "stage": "exec",
                "error": f"代码执行失败：{ex}",
                "traceback": traceback.format_exc(limit=3)}

    fn = ns.get("make_weights")
    if not callable(fn):
        return {"ok": False, "stage": "contract",
                "error": "缺少 make_weights(panel) 函数",
                "hint": CONTRACT.strip()}

    try:
        t1 = time.perf_counter()
        raw = fn(panel)
        build_s = time.perf_counter() - t1
    except Exception as ex:
        return {"ok": False, "stage": "make_weights",
                "error": f"make_weights 抛错：{ex}",
                "traceback": traceback.format_exc(limit=3)}

    val = validate_weights(raw, panel)
    if not val["ok"]:
        return {"ok": False, "stage": "weights",
                "error": "；".join(val["issues"]), "issues": val["issues"]}

    weights = val["cleaned"]
    try:
        bt = Backtest(panel, capital=capital, costs=costs or Costs(), verbose=False)
        res = bt.run(weights, rebalance=rebalance, max_positions=max_positions)
    except Exception as ex:
        return {"ok": False, "stage": "backtest",
                "error": f"回测失败：{ex}",
                "traceback": traceback.format_exc(limit=3)}

    if res.get("error"):
        return {"ok": False, "stage": "backtest", "error": res["error"]}

    elapsed = time.perf_counter() - t0
    if elapsed > timeout_s:
        # 已经跑完，只做提示
        pass

    try:
        bench = benchmark_ew(panel)
    except Exception:
        bench = None

    return {
        "ok": True,
        "metrics": {k: v for k, v in res.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "nav": res.get("nav"),
        "returns": res.get("returns"),
        "turnover_series": res.get("turnover"),
        "weights": weights,
        "bench": bench,
        "issues": val["issues"],
        "elapsed_s": round(elapsed, 2),
        "build_weights_s": round(build_s, 3),
    }


# ══════════════════════════════════════════════════════════
def full_checkup(code_str: str, panel: dict, capital: float = 1_000_000.0,
                 rf_annual: float = 0.0,
                 fundamentals_ticker: str | None = None) -> dict:
    """
    一条龙：回测 → 六维体检 → 综合报告。

    六个维度全部用真实计算结果，任何一维失败都降级为 N/A（不编数）。
    """
    import purity_tester as PT
    import scoring as SC
    import stress_tester as ST
    import validator as V

    run = run_from_code(code_str, panel, capital=capital)
    if not run.get("ok"):
        return {"ok": False, "stage": run.get("stage"),
                "error": run.get("error"), "hint": run.get("hint")}

    rets = run["returns"]
    weights = run["weights"]
    nav = run["nav"]
    bench = run.get("bench")

    # 1) 未来函数
    findings = V.scan_code(code_str)
    code_ok = len(findings) == 0

    # 2) 过拟合（前后半段）
    overfit = None
    try:
        h = len(rets) // 2
        overfit = V.calculate_overfitting(rets.iloc[:h], rets.iloc[h:],
                                          n_trials=1, verbose=False)
    except Exception:
        overfit = None

    # 3) 成本（用回测出来的年换手）
    ann_turn = float(run["metrics"].get("ann_turnover") or 0.0)
    ann_turn = max(ann_turn, 0.0)
    slip = None
    try:
        slip = ST.apply_slippage(pd.DataFrame({"ret": rets}),
                                 slippage_rate=0.0005,
                                 turnover=ann_turn,
                                 turnover_period="year", verbose=False)
    except Exception:
        slip = None

    # 4) 抗压（取最差窗口）
    crash = None
    try:
        suite = ST.run_crash_suite(pd.DataFrame({"ret": rets}), verbose=False)
        ok = [x for x in suite if isinstance(x, dict) and x.get("ok")]
        if ok:
            crash = max(ok, key=lambda x: (x.get("vs_full") or 0))
    except Exception:
        crash = None

    # 5) 纯度（把月度权重铺到日频）
    purity = None
    try:
        wd = weights.reindex(nav.index, method="ffill").fillna(0.0)
        purity = PT.purity_report(weights=wd, returns=rets, bench=bench,
                                  verbose=False)
    except Exception:
        purity = None

    # 6) 稳健性（换调仓频率看夏普波动）
    sens = None
    try:
        sharpes = []
        for rb in ("W", "M", "Q"):
            r = run_from_code(code_str, panel, capital=capital, rebalance=rb)
            if r.get("ok"):
                sharpes.append(float(r["metrics"].get("sharpe") or np.nan))
        sharpes = [x for x in sharpes if np.isfinite(x)]
        if len(sharpes) >= 2:
            sens = {"min": round(min(sharpes), 4), "max": round(max(sharpes), 4),
                    "median": round(float(np.median(sharpes)), 4)}
    except Exception:
        sens = None

    # 可选：带入美股基本面（第七维）。失败不影响主结论，只记警告。
    fund, fund_warn = None, None
    if fundamentals_ticker:
        try:
            import quality as QL
            fund = QL.quality_report(str(fundamentals_ticker).upper())
        except Exception as ex:
            fund_warn = f"基本面取数失败：{str(ex)[:120]}"

    rep = SC.full_report(code_ok=code_ok, findings=findings, overfit=overfit,
                         slippage=slip, crash=crash, purity=purity,
                         sensitivity=sens, fundamentals=fund)

    rep.update({
        "ok": True,
        "fundamentals": fund,
        "fundamentals_warning": fund_warn,
        "backtest": run["metrics"],
        "weight_issues": run.get("issues"),
        "cost_turnover": round(ann_turn, 3),
        "elapsed_s": run.get("elapsed_s"),
    })
    return rep


def format_report(rep: dict) -> str:
    L = ["=" * 68, "策略体检（代码 → 回测 → 六维）", "=" * 68]
    if not rep.get("ok"):
        L.append(f"  失败于 {rep.get('stage')}：{rep.get('error')}")
        if rep.get("hint"):
            L.append("  期望的代码契约：")
            for ln in str(rep["hint"]).splitlines():
                L.append("    " + ln)
        L.append("=" * 68)
        return chr(10).join(L)

    bt = rep.get("backtest") or {}
    L.append(f"  回测 {bt.get('days', '-')} 天   期末净值 {bt.get('final_nav', 0):,.0f}")
    L.append(f"  累计收益 {bt.get('total_return', 0) * 100:+.2f}%   "
             f"年化 {bt.get('cagr', 0) * 100:+.2f}%   "
             f"夏普 {bt.get('sharpe', 0):.2f}")
    L.append(f"  最大回撤 {bt.get('max_drawdown', 0) * 100:.2f}%   "
             f"年换手 {rep.get('cost_turnover')}   "
             f"持仓数 {bt.get('n_positions', '-')}")
    L.append("-" * 68)
    L.append(f"  评分 {rep.get('score')}   等级 {rep.get('grade')}")
    for d in rep.get("dimensions", []):
        s = "-" if d["score"] is None else format(d["score"], ".0f")
        L.append("  " + f"{d['label']:<10}{s:>5}  {d['status']:<10}"
                 + (d.get("note") or "")[:36])
    if rep.get("grade_caps"):
        L.append("-" * 68)
        for c in rep["grade_caps"]:
            L.append("  [封顶] " + c)
    if rep.get("weight_issues"):
        L.append("-" * 68)
        for w in rep["weight_issues"]:
            L.append("  [权重] " + w)
    L.append("=" * 68)
    return chr(10).join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    def synth(n=400, ret=0.0005, seed=7):
        rng = np.random.default_rng(seed)
        px = 10 * np.cumprod(1 + rng.normal(ret, 0.012, n))
        d = pd.bdate_range("2022-01-03", periods=n)
        return pd.DataFrame({
            "date": d, "close": px, "open": px * 0.999,
            "high": px * 1.012, "low": px * 0.988,
            "volume": 1e6, "amount": px * 1e6})

    panel = {"600519": synth(seed=7), "000001": synth(seed=11)}

    GOOD = """
def make_weights(panel):
    import pandas as pd
    out = {}
    for code, df in panel.items():
        ma = df['close'].rolling(20).mean()
        out[code] = (df['close'] > ma).astype(float)
    w = pd.DataFrame(out)
    w.index = pd.to_datetime(panel[list(panel)[0]]['date'])
    return w.div(w.sum(axis=1).replace(0, 1), axis=0) * 0.9
"""
    rep = full_checkup(GOOD, panel)
    print(format_report(rep))
    assert rep["ok"], rep

    BAD = "def make_weights(panel):\n    return None\n"
    bad = full_checkup(BAD, panel)
    assert not bad["ok"] and bad["stage"] == "weights", bad
    print(format_report(bad))

    NOFN = "x = 1\n"
    nf = full_checkup(NOFN, panel)
    assert not nf["ok"] and nf["stage"] == "contract", nf
    print(f"缺函数提示正常：{nf['error']}")
    print(f"OK v{__version__}")
