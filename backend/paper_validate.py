# -*- coding: utf-8 -*-
"""
主动交易验证流水线（M42）
====================================================
给「我就是要交易」的人一条**可执行的准入流程**。

不是保证赚钱，而是把「不该上真钱」的策略挡在外面。
五道闸门，**全部通过**才允许上真钱：

| 闸门 | 要求 | 为什么 |
|---|---|---|
| ① 样本量 | 交易日 ≥ 60 且 交易笔数 ≥ 30 | 太少说明不了任何事 |
| ② 扣成本后为正 | 年化净收益 > 0 | 毛收益正、净收益负是常态 |
| ③ 跑赢基准 | 超额年化 > 0 | 不然直接买指数 |
| ④ 稳健性 | walk-forward 正收益折 ≥ 60% | 不能靠某一段行情 |
| ⑤ 不靠运气 | 去掉最好的 1% 交易日后仍盈利 | 否则是几笔蒙对 |

**硬止损规则（最重要，且不可谈判）**
------------------------------------
连续 3 个自然月跑不赢「同仓位买宽基指数」→ **永久停手**。

这条规则的价值在于：它把「我可能没有能力」变成一个
**可执行、可审计、不依赖自知之明**的动作。

⚠ 诚实的边界
-------------
* 通过五道闸门**不等于**未来会赚钱，只表示"历史上没发现明显缺陷"
* 本工具**不提供**策略，只做准入判定
* 胜率优势很小的策略，扣成本后经常转负——这是最常见的死法
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["GATES", "check_sample", "check_net", "check_excess",
           "check_robust", "check_concentration", "validate",
           "format_report", "HARD_STOP_MONTHS"]

TD = 252
HARD_STOP_MONTHS = 3      # 连续跑输几个月就永久停手

GATES = [
    ("① 样本量", "交易日 ≥ 60 且 笔数 ≥ 30"),
    ("② 扣成本后为正", "年化净收益 > 0"),
    ("③ 跑赢基准", "相对宽基超额年化 > 0"),
    ("④ 稳健性", "walk-forward 正收益折 ≥ 60%"),
    ("⑤ 不靠运气", "去掉最好的 1% 交易日后仍盈利"),
]


def _annual_cost(turnover_year: float, cost_per_turn: float = 0.0015) -> float:
    """
    年换手 → **年化**成本。

    ⚠ 2026 修复：这里原来返回的是「每日成本」并直接拿去减年化收益，
    导致 500 次/年的换手只扣掉 0.3% 而不是 75%。
    与 `backtest.apply_slippage` 踩过的坑是同一类（日/年口径混用）。
    """
    return float(turnover_year) * float(cost_per_turn)


def check_sample(returns: pd.Series, n_trades: int | None = None) -> dict:
    n = int(len(pd.Series(returns).dropna()))
    ok_days = n >= 60
    if n_trades is None:
        trades_ok = True
        note = f"{n} 个交易日"
    else:
        trades_ok = int(n_trades) >= 30
        note = f"{n} 个交易日 / {int(n_trades)} 笔"
    return {"gate": "① 样本量", "pass": bool(ok_days and trades_ok),
            "n_days": n, "n_trades": n_trades, "note": note,
            "detail": ("样本足够" if (ok_days and trades_ok) else
                       f"样本不足（需 ≥60 日且 ≥30 笔），当前 {note}")}


def check_net(returns: pd.Series, turnover_year: float = 0.0,
              cost_per_turn: float = 0.0015) -> dict:
    r = pd.Series(returns).dropna()
    if len(r) < 20:
        return {"gate": "② 扣成本后为正", "pass": False,
                "note": "样本不足", "detail": "无法计算"}
    gross = float(r.mean()) * TD
    cost = _annual_cost(turnover_year, cost_per_turn)
    net = gross - cost
    return {"gate": "② 扣成本后为正", "pass": bool(net > 0),
            "gross_annual": round(gross, 6),
            "cost_annual": round(cost, 6),
            "net_annual": round(net, 6),
            "turnover_year": round(float(turnover_year), 3),
            "note": f"毛 {gross*100:.2f}% − 成本 {cost*100:.2f}% = 净 {net*100:.2f}%",
            "detail": ("扣成本后仍为正" if net > 0 else
                       "扣成本后转负——最常见的死法")}


def check_excess(returns: pd.Series, benchmark: pd.Series) -> dict:
    r = pd.Series(returns).dropna()
    b = pd.Series(benchmark).dropna()
    j = pd.concat([r.rename("s"), b.rename("b")], axis=1).dropna()
    if len(j) < 20:
        return {"gate": "③ 跑赢基准", "pass": False,
                "note": "重叠期不足", "detail": "无法比较"}
    sr = float(j["s"].mean()) * TD
    br = float(j["b"].mean()) * TD
    ex = sr - br
    return {"gate": "③ 跑赢基准", "pass": bool(ex > 0),
            "strategy_annual": round(sr, 6),
            "benchmark_annual": round(br, 6),
            "excess_annual": round(ex, 6),
            "n_overlap": int(len(j)),
            "note": f"策略 {sr*100:.2f}% vs 基准 {br*100:.2f}% = 超额 {ex*100:+.2f}%",
            "detail": ("有正超额" if ex > 0 else
                       "没有超额——不如直接买指数")}


def check_robust(returns: pd.Series, n_folds: int = 5) -> dict:
    r = pd.Series(returns).dropna()
    if len(r) < 100:
        return {"gate": "④ 稳健性", "pass": False,
                "note": "样本不足", "detail": "无法分折"}
    bounds = np.linspace(0, len(r), n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        seg = r.iloc[bounds[i]:bounds[i + 1]]
        if len(seg) < 10:
            continue
        ann = float(seg.mean()) * TD
        folds.append({"fold": i + 1, "annual": round(ann, 4),
                      "pass": bool(ann > 0)})
    if not folds:
        return {"gate": "④ 稳健性", "pass": False, "note": "无有效折",
                "detail": "无法分折"}
    pos = sum(1 for f in folds if f["pass"])
    rate = pos / len(folds)
    return {"gate": "④ 稳健性", "pass": bool(rate >= 0.6),
            "folds": folds, "positive_rate": round(rate, 3),
            "note": f"{pos}/{len(folds)} 折为正（{rate*100:.0f}%）",
            "detail": ("多数时间段有效" if rate >= 0.6 else
                       "只在部分时段有效——靠行情而不是靠能力")}


def check_concentration(returns: pd.Series) -> dict:
    """
    去掉最好的 1% 交易日后，是否仍盈利。

    早期版本用「前 10% 交易日贡献 <60%」当门槛，实测会把
    **正常策略全部否掉**——500 天里前 50 天贡献 240% 是常见现象
    （噪声本来就集中在少数几天）。改成剔除极端日后看剩余收益。
    """
    r = pd.Series(returns).dropna()
    if len(r) < 50:
        return {"gate": "⑤ 不靠运气", "pass": False,
                "note": "样本不足", "detail": "无法评估"}
    total = float(r.sum())
    if abs(total) <= 1e-12:
        return {"gate": "⑤ 不靠运气", "pass": False, "note": "总收益为 0",
                "detail": "无法评估"}
    if total < 0:
        return {"gate": "⑤ 不靠运气", "pass": False, "total": round(total, 6),
                "note": "总收益为负", "detail": "亏损的策略谈不上运气问题"}

    k = max(1, int(len(r) * 0.01))
    top_sum = float(r.nlargest(k).sum())
    without = total - top_sum
    share = top_sum / total
    ok = without > 0
    return {"gate": "⑤ 不靠运气", "pass": bool(ok),
            "top_n": k, "top_share": round(share, 4),
            "total": round(total, 6), "without_top": round(without, 6),
            "note": (f"去掉最好 {k} 天后收益 {without*100:+.2f}%"
                     f"（原 {total*100:+.2f}%，前 1% 贡献 {share*100:.0f}%）"),
            "detail": ("剔除极端日后仍盈利" if ok else
                       "收益全靠少数几天——大概率是运气")}


def validate(returns: pd.Series, benchmark: pd.Series | None = None,
             turnover_year: float = 0.0, n_trades: int | None = None,
             cost_per_turn: float = 0.0015) -> dict:
    """
    跑完五道闸门，给出 GO / NO-GO。
    """
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return {"ok": False, "reason": "有效交易日不足 5 天"}

    checks = [check_sample(r, n_trades)]
    checks.append(check_net(r, turnover_year, cost_per_turn))
    if benchmark is not None:
        checks.append(check_excess(r, benchmark))
    else:
        checks.append({"gate": "③ 跑赢基准", "pass": False,
                       "note": "未提供基准",
                       "detail": "必须和宽基指数对比，否则无法判断是否有意义"})
    checks.append(check_robust(r))
    checks.append(check_concentration(r))

    passed = [c for c in checks if c.get("pass")]
    all_pass = len(passed) == len(checks)
    failed = [c["gate"] for c in checks if not c.get("pass")]

    if all_pass:
        verdict = "GO：五道闸门全部通过，可以小仓位上真钱"
        advice = ("先用不超过总资金 10% 试运行；"
                  f"连续 {HARD_STOP_MONTHS} 个月跑不赢指数就永久停手")
    elif len(failed) == 1:
        verdict = f"NO-GO：{failed[0]} 未通过"
        advice = "先修这一项，别急着上真钱"
    else:
        verdict = f"NO-GO：{len(failed)} 项未通过（{', '.join(failed)}）"
        advice = "多项不达标，大概率不是策略问题而是运气问题"

    return {"ok": True, "n_days": int(len(r)),
            "gates": checks, "n_pass": len(passed), "n_gate": len(checks),
            "all_pass": all_pass, "failed": failed,
            "verdict": verdict, "advice": advice,
            "hard_stop_rule": (f"连续 {HARD_STOP_MONTHS} 个自然月"
                               f"跑不赢同仓位宽基指数 → 永久停手")}


def format_report(res: dict) -> str:
    L = ["=" * 70, "主动交易准入判定", "=" * 70]
    if not res.get("ok"):
        L.append(f"  无法判定：{res.get('reason')}")
        L.append("=" * 70)
        return "\n".join(L)

    L.append(f"  样本 {res['n_days']} 个交易日")
    L.append("-" * 70)
    for c in res["gates"]:
        mark = "通过" if c.get("pass") else "未过"
        L.append(f"  [{mark}] {c['gate']}")
        L.append(f"         {c.get('note', '')}")
        if not c.get("pass"):
            L.append(f"         → {c.get('detail', '')}")
    L.append("-" * 70)
    L.append(f"  {res['n_pass']}/{res['n_gate']} 通过")
    L.append(f"  {res['verdict']}")
    L.append(f"  {res['advice']}")
    L.append("-" * 70)
    L.append(f"  硬止损规则：{res['hard_stop_rule']}")
    L.append("=" * 70)
    L.append("  ⚠ 通过闸门 ≠ 未来会赚钱，只表示「历史上没发现明显缺陷」。")
    L.append("     本工具不提供策略，只做准入判定。")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=500)

    # 好的策略：有正漂移，收益分布均匀
    # 用「固定漂移 + 噪声」而不是纯随机，避免样本均值偶然偏负
    good = pd.Series(0.0008 + rng.normal(0, 0.004, 500), index=idx)
    bench = pd.Series(0.0003 + rng.normal(0, 0.004, 500), index=idx)
    g = validate(good, bench, turnover_year=3.0, n_trades=60)
    print(format_report(g))
    assert g["ok"] and g["all_pass"], g["failed"]

    # 差的策略：收益全来自一天
    bad = pd.Series(-0.0001 + rng.normal(0, 0.004, 500), index=idx)
    bad.iloc[100] = 0.35
    b = validate(bad, bench, turnover_year=20.0, n_trades=500)
    print("\n" + format_report(b))
    assert not b["all_pass"]
    assert any("运气" in x for x in b["failed"]) or any(
        "成本" in x for x in b["failed"]), b["failed"]

    # 样本不足
    s = validate(pd.Series(rng.normal(0.001, 0.01, 20)), bench[:20])
    assert not s["all_pass"] and any("样本量" in x for x in s["failed"]), s["failed"]
    print("\n样本不足正确拦截 ✓")
    print("OK v" + __version__)
