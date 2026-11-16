# -*- coding: utf-8 -*-
"""
第 3 层验收：风险预算与风控对回测的影响
====================================================
同一因子（low_vol top10 月调仓），只改**权重构建**与**总仓位控制**：

| 变体 | 权重 | 总仓位 |
|---|---|---|
| A 基线 | 等权 | 恒 100% |
| B 逆波动 | 1/σ 加权 | 恒 100% |
| C 等风险贡献 | ERC | 恒 100% |
| D 逆波动 + 波动率目标 | 1/σ | cap = 15%/已实现波动 |
| E = D + 回撤熔断 | 1/σ | 再乘 0/1 闸门 |

⚠ 波动率目标与熔断都基于**同池等权组合**的历史（T 日及之前），
  属于"用可观测的基准状态调仓"，不是用未来收益，无前视。
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import attribution as AT
import backtest_strict as BS
import param_walkforward as PW
import risk_budget as RB
import risk_control as RC
import strategy_lab as SL

START, END = "2019-01-02", "2023-12-29"


def ann_of(nav):
    s = pd.Series(nav).dropna()
    if len(s) < 2:
        return None
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)


def main():
    raw = {c: __import__("store").read_bars("cn", c)
           for c in __import__("store").list_codes("cn")}
    raw = {c: d for c, d in raw.items() if len(d) >= 100}
    panel, info = PW.align_panel(raw, min_cover=0.8, max_missing=0.05)
    if not info.get("ok"):
        print("面板对齐失败:", info)
        return
    print(f"原始 {info['n_in']} 只 → 对齐后 {info['n_out']} 只   "
          f"{info['start']} ~ {info['end']}")
    global START, END
    START, END = info["start"], info["end"]

    CL = pd.DataFrame({c: d.set_index("date")["close"]
                       for c, d in panel.items()}).sort_index()
    ew_ret = CL.pct_change(fill_method=None).mean(axis=1)
    ew_nav = (1 + ew_ret.fillna(0)).cumprod()

    sc = SL.build_scores(panel, "low_vol", 20)
    LB, TN, RBAL = 20, 10, 20

    L = RC.Limits(max_weight=0.15, max_industry=0.35, max_leverage=1.0,
                  dd_stop=0.15, dd_cooldown=20)

    cap_vol = RB.vol_target_cap(ew_ret, target_vol=0.18, window=20,
                                max_leverage=1.0)
    gate = RC.circuit_breaker(ew_nav, L)
    cap_mix = (cap_vol * gate).clip(0, 1)

    def _apply_limits(w):
        """对每个调仓日套事前风控（单票 15%）。"""
        if w is None or len(w) == 0:
            return w
        out = w.copy()
        for dt in out.index:
            r = RC.check_pre_trade(out.loc[dt], limits=L)
            out.loc[dt] = r["adjusted"].reindex(out.columns).fillna(0.0)
        return out

    variants = {
        "A 等权(基线)": (SL.factor_weights(sc, "low_vol", TN, RBAL), None),
        "B 逆波动": (RB.apply_risk_budget(sc, CL, TN, "inv_vol", 20, RBAL), None),
        "C 等风险贡献": (RB.apply_risk_budget(sc, CL, TN, "erc", 20, RBAL), None),
        "D 逆波动+波动目标": (RB.apply_risk_budget(sc, CL, TN, "inv_vol", 20, RBAL),
                        cap_vol),
        "E D+回撤熔断": (RB.apply_risk_budget(sc, CL, TN, "inv_vol", 20, RBAL),
                     cap_mix),
    }

    rows = []
    results = {}
    for name, (w, cap) in variants.items():
        if w is None or len(w) == 0:
            print(f"  {name}: 无权重，跳过")
            continue
        w = _apply_limits(w)
        try:
            r = BS.StrictBacktest(panel, capital=1e6).run(
                w, max_positions=TN, cap=cap)
        except Exception as ex:
            print(f"  {name}: 失败 {type(ex).__name__}: {ex}")
            continue
        nav = r["nav"]
        results[name] = r
        dd = (nav / nav.cummax() - 1).min()
        rows.append({
            "变体": name,
            "年化%": round(r["annual"] * 100, 2),
            "夏普": round(r["sharpe"], 3),
            "最大回撤%": round(dd * 100, 2),
            "Calmar": round(r["annual"] / abs(dd), 2) if dd < 0 else None,
            "换手%": round(float(r.get("avg_turnover") or 0) * 100, 1),
            "投资率%": round(float(r.get("avg_invested") or 0) * 100, 1),
            "空仓日%": round(float(r.get("flat_ratio") or 0) * 100, 1),
        })

    df = pd.DataFrame(rows)
    print("\n=== 对照表 ===")
    print(df.to_string(index=False))

    # ── 归因（用 E 或 A）
    best = "E D+回撤熔断" if "E D+回撤熔断" in results else "A 等权(基线)"
    r = results[best]
    print(f"\n=== 归因（{best}）===")
    contrib = r.get("contrib_by_code") or {}
    st = AT.by_stock(contrib, 1e6, min_pp=0.01)
    try:
        ind = BS.industry_map()
    except Exception:
        ind = {}
    grp = AT.by_group(contrib, ind, 1e6) if ind else None
    nav = r["nav"]
    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
    bench = AT.vs_benchmark(nav, ew_nav)
    recon = AT.recombination_check(contrib, total_ret, 1e6)
    print(AT.format_cross_report(st, grp, bench, recon))
    mon = RC.monitor(nav, None, ind, L)
    print("事中监控:", {k: v for k, v in mon.items() if k != "last_exposure"})

    out = Path(__file__).resolve().parent.parent / "reports"
    out.mkdir(exist_ok=True)
    (out / "risk_layer3.md").write_text(
        "# 第 3 层验收：风险预算与风控\n\n" + df.to_markdown(index=False) +
        "\n\n## 归因\n\n```\n" + AT.format_cross_report(st, grp, bench, recon) +
        "\n```\n", encoding="utf-8")
    print(f"\n[报告] {out / 'risk_layer3.md'}")


if __name__ == "__main__":
    main()
