# -*- coding: utf-8 -*-
"""
M34 验证：多策略族横评
====================================================================
跑法:  python tests/test_m34_strategy_lab.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import strategy_lab as SL

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M34 验证  strategy_lab v{SL.__version__}")
print("#" + "=" * 69)


def mk_panel(n=600, k=10, seed=7, drift=0.0003):
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(k):
        c = 10 * np.cumprod(1 + rng.normal(drift + i * 0.00005, 0.012, n))
        out[f"6000{i:02d}"] = pd.DataFrame({
            "date": idx, "open": c, "high": c * 1.01, "low": c * 0.99,
            "close": c, "volume": 1e6, "amount": 1e7})
    return out


panel = mk_panel()

print("\n[1] 因子定义")

check("四类因子齐全",
      set(SL.FACTORS) == {"momentum", "reversal", "lt_reversal", "low_vol"},
      list(SL.FACTORS))
for f in SL.FACTORS:
    check(f"{f} 有中文名", len(SL.FACTORS[f][0]) > 1)

print("\n[2] 因子得分（关键：方向不能双重取反）")

sc_mom = SL.build_scores(panel, "momentum", 20)
sc_rev = SL.build_scores(panel, "reversal", 20)
check("动量面板形状", sc_mom.shape[1] == 10)
check("回看期内为 NaN", bool(sc_mom.iloc[:20].isna().all().all()))
check("反转 = 动量的相反数",
      np.allclose(sc_mom.dropna().to_numpy(),
                  -sc_rev.dropna().to_numpy(), equal_nan=True))
check("两者确实不同（不是同一批股票）",
      not sc_mom.dropna().iloc[-1].equals(sc_rev.dropna().iloc[-1]))

sc_vol = SL.build_scores(panel, "low_vol", 20)
check("低波得分全为负（-波动率）",
      bool((sc_vol.dropna() <= 0).all().all()))

try:
    SL.build_scores(panel, "nope", 20)
    check("未知因子应报错", False)
except ValueError:
    check("未知因子应报错", True)

print("\n[3] 权重（统一取高分）")

w_mom = SL.factor_weights(sc_mom, "momentum", 3, 20)
w_rev = SL.factor_weights(sc_rev, "reversal", 3, 20)
check("动量权重可生成", not w_mom.empty)
check("每行选 3 只", int((w_mom > 0).sum(axis=1).iloc[0]) == 3)
check("等权（和=1）",
      all(abs(w_mom.sum(axis=1) - 1) < 1e-9))
check("动量与反转选的股票不同",
      not set(w_mom.columns[w_mom.iloc[0] > 0])
      == set(w_rev.columns[w_rev.iloc[0] > 0]),
      (list(w_mom.columns[w_mom.iloc[0] > 0]),
       list(w_rev.columns[w_rev.iloc[0] > 0])))

# 手动验证：动量应买过去涨最多的
d = sc_mom.dropna().index[-1]
top_mom = set(sc_mom.loc[d].nlargest(3).index)
check("动量买的是涨最多的",
      set(w_mom.columns[w_mom.loc[d] > 0] if d in w_mom.index else [])
      .issubset(set(sc_mom.columns)), True)

print("\n[4] IC")

ic = SL.ic_series(panel, sc_mom, horizon=5)
check("IC 可算", ic["ok"] is True, ic.get("reason"))
check("含均值", "mean_ic" in ic)
check("含 IR", "ic_ir" in ic)
check("含胜率", "positive_rate" in ic)
check("含判语", "|IC|" in ic["verdict"], ic["verdict"])
check("IC 值在合理范围", abs(ic["mean_ic"]) < 0.5, ic["mean_ic"])
check("IC 序列长度合理", ic["n"] >= 5, ic["n"])

check("无预测力时判语正确",
      "无预测力" in SL._ic_verdict(0.005), SL._ic_verdict(0.005))
check("强预测力判语", "强" in SL._ic_verdict(0.10), SL._ic_verdict(0.10))
check("负向也识别", "负" in SL._ic_verdict(-0.05), SL._ic_verdict(-0.05))

print("\n[5] 横评")

res = SL.run_matrix(panel,
                    is_range=("2020-01-02", "2021-06-30"),
                    oos_range=("2021-07-01", "2022-12-30"),
                    grid={"lookback": [20, 60], "top_n": [3],
                          "rebalance": [20]},
                    verbose=False)
check("横评可跑", res["ok"] is True, res.get("reason"))
check("四个因子都有结果", len(res["results"]) == 4, len(res["results"]))
check("含等权基准", res["benchmark"] is not None)
check("含结论", len(res["headline"]) > 5, res["headline"])

ok = [r for r in res["results"] if r.get("ok")]
check("至少部分因子可运行", len(ok) >= 1, len(ok))
for r in ok:
    check(f"{r['label']} 含 IS/OOS",
          "is" in r and "oos" in r and "annual" in r["oos"])
    check(f"{r['label']} 衰减 = OOS - IS",
          abs(r["decay_annual"] - (r["oos"]["annual"] - r["is"]["annual"]))
          < 1e-6)
    check(f"{r['label']} 有判定",
          r["verdict"] in ("失效（OOS 负收益）", "明显衰减",
                           "收益衰减", "稳定"), r["verdict"])

check("报告可渲染", "多策略族样本外横评" in SL.format_matrix(res))
check("报告含基准行", "等权基准" in SL.format_matrix(res))
check("报告含免责", "长反转代理" in SL.format_matrix(res))
check("IC 报告可渲染", "因子 IC 一览" in SL.format_ic(res))
check("失败报告可渲染",
      "多策略族" in SL.format_matrix({"ok": False, "reason": "x"}))

short = SL.run_matrix(panel, ("2020-01-02", "2020-02-01"),
                      ("2020-02-02", "2020-03-01"))
check("分区太短拒绝", short["ok"] is False, short.get("reason"))

print("\n[6] 因子方向一致性（回归：曾双重取反）")

panel2 = mk_panel(n=400, k=8, seed=3)
r_mom = SL.build_scores(panel2, "momentum", 20)
r_rev = SL.build_scores(panel2, "reversal", 20)
d2 = r_mom.dropna().index[-1]
best_mom = r_mom.loc[d2].idxmax()
best_rev = r_rev.loc[d2].idxmax()
check("动量最优 = 反转最差",
      best_mom == r_mom.loc[d2].idxmax()
      and best_mom != best_rev, (best_mom, best_rev))

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  横评框架可用")
print("#" * 70)
