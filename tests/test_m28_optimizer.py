# -*- coding: utf-8 -*-
"""
M28 验证：马科维茨稳定性 / 凯利仓位上限
====================================================================
跑法:  python tests/test_m28_optimizer.py   （不联网，用合成数据）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import optimizer as OP

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M28 验证  optimizer v{OP.__version__}")
print("#" + "=" * 69)

print("\n[1] 净值 → 收益")


def mk_nav(mean, vol, n=1500, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(mean, vol, n)),
                     index=pd.bdate_range("2015-01-02", periods=n))


navs = {"A": mk_nav(0.0004, 0.010, seed=7),
        "B": mk_nav(0.0005, 0.016, seed=11),
        "C": mk_nav(0.0003, 0.008, seed=13)}
R = OP.to_returns(navs)
check("行数少 1（净值→收益）", R.shape[0] == 1499, R.shape)
check("数值是小数不是价格", float(R.abs().max().max()) < 0.5,
      float(R.abs().max().max()))
check("列名保持", list(R.columns) == ["A", "B", "C"])
check("无 NaN", not R.isna().any().any())

R2 = OP.to_returns(R)
check("传入已是收益时原样返回", abs(R2.shape[0] - R.shape[0]) <= 1, R2.shape)

print("\n[2] 马科维茨权重")

w = OP.markowitz_weights(R)
check("权重非负", all(v >= 0 for v in w.values()), w)
check("权重合计 1", abs(sum(w.values()) - 1) < 1e-3, sum(w.values()))
check("返回三个资产", len(w) == 3)

print("\n[3] 稳定性（关键：单位不能混）")

st = OP.markowitz_stability(R, window=500, step=60)
check("可算", st["ok"] is True, st.get("reason"))
check("有多个窗口", st["n_windows"] >= 5, st["n_windows"])
check("摆幅以小数存储", all(v <= 1.0001 for v in st["swing"].values()),
      {k: round(v, 3) for k, v in st["swing"].items()})
check("同时提供百分点", "avg_swing_pp" in st, st.get("avg_swing_pp"))
check("百分点 = 小数×100",
      abs(st["avg_swing_pp"] - st["avg_swing"] * 100) < 0.1,   # 都是 1~4 位小数
      (st["avg_swing_pp"], st["avg_swing"]))
check("unstable 与百分点比较一致",
      st["unstable"] == (st["avg_swing_pp"] > OP.STABILITY_LIMIT),
      (st["unstable"], st["avg_swing_pp"], OP.STABILITY_LIMIT))
check("结论文字与 unstable 一致",
      ("不稳定" in st["verdict"]) == st["unstable"],
      st["verdict"][:30])
check("指出最不稳资产", st["worst_asset"] in w, st["worst_asset"])

# 极端：三个完全相同序列 → 摆幅应接近 0（优化器无唯一解但不该乱跳）
same = {c: mk_nav(0.0004, 0.01, seed=7) for c in ("X", "Y", "Z")}
Rs = OP.to_returns(same)
st_same = OP.markowitz_stability(Rs, window=500, step=60)
check("相同序列摆幅很小",
      st_same["ok"] and st_same["avg_swing_pp"] < 20,
      st_same.get("avg_swing_pp"))

check("样本不足返回 ok=False",
      OP.markowitz_stability(R.iloc[:400])["ok"] is False)

print("\n[4] 凯利")

ks = OP.kelly_sizing(R)
check("可算", ks["ok"] is True)
check("三个资产都有", len(ks["assets"]) == 3)
for c, v in ks["assets"].items():
    check(f"{c} 含满凯利", "kelly_full" in v)
    check(f"{c} 1/4 凯利 = 满凯利/4",
          abs(v["kelly_quarter"] - v["kelly_full"] * 0.25) < 1e-3,  # 字段保留 4 位
          (v["kelly_quarter"], v["kelly_full"]))
    check(f"{c} 含年化与波动", v["annual_vol"] > 0)

check("组合凯利存在", ks["portfolio_kelly_full"] is not None)
check("标注隐含杠杆资产", isinstance(ks["assets_implying_leverage"], list),
      ks["assets_implying_leverage"])
check("结论提到杠杆", "杠杆" in ks["verdict"] or "上限" in ks["verdict"],
      ks["verdict"])

# 低收益高波动 → 凯利为负
bad = {"X": pd.Series(100 * np.cumprod(
    1 + np.random.default_rng(5).normal(-0.0004, 0.02, 1500)),
    index=pd.bdate_range("2015-01-02", periods=1500))}
Rb = OP.to_returns(bad)
kb = OP.kelly_sizing(Rb)
check("负期望时凯利为负",
      kb["assets"]["X"]["kelly_full"] < 0,
      kb["assets"]["X"]["kelly_full"])

print("\n[5] 综合报告")


def mk_nav2(mean, vol, seed, n=1200):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(mean, vol, n)),
                     index=pd.bdate_range("2018-01-02", periods=n))


navs2 = {"甲": mk_nav2(0.0005, 0.011, 7), "乙": mk_nav2(0.0004, 0.015, 11),
         "丙": mk_nav2(0.0003, 0.008, 13)}
R3 = OP.to_returns(navs2)
rep = OP.optimize_report(R3)
check("报告可生成", rep["ok"] is True, rep.get("reason"))
check("含马科维茨段", "markowitz" in rep)
check("含凯利段", "kelly" in rep)
check("含四方案对比", len(rep["schemes"]) == 4, len(rep["schemes"]))
check("给出推荐", rep["recommend"] in ("等权", "逆波动", "最小方差", "风险平价"),
      rep["recommend"])
check("给出理由", len(rep["reason"]) > 5, rep["reason"])
check("含凯利仓位上限",
      rep["kelly_position_cap"] is None or 0 <= rep["kelly_position_cap"] <= 1,
      rep["kelly_position_cap"])

# 推荐逻辑：不稳定时必须推等权
if rep["markowitz"]["unstable"]:
    check("不稳定时推荐等权", rep["recommend"] == "等权", rep["recommend"])

check("报告可渲染", "马科维茨" in OP.format_report(rep))
check("报告含免责", "期望收益无法可靠估计" in OP.format_report(rep))
check("失败报告可渲染",
      "无法评估" in OP.format_report({"ok": False, "reason": "数据不足"}))

print("\n[6] CLI / API")

import api_v2 as API
from fastapi.testclient import TestClient
import overseas as OS

_orig = OS.index_history
fake_navs = {k: mk_nav2(0.0004, 0.012, i) for i, k in enumerate(("sp500", "ndx"), 1)}
OS.index_history = lambda key, days=20000, refresh=False: {
    "ok": True, "key": key, "name": key.upper(),
    "series": fake_navs.get(key, mk_nav2(0.0004, 0.012, 9)),
    "source": "fake", "errors": []}
try:
    cli = TestClient(API.app)
    r = cli.get("/api/optimizer?keys=sp500,ndx")
    check("接口 200", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        b = r.json()
        check("返回报告", b.get("ok") is True)
        check("含文本", "马科维茨" in b.get("text", ""))
        check("含推荐", b["report"].get("recommend") is not None)

    r2 = cli.get("/api/optimizer?keys=onlyone")
    check("单指数返回 400", r2.status_code == 400, r2.status_code)
finally:
    OS.index_history = _orig

import quantpilot as QP
check("CLI 注册 optimize", "optimize" in QP.COMMANDS)
pp = QP.build_parser()
check("解析 --keys", pp.parse_args(["optimize", "--keys", "sp500,ftse"]).keys
      == "sp500,ftse")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  优化器结论可验证")
print("#" * 70)
