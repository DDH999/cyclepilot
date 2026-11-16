# -*- coding: utf-8 -*-
"""
M17 验证：策略代码 → 回测 → 六维体检
====================================================================
跑法:  python tests/test_m17_strategy_runner.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import strategy_runner as SR

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M17 验证  strategy_runner v{SR.__version__}")
print("#" + "=" * 69)


def synth(n=400, ret=0.0005, seed=7):
    rng = np.random.default_rng(seed)
    px = 10 * np.cumprod(1 + rng.normal(ret, 0.012, n))
    d = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"date": d, "close": px, "open": px * 0.999,
                         "high": px * 1.012, "low": px * 0.988,
                         "volume": 1e6, "amount": px * 1e6})


PANEL = {"600519": synth(seed=7), "000001": synth(seed=11)}

GOOD = '''
def make_weights(panel):
    import pandas as pd
    out = {}
    for code, df in panel.items():
        ma = df["close"].rolling(20).mean()
        out[code] = (df["close"] > ma).astype(float)
    w = pd.DataFrame(out)
    w.index = pd.to_datetime(panel[list(panel)[0]]["date"])
    return w.div(w.sum(axis=1).replace(0, 1), axis=0) * 0.9
'''

ZERO_BUG = '''
def make_weights(panel):
    import pandas as pd
    out = {}
    for code, df in panel.items():
        out[code] = (df["close"] > df["close"].rolling(20).mean()).astype(float)
    # 错误写法：RangeIndex 的 Series 配 DatetimeIndex → 重索引后全 NaN
    return pd.DataFrame(out, index=panel[list(panel)[0]]["date"])
'''

print("\n[1] 权重校验")

w_ok = SR.validate_weights(
    pd.DataFrame({"600519": 0.5, "000001": 0.4},
                 index=pd.bdate_range("2024-01-01", periods=30)),
    PANEL)
check("合法权重通过", w_ok["ok"] is True)

allzero = pd.DataFrame({"600519": 0.0, "000001": 0.0},
                       index=pd.bdate_range("2024-01-01", periods=30))
z = SR.validate_weights(allzero, PANEL)
check("全零权重被拒", z["ok"] is False)
check("全零给出索引提示",
      any("索引" in i for i in z["issues"]), z["issues"])

neg = pd.DataFrame({"600519": -0.2, "000001": 0.6},
                   index=pd.bdate_range("2024-01-01", periods=30))
n = SR.validate_weights(neg, PANEL)
check("负权重被截为 0", n["ok"] is True and float(n["cleaned"].min().min()) >= 0)
check("负权重有提示", any("负权重" in i for i in n["issues"]))

over = pd.DataFrame({"600519": 0.8, "000001": 0.8},
                    index=pd.bdate_range("2024-01-01", periods=30))
o = SR.validate_weights(over, PANEL)
check("超 1 被缩放", o["ok"] is True and float(o["cleaned"].sum(axis=1).max()) <= 1.0001)
check("超 1 有提示", any("缩放" in i for i in o["issues"]))

mismatch = pd.DataFrame({"999999": 0.5},
                        index=pd.bdate_range("2024-01-01", periods=30))
m = SR.validate_weights(mismatch, PANEL)
check("列不匹配被拒", m["ok"] is False)

check("非 DataFrame 被拒", SR.validate_weights(None, PANEL)["ok"] is False)
check("空矩阵被拒",
      SR.validate_weights(pd.DataFrame(), PANEL)["ok"] is False)

print("\n[2] 代码执行")

r1 = SR.run_from_code("raise ValueError('炸了')", PANEL)
check("exec 异常被捕获", r1["ok"] is False and r1["stage"] == "exec")
check("异常带上下文", "炸了" in r1["error"])

r2 = SR.run_from_code("x = 1", PANEL)
check("缺函数被识别", r2["ok"] is False and r2["stage"] == "contract")
check("给出契约示例", "make_weights" in (r2.get("hint") or ""))

r3 = SR.run_from_code("def make_weights(p):\n    return None\n", PANEL)
check("返回 None 被拒", r3["ok"] is False and r3["stage"] == "weights")

r4 = SR.run_from_code(ZERO_BUG, PANEL)
check("索引陷阱被拒（不静默空转）", r4["ok"] is False and r4["stage"] == "weights")
check("索引陷阱提示明确",
      "索引" in str(r4.get("error")), str(r4.get("error"))[:60])

print("\n[3] 正常回测")

r5 = SR.run_from_code(GOOD, PANEL)
check("回测成功", r5["ok"] is True, r5.get("error"))
check("产出净值序列", isinstance(r5.get("nav"), pd.Series) and len(r5["nav"]) > 100)
check("产出收益序列", isinstance(r5.get("returns"), pd.Series))
check("含夏普", "sharpe" in r5["metrics"])
check("含年换手", "ann_turnover" in r5["metrics"])
check("有实际成交（不再空转）",
      abs(float(r5["metrics"].get("total_return") or 0)) > 1e-6,
      r5["metrics"].get("total_return"))
check("含基准", r5.get("bench") is not None)

print("\n[4] 一体化体检")

rep = SR.full_checkup(GOOD, PANEL)
check("体检成功", rep.get("ok") is True, rep.get("error"))
check("含评分", rep.get("score") is not None)
check("含等级", rep.get("grade") in ("A", "B", "C", "D"), rep.get("grade"))
check("七个维度齐全（六维 + 基本面）",
      len(rep.get("dimensions") or []) == 7,
      [d["key"] for d in rep.get("dimensions") or []])
check("含回测指标", "backtest" in rep and rep["backtest"].get("days") > 100)
check("报告可渲染", "策略体检" in SR.format_report(rep))
check("报告含等级", str(rep.get("grade")) in SR.format_report(rep))

bad_rep = SR.full_checkup("x = 1", PANEL)
check("坏代码给出失败阶段", bad_rep.get("ok") is False and bad_rep.get("stage"))
check("失败报告可渲染", "失败于" in SR.format_report(bad_rep))

print("\n[5] API 接口")

import api_v2 as API
from fastapi.testclient import TestClient

client = TestClient(API.app)
_orig_build = SR.build_panel
try:
    SR.build_panel = lambda codes, start=None, end=None, loader=None: {
        "panel": PANEL, "missing": [], "n_days": 400}

    r = client.post("/api/checkup_code",
                    json={"code": GOOD, "codes": ["600519", "000001"]})
    check("接口 200", r.status_code == 200, r.status_code)
    body = r.json()
    check("返回 ok", body.get("ok") is True)
    check("返回报告", "report" in body and body["report"].get("score") is not None)
    check("返回文本", "策略体检" in body.get("text", ""))
    check("返回标的数", body.get("n_symbols") == 2)
    check("不含 pandas 对象", "nav" not in (body["report"].get("backtest") or {}))

    r400 = client.post("/api/checkup_code", json={"codes": ["600519"]})
    check("缺 code 返回 400", r400.status_code == 400, r400.status_code)
finally:
    SR.build_panel = _orig_build

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  代码→回测→体检 已打通")
print("#" * 70)
