# -*- coding: utf-8 -*-
"""
M12 验证：真实持仓台账
====================================================================
跑法:  python tests/test_m12_portfolio.py   （不联网，价格用注入的 loader）
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import portfolio as PF

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m12"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M12 验证  portfolio v{PF.__version__}")
print("#" + "=" * 69)

print("\n[1] 读取持仓")

f1 = TMP / "h1.csv"
f1.write_text("code,shares,cost\n600519,100,1400\n000001,2000,11.2\n", encoding="utf-8")
d1 = PF.load_holdings(f1)
check("读两只", len(d1["holdings"]) == 2, len(d1["holdings"]))
check("无现金时 cash=0", d1["cash"] == 0)
check("成本被解析", abs(d1["holdings"][0].cost - 1400) < 1e-9)
check("代码补零保留", d1["holdings"][1].code == "000001")

f2 = TMP / "h2.csv"
f2.write_text("# 注释\ncode,shares,cost\n600519,100,1400\nCASH,30000,\n"
              "# 又一行注释\n000001,500,\n", encoding="utf-8")
d2 = PF.load_holdings(f2)
check("忽略注释行", len(d2["holdings"]) == 2 and d2["warnings"] == [],
      d2["warnings"])
check("识别现金行", abs(d2["cash"] - 30000) < 1e-6)
check("成本可留空", d2["holdings"][1].cost is None)

f3 = TMP / "h3.csv"
f3.write_text("code,shares\n600519,100\n", encoding="utf-8")
check("成本列缺失也能读", PF.load_holdings(f3)["holdings"][0].cost is None)

f4 = TMP / "h4.csv"
f4.write_text("code,shares,cost\n600519,abc,1\n000001,100,11\n", encoding="utf-8")
d4 = PF.load_holdings(f4)
check("非法股数被跳过并告警", len(d4["holdings"]) == 1 and d4["warnings"], d4["warnings"])

f5 = TMP / "h5.csv"
f5.write_text("code,shares,cost\n12345,100,11\nAAPL,10,1", encoding="utf-8")
try:
    PF.load_holdings(f5)
    check("全非法代码 → 应报错", False)
except ValueError as ex:
    check("全非法代码 → 应报错", "持仓为空" in str(ex), str(ex))

f6 = TMP / "h6.csv"
f6.write_text("# 只有注释\n", encoding="utf-8")
try:
    PF.load_holdings(f6)
    check("空文件报错", False)
except ValueError:
    check("空文件报错", True)

try:
    PF.load_holdings(TMP / "nope.csv")
    check("不存在报错", False)
except FileNotFoundError:
    check("不存在报错", True)

f7 = TMP / "h7.csv"
f7.write_text("a,b\n1,2\n", encoding="utf-8")
try:
    PF.load_holdings(f7)
    check("缺必要列报错", False)
except ValueError:
    check("缺必要列报错", True)

print("\n[2] 市值与盈亏")

px = {"600519": 1453.0, "000001": 11.81}
mtm = PF.mark_to_market(d1, px)
check("估值成功", mtm["ok"] is True)
check("市值正确",
      abs(mtm["market_value"] - (100 * 1453 + 2000 * 11.81)) < 1e-6,
      round(mtm["market_value"], 2))
check("总资产 = 市值 + 现金", abs(mtm["total_value"] - mtm["market_value"]) < 1e-6)
check("权重和为 1", abs(sum(mtm["weights"].values()) - 1) < 1e-5)
check("含盈亏", mtm["total_pnl"] is not None)
check("盈亏率正确",
      abs(mtm["total_pnl_pct"] - (mtm["market_value"] / mtm["cost_value"] - 1)) < 1e-5,
      (mtm["total_pnl_pct"], round(mtm["market_value"] / mtm["cost_value"] - 1, 6)))
check("按市值降序", mtm["holdings"][0]["market_value"]
      >= mtm["holdings"][1]["market_value"])

mtm_cash = PF.mark_to_market(d2, px)
check("含现金时现金比例正确",
      abs(mtm_cash["cash_ratio"] - 30000 / mtm_cash["total_value"]) < 1e-5)
check("权重 + 现金 = 1",
      abs(sum(mtm_cash["weights"].values()) + mtm_cash["cash_ratio"] - 1) < 1e-5,
      round(sum(mtm_cash["weights"].values()) + mtm_cash["cash_ratio"], 8))

mtm_miss = PF.mark_to_market(d1, {"600519": 1453.0})
check("缺价不计入市值", abs(mtm_miss["market_value"] - 145300) < 1e-6)
check("缺价有记录", mtm_miss["missing"] == ["000001"], mtm_miss["missing"])
check("缺价有告警", any("取不到价格" in w for w in mtm_miss["warnings"]))

mtm_err = PF.mark_to_market(d1, {})
check("全部缺价返回 ok=False", mtm_err["ok"] is False)

check("报告可渲染", "持仓概览" in PF.format_report(mtm))

print("\n[3] 权重序列")

w = PF.weights_of(mtm)
check("权重和为 1", abs(w.sum() - 1) < 1e-5)
check("不含现金", "CASH" not in w.index)

w2 = PF.weights_of(mtm_cash, with_cash=True)
check("可含现金", "CASH" in w2.index)
check("含现金后和仍为 1", abs(w2.sum() - 1) < 1e-5)

empty = PF.weights_of({"ok": False})
check("不可估值返回空序列", len(empty) == 0)

print("\n[4] 价格缓存")

cache = TMP / "px.json"
cache.write_text(json.dumps({"600519": 1453.0, "000001": 11.81}), encoding="utf-8")
got = PF.fetch_last_prices(["600519", "000001"], loader=None, cache=cache)
check("缓存命中", abs(got["600519"] - 1453.0) < 1e-9)

calls = []


def fake_loader(code):
    calls.append(code)
    return {"600519": 1500.0, "000001": 12.0}.get(code, np.nan)


cache2 = TMP / "px2.json"
got2 = PF.fetch_last_prices(["600519", "000001"], loader=fake_loader, cache=cache2)
check("无缓存时调用 loader", len(calls) == 2, calls)
check("取到注入价格", abs(got2["600519"] - 1500.0) < 1e-9)
check("缓存已写入", cache2.exists())

cache3 = TMP / "px3.json"
got3 = PF.fetch_last_prices(["600519", "999999"], loader=fake_loader, cache=cache3)
check("取不到的标的不进入结果", "999999" not in got3, got3)

refresh_calls = []
PF.fetch_last_prices(["600519"], loader=lambda c: (refresh_calls.append(c), 1600.0)[1],
                     cache=cache2, refresh=True)
check("refresh 强制重取", len(refresh_calls) == 1)
check("重新取值生效",
      abs(json.loads(cache2.read_text(encoding="utf-8"))["600519"] - 1600.0) < 1e-9)

print("\n[5] 接口（进程内）")

import api_v2 as API
from fastapi.testclient import TestClient

client = TestClient(API.app)

_orig_holdings = ROOT / "holdings.csv"
_backup = None
if _orig_holdings.exists():
    _backup = _orig_holdings.read_text(encoding="utf-8")

try:
    # 用测试持仓替换
    (ROOT / "holdings.csv").write_text(
        "code,shares,cost\n600519,100,1400\n000001,2000,11.20\nCASH,50000,\n",
        encoding="utf-8")
    # 预置价格缓存，避免联网
    PF.PRICE_CACHE.write_text(
        json.dumps({"600519": 1453.0, "000001": 11.81}), encoding="utf-8")

    r = client.get("/api/portfolio")
    check("持仓接口 200", r.status_code == 200, r.status_code)
    body = r.json()
    check("返回 ok", body.get("ok") is True)
    check("含持仓概览", "portfolio" in body and body["portfolio"].get("ok"))
    check("含总资产", body["portfolio"]["total_value"] > 0)
    check("含报告文本", "持仓概览" in body.get("text", ""))

    r2 = client.get("/api/risk?turnover=12")
    check("风控接口 200", r2.status_code == 200, r2.status_code)
    if r2.status_code == 200:
        rk = r2.json()
        check("风控用真实持仓权重",
              rk["inputs"].get("weights_source") == "holdings",
              rk["inputs"].get("weights_source"))
        check("拥挤度可算", rk["crowding"].get("score") is not None)
finally:
    if _backup is not None:
        (ROOT / "holdings.csv").write_text(_backup, encoding="utf-8")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  持仓台账可用")
print("#" * 70)
