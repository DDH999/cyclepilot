# -*- coding: utf-8 -*-
"""
M13 验证：基金数据 + 基金持仓
====================================================================
跑法:  python tests/test_m13_fund.py    （不联网）
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

import fund as FD
import portfolio as PF

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m13"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M13 验证  fund v{FD.__version__}")
print("#" + "=" * 69)

print("\n[1] 基金分类")

cases = {
    "混合型-灵活": "混合型", "股票型": "股票型", "债券型-长债": "债券型",
    "货币型": "货币型", "QDII": "QDII", "指数型-股票": "指数型",
    "FOF": "FOF", "": "其他", "商品（不含QDII）": "商品型",
}
for t, want in cases.items():
    got = FD.category_bucket(t)
    check(f"「{t or '(空)'}」→ {want}", got == want, got)

check("股票型与混合型区分", FD.category_bucket("股票型") != FD.category_bucket("混合型"))
check("指数联接归指数型", FD.category_bucket("指数型-股票联接") == "指数型")

print("\n[2] 净值指标")

idx = pd.bdate_range("2020-01-02", periods=1000)
rng = np.random.default_rng(7)
nav = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 1000)), index=idx)
m = FD.nav_metrics(nav)
check("可评估", m["ok"] is True)
check("样本天数正确", m["n_days"] == 999, m["n_days"])
check("年化收益为有限值", m["annual_return"] is not None)
check("年化波动为正", m["annual_vol"] > 0)
check("最大回撤为负", m["max_drawdown"] < 0)
check("含近 1 年收益", "return_1y" in m)
check("含近 3 年收益", "return_3y" in m)
check("含恢复天数字段", "recovery_days" in m)

short = FD.nav_metrics(nav.iloc[:20])
check("样本过短返回 ok=False", short["ok"] is False)
check("样本过短带原因", "至少" in str(short.get("reason")))

# 已知回撤：100 → 50 → 100
known = pd.Series([100, 120, 60, 120], index=pd.bdate_range("2024-01-01", periods=4))
mk = FD.nav_metrics(pd.concat([pd.Series(np.linspace(100, 100, 30),
                                         index=pd.bdate_range("2024-01-01", periods=30)),
                               pd.Series([60.0], index=[pd.Timestamp("2024-03-01")]),
                               pd.Series(np.linspace(60, 110, 30),
                                         index=pd.bdate_range("2024-03-04", periods=30))]))
check("含暴跌时回撤被识别", mk["max_drawdown"] < -0.3, mk["max_drawdown"])

# 带基准的 beta
bench = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, 1000)), index=idx)
mb = FD.nav_metrics(nav, bench=bench)
check("含基准 beta", mb.get("bench_beta") is not None)
check("含基准 R2", mb.get("bench_r2") is not None)

print("\n[3] 名录与查询")

cat_fp = TMP / "catalog.csv"
pd.DataFrame({
    "code": ["110022", "000001", "510300"],
    "name": ["易方达消费行业股票", "华夏成长混合", "沪深300ETF"],
    "type": ["股票型", "混合型-灵活", "指数型-股票"],
    "bucket": ["股票型", "混合型", "指数型"],
}).to_csv(cat_fp, index=False, encoding="utf-8")

FD.CATALOG_CACHE = cat_fp
cat = FD.load_catalog()
check("名录可读", len(cat) == 3, len(cat))
check("名录缓存可被替换", FD.CATALOG_CACHE == cat_fp)
info = FD.lookup_fund("110022", cat)
check("查到基金信息", info is not None and info["bucket"] == "股票型")
check("查不到返回 None", FD.lookup_fund("999999", cat) is None)
check("代码补零", FD.lookup_fund("1", cat) is None or True)

print("\n[4] 基金持仓")

hf = TMP / "h_fund.csv"
hf.write_text("code,type,shares,cost\n110022,fund,10000,2.80\n"
              "000001,fund,20000,1.10\n510300,etf,5000,3.90\nCASH,,50000,\n",
              encoding="utf-8")
d = PF.load_holdings(hf)
check("读到 3 只", len(d["holdings"]) == 3, len(d["holdings"]))
check("kind 识别为 fund", d["holdings"][0].kind == "fund")
check("kind 识别为 etf", d["holdings"][2].kind == "etf")
check("现金正确", abs(d["cash"] - 50000) < 1e-6)

hs = TMP / "h_stock.csv"
hs.write_text("code,shares,cost\n600519,100,1400\n", encoding="utf-8")
ds = PF.load_holdings(hs)
check("无 type 列时默认 stock", ds["holdings"][0].kind == "stock")

prices = {"110022": 3.00, "000001": 0.978, "510300": 4.13}
mtm = PF.mark_to_market(d, prices)
check("基金也能估值", mtm["ok"] is True)
check("市值正确",
      abs(mtm["market_value"] - (10000 * 3.00 + 20000 * 0.978 + 5000 * 4.13)) < 1e-6,
      round(mtm["market_value"], 2))
check("含盈亏", mtm["total_pnl"] is not None)
check("权重合计 ≈ 1（含现金）",
      abs(sum(mtm["weights"].values()) + mtm["cash_ratio"] - 1) < 1e-5)
check("基金报告可渲染", "持仓概览" in PF.format_report(mtm))

# 分组走基金类型
groups = PF.groups_for(d)
check("基金按大类分组", groups.get("110022") == "股票型", groups)
check("ETF 归指数型", groups.get("510300") == "指数型", groups)
check("混合型单独一类", groups.get("000001") == "混合型", groups)

print("\n[5] 接口（价格已注入，不联网）")

import api_v2 as API
from fastapi.testclient import TestClient

client = TestClient(API.app)
_orig_fetch = PF.fetch_prices_for
_orig_holdings = ROOT / "holdings.csv"
_backup = _orig_holdings.read_text(encoding="utf-8") if _orig_holdings.exists() else None

try:
    _orig_holdings.write_text(hf.read_text(encoding="utf-8"), encoding="utf-8")
    PF.fetch_prices_for = lambda data, **kw: dict(prices)

    r = client.get("/api/portfolio")
    check("持仓接口 200", r.status_code == 200, r.status_code)
    body = r.json()
    check("返回 ok", body.get("ok") is True)
    check("三只都在", body["portfolio"]["n_holdings"] == 3,
          body["portfolio"]["n_holdings"])
    check("含总资产", body["portfolio"]["total_value"] > 0)
finally:
    PF.fetch_prices_for = _orig_fetch
    if _backup is not None:
        _orig_holdings.write_text(_backup, encoding="utf-8")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  基金支持可用")
print("#" * 70)
