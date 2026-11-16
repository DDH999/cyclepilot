# -*- coding: utf-8 -*-
"""
M18 验证：数据源冗余与熔断
====================================================================
跑法:  python tests/test_m18_datasource.py   （不联网）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import datasource as DS

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M18 验证  datasource v{DS.__version__}")
print("#" + "=" * 69)

print("\n[1] 市场前缀")

cases = {"600519": "sh600519", "688981": "sh688981", "000001": "sz000001",
         "300750": "sz300750", "830799": "bj830799"}
for code, want in cases.items():
    got = DS.market_prefix(code)
    check(f"{code} → {want}", got == want, got)
check("裸数字补零为 6 位", DS.market_prefix("1") == "sz000001",
      DS.market_prefix("1"))

print("\n[2] 列名归一")

raw = pd.DataFrame({"日期": ["2025-01-02", "2025-01-03"],
                    "开盘": [1, 2], "最高": [2, 3], "最低": [0.5, 1.5],
                    "收盘": [1.5, 2.5], "成交量": [100, 200]})
n = DS.normalize_daily(raw)
check("标准列齐全", list(n.columns) == DS.STD_COLS, list(n.columns))
check("行数正确", len(n) == 2)
check("缺 amount 用 close×volume 估算",
      abs(n["amount"].iloc[0] - 1.5 * 100) < 1e-9, n["amount"].iloc[0])
check("日期已转 datetime", pd.api.types.is_datetime64_any_dtype(n["date"]))

en = pd.DataFrame({"date": ["2025-01-02"], "open": [1], "high": [2],
                   "low": [0.5], "close": [1.5], "volume": [100],
                   "amount": [150]})
check("英文列名也认", len(DS.normalize_daily(en)) == 1)

try:
    DS.normalize_daily(pd.DataFrame())
    check("空数据抛错", False)
except ValueError:
    check("空数据抛错", True)

try:
    DS.normalize_daily(pd.DataFrame({"日期": ["2025-01-02"], "收盘": [1]}))
    check("缺 OHLC 抛错", False)
except ValueError:
    check("缺 OHLC 抛错", True)

print("\n[3] 熔断器")

b = DS.Breaker(fail_threshold=3, cooldown=60)
check("初始放行", b.allow("s"))
b.record("s", False, "e1")
b.record("s", False, "e2")
check("2 次失败仍放行", b.allow("s"))
b.record("s", False, "e3")
check("3 次失败后熔断", not b.allow("s"))
snap = b.snapshot()["s"]
check("快照含连续失败数", snap["consecutive_fails"] == 3, snap)
check("快照标记熔断", snap["open"] is True)
check("快照含剩余冷却", snap["cooldown_left_s"] > 0, snap["cooldown_left_s"])

b2 = DS.Breaker(fail_threshold=2, cooldown=0.2)
b2.record("t", False, "x")
b2.record("t", False, "y")
check("冷却期内拒绝", not b2.allow("t"))
import time
time.sleep(0.25)
check("冷却结束恢复", b2.allow("t"))

b3 = DS.Breaker(fail_threshold=2)
b3.record("u", False, "x")
b3.record("u", True)
check("成功后计数清零", b3.snapshot()["u"]["consecutive_fails"] == 0)

print("\n[4] 多源降级")

DS.reset()


def boom(*a, **k):
    raise ValueError("主源挂了")


def good(*a, **k):
    return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=50),
                         "open": 1.0, "high": 1.1, "low": 0.9,
                         "close": 1.0, "volume": 1e6, "amount": 1e6})


res, src, errs = DS._try_sources("k1", [("primary", boom), ("backup", good)])
check("降级到备源", src == "backup", src)
check("备源数据可用", len(res) == 50)
check("记录失败明细", len(errs) == 1 and "主源挂了" in errs[0], errs)

DS.reset()
try:
    DS._try_sources("k2", [("a", boom), ("b", boom)])
    check("全挂抛错", False)
except RuntimeError as ex:
    check("全挂抛错", "所有数据源均失败" in str(ex))
    check("错误里含各源原因",
          "a:" in str(ex) and "b:" in str(ex), str(ex)[:80])

# 熔断后应跳过
DS.reset()
calls = []


def counting(*a, **k):
    calls.append(1)
    raise ValueError("x")


DS._try_sources.__globals__["BREAKER"] = DS.Breaker(fail_threshold=2, cooldown=60)
try:
    DS._try_sources("k3", [("c", counting), ("d", good)])
    DS._try_sources("k3", [("c", counting), ("d", good)])
    n_before = len(calls)
    DS._try_sources("k3", [("c", counting), ("d", good)])
    check("熔断后不再调用坏源", len(calls) == n_before, (n_before, len(calls)))
except Exception as ex:
    check("熔断测试未崩", False, str(ex)[:60])

print("\n[5] 对外接口")

DS.reset()
DS.SOURCE_STOCK = [("fake_bad", boom), ("fake_good", good)]
got = DS.get_stock_daily("600519", days=100)
check("get_stock_daily 返回 dict", isinstance(got, dict))
check("含降级标记", got["downgraded"] is True)
check("标记来源", got["source"] == "fake_good", got.get("source"))

st = DS.sources_status()
check("状态含版本", "version" in st)
check("状态含统计", "stats" in st and "stock_daily" in st["stats"])
check("状态含熔断", "breaker" in st)
check("统计记了降级次数",
      st["stats"]["stock_daily"]["fallbacks"] >= 1,
      st["stats"]["stock_daily"].get("fallbacks"))

print("\n[6] daily_job 接线（回归：曾只返回 close）")

import daily_job as DJ

_orig = DS.SOURCE_STOCK
DS.SOURCE_STOCK = [("fake", good)]
try:
    df = DJ.fetch_daily("600519", days=50)
    check("fetch_daily 返回非空", len(df) > 0, len(df))
    check("含 OHLC（曾缺失导致回测全缺数据）",
          {"open", "high", "low", "close"}.issubset(df.columns), list(df.columns))
    check("索引是日期", isinstance(df.index, pd.DatetimeIndex))
finally:
    DS.SOURCE_STOCK = _orig

print("\n[7] API")

import api_v2 as API
from fastapi.testclient import TestClient

cli = TestClient(API.app)
r = cli.get("/api/sources")
check("数据源接口 200", r.status_code == 200, r.status_code)
body = r.json()
check("返回 sources", "sources" in body)
check("返回 stats", "stats" in body)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  多源与熔断可用")
print("#" * 70)
