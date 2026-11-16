# -*- coding: utf-8 -*-
"""
M21 验证：zzshare 数据源集成
====================================================================
跑法:  python tests/test_m21_zzshare.py   （不联网，用假模块替换）
"""
import shutil
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import datasource as DS
import zz_source as ZZ

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m21"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M21 验证  zz_source v{ZZ.__version__}")
print("#" + "=" * 69)

print("\n[1] 可用性检测")

info = ZZ.available()
check("返回 dict", isinstance(info, dict))
check("含 available 字段", "available" in info)
check("本机已安装 zzshare", info["available"] is True, info)
check("版本号非空", bool(info.get("version")), info.get("version"))

print("\n[2] 代码格式互转")

cases = {"600519": "600519.SH", "688981": "688981.SH",
         "000001": "000001.SZ", "300750": "300750.SZ",
         "830799": "830799.BJ"}
for c, want in cases.items():
    got = ZZ.to_ts_code(c)
    check(f"{c} → {want}", got == want, got)
    check(f"{want} → {c}", ZZ.to_symbol(want) == c, ZZ.to_symbol(want))
check("裸数字补零", ZZ.to_ts_code("1") == "000001.SZ", ZZ.to_ts_code("1"))

print("\n[3] 日线归一（假数据）")

_orig_zz = sys.modules.get("zzshare")


def install_fake(day_df=None, val_df=None, fail_daily=False):
    fake = types.ModuleType("zzshare")

    def daily(ts_code=None, start_date=None, end_date=None, **kw):
        if fail_daily:
            raise RuntimeError("模拟上游失败")
        return day_df

    def finance_valuation(date_value=None, **kw):
        return val_df

    fake.daily = daily
    fake.finance_valuation = finance_valuation
    sys.modules["zzshare"] = fake
    return fake


RAW = pd.DataFrame({
    "ts_code": ["600519.SH"] * 5,
    "trade_date": ["20261106", "20261109", "20261110", "20261111", "20261112"],
    "open": [1400, 1405, 1410, 1415, 1420],
    "high": [1420, 1425, 1430, 1435, 1440],
    "low": [1390, 1395, 1400, 1405, 1410],
    "close": [1410, 1415, 1420, 1425, 1430],
    "vol": [1000, 1100, 1200, 1300, 1400],
})
install_fake(day_df=RAW)

df = ZZ.stock_daily("600519", days=10)
check("返回非空", len(df) == 5, len(df))
check("标准列齐全",
      {"date", "open", "high", "low", "close", "volume", "amount"}.issubset(df.columns),
      list(df.columns))
check("日期已解析", pd.api.types.is_datetime64_any_dtype(df["date"]))
check("按日期升序", df["date"].is_monotonic_increasing)
check("缺 amount 时用 close×volume 估算",
      abs(df["amount"].iloc[0] - 1410 * 1000 * 100) < 1e-6, df["amount"].iloc[0])

install_fake(day_df=pd.DataFrame())
try:
    ZZ.stock_daily("600519")
    check("空数据抛错", False)
except ValueError:
    check("空数据抛错", True)

install_fake(fail_daily=True)
try:
    ZZ.stock_daily("600519")
    check("上游异常抛错", False)
except RuntimeError:
    check("上游异常抛错", True)

print("\n[4] 全市场估值（假数据）")

n = 200
VAL = pd.DataFrame({
    "code": [f"{i:06d}.SZ" for i in range(n)],
    "pe_ratio": np.linspace(5, 95, n),
    "pb_ratio": np.linspace(0.5, 9.5, n),
    "pcf_ratio": np.linspace(-20, 180, n),
    "market_cap": np.linspace(1e9, 1e12, n),
})
install_fake(day_df=RAW, val_df=VAL)

got = ZZ.market_valuation("2026-11-10")
check("返回 ok", got["ok"] is True, got.get("reason"))
check("含日期", got["date"] == "2026-11-10", got.get("date"))
check("行数正确", got["n"] == n, got["n"])
check("含 symbol 列", "symbol" in got["df"].columns)
check("代码转成 6 位", got["df"]["symbol"].iloc[0] == "000000",
      got["df"]["symbol"].iloc[0])

snap = ZZ.valuation_snapshot("2026-11-10")
check("快照可算", snap["ok"] is True)
check("含 PE 中位数", snap["PE"]["median"] is not None, snap.get("PE"))
check("中位数在区间内", 5 <= snap["PE"]["median"] <= 95, snap["PE"]["median"])
check("含 P25/P75", snap["PB"]["p25"] < snap["PB"]["p75"])
check("统计非正 PE",
      snap["PE"]["n_negative"] >= 0, snap.get("PE"))
check("格式化报告可渲染", "zzshare 数据源" in ZZ.format_report(info, snap))

install_fake(val_df=pd.DataFrame())
empty = ZZ.market_valuation("2026-11-10", max_lookback_days=0)
check("无数据时 ok=False", empty["ok"] is False)
check("记录尝试过的日期", len(empty["tried"]) >= 1, empty.get("tried"))

print("\n[5] 作为数据源降级")

install_fake(day_df=RAW)
DS.reset()
orig = DS.SOURCE_STOCK


def boom(code, days=800):
    raise RuntimeError("东财挂了")


DS.SOURCE_STOCK = [("eastmoney", boom), ("zzshare", DS._src_zzshare_daily)]
try:
    res = DS.get_stock_daily("600519", days=10)
    check("降级到 zzshare", res["source"] == "zzshare", res["source"])
    check("标记降级", res["downgraded"] is True)
    check("记录失败原因", any("东财挂了" in e for e in res["errors"]), res["errors"])
    check("数据可用", len(res["df"]) == 5, len(res["df"]))
finally:
    DS.SOURCE_STOCK = orig

st = DS.sources_status()
check("健康状态含 zzshare",
      "zzshare" in st["sources"]["stock_daily"], st["sources"])
check("健康状态标记 zzshare 可用",
      st["sources"].get("zzshare_available") is True)

print("\n[6] 行业映射（假数据 + 缓存）")

fake2 = sys.modules["zzshare"]
fake2.plates_list = lambda plate_type=14: pd.DataFrame({
    "plate_code": ["881101", "881102"],
    "plate_name": ["种植业与林业", "养殖业"]})
fake2.plates_stocks = lambda plate_type=14, plate_code=None, date=None: pd.DataFrame({
    "code": ["600519.SH", "000001.SZ"] if plate_code == "881101"
            else ["300750.SZ"]})

cache = TMP / "industry.csv"
got2 = ZZ.industry_map("2026-11-10", cache=cache, refresh=True)
check("行业映射成功", got2["ok"] is True, got2.get("reason"))
check("映射数量正确", got2["n"] == 3, got2["n"])
check("600519 归到种植业",
      got2["mapping"].get("600519") == "种植业与林业", got2["mapping"].get("600519"))
check("缓存文件已写", cache.exists())

got3 = ZZ.industry_map("2026-11-10", cache=cache)
check("第二次走缓存", got3["source"] == "cache", got3.get("source"))
check("缓存结果一致", got3["n"] == got2["n"])

# 回归 1：zzshare 真实返回 list[dict]，不是 DataFrame
fake2.plates_list = lambda plate_type=14: [
    {"plate_code": "881101", "plate_name": "白酒"},
    {"plate_code": "881102", "plate_name": "银行"}]
check("plates_list 返回 list 也能处理",
      len(ZZ.plates(14)) == 2, len(ZZ.plates(14)))

# 回归 2：成分股代码列名是 stock_code，不是 code/ts_code
fake2.plates_stocks = lambda plate_type=14, plate_code=None, date=None: [
    {"stock_code": "600519", "stock_name": "贵州茅台"},
    {"stock_code": "000001", "stock_name": "平安银行"}]
lm = ZZ.industry_map("2026-11-10", cache=TMP / "lm.csv", refresh=True)
check("stock_code 字段被识别", lm["ok"] and lm["n"] == 2, lm.get("n"))
check("映射值正确", lm["mapping"].get("600519") == "白酒", lm.get("mapping"))

fake2.plates_stocks = lambda **kw: pd.DataFrame()
failed = ZZ.industry_map("2026-11-10", cache=TMP / "none.csv", refresh=True)
check("全空时 ok=False", failed["ok"] is False)

print("\n[7] API")

# 前面把桩改坏了（用于测异常路径），这里恢复成可用状态再测接口
# 同时把「真实缓存路径」重定向到临时目录 —— 否则测试会往 .cache 写假数据，
# 污染下一次真实构建（已经发生过一次）。
ZZ.INDUSTRY_CACHE = TMP / "api_industry.csv"
install_fake(day_df=RAW, val_df=VAL)
sys.modules["zzshare"].plates_list = lambda plate_type=14: pd.DataFrame({
    "plate_code": ["881101", "881102"],
    "plate_name": ["种植业与林业", "养殖业"]})
sys.modules["zzshare"].plates_stocks = lambda plate_type=14, plate_code=None, date=None:     pd.DataFrame({"code": ["600519.SH"]})

from fastapi.testclient import TestClient
import api_v2 as API

cli = TestClient(API.app)
r = cli.get("/api/zz")
check("/api/zz 200", r.status_code == 200, r.status_code)
check("/api/zz 返回版本", r.json().get("version") == "0.4.13", r.json().get("version"))

r2 = cli.get("/api/valuation?date=2026-11-10")
check("/api/valuation 200", r2.status_code == 200, r2.status_code)
if r2.status_code == 200:
    b = r2.json()
    check("估值返回 ok", b.get("ok") is True)
    check("估值含快照", b["snapshot"].get("n") == n, b["snapshot"].get("n"))
    check("估值含文本", "zzshare" in b.get("text", ""))

r3 = cli.get("/api/industry_map?refresh=true")
check("/api/industry_map 200", r3.status_code == 200, r3.status_code)

import quantpilot as QP
check("CLI 注册 valuation", "valuation" in QP.COMMANDS)
check("CLI 注册 industry", "industry" in QP.COMMANDS)

# 还原真实 zzshare
if _orig_zz is not None:
    sys.modules["zzshare"] = _orig_zz
else:
    sys.modules.pop("zzshare", None)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  zzshare 已融入")
print("#" * 70)
