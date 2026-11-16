# -*- coding: utf-8 -*-
"""
M23 验证：凭据管理 + 美股数据源（ROIC.ai / xfinlink）
====================================================================
跑法:  python tests/test_m23_us.py   （不联网、不消耗 API 配额）
"""
import json
import os
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

import credentials as CR
import us_source as US

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m23"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M23 验证  credentials v{CR.__version__} / us_source v{US.__version__}")
print("#" + "=" * 69)

_orig_cfg = CR.CONFIG_PATH
_orig_env = {k: os.environ.get(v["env"]) for k, v in CR.KNOWN_KEYS.items()}
for v in CR.KNOWN_KEYS.values():
    os.environ.pop(v["env"], None)

print("\n[1] 掩码与存储")

check("短 Key 也掩码", CR.mask("abc") == "ab****", CR.mask("abc"))
check("长 Key 前后各留 4 位",
      CR.mask("I8H2uia0JgqovL2B3eFEYk55coROjVj5") == "I8H2****jVj5",
      CR.mask("I8H2uia0JgqovL2B3eFEYk55coROjVj5"))
check("空 Key 显示未设置", CR.mask(None) == "(未设置)")
check("掩码不含中间内容",
      "uia0JgqovL2B3eFEYk55" not in CR.mask("I8H2uia0JgqovL2B3eFEYk55coROjVj5"))

CR.CONFIG_PATH = TMP / "keys.json"
r = CR.save_key("roicai", "TESTKEY_roicai_123456")
check("保存成功", r["ok"] is True)
check("保存返回掩码", "****" in r["masked"], r["masked"])
check("配置文件已写", CR.CONFIG_PATH.exists())
check("文件内容可解析", isinstance(json.loads(CR.CONFIG_PATH.read_text("utf-8")), dict))
check("读回一致", CR.load_key("roicai") == "TESTKEY_roicai_123456")

CR.save_key("xfinlink", "TESTKEY_xfin_9999")
data = json.loads(CR.CONFIG_PATH.read_text("utf-8"))
check("保存第二个不覆盖第一个",
      "roicai" in data and "xfinlink" in data, list(data))
check("读回第二个", CR.load_key("xfinlink") == "TESTKEY_xfin_9999")

try:
    CR.load_key("nope")
    check("未知服务抛错", False)
except ValueError:
    check("未知服务抛错", True)

print("\n[2] 环境变量优先")

os.environ["ROICAI_API_KEY"] = "FROM_ENV_KEY"
check("环境变量覆盖文件", CR.load_key("roicai") == "FROM_ENV_KEY", CR.load_key("roicai"))
st = CR.status()
check("状态标明来源 env",
      st["keys"]["roicai"]["source"] == "env", st["keys"]["roicai"]["source"])
check("状态只给掩码",
      "FROM_ENV_KEY" not in json.dumps(st, ensure_ascii=False))
os.environ.pop("ROICAI_API_KEY")
check("移除后回落到文件", CR.load_key("roicai") == "TESTKEY_roicai_123456")
check("报告可渲染", "API 凭据" in CR.format_report(CR.status()))

print("\n[3] 美股源基础")

av = US.available()
check("含两个源", set(av) == {"roicai", "xfinlink"}, list(av))
check("roicai 已安装", av["roicai"]["installed"] is True, av["roicai"])
check("xfinlink 已安装", av["xfinlink"]["installed"] is True, av["xfinlink"])
check("有版本号", bool(av["roicai"]["version"]), av["roicai"]["version"])

check("AAPL → NASDAQ:AAPL", US._roic_symbol("AAPL") == "NASDAQ:AAPL")
check("小写也转", US._roic_symbol("aapl") == "NASDAQ:AAPL")
check("已带前缀原样", US._roic_symbol("NYSE:IBM") == "NYSE:IBM")
check("去空格", US._roic_symbol("  AAPL ") == "NASDAQ:AAPL")

th = US.throttle_state()
check("节流状态含 roicai", "roicai" in th)
check("ROIC 最小间隔 12s", th["roicai"]["min_interval_s"] == 12.0,
      th["roicai"]["min_interval_s"])
check("报告可渲染", "美股数据源" in US.format_report(US.status()))

print("\n[4] ROIC 客户端（假 SDK）")


class _FakeProfile:
    def model_dump(self):
        return {"short_description": "Apple Inc.", "sector": "Electronic technology",
                "industry": "Telecommunications equipment", "ceo": "John Ternus",
                "number_of_employees": 167000, "fundamental_currency": "USD",
                "ai_description": "Apple designs consumer electronics.",
                "website": "http://www.apple.com"}


class _FakeQuote:
    def model_dump(self):
        return {"symbol": "NASDAQ:AAPL", "date": "2026-11-12", "open": 355.36,
                "high": 355.36, "low": 355.36, "close": 355.36,
                "volume": 61987.0, "currency": "USD"}


_orig_roic = sys.modules.get("roicai")
fake_roic = types.ModuleType("roicai")
calls = []


class _Profiles:
    def retrieve(self, sym):
        calls.append(("profile", sym))
        return _FakeProfile()


class _Latest:
    def retrieve(self, sym):
        calls.append(("latest", sym))
        return _FakeQuote()


class _Companies:
    profiles = _Profiles()


class _Prices:
    latest = _Latest()


class _FakeCli:
    def __init__(self, **kw):
        self.kw = kw
        self.companies = _Companies()
        self.stock_prices = _Prices()


fake_roic.Roicai = _FakeCli
sys.modules["roicai"] = fake_roic
US._LAST_CALL["roicai"] = 0.0          # 跳过节流等待

p = US.roic_profile("AAPL")
check("资料成功", p["ok"] is True)
check("公司名来自 short_description", p["company_name"] == "Apple Inc.", p["company_name"])
check("行业正确", p["sector"] == "Electronic technology")
check("CEO 解析", p["ceo"] == "John Ternus")
check("员工数解析", p["employees"] == 167000)
check("传入的符号带交易所", calls[0][1] == "NASDAQ:AAPL", calls[0])

q = US.roic_latest("AAPL")
check("最新价成功", q["ok"] is True)
check("收盘价正确", abs(float(q["close"]) - 355.36) < 1e-6, q["close"])
check("日期正确", q["date"] == "2026-11-12", q["date"])

check("缺 Key 时抛错", True)
_saved_key = CR.load_key("roicai")
os.environ.pop("ROICAI_API_KEY", None)
CR.CONFIG_PATH = TMP / "empty.json"
try:
    US.roicai_client()
    check("无 Key 抛错", False)
except RuntimeError as ex:
    check("无 Key 抛错", "未配置" in str(ex), str(ex)[:40])
CR.CONFIG_PATH = TMP / "keys.json"

print("\n[5] xfinlink（假 SDK）")

_orig_x = sys.modules.get("xfinlink")
fake_x = types.ModuleType("xfinlink")
fake_x._api_key = None
fake_x.set_api_key = lambda k: setattr(fake_x, "_api_key", k)
fake_x.set_timeout = lambda s: None
_XDF = pd.DataFrame({
    "date": pd.bdate_range("2026-11-01", periods=5).astype(str),
    "ticker": "AAPL", "gics_sector": "Information Technology",
    "close": [350.0, 352.0, 351.0, 354.0, 355.36],
    "adj_close": [350.0, 352.0, 351.0, 354.0, 355.36],
    "volume": [1e6, 1.1e6, 0.9e6, 1.2e6, 6.2e4],
})
fake_x.prices = lambda t, start=None, end=None: _XDF.copy()
fake_x.search = lambda q: pd.DataFrame({
    "ticker": ["AAPL", "AAPY"], "entity_name": ["Apple Inc.", "Other"],
    "gics_sector": ["IT", "IT"], "country": ["US", "US"]})
sys.modules["xfinlink"] = fake_x
US.CACHE_DIR = TMP / "us_cache"

got = US.xfin_prices("AAPL", "2026-11-01", "2026-11-12", refresh=True)
check("价格成功", got["ok"] is True, got.get("reason"))
check("来源 live", got["source"] == "live", got["source"])
check("行数正确", got["n"] == 5, got["n"])
check("传给 SDK 的 Key 正确", fake_x._api_key == "TESTKEY_xfin_9999",
      CR.mask(fake_x._api_key))

got2 = US.xfin_prices("AAPL", "2026-11-01", "2026-11-12")
check("第二次走缓存", got2["source"] == "cache", got2["source"])
check("缓存内容一致", got2["n"] == 5, got2["n"])

sr = US.xfin_search("AAPL")
check("搜索成功", sr["ok"] is True)
check("返回行数", sr["n"] == 2, sr["n"])
check("含关键列", "ticker" in (sr["rows"][0] if sr["rows"] else {}),
      sr["rows"][:1])

print("\n[6] API")

from fastapi.testclient import TestClient
import api_v2 as API

cli = TestClient(API.app)

r1 = cli.get("/api/keys")
check("/api/keys 200", r1.status_code == 200, r1.status_code)
body = r1.json()
txt = json.dumps(body, ensure_ascii=False)
check("接口也只给掩码", "TESTKEY_roicai_123456" not in txt)
check("返回两个服务", set(body["keys"]) == {"roicai", "xfinlink"})

r2 = cli.get("/api/us?ticker=AAPL&what=profile")
check("/api/us 200", r2.status_code == 200, r2.status_code)
if r2.status_code == 200:
    b = r2.json()
    check("返回 profile", b.get("profile") is not None, b.get("errors"))
    check("公司名正确", b["profile"]["company_name"] == "Apple Inc.")

r3 = cli.get("/api/us?what=status")
check("status 模式可用", r3.status_code == 200 and "status" in r3.json())

import quantpilot as QP
check("CLI 注册 us", "us" in QP.COMMANDS)
check("CLI 注册 keys", "keys" in QP.COMMANDS)
pp = QP.build_parser()
check("us 解析 ticker", pp.parse_args(["us", "--ticker", "MSFT"]).ticker == "MSFT")
check("keys 解析 --set",
      pp.parse_args(["keys", "--set", "roicai=abc"]).set == "roicai=abc")

# 还原
CR.CONFIG_PATH = _orig_cfg
if _orig_roic is not None:
    sys.modules["roicai"] = _orig_roic
else:
    sys.modules.pop("roicai", None)
if _orig_x is not None:
    sys.modules["xfinlink"] = _orig_x
else:
    sys.modules.pop("xfinlink", None)
for k, v in _orig_env.items():
    if v:
        os.environ[CR.KNOWN_KEYS[k]["env"]] = v

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  凭据与美股源可用")
print("#" * 70)
