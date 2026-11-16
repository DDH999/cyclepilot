# -*- coding: utf-8 -*-
"""
M22 验证：海外指数（overseas）+ Baostock
====================================================================
跑法:  python tests/test_m22_overseas_bs.py   （不联网，用假模块）
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

import baostock_source as BS
import datasource as DS
import overseas as OS

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m22"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M22 验证  overseas v{OS.__version__} / baostock v{BS.__version__}")
print("#" + "=" * 69)

print("\n[1] 依赖与指数表")

av = OS.available()
check("返回依赖状态", isinstance(av, dict))
check("含 yfinance", "yfinance" in av)
check("含 akshare", "akshare" in av)
check("yfinance 已安装", av["yfinance"]["ok"] is True, av["yfinance"])
check("akshare 已安装", av["akshare"]["ok"] is True)
check("指数表含四个目标", all(k in OS.INDEXES for k in
                          ("sp500", "ndx", "nikkei", "ftse")))
check("每个指数有中文名", all("name" in v for v in OS.INDEXES.values()))

print("\n[2] 基金名 → 指数")

for nm, want in [("易方达标普500指数(人民币)A", "sp500"),
                 ("广发纳斯达克100ETF联接(QDII)C", "ndx"),
                 ("华安三菱日联日经225ETF发起式联接", "nikkei"),
                 ("建信富时100指数(QDII)A人民币", "ftse"),
                 ("天弘标普500发起(QDII-FOF)A", "sp500")]:
    m = OS.match_fund_index(nm)
    check(f"「{nm[:16]}…」→ {want}", m.get("key") == want, m.get("key"))

unknown = OS.match_fund_index("华夏成长混合")
check("无关键词时明确说认不出", unknown["ok"] is False, unknown)
check("给出原因", "关键词" in unknown.get("reason", ""), unknown.get("reason"))
check("空名字不崩", OS.match_fund_index("")["ok"] is False)

print(chr(10) + "[2.5] 东财直连与代理")

import os as _os


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def json(self):
        return self._p


class _FakeHttp:
    def __init__(self, payload):
        self._p = payload
        self.calls = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return _FakeResp(self._p)


KL = [f"2026-11-{d:02d},1,{1000 + d},2,3,100,200" for d in range(1, 21)]
_orig_http = OS._http
_fh = _FakeHttp({"data": {"klines": KL, "name": "标普500"}})
OS._http = lambda: _fh
try:
    em = OS._from_eastmoney("sp500", 10)
    check("东财解析成功", len(em) == 10, len(em))
    check("东财取第 3 列为收盘", abs(float(em.iloc[-1]) - 1020) < 1e-6, em.iloc[-1])
    check("请求带 secid", any("100.SPX" in c[0] for c in _fh.calls))
    check("请求带 Referer",
          any("Referer" in (c[1].get("headers") or {}) for c in _fh.calls))
finally:
    OS._http = _orig_http

_fh2 = _FakeHttp({"data": {"klines": [], "rc": 0}})
OS._http = lambda: _fh2
try:
    OS._from_eastmoney("sp500", 10)
    check("空 klines 抛错", False)
except RuntimeError as ex:
    check("空 klines 抛错", "空" in str(ex), str(ex)[:40])
finally:
    OS._http = _orig_http

check("默认提供者顺序：东财在前",
      OS.PROVIDERS[0][0] == "eastmoney", [n for n, _ in OS.PROVIDERS])
check("yfinance 排在最后",
      OS.PROVIDERS[-1][0] == "yfinance", [n for n, _ in OS.PROVIDERS])

_saved = _os.environ.pop("QP_PROXY", None)
check("无代理时返回 None", OS.proxy_of() is None, OS.proxy_of())
_os.environ["QP_PROXY"] = "http://127.0.0.1:7890"
check("读到 QP_PROXY", OS.proxy_of() == "http://127.0.0.1:7890", OS.proxy_of())
if _saved:
    _os.environ["QP_PROXY"] = _saved
else:
    _os.environ.pop("QP_PROXY", None)

print("\n[3] 指数取数回退链（假提供者）")

_orig_prov = OS.PROVIDERS
_orig_cache = OS.CACHE_DIR
OS.CACHE_DIR = TMP / "cache"

IDX = pd.bdate_range("2024-01-01", periods=300)
FAKE_S = pd.Series(np.linspace(100, 150, 300), index=IDX)


def yf_fail(key, days):
    raise RuntimeError("YFRateLimitError: Too Many Requests")


def sina_ok(key, days):
    return FAKE_S.tail(days)


OS.PROVIDERS = [("yfinance", yf_fail), ("sina", sina_ok)]
res = OS.index_history("sp500", days=200, refresh=True)
check("yfinance 失败后回退", res["ok"] is True, res.get("reason"))
check("来源标为 sina", res["source"] == "sina", res.get("source"))
check("记录降级原因", any("RateLimit" in e for e in res["errors"]), res["errors"])
check("返回 200 点", len(res["series"]) == 200, len(res["series"]))

res2 = OS.index_history("sp500", days=200)
check("第二次走缓存", res2["source"] == "cache", res2.get("source"))

OS.PROVIDERS = [("a", yf_fail), ("b", yf_fail)]
allfail = OS.index_history("ndx", days=100, refresh=True)
check("全失败返回 ok=False", allfail["ok"] is False)
check("全失败保留各源错误", len(allfail["errors"]) == 2, allfail["errors"])
check("全失败不返回数据", allfail["series"] is None)

try:
    OS.index_history("nope")
    check("未知指数抛错", False)
except ValueError:
    check("未知指数抛错", True)
OS.PROVIDERS = _orig_prov

print("\n[4] 基金 vs 基准")

rng = np.random.default_rng(7)
bench = pd.Series(rng.normal(0.0004, 0.010, 500), index=pd.bdate_range("2023-01-02", periods=500))
fund = 1.2 * bench + rng.normal(0.0001, 0.002, 500)     # 已知 beta≈1.2
cmp_ = OS.fund_vs_benchmark(fund, bench)
check("可对比", cmp_["ok"] is True, cmp_.get("reason"))
check("beta 接近 1.2", abs(cmp_["beta"] - 1.2) < 0.1, cmp_["beta"])
check("相关性很高", cmp_["correlation"] > 0.95, cmp_["correlation"])
check("跟踪误差为正", cmp_["tracking_error"] > 0, cmp_["tracking_error"])
check("含超额", "excess_annual" in cmp_)
check("样本不足返回 ok=False",
      OS.fund_vs_benchmark(fund.iloc[:20], bench.iloc[:20])["ok"] is False)

print("\n[5] Baostock 代码转换")

for c, want in [("600519", "sh.600519"), ("000001", "sz.000001"),
                ("300750", "sz.300750"), ("830799", "bj.830799"),
                ("688981", "sh.688981")]:
    got = BS.to_bs_code(c)
    check(f"{c} → {want}", got == want, got)
check("裸数字补零", BS.to_bs_code("1") == "sz.000001", BS.to_bs_code("1"))

print("\n[6] Baostock 日线（假模块）")


class FakeRS:
    def __init__(self, fields, rows, error_code="0", error_msg="ok"):
        self.fields = fields
        self._rows = rows
        self._i = -1
        self.error_code = error_code
        self.error_msg = error_msg

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


_orig_bs = sys.modules.get("baostock")


def install_fake_bs(rows=None, login_ok=True, empty=False):
    fake = types.ModuleType("baostock")
    fake.login = lambda: FakeRS([], [], "0" if login_ok else "1",
                                "ok" if login_ok else "登录失败")
    fake.logout = lambda: None
    if empty:
        fake.query_history_k_data_plus = lambda *a, **k: FakeRS(
            ["date", "open", "high", "low", "close", "volume", "amount"], [])
    else:
        fake.query_history_k_data_plus = lambda *a, **k: FakeRS(
            ["date", "open", "high", "low", "close", "volume", "amount"],
            rows or [])
    fake.query_stock_industry = lambda: FakeRS(
        ["updateDate", "code", "code_name", "industry", "industryClassification"],
        [["2026-11-09", "sh.600519", "贵州茅台", "C15酒、饮料和精制茶制造业", "证监会"],
         ["2026-11-09", "sz.000001", "平安银行", "J66货币金融服务", "证监会"],
         ["2026-11-09", "sh.600001", "某公司", "", "证监会"]])
    sys.modules["baostock"] = fake
    BS._LOGGED_IN = False
    return fake


ROWS = [["2026-11-10", "1400", "1410", "1395", "1405", "1000", "140500"],
        ["2026-11-11", "1405", "1420", "1400", "1418", "1200", "170160"],
        ["2026-11-12", "1418", "1425", "1408", "1410", "900", "126900"]]
install_fake_bs(ROWS)

df = BS.stock_daily("600519", days=10)
check("日线返回非空", len(df) == 3, len(df))
check("标准列齐全",
      {"date", "open", "high", "low", "close", "volume", "amount"}.issubset(df.columns),
      list(df.columns))
check("数值已转 float", df["close"].dtype.kind == "f", df["close"].dtype)
check("按日期升序", df["date"].is_monotonic_increasing)
check("最新收盘 1410", abs(float(df["close"].iloc[-1]) - 1410) < 1e-6)

install_fake_bs(empty=True)
try:
    BS.stock_daily("600519")
    check("空数据抛错", False)
except ValueError:
    check("空数据抛错", True)

install_fake_bs(ROWS, login_ok=False)
BS._LOGGED_IN = False
try:
    BS.stock_daily("600519")
    check("登录失败抛错", False)
except RuntimeError as ex:
    check("登录失败抛错", "登录失败" in str(ex), str(ex)[:50])

print("\n[7] Baostock 行业映射")

install_fake_bs(ROWS)
cache = TMP / "bs_industry.csv"
ind = BS.industry_map(cache=cache, refresh=True)
check("行业映射成功", ind["ok"] is True, ind.get("reason"))
check("空行业被剔除", ind["n"] == 2, ind["n"])
check("600519 → 酒类", "酒" in ind["mapping"]["600519"], ind["mapping"].get("600519"))
check("000001 → 金融", "货币金融" in ind["mapping"]["000001"])
check("缓存已写", cache.exists())
ind2 = BS.industry_map(cache=cache)
check("第二次走缓存", ind2["source"] == "cache", ind2.get("source"))

print("\n[8] 接入数据源层")

check("数据源含 baostock",
      "baostock" in [n for n, _ in DS.SOURCE_STOCK],
      [n for n, _ in DS.SOURCE_STOCK])
st = DS.sources_status()
check("健康状态标记 baostock 可用",
      st["sources"].get("baostock_available") is True, st["sources"])

# 模拟前三个源全挂，baostock 接管
DS.reset()
orig_src = DS.SOURCE_STOCK


def boom(code, days=800):
    raise RuntimeError("挂了")


DS.SOURCE_STOCK = [("a", boom), ("b", boom), ("c", boom),
                   ("baostock", DS._src_baostock_daily)]
try:
    got = DS.get_stock_daily("600519", days=10)
    check("baostock 能接管", got["source"] == "baostock", got["source"])
    check("标记降级", got["downgraded"] is True)
    check("数据可用", len(got["df"]) == 3, len(got["df"]))
finally:
    DS.SOURCE_STOCK = orig_src

print("\n[9] API / CLI")

from fastapi.testclient import TestClient
import api_v2 as API

cli = TestClient(API.app)
r = cli.get("/api/baostock")
check("/api/baostock 200", r.status_code == 200, r.status_code)

OS.CACHE_DIR = TMP / "cache"      # 已有 sp500 缓存
r2 = cli.get("/api/overseas?key=sp500&days=100")
check("/api/overseas 200", r2.status_code == 200, r2.status_code)
if r2.status_code == 200:
    b = r2.json()
    check("返回 ok", b.get("ok") is True, b.get("detail"))
    check("含最新值", "latest" in b["result"], b["result"].keys())
    check("含曲线尾部", len(b["result"].get("series_tail") or []) > 0)
    check("不含 Series 对象", "series" not in b["result"])

r3 = cli.get("/api/overseas?key=nope")
check("未知指数返回 400", r3.status_code == 400, r3.status_code)

import quantpilot as QP
check("CLI 注册 overseas", "overseas" in QP.COMMANDS)
check("CLI 注册 industry", "industry" in QP.COMMANDS)
p = QP.build_parser()
check("industry 默认 zz", p.parse_args(["industry"]).source == "zz")
check("industry 可选 bs", p.parse_args(["industry", "--source", "bs"]).source == "bs")
check("overseas 支持 keys",
      p.parse_args(["overseas", "--keys", "sp500,ftse"]).keys == "sp500,ftse")

OS.CACHE_DIR = _orig_cache
if _orig_bs is not None:
    sys.modules["baostock"] = _orig_bs
else:
    sys.modules.pop("baostock", None)
BS._LOGGED_IN = False

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  Baostock 与海外指数已融入")
print("#" * 70)
