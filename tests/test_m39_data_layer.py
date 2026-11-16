# -*- coding: utf-8 -*-
"""
M39 验证：数据层（Parquet 存储 / 质量监控 / 入库）
====================================================================
跑法:  python tests/test_m39_data_layer.py   （不联网）
"""
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import data_quality as DQ
import store

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m39"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)

# 把存储根重定向到临时目录，避免污染真实库
store.ROOT = TMP / ".data"
store.PARQUET = store.ROOT / "parquet"
store.META = store.ROOT / "meta"


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M39 验证  store v{store.__version__} / quality v{DQ.__version__}")
print("#" + "=" * 69)


def mk(n=300, seed=7):
    # 以「今天」为终点，避免"陈旧"误判，也让追加测试的重叠段可预测
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    rng = np.random.default_rng(seed)
    c = 10 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({"date": idx, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c,
                         "volume": 1e6, "amount": 1e7})


print("\n[1] Parquet 写入 / 读取")

df = mk()
r1 = store.write_bars("cn", "600519", df, source="t1")
check("写入成功", r1["ok"] is True, r1.get("error"))
check("新增行数正确", r1["added"] == 300, r1["added"])
check("总行数正确", r1["n"] == 300, r1["n"])

r2 = store.write_bars("cn", "600519", df, source="t2")
check("重复写入幂等（新增 0）", r2["added"] == 0, r2["added"])
check("总行数不变", r2["n"] == 300, r2["n"])

back = store.read_bars("cn", "600519")
check("能读回", len(back) == 300, len(back))
check("列齐全", all(c in back.columns for c in
                  ("date", "open", "high", "low", "close", "volume", "amount")))
check("日期升序", back["date"].is_monotonic_increasing)
check("记录了来源", "source" in back.columns and back["source"].iloc[-1] == "t2")

cut = str(df["date"].iloc[150].date())
part = store.read_bars("cn", "600519", start=cut)
check("条件读取生效", 0 < len(part) < 300, len(part))
check("起点正确", part["date"].min() >= pd.Timestamp(cut))

# 增量追加
df2 = mk(n=350, seed=8)
r3 = store.write_bars("cn", "600519", df2, source="t3")
check("增量追加新增 50 行", r3["added"] == 50, r3["added"])
check("总行数 350", r3["n"] == 350, r3["n"])

print("\n[2] 多市场 / 清单 / 统计")

store.write_bars("us", "AAPL", mk(seed=11), source="t1")
store.write_bars("hk", "00700", mk(seed=12), source="t1")
check("A股清单", store.list_codes("cn") == ["600519"], store.list_codes("cn"))
check("美股清单", store.list_codes("us") == ["AAPL"])
check("未知市场返回空", store.list_codes("nope") == [])

cov = store.coverage("cn")
check("覆盖清单有数据", len(cov) == 1, len(cov))
check("覆盖清单含列", all(c in cov.columns for c in ("code", "n", "source")))

st = store.stats()
check("统计三个市场", len(st["markets"]) == 3, list(st["markets"]))
check("总标的 3", st["total_codes"] == 3, st["total_codes"])
check("总行数 > 900", st["total_rows"] > 900, st["total_rows"])

print("\n[3] DuckDB 查询")

q = store.query("SELECT market, count(DISTINCT code) c, count(*) n "
                "FROM bars GROUP BY 1 ORDER BY 1")
check("查询可跑", len(q) == 3, len(q))
check("市场识别正确", set(q["market"]) == {"cn", "us", "hk"}, set(q["market"]))
qn = store.query("SELECT count(*) n FROM bars WHERE market='cn'")
check("按市场过滤", int(qn["n"].iloc[0]) == 350, qn["n"].iloc[0])

print("\n[4] 质量监控")

good = mk()
r = DQ.check_frame(good, "600519")
check("正常数据通过", r["ok"] is True, r["issues"])
check("正常数据无问题", len(r["issues"]) == 0, r["issues"])

bad = good.copy()
bad.loc[10, "close"] = -1.0
bad.loc[20, "close"] = bad.loc[19, "close"] * 1.5
bad = pd.concat([bad, bad.iloc[[30]]], ignore_index=True)
r2 = DQ.check_frame(bad, "000002")
kinds = {i["kind"] for i in r2["issues"]}
check("负价被捕获", "nonpos" in kinds, sorted(kinds))
check("重复日期被捕获", "dup_date" in kinds)
check("跳变被捕获", "jump" in kinds)
check("问题数据不通过", r2["ok"] is False)

check("空数据判死", DQ.check_frame(pd.DataFrame(), "x")["ok"] is False)
check("行数不足判死", DQ.check_frame(mk(10), "x")["ok"] is False,
      [i["kind"] for i in DQ.check_frame(mk(10), "x")["issues"]])

ohlc = good.copy()
ohlc.loc[5, "high"] = ohlc.loc[5, "close"] * 0.5      # high < close
r3 = DQ.check_frame(ohlc, "x")
check("OHLC 逻辑错被捕获",
      any(i["kind"] == "ohlc_logic" for i in r3["issues"]),
      [i["kind"] for i in r3["issues"]])

print("\n[5] 全库体检")

man = DQ.check_market("cn")
check("单市场体检可跑", man["ok"] is True)
check("统计标的数", man["n_codes"] == 1, man["n_codes"])
check("健康数正确", man["n_ok"] == 1, man["n_ok"])
check("报告可渲染", "数据质量报告" in DQ.format_report(man))

allq = DQ.check_store()
check("全库体检可跑", allq["ok"] is True)
check("覆盖三市场", len(allq["markets"]) == 3, list(allq["markets"]))
check("全库报告可渲染", "总标的 3" in DQ.format_report(allq),
      DQ.format_report(allq)[:80])

print("\n[6] 删除与边界")

d = store.delete_code("us", "AAPL")
check("删除返回文件数", d["removed_files"] >= 1, d["removed_files"])
check("删除后清单为空", store.list_codes("us") == [], store.list_codes("us"))
check("重复删除不报错", store.delete_code("us", "AAPL")["removed_files"] == 0)

check("空 market 读取返回空表", len(store.read_bars("nope", "x")) == 0)
check("空数据写入被拒", store.write_bars("cn", "x", pd.DataFrame())["ok"] is False)
check("未知市场体检返回 ok=False",
      DQ.check_market("nope")["ok"] is False)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  数据层可用")
print("#" * 70)
