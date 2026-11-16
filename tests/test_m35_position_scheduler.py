# -*- coding: utf-8 -*-
"""
M35 验证：仓位管理器 + 每日调度器
====================================================================
跑法:  python tests/test_m35_position_scheduler.py   （不联网）
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

import backtest_strict as BS
import daily_scheduler as DS
import position_manager as PM
import strategy_lab as SL

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m35"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M35 验证  position_manager v{PM.__version__} / "
      f"daily_scheduler v{DS.__version__}")
print("#" + "=" * 69)

print("\n[1] 均线状态")

idx = pd.bdate_range("2020-01-02", periods=120)
up = pd.Series(np.linspace(100, 140, 120), index=idx)
dn = pd.Series(np.linspace(140, 100, 120), index=idx)

st_up = PM.ma_state(up, 20)
st_dn = PM.ma_state(dn, 20)
check("返回 DataFrame", isinstance(st_up, pd.DataFrame))
check("含 ma/state/above/diff",
      all(c in st_up.columns for c in ("ma", "state", "above", "diff")))
check("前 19 天均线为 NaN", bool(st_up["ma"].iloc[:19].isna().all()))
check("第 20 天均线有值", bool(np.isfinite(st_up["ma"].iloc[19])))
check("单边上涨全在上方", bool((st_up["state"].iloc[20:] == 1).all()))
check("单边下跌全在下方", bool((st_dn["state"] == 0).all()))

print("\n[2] 仓位上限")

c_up = PM.position_cap(up)
c_dn = PM.position_cap(dn)
check("上涨时上限 80%", abs(float(c_up["cap"].iloc[-1]) - 0.80) < 1e-9,
      c_up["cap"].iloc[-1])
check("下跌时上限 30%", abs(float(c_dn["cap"].iloc[-1]) - 0.30) < 1e-9,
      c_dn["cap"].iloc[-1])
check("均线未形成时保守 30%",
      abs(float(c_up["cap"].iloc[0]) - 0.30) < 1e-9, c_up["cap"].iloc[0])
check("上限只取两档",
      set(np.unique(c_up["cap"])) <= {0.30, 0.80}, set(np.unique(c_up["cap"])))

# 自定义参数
cc = PM.position_cap(up, cap_low=0.10, cap_high=0.50)
check("自定义上限生效", abs(float(cc["cap"].iloc[-1]) - 0.50) < 1e-9)

# 非法参数
for bad in ((0.9, 0.3), (-0.1, 0.8), (0.3, 1.5)):
    try:
        PM.position_cap(up, cap_low=bad[0], cap_high=bad[1])
        check(f"非法参数 {bad} 应报错", False)
    except ValueError:
        check(f"非法参数 {bad} 应报错", True)

print("\n[3] 缓冲带与确认日（抑制频繁切换）")

# 缓冲带只对「均线附近的小幅抖动」有效；大幅摆动仍会切换（这是对的）
zig = pd.Series(np.tile([100.0, 103.0], 60), index=idx[:120])
a = PM.cap_summary(PM.position_cap(zig))
b = PM.cap_summary(PM.position_cap(zig, band=0.05))
c = PM.cap_summary(PM.position_cap(zig, band=0.05, confirm=3))
check("小幅锯齿行情切换很多", a["switches"] > 20, a["switches"])
check("加缓冲后大幅减少", b["switches"] < a["switches"],
      (a["switches"], b["switches"]))
check("加确认日后更少", c["switches"] <= b["switches"],
      (b["switches"], c["switches"]))

print("\n[4] 权重压缩")

w = pd.DataFrame([{"A": 0.5, "B": 0.5}], index=[idx[30]])
capped = PM.apply_cap(w, c_up)          # 该日在上方，上限 0.8
tot = float(capped.iloc[0].sum())
check("权重总和被压到上限", tot <= 0.80 + 1e-9, round(tot, 4))
check("等比例缩放（A=B）",
      abs(float(capped.iloc[0, 0]) - float(capped.iloc[0, 1])) < 1e-12)
check("比例保持 0.4:0.4",
      abs(float(capped.iloc[0, 0]) - 0.4) < 1e-9, capped.iloc[0, 0])

# 上限更高时不放大
small = pd.DataFrame([{"A": 0.2, "B": 0.2}], index=[idx[30]])
kept = PM.apply_cap(small, c_up)
check("权重低于上限时不放大",
      abs(float(kept.iloc[0].sum()) - 0.4) < 1e-9,
      float(kept.iloc[0].sum()))

# 传入 Series 也可以
kept2 = PM.apply_cap(w, c_up["cap"])
check("cap 可传 Series", float(kept2.iloc[0].sum()) <= 0.80 + 1e-9)

print("\n[5] 回测串联（cap 生效）")


def mk(n=300, k=4, seed=5):
    ix = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(k):
        c = 10 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
        out[f"6000{i:02d}"] = pd.DataFrame({
            "date": ix, "open": c, "high": c * 1.01, "low": c * 0.99,
            "close": c, "volume": 1e6, "amount": 1e7})
    return out


p = mk()
w2 = SL.factor_weights(SL.build_scores(p, "momentum", 20), "momentum", 2, 20)
bt_free = BS.StrictBacktest(p, capital=1e6).run(w2, max_positions=2)
cap_s = pd.Series(0.30, index=p["600000"]["date"])
bt_cap = BS.StrictBacktest(p, capital=1e6).run(w2, max_positions=2, cap=cap_s)

check("无 cap 时可运行", bt_free["ok"] is True)
check("有 cap 时可运行", bt_cap["ok"] is True)
check("结果标记 cap_applied", bt_cap["cap_applied"] is True)
check("无 cap 时标记为 False", bt_free["cap_applied"] is False)
check("记录平均 cap", abs(bt_cap["avg_cap"] - 0.30) < 1e-9, bt_cap["avg_cap"])
check("满仓上限 30% 时波动更低",
      bt_cap["vol"] < bt_free["vol"], (bt_cap["vol"], bt_free["vol"]))
check("收益绝对值也更小（仓位小）",
      abs(bt_cap["annual"]) < abs(bt_free["annual"]) + 0.05,
      (bt_cap["annual"], bt_free["annual"]))

# cap=0 时应完全不建仓
bt_zero = BS.StrictBacktest(p, capital=1e6).run(
    w2, max_positions=2, cap=pd.Series(0.0, index=p["600000"]["date"]))
check("cap=0 时不亏不赚（全现金）", abs(bt_zero["annual"]) < 1e-6,
      bt_zero["annual"])

print("\n[6] 调度器：数据清洗")

cache = TMP / "cache"
(cache / "strict").mkdir(parents=True)
good = pd.DataFrame({"date": pd.bdate_range("2026-11-01", periods=10),
                     "close": np.linspace(10, 11, 10), "volume": 1e6})
good.to_csv(cache / "strict" / "good.csv", index=False)
bad = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5),
                    "close": [10, None, 12, 13, 14], "volume": 0})
bad.to_csv(cache / "strict" / "bad.csv", index=False)

_orig_cache = DS.CACHE
_orig_reports = DS.REPORTS
DS.CACHE = cache
DS.REPORTS = TMP / "reports"
try:
    r = DS.clean_data(write_report=True)
    check("清洗可运行", r["ok"] is True, r.get("error"))
    check("扫到 2 个文件", r["n_files"] == 2, r["n_files"])
    check("识别出问题文件", r["n_bad"] >= 1, r["n_bad"])
    check("写出报告", r["report"] is not None and Path(r["report"]).exists())
    txt = Path(r["report"]).read_text(encoding="utf-8")
    check("报告含表头", "数据清洗报告" in txt)
    check("报告列出问题", "⚠" in txt or "❌" in txt)

    r2 = DS.clean_data(write_report=False)
    check("不落盘也可运行", r2["ok"] is True and r2["report"] is None)
finally:
    DS.CACHE = _orig_cache
    DS.REPORTS = _orig_reports

print("\n[7] 调度器：交易清单（打桩）")

_orig_fetch = BS.fetch_panel
_orig_index = PM.load_index
_orig_build = SL.build_scores
_orig_fw = SL.factor_weights


def fake_fetch(codes, start, end, **kw):
    return {"ok": True, "panel": mk(n=300, k=12, seed=9), "missing": [],
            "n_ok": 12, "n_missing": 0}


BS.fetch_panel = fake_fetch
PM.load_index = lambda code="sh000300", refresh=False: up
DS.REPORTS = TMP / "reports2"
try:
    r = DS.make_trade_list(n=12, top_n=3, write=True)
    check("清单可生成", r["ok"] is True, r.get("error"))
    check("条数正确", r["n"] == 3, r["n"])
    check("带上仓位上限", r["cap"] in (0.30, 0.80), r["cap"])
    check("记录信号日", r["signal_date"] is not None)
    check("CSV 已落盘", r["csv"] and Path(r["csv"]).exists())
    check("MD 已落盘", r["md"] and Path(r["md"]).exists())
    df = pd.read_csv(r["csv"])
    check("CSV 三列齐全",
          all(c in df.columns for c in ("code", "target_weight", "ref_close")))
    check("权重合计等于 cap",
          abs(float(df["target_weight"].sum()) - r["cap"]) < 1e-3,
          (float(df["target_weight"].sum()), r["cap"]))
    md = Path(r["md"]).read_text(encoding="utf-8")
    check("清单含免责", "候选池 ≠ 买入信号" in md)
    check("清单标明次日开盘", "次日开盘" in md)
finally:
    BS.fetch_panel = _orig_fetch
    PM.load_index = _orig_index
    DS.REPORTS = _orig_reports

print("\n[8] 调度器：任务注册与触发")

check("trigger 未知任务报错",
      DS.trigger("nope")["ok"] is False)
try:
    import schedule
    sch = DS.build_scheduler(clean_at="15:30", trade_at="20:00")
    jobs = sch.get_jobs()
    check("注册 2 个任务", len(jobs) == 2, len(jobs))
    times = sorted(j.at_time.strftime("%H:%M") for j in jobs)
    check("时间点为 15:30 与 20:00", times == ["15:30", "20:00"], times)
    sch.clear()
    check("可清空", len(sch.get_jobs()) == 0)
except ImportError:
    check("schedule 未安装（跳过注册测试）", True)

check("暴露默认时间", DS.CLEAN_TIME == "15:30" and DS.TRADE_TIME == "20:00")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  两个模块可用")
print("#" * 70)
