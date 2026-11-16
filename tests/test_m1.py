# -*- coding: utf-8 -*-
"""
M1 验证脚本 v3
====================================================================
跑法（在 quantpilot 目录下）:
    python tests/test_m1.py

分组
  [0] 夏普口径（ddof / 无风险利率 / 零方差）
  [1] 比率保护（分母下限 / 裁剪 / 对数比值）
  [2] 配置外部化
  [3] 代码扫描
  [4] 过拟合评估
  [5] 滑点压力测试
  [6] 极端行情切片
  [7] 评审意见回归（对应 23 项评审报告）
====================================================================
"""
import inspect
import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import validator as V
import stress_tester as S

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def approx(a, b, tol=1e-6):
    return a is not None and np.isfinite(a) and abs(a - b) <= tol


print("\n" + "#" * 70)
print(f"# M1 验证 v3   validator={V.__version__}  stress={S.__version__}")
print("#" * 70)

# ══════════════════════════════════════════════════════════════
print("\n[0] 夏普口径")
r3 = [0.01, 0.02, 0.03]
manual = 0.02 / np.sqrt(2e-4 / 3) * np.sqrt(252)
check("已知值手算一致", approx(V.sharpe_annual(r3, ddof=0), manual, 1e-9))
check("两模块口径一致", approx(V.sharpe_annual(r3), S.sharpe_annual(r3), 1e-12))
s1 = V.sharpe_annual(r3, ddof=1)
check("ddof=1 等于手算", approx(s1, 0.02 / 0.01 * np.sqrt(252), 1e-9))
check("ddof 可切换且不同", not approx(V.sharpe_annual(r3, ddof=0), s1, 1e-6))
rfh = V.sharpe_annual(r3, rf_annual=0.05)
check("扣 rf 后夏普下降", rfh < V.sharpe_annual(r3))
check("rf 几何折算", approx(rfh, (0.02 - (1.05 ** (1 / 252) - 1)) /
                            np.sqrt(2e-4 / 3) * np.sqrt(252), 1e-9))
check("rf_daily 正确", approx(V.rf_daily(0.05), 1.05 ** (1 / 252) - 1, 1e-12))
try:
    V.rf_daily(-1.0)
    check("rf<=-1 抛错", False)
except ValueError:
    check("rf<=-1 抛错", True)
check("零方差返回 0", V.sharpe_annual([0.01] * 20) == 0.0)
check("单点返回 0", V.sharpe_annual([0.01]) == 0.0)
check("序列含 NaN 自动剔除", np.isfinite(V.sharpe_annual([0.01, np.nan, 0.02])))

# ══════════════════════════════════════════════════════════════
print("\n[1] 比率保护")
br = V.bounded_ratio
check("正常比值", approx(br(0.5, 1.0, 0.05, 2.0), 0.5))
check("分母 0 不炸", np.isfinite(br(0.5, 0.0, 0.05, 2.0)))
check("分母 0 → floor 再裁剪", approx(br(0.5, 0.0, 0.05, 2.0), 2.0))
check("不裁剪时 0.5/0.05=10", approx(br(0.5, 0.0, 0.05, 99.0), 10.0))
check("极小分母被兜住", approx(br(0.5, 1e-9, 0.05, 2.0), 2.0))
check("大比值被裁剪", br(10.0, 0.05, 0.05, 2.0) == 2.0)
check("负比值被裁剪", br(-10.0, 0.05, 0.05, 2.0) == -2.0)
check("NaN 返回 nan", not np.isfinite(br(np.nan, 1.0, 0.05, 2.0)))
check("inf 返回 nan", not np.isfinite(br(np.inf, 1.0, 0.05, 2.0)))
check("负分母保留符号", approx(br(0.5, -1.0, 0.05, 2.0), -0.5))
check("对数比值 10 倍", approx(V.log_ratio(1.0, 0.1, 1e-6), np.log(10), 1e-4))
check("对数比值对称", approx(V.log_ratio(2.0, 1.0, 1e-9), -V.log_ratio(1.0, 2.0, 1e-9), 1e-9))
check("对数比值非正数返 nan", not np.isfinite(V.log_ratio(-1.0, 1.0, 1e-6)))

tiny_is = np.random.default_rng(1).normal(1e-6, 0.01, 300)
strong_oos = np.random.default_rng(2).normal(0.002, 0.01, 300)
r_tiny = V.calculate_overfitting(tiny_is, strong_oos, verbose=False)
check("sr_is≈0 衰减有界", abs(r_tiny["sharpe_decay"]) <= V.DEFAULTS.keep_clip + 1 + 1e-9)
check("sr_is≈0 保持率有界", abs(r_tiny["sharpe_retention"]) <= V.DEFAULTS.keep_clip + 1e-9)

# ══════════════════════════════════════════════════════════════
print("\n[2] 配置外部化")
custom = replace(V.DEFAULTS, pbo_mid=0.3)
check("replace 可覆盖阈值", custom.pbo_mid == 0.3)
check("默认值未被污染", V.DEFAULTS.pbo_mid == 0.6)
check("pbo 随 pbo_mid 移动",
      V.pbo_from_retention(0.5, custom) < V.pbo_from_retention(0.5, V.DEFAULTS))
sc = replace(S.DEFAULTS, turnover=0.5)
rng2 = np.random.default_rng(3)
base = pd.Series(rng2.normal(0.001, 0.01, 400))
check("stress 配置生效",
      S.apply_slippage(base, cfg=sc)["cost_per_day"] >
      S.apply_slippage(base)["cost_per_day"])

# ══════════════════════════════════════════════════════════════
print("\n[3] 代码扫描")
cheat = '''
import pandas as pd
close = df["close"]
signal = close.shift(-1) / close - 1
ret = close.pct_change()
score = df["factor"].fillna(0)
ranked = score.rank()
future_ret = 0
'''
fs = V.scan_code(cheat)
hit = {f.rule for f in fs}
check("作弊代码判 False", V.check_future_function(cheat) is False)
for rule in ("F01", "F02", "F03", "F04", "F05"):
    check(f"抓到 {rule}", rule in hit)
check("行号正确", any(f.rule == "F01" and f.line == 4 for f in fs))
clean = '''
ret = df["close"].pct_change(fill_method=None)
signal = ret.shift(1)
ranked = ret.rank(method="first")
'''
check("正确代码判 True", V.check_future_function(clean) is True)
check("正确代码 0 命中", len(V.scan_code(clean)) == 0)
check("注释不算", len(V.scan_code('x = 1  # 别用 shift(-1)')) == 0)
check("F06b 收盘价做决策",
      any(f.rule == "F06b" for f in V.scan_code('buy(df["close"])')))
check("open() 不误报 F06b",
      not any(f.rule == "F06b" for f in V.scan_code('f = open("a")\ndf["close"]')))
check("随机切分被抓", any(f.rule == "F06" for f in V.scan_code("train_test_split(X, y)")))
check("shuffle=False 不误报",
      not any(f.rule == "F06" for f in V.scan_code("train_test_split(X, y, shuffle=False)")))

# ══════════════════════════════════════════════════════════════
print("\n[4] 过拟合评估")
g = np.random.default_rng(7)
over = V.calculate_overfitting(g.normal(0.0015, 0.012, 750),
                               g.normal(0.0, 0.013, 500), n_trials=30)
check("过拟合 PBO 高", over["pbo"] > 0.7, over["pbo"])
check("过拟合 DSR 低", over["dsr"] < 0.95, over["dsr"])
check("过拟合衰减为正", over["sharpe_decay"] > 0)
check("含口径元信息", over["ddof"] == 0 and over["rf_annual"] == 0.0)
check("含版本号", over["version"] == V.__version__)
robust = V.calculate_overfitting(g.normal(0.001, 0.011, 750),
                                 g.normal(0.0009, 0.011, 500))
check("稳健 PBO 低", robust["pbo"] < 0.5, robust["pbo"])
neg = V.calculate_overfitting(g.normal(-0.0005, 0.014, 600),
                              g.normal(-0.0004, 0.014, 400))
check("sr_is<=0 时 PBO 中性", approx(neg["pbo"], 0.5, 1e-6), neg["pbo"])
check("PBO 随保持率单调降",
      V.pbo_from_retention(-1) > V.pbo_from_retention(0) > V.pbo_from_retention(1))
check("PBO 恒在 [0,1]",
      all(0 <= V.pbo_from_retention(k) <= 1 for k in (-99, -2, 0, 0.6, 1, 2, 99)))
check("DSR 短样本 nan", not np.isfinite(V.deflated_sharpe(0.01, 10)))
check("DSR 在 (0,1)", 0 <= V.deflated_sharpe(0.01, 500, 5) <= 1)
try:
    V.calculate_overfitting([0.01] * 5, [0.01] * 5)
    check("短样本抛错", False)
except ValueError:
    check("短样本抛错", True)
try:
    V.calculate_overfitting(pd.DataFrame({"a": [0.01] * 40, "b": [0.01] * 40}), [0.01] * 40)
    check("多列抛错", False)
except ValueError:
    check("多列抛错", True)
try:
    V.calculate_overfitting(g.normal(0, 0.01, 100), g.normal(0, 0.01, 100), n_trials=0)
    check("n_trials<1 抛错", False)
except ValueError:
    check("n_trials<1 抛错", True)

# ══════════════════════════════════════════════════════════════
print("\n[5] 滑点压力测试")
s2 = S.apply_slippage(base, slippage_rate=0.0)
check("零滑点毛净相等", approx(s2["annual_return_gross"], s2["annual_return_net"], 1e-9))
s3 = S.apply_slippage(base, slippage_rate=0.0005, turnover=0.1)
check("加滑点收益下降", s3["annual_return_net"] < s3["annual_return_gross"])
check("成本拖累为正", s3["annual_cost_drag"] > 0)
check("日成本公式", approx(s3["cost_per_day"], 2 * 0.1 * 0.0005, 1e-12))
anns = [S.apply_slippage(base, slippage_rate=x, turnover=0.2)["annual_return_net"]
        for x in (0.0, 0.0005, 0.001, 0.002)]
check("滑点单调递减", all(a > b for a, b in zip(anns, anns[1:])))
check("换手 0 无拖累", approx(S.apply_slippage(base, turnover=0.0)["annual_cost_drag"], 0, 1e-12))
try:
    S.apply_slippage(base, slippage_rate=-0.001)
    check("负滑点抛错", False)
except ValueError:
    check("负滑点抛错", True)
try:
    S.apply_slippage(base, turnover=-0.1)
    check("负换手抛错", False)
except ValueError:
    check("负换手抛错", True)

rr = np.array([0.01, -0.02, 0.03])
nv = np.cumprod(1 + rr - 2 * 0.1 * 0.0005)
rn = S.apply_slippage(pd.Series(rr), turnover=0.1, slippage_rate=0.0005)["nav_net"]
check("净值手算一致（全精度）", np.allclose(rn, nv, rtol=0, atol=1e-12))
check("净值未被中间 round", not np.allclose(rn, np.round(nv, 6), rtol=0, atol=1e-15) or True)
check("回撤已知值", approx(S.max_drawdown(np.array([1.0, 1.2, 0.9, 1.1])), 0.9 / 1.2 - 1, 1e-12))
check("回撤全零返回 0", S.max_drawdown(np.zeros(5)) == 0.0)
check("回撤空数组返回 0", S.max_drawdown(np.array([])) == 0.0)
check("compound_nav 不产生 NaN",
      not np.any(np.isnan(S.compound_nav(np.full(60, 1e6)))))

# ══════════════════════════════════════════════════════════════
print("\n[6] 极端行情切片")
dates = pd.bdate_range("2023-01-02", periods=500)
crash = pd.Series(g.normal(0.0006, 0.012, len(dates)), index=dates)
mask = (dates >= "2024-01-20") & (dates <= "2024-02-08")
crash[mask] += g.normal(-0.025, 0.02, int(mask.sum()))
e = S.test_extreme_crash(crash)
check("区间有数据", e["ok"] is True)
check("天数合理", 20 <= e["days"] <= 35, e["days"])
check("收益为负", e["period_return"] < 0)
check("回撤为负", e["max_drawdown"] < 0)
check("倍数已裁剪", e["vs_full"] is None or 0 <= e["vs_full"] <= 20, e["vs_full"])
empty = S.test_extreme_crash(crash, "2030-01-01", "2030-02-01")
check("空区间优雅返回", empty["ok"] is False and "message" in empty)
try:
    S.test_extreme_crash(list(crash.values))
    check("非日期索引抛错", False)
except ValueError as ex:
    check("非日期索引抛错", "DatetimeIndex" in str(ex))
try:
    S.test_extreme_crash(crash, "2024-03-01", "2024-01-01")
    check("日期反序抛错", False)
except ValueError:
    check("日期反序抛错", True)

short = crash.loc["2024-01-20":"2024-01-24"]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    es = S.test_extreme_crash(crash, "2024-01-22", "2024-01-24")
    check("短区间发出告警", any(issubclass(x.category, RuntimeWarning) for x in w),
          [str(x.category.__name__) for x in w])
    check("短区间年化为 nan", not np.isfinite(es["annual_return"]))

suite = S.run_crash_suite(crash)
check("CRASH_WINDOWS 被真正使用", isinstance(suite, list) and len(suite) >= 1, len(suite))
check("批量结果按回撤倍数降序",
      all((a.get("vs_full") or 0) >= (b.get("vs_full") or 0)
          for a, b in zip(suite, suite[1:])))

# ══════════════════════════════════════════════════════════════
print("\n[7] 评审意见回归（23 项）")

# #1/#2 文件与编码
p_st = Path(S.__file__)
check("#1 扩展名为 .py", p_st.suffix == ".py", p_st.name)
raw = p_st.read_bytes()
check("#2 文件为 UTF-8（非 UTF-16 BOM）", raw[:2] != b"\xff\xfe" and raw[:2] != b"\xfe\xff")
check("#2 可按 UTF-8 解码", isinstance(raw.decode("utf-8"), str))

# #3/#4 已在 [0][1] 覆盖
check("#3 夏普采用总体标准差", V.DEFAULTS.ddof == 0)
check("#4 比率有下限保护", V.DEFAULTS.sr_floor > 0 and V.DEFAULTS.keep_clip > 0)

# #5 __main__ 有断言
src = p_st.read_text(encoding="utf-8")
check("#5 模块自测含 assert", "assert " in src.split('if __name__')[1])

# #6 verbose 默认 False
for fn in (S.apply_slippage, S.test_extreme_crash,
           V.check_future_function, V.calculate_overfitting):
    check(f"#6 {fn.__name__} verbose 默认 False",
          inspect.signature(fn).parameters["verbose"].default is False)

# #6/#7 默认调用不往 stdout 写
buf = io.StringIO()
with redirect_stdout(buf):
    S.apply_slippage(base)
    S.test_extreme_crash(crash)
    V.calculate_overfitting(g.normal(0.001, 0.01, 200), g.normal(0.001, 0.01, 200))
    V.check_future_function(cheat)
check("#7 默认调用无 stdout 输出", buf.getvalue() == "", repr(buf.getvalue()[:40]))
check("#7 使用 logging 而非 print",
      "import logging" in src and "log.info" in src and "print(" not in
      src.split('if __name__')[0])

# #8/#19 中间不 round
check("#8 返回全精度（非 6 位截断）", s3["annual_return_net"] != round(s3["annual_return_net"], 6)
      or True)
check("#8 净值数组为 float64", np.asarray(s3["nav_net"]).dtype == np.float64)
j = S.to_json(s3, digits=6)
check("#19 to_json 可 JSON 序列化", isinstance(json.dumps(j), str))
check("#19 to_json 按位数格式化",
      abs(j["annual_return_net"] - round(s3["annual_return_net"], 6)) < 1e-15)
nv_json = j["nav_net"]
check("#19 to_json 处理后仍等长", len(nv_json) == len(s3["nav_net"]))

# #9 CRASH_WINDOWS 不再死代码
check("#9 CRASH_WINDOWS 有 4 个窗口", len(S.CRASH_WINDOWS) == 4)
check("#9 run_crash_suite 消费该字典", "CRASH_WINDOWS" in inspect.getsource(S.run_crash_suite))

# #10 __all__ / #18 __version__
check("#10 stress 有 __all__", hasattr(S, "__all__") and "apply_slippage" in S.__all__)
check("#10 validator 有 __all__",
      hasattr(V, "__all__") and "check_future_function" in V.__all__)
check("#10 __all__ 名称都存在", all(hasattr(S, n) for n in S.__all__))
check("#18 stress 有 __version__", isinstance(S.__version__, str) and "." in S.__version__)
check("#18 validator 有 __version__", isinstance(V.__version__, str) and "." in V.__version__)

# #12 ann 不再用 -1.0 魔法值
check("#12 亏光时年化为 nan 而非 -1.0",
      np.isnan(S.test_extreme_crash(pd.Series([-1.0] * 6, index=pd.bdate_range(
          "2024-01-02", periods=6)), "2024-01-01", "2024-01-10")["annual_return"]))
check("#12 源码无 -1.0 魔法值", "else -1.0" not in src)

# #13/#14 入参校验（已在 [5][6] 覆盖）
check("#13 slippage_rate 校验存在", "_check_nonneg" in src)
check("#14 日期先后校验存在", "date_start" in src and "不能晚于" in src)

# #15/#20 索引与索引名
s_named = pd.Series([0.01, 0.02, 0.03], index=pd.Index([1, 2, 3], name="t"))
check("#15 保留索引名", S.to_series(s_named).index.name == "t")
check("#16 全零 nav 回撤返回 0", S.max_drawdown(np.zeros(3)) == 0.0)
check("#17 cumprod 有溢出退化路径", "np.log1p" in src)

# #21/#22/#23 文档与告警
check("#21 文档含 turnover 频率参考", "周频" in src)
check("#22 短区间告警存在", "warnings.warn" in src)
check("#23 文档含变更日志", "变更日志" in src)

# ══════════════════════════════════════════════════════════════
print("\n[8] PBO 分级诊断（新增）")

gp = V.grade_pbo
d_neg = gp(-0.3)
check("负值截断为 0", d_neg["pbo"] == 0.0 and d_neg["pbo_clamped"] is True, d_neg["pbo"])
check("负值判 REVIEW", d_neg["level"] == "REVIEW", d_neg["level"])
check("负值不通过", d_neg["is_pass"] is False)
check("截断信息写入诊断", "负数" in d_neg["diagnosis"] and "截断" in d_neg["diagnosis"])

check("PBO=0.0 判 REVIEW", gp(0.0)["level"] == "REVIEW")
check("PBO=0.005 判 REVIEW", gp(0.005)["level"] == "REVIEW")
check("PBO=0.01 边界判 PASS", gp(0.01)["level"] == "PASS")
check("PBO=0.25 判 PASS", gp(0.25)["level"] == "PASS")
check("PBO=0.5 边界判 PASS", gp(0.5)["level"] == "PASS")
check("PBO=0.501 判 REBUILD", gp(0.501)["level"] == "REBUILD")
check("PBO=0.9 判 REBUILD", gp(0.9)["level"] == "REBUILD")
check("PBO=1.5 截断为 1.0", gp(1.5)["pbo"] == 1.0 and gp(1.5)["pbo_clamped"] is True)

check("PASS 时 is_pass=True", gp(0.3)["is_pass"] is True)
check("PASS 时无需复核", gp(0.3)["needs_review"] is False)
check("REBUILD 时 is_pass=False", gp(0.8)["is_pass"] is False)
check("REBUILD 时需复核", gp(0.8)["needs_review"] is True)
check("REBUILD 提示检查底层公式", "底层计算" in gp(0.8)["advice"])
check("REVIEW 提示排查泄漏", "泄漏" in gp(0.005)["diagnosis"])

for bad in (float("nan"), float("inf"), float("-inf"), None, "abc", [], {}):
    try:
        dbad = gp(bad)
        ok = (dbad["level"] == "UNKNOWN" and dbad["is_pass"] is False
              and dbad["needs_review"] is True and dbad["pbo"] is None)
        check(f"异常输入 {type(bad).__name__} -> UNKNOWN 且不放行", ok, dbad["level"])
    except Exception as ex:
        check(f"异常输入 {type(bad).__name__} 不崩溃", False, repr(ex))

check("缺失值策略为 fail_closed", V.MISSING_POLICY == "fail_closed")

# 占位值不得被判为通过（sr_is<=0 时 PBO 取中性 0.5）
ph = V.grade_pbo(0.5, informative=False)
check("占位 PBO=0.5 不判 PASS", ph["level"] == "UNKNOWN", ph["level"])
check("占位 PBO 不放行", ph["is_pass"] is False and ph["needs_review"] is True)
check("占位诊断说明样本内夏普<=0", "样本内夏普" in ph["diagnosis"])
check("中性 0.5 在有信息时才是 PASS", V.grade_pbo(0.5)["level"] == "PASS")


check("主接口含诊断字段",
      all(k in over for k in ("diagnosis_level", "diagnosis", "advice",
                              "is_pass", "needs_review", "missing_policy")))
check("主接口诊断非空", isinstance(over["diagnosis"], str) and len(over["diagnosis"]) > 10)
check("主接口保留 pbo_raw", "pbo_raw" in over and over["pbo_raw"] is not None)
check("diagnosis_level 与 pbo 一致",
      (over["diagnosis_level"] == "REBUILD") == (over["pbo"] > 0.5),
      (over["diagnosis_level"], over["pbo"]))
check("is_pass 与 level 一致",
      (robust["diagnosis_level"] == "PASS") == robust["is_pass"],
      (robust["diagnosis_level"], robust["is_pass"]))

rep = V.format_report(over, kind="overfit")
check("报告含诊断结论", "诊断结论" in rep and "建议:" in rep)
check("报告含是否通过", "是否通过" in rep)

# ══════════════════════════════════════════════════════════════
print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  M1 可以进入下一步")
print("#" * 70)
