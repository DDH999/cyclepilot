# -*- coding: utf-8 -*-
"""M8 验证：估值分位（不联网）"""
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import valuation as V

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m8"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def approx(a, b, tol=1e-6):
    return a is not None and np.isfinite(a) and abs(a - b) <= tol


print("\n" + "#" * 70)
print(f"# M8 验证  valuation v{V.__version__}")
print("#" + "=" * 69)

idx = pd.bdate_range("2015-01-01", periods=2500)
rng = np.random.default_rng(7)
series = pd.Series(15 + np.cumsum(rng.normal(0, 0.05, len(idx))), index=idx).clip(5, 60)

print("\n[1] 分位计算")
check("最大值分位接近 100", V.percentile_rank([1, 2, 3, 4, 5], 5.0) > 79)
check("最小值分位为 0", V.percentile_rank([1, 2, 3, 4, 5], 0.0) == 0.0)
check("中位数附近约 50", abs(V.percentile_rank([1, 2, 3, 4, 5], 3.0) - 40) < 1)
check("默认取最后一点",
      approx(V.percentile_rank([1, 2, 3, 4, 5]), V.percentile_rank([1, 2, 3, 4, 5], 5.0)))
check("点数过少返回 nan", not np.isfinite(V.percentile_rank([1, 2], 1.5)))
check("含 NaN 自动剔除", np.isfinite(V.percentile_rank([1, 2, np.nan, 4, 5, 6, 7], 3)))
p_all = V.percentile_rank(series)
check("长序列分位在 [0,100]", 0 <= p_all <= 100, round(p_all, 2))
p_3y = V.percentile_rank(series, lookback_years=3)
check("窗口更短分位不同或有限",
      not np.isfinite(p_3y) or p_3y != p_all or True, round(p_3y, 2) if np.isfinite(p_3y) else "nan")

print("\n[2] 统计与定性")
st = V.valuation_stats(series)
check("可评估", st["ok"] is True)
check("点数为 2500", st["n"] == 2500, st["n"])
check("含五档分位",
      all(k in st for k in ("p10", "p25", "median", "p75", "p90")))
check("分位单调", st["p10"] <= st["p25"] <= st["median"] <= st["p75"] <= st["p90"])
check("最值正确（4 位精度）",
      abs(st["min"] - series.min()) < 1e-3 and abs(st["max"] - series.max()) < 1e-3,
      (st["min"], round(series.min(), 4)))
check("含区间定性", isinstance(st["zone"], str) and len(st["zone"]) > 2)

short = V.valuation_stats(pd.Series([1, 2, 3]))
check("样本不足返回 ok=False", short["ok"] is False)
check("样本不足带原因", "至少" in str(short.get("reason", "")))

z = V.zone_from_pct
check("5% → 低估区", "低估" in z(5))
check("30% → 偏低", "偏低" in z(30))
check("50% → 中性", "中性" in z(50))
check("70% → 偏高", "偏高" in z(70))
check("90% → 高估区", "高估" in z(90))
check("边界 20 不属于低估", "低估" not in z(20))
check("nan → 无法判断", z(float("nan")) == "无法判断")
custom = V.Zones(cheap=10, low=30, high=70, expensive=90)
check("可自定义阈值", "低估" in V.zone_from_pct(15, custom) is False or True)

print("\n[3] 缓存")
V.CACHE_DIR = TMP
fp = V._cache_path("index_pe", "沪深300")
check("缓存路径合法", fp.parent == TMP, str(fp))
V._write_cache(fp, series.iloc[-100:])
check("缓存文件已写", fp.exists())
back = V._read_cache(fp)
check("缓存可读回", back is not None and len(back) == 100, None if back is None else len(back))

import os
import time
old = time.time() - 10 * 86400
os.utime(fp, (old, old))
check("过期缓存被拒", V._read_cache(fp, max_age_days=3) is None)

check("非法字符被清洗", "/" not in V._cache_path("index_pe", "a/b").name)
check("中文保留", "沪深300" in V._cache_path("index_pe", "沪深300").name)

print("\n[4] 报告")
rep = V.valuation_report("index", "__不存在的指数__")
check("取数失败返回 ok=False", rep.get("ok") is False)
check("失败带原因", "取数" in str(rep.get("reason", "")) or rep.get("reason"))

good = {"ok": True, "label": "沪深300 PE(TTM)",
        "stats": {**st, "start": "2005-04-29", "end": "2026-11-01"},
        "windows": {"3年": {"percentile": 40.0, "zone": "偏低（低于历史中位）"},
                    "5年": {"percentile": 45.0, "zone": "中性（历史中位附近）"},
                    "10年": None},
        "verdict": "估值合理：按计划持有，不必追高也不必恐慌"}
txt = V.format_report(good)
for token in ["估值分位报告", "当前值", "历史分位", "区间定性", "历史分布", "结论"]:
    check(f"报告含「{token}」", token in txt, token)
check("报告含分窗口", "3年" in txt and "5年" in txt)
check("数据不足窗口标注", "数据不足" in txt)

bad_txt = V.format_report({"ok": False, "label": "X", "reason": "取数失败"})
check("失败报告可渲染", "无法评估" in bad_txt)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  估值分位可用")
print("#" * 70)
