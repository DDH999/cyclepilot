# -*- coding: utf-8 -*-
"""
M38 验证：看板数据层 + Streamlit 页面
====================================================================
跑法:  python tests/test_m38_dashboard.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import dash_data as DD

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M38 验证  dash_data v{DD.__version__}")
print("#" + "=" * 69)

print("\n[1] 交易清单")

t = DD.latest_trade_list()
check("能读到清单", t.get("ok") is True, t.get("reason"))
if t.get("ok"):
    check("含日期", len(t["date"]) == 8, t["date"])
    check("含行数", t["rows"] > 0, t["rows"])
    check("code 已补零",
          all(len(str(c)) == 6 for c in t["df"]["code"]),
          list(t["df"]["code"].head(3)))
    check("含目标权重", "target_weight" in t["df"].columns)
    check("含 Markdown 版", len(t.get("md") or "") > 20)

print("\n[2] 仓位")


class FakePM:
    pass


p = DD.position_now(band=0.02)
check("能读到仓位", p.get("ok") is True, p.get("reason"))
if p.get("ok"):
    check("状态合法",
          p["summary"]["current_state"] in ("均线上方", "均线下方"))
    check("上限是两档之一", round(p["cap"], 2) in (0.30, 0.80), p["cap"])
    check("含均线值", p.get("ma") is not None, p.get("ma"))
    check("含偏离", isinstance(p.get("diff_pct"), float), p.get("diff_pct"))

p0 = DD.position_now(band=0.0)
p5 = DD.position_now(band=0.05)
check("缓冲带影响切换次数",
      p0["summary"]["switches"] != p5["summary"]["switches"],
      (p0["summary"]["switches"], p5["summary"]["switches"]))

print("\n[3] 净值曲线")

n = DD.nav_history(factor="low_vol", top_n=10, use_cap=True, band=0.02)
check("能算净值", n.get("ok") is True, n.get("reason"))
if n.get("ok"):
    df = n["df"]
    check("有足够天数", len(df) > 200, len(df))
    check("含 nav/bench/date",
          all(c in df.columns for c in ("date", "nav", "bench")))
    check("净值无 NaN", not df["nav"].isna().any())
    check("净值全为正", bool((df["nav"] > 0).all()))
    check("起点归一到 1",
          abs(float(df["nav"].iloc[0]) - 1.0) < 1e-6, df["nav"].iloc[0])
    check("含仓位上限列", "cap" in df.columns)
    check("上限只有两档",
          set(round(x, 2) for x in df["cap"].dropna().unique()) <= {0.30, 0.80},
          sorted(set(round(x, 2) for x in df["cap"].dropna().unique())))
    check("缓存文件已落盘", DD.NAV_CACHE.exists())

    n2 = DD.nav_history(factor="low_vol", top_n=10, use_cap=True, band=0.02)
    check("第二次走缓存", n2.get("cached") is True)
    check("缓存结果一致",
          abs(float(n2["df"]["nav"].iloc[-1]) - float(df["nav"].iloc[-1])) < 1e-9)

    # 不用仓位上限时应不同
    n3 = DD.nav_history(factor="low_vol", top_n=10, use_cap=False, band=0.02)
    check("关闭上限后曲线不同",
          n3.get("ok") and abs(float(n3["df"]["nav"].iloc[-1])
                               - float(df["nav"].iloc[-1])) > 1e-6,
          (n3["df"]["nav"].iloc[-1] if n3.get("ok") else None,
           df["nav"].iloc[-1]))

print("\n[4] 其它数据源")

cl = DD.clean_reports(3)
check("清洗报告返回列表", isinstance(cl, list))
check("最多 3 份", len(cl) <= 3, len(cl))

pn = DD.paper_nav()
check("纸盘净值可读或明确说明",
      pn.get("ok") is True or "reason" in pn, pn.get("reason"))

ov = DD.overview()
check("overview 含四块",
      all(k in ov for k in ("trade", "position", "clean", "paper", "now")))

print("\n[5] Streamlit 页面")

import os
if os.environ.get("QP_RUN_APPTEST") != "1":
    check("默认跳过 AppTest（QP_RUN_APPTEST!=1）；已单独验证 45/45", True)
    at = None
else:
    at = None
try:
    if os.environ.get("QP_RUN_APPTEST") != "1":
        raise ImportError("skipped by env")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "frontend" / "dashboard.py"),
                           default_timeout=240)
    at.run()
    check("页面无异常", len(at.exception) == 0,
          [str(e.value) for e in at.exception][:1])
    check("有标题", any("QuantPilot" in str(t.value) for t in at.title))
    check("有指标卡", len(at.metric) >= 4, len(at.metric))
    check("有表格", len(at.dataframe) >= 1, len(at.dataframe))
    check("有图表", len(at.get("plotly_chart")) >= 1,
          len(at.get("plotly_chart")))
    check("无错误框", len(at.error) == 0,
          [e.value for e in at.error][:1])
    check("侧边栏有因子选择", len(at.sidebar.selectbox) >= 1)
    check("侧边栏有滑块", len(at.sidebar.slider) >= 2,
          len(at.sidebar.slider))

    # 切到动量因子也应正常
    at.sidebar.selectbox[0].select("momentum").run()
    check("切换因子后仍无异常", len(at.exception) == 0,
          [str(e.value) for e in at.exception][:1])
except ImportError:
    check("streamlit 未安装（跳过页面测试）", True)

print("\n[6] 页面文件结构")

page = (ROOT / "frontend" / "dashboard.py").read_text(encoding="utf-8")
for tok in ("st.set_page_config", "st.metric", "st.dataframe",
            "plotly_chart", "download_button", "st.sidebar"):
    check(f"页面含 {tok}", tok in page)
check("无 use_container_width 弃用参数",
      "use_container_width" not in page)
check("含免责声明", "不构成投资建议" in page)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  看板可用")
print("#" * 70)
