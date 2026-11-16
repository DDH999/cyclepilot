# -*- coding: utf-8 -*-
"""
M36 验证：仓位标签页 + /api/position
====================================================================
跑法:  python tests/test_m36_position_tab.py
"""
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print("# M36 验证：仓位标签页")
print("#" + "=" * 69)

html = (ROOT / "frontend" / "checkup.html").read_text(encoding="utf-8")

print("\n[1] 页面结构")

check("页面体积合理", len(html) > 55000, len(html))
for tok in ("tab-pos", "view-pos", "loadPosition", "p-state", "p-cap",
            "p-close", "p-ma", "t-pos", "/api/position"):
    check(f"含 {tok}", tok in html)

tabs = re.findall(r'id="tab-([a-z]+)"', html)
views = re.findall(r'id="view-([a-z]+)"', html)
check("七个标签页", len(tabs) == 7, tabs)
check("七个视图", len(views) == 7, views)
check("标签与视图一一对应", tabs == views, (tabs, views))
check("含 pos", "pos" in tabs)
check("保留了原有六个",
      all(x in tabs for x in ("checkup", "daily", "risk", "portfolio",
                              "code", "market")))

check("views 映射含 pos", 'pos:"view-pos"' in html)
check("切页会加载仓位", 'if(name === "pos") loadPosition();' in html)
check("有 tab-pos 监听", '$("tab-pos").addEventListener' in html)
check("有 hash 路由", 'location.hash === "#pos"' in html)

print("\n[2] 参数控件")

for pid, lo, hi in (("p-window", 20, 20), ("p-band", 2, 2),
                    ("p-confirm", 1, 1), ("p-high", 80, 80), ("p-low", 30, 30)):
    m = re.search(r'id="%s"[^>]*value="([^"]+)"' % pid, html)
    check(f"{pid} 默认值 {lo}", m is not None and m.group(1) == str(lo),
          m.group(1) if m else "未找到")
check("有刷新按钮", 'id="p-run"' in html)
check("带宽转小数（/100）", '/ 100' in html)

print("\n[3] 接口")

import api_v2 as API
from fastapi.testclient import TestClient
cli = TestClient(API.app)

r = cli.get("/api/position?window=20&band=0.02&days=20")
check("接口 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    b = r.json()
    check("返回 ok", b.get("ok") is True)
    s = b["summary"]
    for k in ("current_state", "current_cap", "avg_cap", "switches",
              "switches_per_year", "as_of"):
        check(f"summary 含 {k}", k in s, s.get(k))
    check("当前状态合法",
          s["current_state"] in ("均线上方", "均线下方"), s["current_state"])
    check("当前上限是两档之一",
          round(float(s["current_cap"]), 2) in (0.30, 0.80),
          s["current_cap"])

    ser = b["series"]
    check("明细至少 5 条", len(ser) >= 5, len(ser))
    check("明细最多 20 条", len(ser) <= 20, len(ser))
    row = ser[-1]
    for k in ("date", "close", "ma", "cap", "above"):
        check(f"明细行含 {k}", k in row, row.get(k))
    check("cap 只有两档",
          set(round(x["cap"], 2) for x in ser) <= {0.30, 0.80},
          set(round(x["cap"], 2) for x in ser))
    check("含文本报告", "仓位管理器" in b.get("text", ""))
    check("记录参数", b["params"]["band"] == 0.02, b["params"])

    # 参数确实生效
    r2 = cli.get("/api/position?band=0.0&days=5")
    b2 = r2.json()
    check("不同参数返回不同切换次数",
          b2["summary"]["switches"] != s["switches"],
          (b2["summary"]["switches"], s["switches"]))

    # 非法参数
    r3 = cli.get("/api/position?cap_low=0.9&cap_high=0.1")
    check("非法仓位参数返回错误", r3.status_code >= 400, r3.status_code)

    # 自定义上限
    r4 = cli.get("/api/position?cap_low=0.1&cap_high=0.5&window=20")
    if r4.status_code == 200:
        caps4 = set(round(x["cap"], 2) for x in r4.json()["series"])
        check("自定义上限生效", caps4 <= {0.10, 0.50}, caps4)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  仓位页可用")
print("#" * 70)
