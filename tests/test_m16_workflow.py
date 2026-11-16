# -*- coding: utf-8 -*-
"""
M16 验证：风控闭环 + 统一入口
====================================================================
跑法:  python tests/test_m16_workflow.py    （不联网）
"""
import io
import json
import shutil
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import quantpilot as QP
import risk_state as RS
import riskcontrol as RC

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m16"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M16 验证  risk_state v{RS.__version__} / quantpilot v{QP.__version__}")
print("#" + "=" * 69)

D_NORM = {"action": "NORMAL", "scale": 1.0, "reasons": [],
          "crowding": {"score": 40}, "drawdown": {"current_drawdown": -0.02}}
D_RED = {"action": "REDUCE", "scale": 0.5,
         "reasons": [{"why": "换手超限"}],
         "crowding": {"score": 60}, "drawdown": {"current_drawdown": -0.13}}
D_HALT = {"action": "HALT", "scale": 0.0,
          "reasons": [{"why": "回撤超上限"}],
          "crowding": {"score": 50}, "drawdown": {"current_drawdown": -0.22}}

print("\n[1] 记录与读取")

sp = TMP / "state.json"
check("无文件返回空", RS.load_history(sp) == [])
rec = RS.record(D_RED, {"weights": {"A": 0.5, "B": 0.5}, "capital": 100000}, path=sp)
check("写入返回记录", rec["action"] == "REDUCE")
check("记录含仓位", abs(rec["exposure"] - 1.0) < 1e-9, rec["exposure"])
check("文件已生成", sp.exists())

hist = RS.load_history(sp)
check("读回 1 条", len(hist) == 1)
check("含时间戳", "time" in hist[0])
check("含原因", hist[0]["reasons"] == ["换手超限"], hist[0]["reasons"])
check("latest 可取", RS.latest(sp)["action"] == "REDUCE")

# 损坏文件不崩
sp_bad = TMP / "bad.json"
sp_bad.write_text("{not json", encoding="utf-8")
check("损坏文件返回空而不崩", RS.load_history(sp_bad) == [])

# 上限截断
sp_many = TMP / "many.json"
for i in range(RS.MAX_RECORDS + 20):
    RS.record(D_NORM, {"weights": {"A": 1.0}}, path=sp_many)
many = RS.load_history(sp_many)
check("记录数被截断到上限", len(many) <= RS.MAX_RECORDS, len(many))
check("保留最近记录", len(many) == RS.MAX_RECORDS, len(many))

print("\n[2] 闭环：建议 vs 实际")

sp2 = TMP / "ft.json"
RS.record(D_NORM, {"weights": {"A": 0.5, "B": 0.5}}, path=sp2)
RS.record(D_RED, {"weights": {"A": 0.5, "B": 0.5}}, path=sp2)
RS.record(D_NORM, {"weights": {"A": 0.2}, "cash_ratio": 0.6}, path=sp2)
ft = RS.followthrough(path=sp2)
check("可判定", ft["ok"] is True)
check("识别 1 次建议", ft["n_advise"] == 1, ft["n_advise"])
check("判定为已执行", ft["n_followed"] == 1, ft["advices"])
check("执行率 100%", abs(ft["follow_rate"] - 1.0) < 1e-9)
check("结论正面", "良好" in ft["verdict"], ft["verdict"])

sp3 = TMP / "ft2.json"
RS.record(D_RED, {"weights": {"A": 0.5, "B": 0.5}}, path=sp3)
RS.record(D_NORM, {"weights": {"A": 0.5, "B": 0.5}}, path=sp3)
ft2 = RS.followthrough(path=sp3)
check("未减仓判定为未执行", ft2["n_followed"] == 0, ft2["advices"])
check("结论严厉", "形同虚设" in ft2["verdict"], ft2["verdict"])

sp4 = TMP / "ft3.json"
RS.record(D_NORM, {"weights": {"A": 1.0}}, path=sp4)
ft4 = RS.followthrough(path=sp4)
check("记录不足给出原因", ft4["ok"] is False and "不足" in ft4["reason"])

sp5 = TMP / "ft4.json"
RS.record(D_NORM, {"weights": {"A": 1.0}}, path=sp5)
RS.record(D_HALT, {"weights": {"A": 1.0}}, path=sp5)
RS.record(D_NORM, {"weights": {"A": 0.3}, "cash_ratio": 0.7}, path=sp5)
ft5 = RS.followthrough(path=sp5)
check("HALT 也被计入建议", ft5["n_halt"] == 1, ft5["n_halt"])
check("HALT 后减仓算执行", ft5["n_followed"] == 1, ft5["advices"])

# 最新一条建议还没有"下一条"可比 → 不应崩
sp6 = TMP / "ft5.json"
RS.record(D_NORM, {"weights": {"A": 1.0}}, path=sp6)
RS.record(D_HALT, {"weights": {"A": 1.0}}, path=sp6)
ft6 = RS.followthrough(path=sp6)
check("最新建议无后续时返回完整结构",
      "n_halt" in ft6 and ft6["n_advise"] == 0, ft6.get("n_advise"))

check("报告可渲染", "风控闭环" in RS.format_report(hist, ft))
check("空历史报告友好", "还没有任何风控记录" in RS.format_report([], None))

print("\n[3] 拥挤度覆盖率（回归：曾经误报 100 分）")

w3 = pd.DataFrame([{"110022": 0.25, "510300": 0.17, "000001": 0.16}])
c3 = RC.crowding_score(w3, turnover_annual=12.0)
check("不再被归一化放大到 100", c3["score"] < 80, c3["score"])
check("报告覆盖率", c3.get("coverage") is not None, c3.get("coverage"))
check("标注低置信", c3.get("low_confidence") is True, c3.get("low_confidence"))
check("提示数据不全", "项可算" in c3["note"], c3["note"])

c4 = RC.crowding_score(w3, turnover_annual=12.0, bench_r2=0.9,
                       participation=0.05)
check("四项齐全时可比", c4.get("n_available") == 4, c4.get("n_available"))
check("四项齐全不高估", c4["score"] <= 100)

print("\n[4] 统一入口")

check("命令已注册", len(QP.COMMANDS) >= 7, len(QP.COMMANDS))
for name in ("status", "portfolio", "dca", "risk", "daily",
             "calibrate", "check", "serve"):
    check(f"注册 {name}", name in QP.COMMANDS)

parser = QP.build_parser()
args = parser.parse_args(["risk", "--turnover", "8", "--capital", "200000"])
check("参数解析 turnover", args.turnover == 8.0)
check("参数解析 capital", args.capital == 200000.0)
check("默认子命令为 status",
      parser.parse_args([]).cmd is None)

buf = io.StringIO()
with redirect_stdout(buf):
    rc = QP.main(["status"])
out = buf.getvalue()
check("status 返回 0", rc == 0, rc)
check("status 打印标题", "QuantPilot 状态" in out)
check("status 列出文件", "holdings.csv" in out)
check("status 列出命令", "calibrate" in out)

buf2 = io.StringIO()
with redirect_stdout(buf2):
    rc2 = QP.main(["__nope__"])
check("未知命令返回 2", rc2 == 2, rc2)

check("status 给出下一步建议", "[下一步建议]" in out)
check("下一步有具体动作", ("- " in out.split("[下一步建议]")[-1]))

print("\n[5] 端到端：risk 记账可被 status 读到")

_orig_state = RS.RISK_STATE
try:
    RS.RISK_STATE = TMP / "e2e_state.json"
    RS.record(D_RED, {"weights": {"A": 0.6}, "cash_ratio": 0.4})
    h = RS.load_history()
    check("端到端能读到", len(h) == 1, len(h))
    ft_e = RS.followthrough(h)
    check("单条记录提示需要更多",
          ft_e["ok"] is False or ft_e.get("n_advise") == 0)
finally:
    RS.RISK_STATE = _orig_state

print("\n" + "#" * 70)
print(chr(10) + "[6] 定投接口 + 新标签页")


def _fake_nav(code, kind="open", **kw):
    idx = pd.bdate_range("2024-01-01", periods=500)
    return pd.Series(np.linspace(2.0, 2.6, 500), index=idx)


import fund as FD
_orig_nav = FD.fetch_nav
FD.fetch_nav = _fake_nav
try:
    import api_v2 as API2
    from fastapi.testclient import TestClient
    _cli = TestClient(API2.app)
    _r = _cli.get("/api/dca")
    check("定投接口 200", _r.status_code == 200, _r.status_code)
    _body = _r.json()
    check("定投返回 ok", _body.get("ok") is True, _body.get("detail"))
    _res = _body.get("result") or {}
    check("含累计投入", _res.get("invested", 0) > 0, _res.get("invested"))
    check("含 XIRR", _res.get("xirr") is not None, _res.get("xirr"))
    check("nav 留空被自动补齐", _res.get("n_missing") == 0, _res.get("missing"))
    check("返回文本", "定投分析" in _body.get("text", ""))
finally:
    FD.fetch_nav = _orig_nav

_html = (ROOT / "frontend" / "checkup.html").read_text(encoding="utf-8")
for _tok, _why in [("tab-portfolio", "持仓标签"), ("tab-code", "代码体检标签"),
                   ("view-portfolio", "持仓视图"), ("view-code", "代码视图"),
                   ("/api/dca", "定投接口"),
                   ("/api/checkup_code", "代码体检接口"),
                   ("runCodeCheckup", "代码执行函数")]:
    check("页面含" + _why, _tok in _html, _tok)
check("五个视图齐全",
      all(("view-" + x) in _html for x in
          ("checkup", "daily", "risk", "portfolio", "code")))

total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  闭环与统一入口可用")
print("#" * 70)
