# -*- coding: utf-8 -*-
"""
M4 验证：API + 前端页面
====================================================================
跑法:  python tests/test_m4.py      （不需要真实网络）

分组
  [1] 收益序列解析（多种格式）
  [2] 权重解析
  [3] 体检主流程 run_checkup
  [4] HTTP 接口（TestClient）
  [5] 前端页面静态检查
====================================================================
"""
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import api_v2 as API

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M4 验证  api={API.__version__}")
print("#" * 70)

DATES = pd.bdate_range("2023-01-02", periods=420)
rng = np.random.default_rng(20261104)
RETS = pd.Series(rng.normal(0.0007, 0.011, len(DATES)), index=DATES)
CODES = [f"{i:06d}" for i in range(20)]


def rets_csv_as_ret():
    return "date,ret\n" + "\n".join(f"{d.date()},{v:.6f}" for d, v in RETS.items())


def rets_csv_as_nav():
    nav = (1 + RETS).cumprod()
    return "date,nav\n" + "\n".join(f"{d.date()},{v:.6f}" for d, v in nav.items())


def weights_csv():
    w = pd.DataFrame(1 / len(CODES), index=DATES, columns=CODES)
    lines = ["date," + ",".join(CODES)]
    for d, row in w.iterrows():
        lines.append(str(d.date()) + "," + ",".join(f"{x:.4f}" for x in row))
    return "\n".join(lines)


def bench_csv():
    b = pd.Series(rng.normal(0.0004, 0.010, len(DATES)), index=DATES)
    return "date,ret\n" + "\n".join(f"{d.date()},{v:.6f}" for d, v in b.items())


# ══════════════════════════════════════════════════════════════
print("\n[1] 收益序列解析")

s1 = API.parse_series_csv(rets_csv_as_ret())
check("date,ret 解析成功", len(s1) == len(RETS), len(s1))
check("索引是日期", isinstance(s1.index, pd.DatetimeIndex))
check("数值接近原值", abs(s1.mean() - RETS.mean()) < 1e-6)

s2 = API.parse_series_csv(rets_csv_as_nav())
check("date,nav 转收益", len(s2) == len(RETS) - 1, len(s2))  # pct_change 丢失首行
check("净值转收益后均值接近", abs(s2.mean() - RETS.mean()) < 1e-4,
      (s2.mean(), RETS.mean()))

s3 = API.parse_series_csv("date,收益\n" + "\n".join(
    f"{d.date()},{v*100:.4f}" for d, v in RETS.items()))
check("中文列名 + 百分数自动折算", abs(s3.mean() - RETS.mean()) < 1e-4, s3.mean())

noheader = "\n".join(f"{d.date()},{v:.6f}" for d, v in RETS.items())
s4 = API.parse_series_csv(noheader)
check("无表头也能解析", len(s4) == len(RETS))

check("BOM 可处理", len(API.parse_series_csv("\ufeff" + rets_csv_as_ret())) == len(RETS))

for bad, why in [("", "空文本"), ("date,ret\n", "无数据行"),
                 ("date,ret\n2023-01-03,0.01\n2023-01-04,0.02", "样本不足")]:
    try:
        API.parse_series_csv(bad)
        check(f"应拒绝：{why}", False)
    except ValueError:
        check(f"应拒绝：{why}", True)

# ══════════════════════════════════════════════════════════════
print("\n[2] 权重解析")

w = API.parse_weights_csv(weights_csv())
check("宽表权重可解析", w is not None and w.shape[1] == len(CODES), None if w is None else w.shape)
check("空文本返回 None", API.parse_weights_csv("") is None)
check("垃圾文本返回 None", API.parse_weights_csv("abc\ndef") is None)
check("单列返回 None", API.parse_weights_csv("date,A\n2023-01-03,1\n2023-01-04,1") is None)

# ══════════════════════════════════════════════════════════════
print("\n[3] 体检主流程")

CODE_CHEAT = ("import pandas as pd\nclose = df['close']\n"
              "sig = close.shift(-1) / close - 1\nret = close.pct_change()\n")
CODE_CLEAN = ("ret = df['close'].pct_change(fill_method=None)\n"
              "sig = ret.shift(1)\n")

r_full = API.run_checkup(code=CODE_CLEAN, returns_csv=rets_csv_as_ret(),
                         weights_csv=weights_csv(), bench_csv=bench_csv(),
                         use_llm=False)
check("完整输入跑通", r_full["ok"] is True)
check("含六维报告", len(r_full["report"]["radar"]) == 6)
check("含诊断三段", all(r_full["diagnosis"][k] for k in ("summary", "fatal", "advice")))
check("诊断走模板（未启用LLM）", r_full["diagnosis"]["source"] == "template")
check("含输入元信息", r_full["report"]["inputs"]["has_code"] is True)
check("权重被识别", r_full["report"]["inputs"]["has_weights"] is True)
check("基准被识别", r_full["report"]["inputs"]["has_bench"] is True)

r_cheat = API.run_checkup(code=CODE_CHEAT, returns_csv=rets_csv_as_ret(),
                          use_llm=False)
check("未来函数 → REJECTED", r_cheat["report"]["verdict"] == "REJECTED",
      r_cheat["report"]["verdict"])
check("严厉文案", "主观作弊" in r_cheat["report"]["summary"],
      r_cheat["report"]["summary"][:30])
check("命中明细非空", len(r_cheat["findings"]) >= 2, len(r_cheat["findings"]))
check("命中含规则号", r_cheat["findings"][0].get("rule", "").startswith("F"))
check("作弊时诊断也严厉", "高危" in r_cheat["diagnosis"]["summary"])

r_nocode = API.run_checkup(code="", returns_csv=rets_csv_as_ret(), use_llm=False)
check("无代码仍可体检", r_nocode["ok"] is True)
check("无代码给提示", any("未提供策略代码" in w for w in r_nocode["warnings"]),
      r_nocode["warnings"])

r_now = API.run_checkup(code=CODE_CLEAN, returns_csv=rets_csv_as_ret(), use_llm=False)
check("无权重时纯度降级", r_now["report"]["inputs"]["has_weights"] is False)
check("无权重不崩溃", r_now["report"]["verdict"] in
      ("PASS", "WARNING", "FAIL", "REJECTED"))

try:
    API.run_checkup(code="", returns_csv="")
    check("空收益序列被拒", False)
except ValueError:
    check("空收益序列被拒", True)

# ══════════════════════════════════════════════════════════════
print("\n[4] HTTP 接口")

from fastapi.testclient import TestClient

client = TestClient(API.app)

h = client.get("/api/health")
check("健康检查 200", h.status_code == 200, h.status_code)
check("健康检查字段", set(["ok", "version", "llm_configured"]) <= set(h.json()))

resp = client.post("/api/checkup", json={
    "code": CODE_CLEAN,
    "returns_csv": rets_csv_as_ret(),
    "weights_csv": weights_csv(),
    "bench_csv": bench_csv(),
    "use_llm": False,
})
check("体检接口 200", resp.status_code == 200, resp.status_code)
body = resp.json()
check("返回 ok", body.get("ok") is True)
check("含 report", "report" in body and "diagnosis" in body)
check("六维齐全", len(body["report"]["radar"]) == 6)

resp_bad = client.post("/api/checkup", json={"code": "", "returns_csv": ""})
check("空输入返回 400", resp_bad.status_code == 400, resp_bad.status_code)
check("400 带错误说明", "收益序列" in str(resp_bad.json().get("detail", "")),
      resp_bad.json().get("detail"))

resp_cheat = client.post("/api/checkup", json={
    "code": CODE_CHEAT, "returns_csv": rets_csv_as_ret(), "use_llm": False})
check("作弊请求 200 且结论 REJECTED",
      resp_cheat.status_code == 200
      and resp_cheat.json()["report"]["verdict"] == "REJECTED")

idx = client.get("/")
check("首页 200", idx.status_code == 200, idx.status_code)
check("首页是 HTML", "text/html" in idx.headers.get("content-type", ""))
check("首页含雷达 canvas", "radar" in idx.text and "<canvas" in idx.text)

# ══════════════════════════════════════════════════════════════
print("\n[5] 前端静态检查")

html = (ROOT / "frontend" / "checkup.html").read_text(encoding="utf-8")
for token, why in [
    ("/api/checkup", "调用体检接口"),
    ("f-code", "代码上传框"),
    ("f-ret", "收益上传框"),
    ("f-w", "权重上传框"),
    ("f-b", "基准上传框"),
    ("getContext(\"2d\")", "Canvas 雷达图"),
    ("合定性", "渲染三段诊断"),
    ("FileReader", "浏览器端读取文件"),
    ("use_llm", "大模型开关"),
]:
    check(f"页面含 {why}", token in html, token)

check("无外部依赖（无 CDN）",
      "http://" not in html.replace("http://127.0.0.1", "")
      and "https://" not in html)
check("页面体积合理", len(html) > 8000, f"{len(html)} 字符")

# ══════════════════════════════════════════════════════════════
print("\n[6] 日更接口（M5）")

import json

import daily_job as DJmod

tmp_daily = ROOT / ".tmp" / "test_m4_daily"
tmp_daily.mkdir(parents=True, exist_ok=True)
tmp_nav = ROOT / ".tmp" / "test_m4_nav.csv"

fixture_latest = {
    "date": "2026-11-07",
    "generated_at": "2026-11-07T15:30:00",
    "watchlist": ["600519", "000001"],
    "nav_days": 400,
    "nav_latest": 112756.3,
    "report": {"verdict": "PASS", "grade": "B", "score": 80.28,
               "radar": {"anti_cheat": None, "anti_overfit": 70.0}},
    "diagnosis": {"summary": "综合定性：中性偏好", "fatal": "核心致命伤：无",
                  "advice": "整改建议：保持", "source": "template"},
    "changes": [{"kind": "first_run", "level": "low", "msg": "首次体检，建立基线"}],
}
(tmp_daily / "latest.json").write_text(
    json.dumps(fixture_latest, ensure_ascii=False), encoding="utf-8")
(tmp_daily / "2026-11-06.json").write_text(json.dumps(
    {**fixture_latest, "date": "2026-11-06",
     "report": {**fixture_latest["report"], "grade": "C", "score": 60.0}},
    ensure_ascii=False), encoding="utf-8")
(tmp_daily / "2026-11-07.json").write_text(
    json.dumps(fixture_latest, ensure_ascii=False), encoding="utf-8")
tmp_nav.write_text("date,nav\n2026-11-05,100000.0\n2026-11-06,101000.0\n"
                   "2026-11-07,102500.0\n", encoding="utf-8")

_old_daily, _old_nav = API.DAILY_DIR, API.NAV_CSV
_old_run = DJmod.run_daily
try:
    API.DAILY_DIR, API.NAV_CSV = tmp_daily, tmp_nav

    r_latest = client.get("/api/daily/latest")
    check("日更 latest 200", r_latest.status_code == 200, r_latest.status_code)
    body = r_latest.json()
    check("latest 含报告", body.get("report", {}).get("grade") == "B")
    check("latest 含变化", isinstance(body.get("changes"), list))

    r_hist = client.get("/api/daily/history?limit=10")
    check("日更 history 200", r_hist.status_code == 200)
    items = r_hist.json().get("items", [])
    check("history 有记录", len(items) == 2, len(items))
    check("history 按日期升序", items[0]["date"] <= items[-1]["date"])
    check("history 含摘要字段",
          all(k in items[0] for k in ("date", "verdict", "grade", "score", "n_changes")))

    r_nav = client.get("/api/paper/nav?limit=100")
    check("净值接口 200", r_nav.status_code == 200)
    nb = r_nav.json()
    check("净值点数正确", nb.get("n") == 3, nb.get("n"))
    check("日期与净值等长", len(nb.get("dates", [])) == len(nb.get("nav", [])))
    check("净值末值正确", abs(nb["nav"][-1] - 102500.0) < 1e-6)

    # 手动触发：打桩，避免真拉网络
    DJmod.run_daily = lambda **kw: {
        "date": "2026-11-08",
        "report": {"verdict": "PASS", "grade": "B", "score": 81.0},
        "diagnosis": {"summary": "综合定性：中性偏好"},
        "changes": [], "nav_latest": 103000.0, "nav_days": 401, "elapsed_ms": 12.3,
    }
    r_run = client.post("/api/daily/run")
    check("手动更新 200", r_run.status_code == 200, r_run.status_code)
    rb = r_run.json()
    check("手动更新返回记录", rb.get("ok") and rb.get("report", {}).get("grade") == "B")
    check("手动更新含耗时", rb.get("elapsed_ms") == 12.3)

    # 日更失败也应给出干净的错误
    DJmod.run_daily = lambda **kw: (_ for _ in ()).throw(FileNotFoundError("缺 watchlist.txt"))
    r_bad = client.post("/api/daily/run")
    check("日更失败返回 400", r_bad.status_code == 400, r_bad.status_code)
    check("错误信息可读", "watchlist" in str(r_bad.json().get("detail", "")))

    # 无记录时应 404 而不是 500
    API.DAILY_DIR = ROOT / ".tmp" / "no_such_dir_xyz"
    r_404 = client.get("/api/daily/latest")
    check("无记录返回 404", r_404.status_code == 404, r_404.status_code)
finally:
    API.DAILY_DIR, API.NAV_CSV = _old_daily, _old_nav
    DJmod.run_daily = _old_run

# 前端新增内容
html2 = (ROOT / "frontend" / "checkup.html").read_text(encoding="utf-8")
for token, why in [
    ("tab-daily", "每日监控标签页"),
    ("view-daily", "监控视图容器"),
    ("navchart", "净值曲线 canvas"),
    ("/api/daily/latest", "调用日更接口"),
    ("/api/daily/run", "手动刷新接口"),
    ("/api/paper/nav", "净值接口"),
    ("立即更新", "刷新按钮"),
]:
    check(f"页面含 {why}", token in html2, token)
check("双视图都在", "view-checkup" in html2 and "view-daily" in html2)
check("无外部 CDN", "http://" not in html2 and "https://" not in html2)
check("页面体积合理", len(html2) > 15000, f"{len(html2)} 字符")

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
print("\n[7] 风控仪表盘接口")

tmp_nav2 = ROOT / ".tmp" / "test_m4_risk_nav.csv"
tmp_nav2.parent.mkdir(parents=True, exist_ok=True)
rows = ["date,nav"]
d0 = pd.Timestamp("2025-01-02")
rng_r = np.random.default_rng(31)
vals = 100000 * np.cumprod(1 + rng_r.normal(0.0006, 0.011, 400))
for v in vals:
    while True:
        d0 = d0 + pd.Timedelta(days=1)
        if d0.weekday() < 5:
            break
    rows.append(f"{d0.date()},{v:.2f}")
tmp_nav2.write_text("\n".join(rows), encoding="utf-8")

_old_nav2 = API.NAV_CSV
try:
    API.NAV_CSV = tmp_nav2
    r_risk = client.get("/api/risk?capital=100000&turnover=12")
    check("风控接口 200", r_risk.status_code == 200, r_risk.status_code)
    rk = r_risk.json()
    check("返回 ok", rk.get("ok") is True)
    for key in ("decision", "drawdown", "crowding", "style", "data",
                "walkforward", "tail", "fragility", "cost", "limits", "inputs"):
        check(f"含 {key}", key in rk, key)
    check("动作合法",
          rk["decision"]["action"] in ("NORMAL", "REDUCE", "HALT"),
          rk["decision"]["action"])
    check("换手 12x 触发减仓", rk["decision"]["action"] == "REDUCE",
          rk["decision"]["action"])
    check("减仓系数 0.5", rk["decision"]["scale"] == 0.5)
    check("拥挤度有分数", rk["crowding"].get("score") is not None)
    check("数据体检有等级", isinstance(rk["data"].get("level"), str))
    check("滚动样本外有分段", len(rk["walkforward"].get("folds", [])) >= 2,
          len(rk["walkforward"].get("folds", [])))
    check("成本为正", rk["cost"]["total"] > 0)
    check("容量上限为正", (rk.get("capacity") or 0) > 0)
    check("含风控阈值", "max_drawdown" in (rk["limits"] or {}))
    check("输入回显正确", rk["inputs"]["turnover"] == 12.0)

    r_low = client.get("/api/risk?capital=100000&turnover=1")
    check("低换手不再因换手触发",
          all("换手" not in x["why"] for x in r_low.json()["decision"]["reasons"]),
          [x["why"] for x in r_low.json()["decision"]["reasons"]])

    API.NAV_CSV = ROOT / ".tmp" / "no_such_nav.csv"
    r_404r = client.get("/api/risk")
    check("无净值返回 404", r_404r.status_code == 404, r_404r.status_code)

    short_fp = ROOT / ".tmp" / "test_m4_short_nav.csv"
    short_fp.write_text("date,nav\n2026-01-05,100000\n2026-01-06,100100\n"
                        "2026-01-07,100200\n", encoding="utf-8")
    API.NAV_CSV = short_fp
    r_400 = client.get("/api/risk")
    check("净值过短返回 400", r_400.status_code == 400, r_400.status_code)
finally:
    API.NAV_CSV = _old_nav2

html3 = (ROOT / "frontend" / "checkup.html").read_text(encoding="utf-8")
for token, why in [
    ("tab-risk", "风控标签页"),
    ("view-risk", "风控视图容器"),
    ("/api/risk", "调用风控接口"),
    ("四道闸门", "闸门表格"),
    ("暂停交易", "熔断文案"),
    ("容量上限", "容量展示"),
    ("loadRisk", "风控加载函数"),
]:
    check(f"页面含 {why}", token in html3, token)
check("三视图齐全",
      all(f"view-{x}" in html3 for x in ("checkup", "daily", "risk")))
check("页面体积合理", len(html3) > 30000, f"{len(html3)} 字符")

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  M4 可用")
print("#" * 70)
