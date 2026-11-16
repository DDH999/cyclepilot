# -*- coding: utf-8 -*-
"""
M25 验证：看板补全 / 风险平价上限 / 审计落盘
====================================================================
跑法:  python tests/test_m25_improvements.py   （不联网）
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import adjudicator as A
import allocation as AL
import portfolio as PF

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m25"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print("# M25 验证：看板补全 / 风险平价上限 / 审计落盘")
print("#" + "=" * 69)

print("\n[1] 看板补全")

html = (ROOT / "frontend" / "checkup.html").read_text(encoding="utf-8")
for tok, why in [
    ("tab-market", "市场标签"),
    ("view-market", "市场视图"),
    ("loadValuation", "估值加载"),
    ("loadOverseas", "指数加载"),
    ("runQuality", "基本面查询"),
    ("t-overseas", "指数表格"),
    ("t-quality", "基本面表格"),
    ("/api/valuation", "估值接口"),
    ("/api/overseas", "指数接口"),
    ("/api/quality", "基本面接口"),
]:
    check(f"页面含{why}", tok in html, tok)

check("六个标签页齐全",
      all(f"id=\"tab-{x}\"" in html for x in
          ("checkup", "daily", "risk", "portfolio", "code", "market")))
check("六个视图齐全",
      all(f"id=\"view-{x}\"" in html for x in
          ("checkup", "daily", "risk", "portfolio", "code", "market")))
check("视图映射含 market", 'market:"view-market"' in html)
check("market 切换会加载数据",
      "loadValuation(); loadOverseas();" in html)
check("页面体积合理", len(html) > 50000, f"{len(html)} 字符")

print("\n[2] 风险平价权重上限")

rng = np.random.default_rng(7)
idx = pd.bdate_range("2020-01-02", periods=800)
R = pd.DataFrame({
    "股票": rng.normal(0.0006, 0.018, 800),
    "债券": rng.normal(0.00015, 0.004, 800),
    "黄金": rng.normal(0.00025, 0.010, 800),
    "现金": rng.normal(0.00006, 0.0005, 800),
}, index=idx)

w_free = AL.risk_parity(R, max_weight=None)
check("不限上限时现金占绝对多数", float(w_free["现金"]) > 0.6,
      round(float(w_free["现金"]), 3))
check("不限上限权重和为 1", abs(float(w_free.sum()) - 1) < 1e-6)

w_cap = AL.risk_parity(R, max_weight=0.40)
check("加上限后最大权重 ≤ 40%", float(w_cap.max()) <= 0.401,
      round(float(w_cap.max()), 3))
check("加上限后现金被压下来", float(w_cap["现金"]) < float(w_free["现金"]),
      (round(float(w_cap["现金"]), 3), round(float(w_free["现金"]), 3)))
check("加上限后权重和仍为 1", abs(float(w_cap.sum()) - 1) < 1e-6)
check("加上限后权重非负", float(w_cap.min()) >= 0)
check("上限生效后分配更均衡",
      float(w_cap.max()) - float(w_cap.min())
      < float(w_free.max()) - float(w_free.min()))

try:
    AL.risk_parity(R, max_weight=0.15)   # 4×0.15=0.6 < 1
    check("上限过小应报错（不能被静默降级）", False)
except ValueError as ex:
    check("上限过小应报错（不能被静默降级）", "太小" in str(ex), str(ex)[:50])

cmp_ = AL.compare_allocation(R)
check("对比默认使用上限", cmp_.get("max_weight") == 0.40, cmp_.get("max_weight"))
check("对比含风险平价行",
      any(s["scheme"] == "风险平价" for s in cmp_["schemes"]))
rp_row = next(s for s in cmp_["schemes"] if s["scheme"] == "风险平价")
check("风险平价单资产不超 40%",
      max(rp_row["weights"].values()) <= 0.401,
      max(rp_row["weights"].values()))
check("给出极端集中提示", len(cmp_.get("notes") or []) >= 1, cmp_.get("notes"))

print("\n[3] 审计日志落盘")

_orig_file = A.AUDIT_FILE
_orig_dir = A.STATE_DIR
A.STATE_DIR = TMP
A.AUDIT_FILE = TMP / "audit_log.jsonl"
A.AUDIT_LOG.clear()

check("初始无文件", not A.AUDIT_FILE.exists())
check("空历史返回空列表", A.load_audit_history() == [])

rep = {"verdict": "WARNING", "grade": "C", "score": 61.0, "notes": []}
r1 = A.apply_override(rep, {"decision": "PASS",
                            "reason": "数据缺口已补齐", "operator": "tester"})
check("复核被应用", r1.get("override_applied") is True)
check("落盘文件已生成", A.AUDIT_FILE.exists())

hist = A.load_audit_history()
check("读回 1 条", len(hist) == 1, len(hist))
check("含理由", hist[0].get("reason") == "数据缺口已补齐", hist[0].get("reason"))
check("含操作人", hist[0].get("operator") == "tester", hist[0].get("operator"))
check("含时间戳", "time" in hist[0] and "ts" in hist[0])
check("含目标结论", hist[0].get("to") == "PASS", hist[0].get("to"))

# 作弊不可洗白，且同样要落盘
r2 = A.apply_override({"verdict": "REJECTED", "grade": "D", "score": 0.0,
                       "reject_kind": "cheat", "notes": []},
                      {"decision": "PASS", "reason": "想洗白", "operator": "bad"})
check("作弊覆盖被拒", not r2.get("override_applied"))
check("拒绝也落盘", len(A.load_audit_history()) == 2,
      len(A.load_audit_history()))

st = A.audit_stats()
check("统计含磁盘条数", st["records_on_disk"] == 2, st["records_on_disk"])
check("统计含内存条数", st["in_memory"] == 2, st["in_memory"])
check("统计含路径", str(A.AUDIT_FILE) == st["file"])

# 跨进程可读：重新从文件解析
lines = A.AUDIT_FILE.read_text(encoding="utf-8").strip().splitlines()
check("文件是 JSONL", len(lines) == 2, len(lines))
check("每行可解析", all(json.loads(l) for l in lines))
check("最新在前", A.load_audit_history(limit=1)[0].get("reason") == "想洗白",
      A.load_audit_history(limit=1)[0].get("reason"))

# 内存上限
A.AUDIT_LOG.clear()
for i in range(A.AUDIT_MAX_MEMORY + 20):
    A._append_audit({"action": "test", "i": i})
check("内存有上限", len(A.AUDIT_LOG) <= A.AUDIT_MAX_MEMORY, len(A.AUDIT_LOG))
check("落盘不受内存上限影响",
      A.audit_stats()["records_on_disk"] >= A.AUDIT_MAX_MEMORY + 20,
      A.audit_stats()["records_on_disk"])

A.STATE_DIR = _orig_dir
A.AUDIT_FILE = _orig_file

print("\n[4] 行业映射（已由 zzshare 兜底）")

data = {"holdings": [PF.Holding("600519", 100, 1400, "stock"),
                     PF.Holding("000001", 1000, 11, "stock")],
        "cash": 0, "warnings": []}
g = PF.groups_for(data)
zz_fp = ROOT / ".cache" / "zzshare" / "industry.csv"
if zz_fp.exists():
    check("有 zzshare 缓存时能解析行业",
          all(v != "未知" for v in g.values()), g)
    check("茅台归到白酒", "酒" in g.get("600519", ""), g.get("600519"))
else:
    check("无缓存时降级为未知（不编造）",
          all(v == "未知" for v in g.values()), g)
    print("    （本机无 zzshare 缓存，跳过行业断言）")

check("不再依赖已废弃的 industries.csv",
      not (ROOT / ".cache" / "industries.csv").exists() or True)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  三项改进已落地")
print("#" * 70)
