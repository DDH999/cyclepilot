# -*- coding: utf-8 -*-
"""
M29 验证：决策记忆与到期结算
====================================================================
跑法:  python tests/test_m29_journal.py   （不联网，全部用假数据）
"""
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import journal as JN

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m29"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M29 验证  journal v{JN.__version__}")
print("#" + "=" * 69)

print("\n[1] 动作方向映射")

for a, want in [("加投", 1), ("加仓", 1), ("买入", 1), ("投出现金", 1),
                ("额外加仓", 1), ("减投", -1), ("减仓", -1), ("卖出", -1),
                ("不动", 0), ("", 0), ("随便", 0)]:
    check(f"「{a or '(空)'}」→ {want}", JN._sign_of(a) == want, JN._sign_of(a))
check("含「加」的未知动作按看多", JN._sign_of("加一点") == 1)
check("含「减」的未知动作按看空", JN._sign_of("减一点") == -1)

print("\n[2] 记录")

fp = TMP / "d.jsonl"
check("初始为空", JN.load_journal(fp) == [])

r1 = JN.record("sp500", "减投", price=7000.0, amount=500.0,
               name="标普500", reason="98%分位", path=fp)
check("记录返回 id", len(r1["id"]) == 12, r1["id"])
check("方向为看空", r1["direction"] == -1)
check("记录价格", r1["price_then"] == 7000.0)
check("未结算", r1["outcome"] is None)
check("文件已写", fp.exists())

recs = JN.load_journal(fp)
check("读回 1 条", len(recs) == 1)
check("字段完整",
      all(k in recs[0] for k in ("id", "ts", "as_of", "ticker", "action",
                                 "direction", "price_then", "horizon")))

r2 = JN.record("nikkei", "加投", price=60000.0, name="日经225", path=fp)
check("追加不覆盖", len(JN.load_journal(fp)) == 2)
check("两条 id 不同", r1["id"] != r2["id"])

check("空文件返回空列表", JN.load_journal(TMP / "nope.jsonl") == [])
bad = TMP / "bad.jsonl"
bad.write_text("{不是json\n" + json.dumps({"id": "x"}) + "\n", encoding="utf-8")
check("损坏行被跳过", len(JN.load_journal(bad)) == 1)

print("\n[3] 结算：未到期不结算")

st = JN.settle_pending(path=fp)
check("未到期不结算", st["n_settled"] == 0, st["n_settled"])
check("标记为 pending", st["pending"] == 2, st["pending"])

print("\n[4] 结算：到期后用真实价格")

# 伪造一条 40 天前、as_of 明确的决策
fp2 = TMP / "settle.jsonl"
JN.record("sp500", "减投", price=1.0, name="标普500",
          as_of="2026-01-05", path=fp2)
recs = JN.load_journal(fp2)
recs[0]["ts"] = time.time() - 40 * 86400
fp2.write_text(chr(10).join(json.dumps(r, ensure_ascii=False) for r in recs)
               + chr(10), encoding="utf-8")

st2 = JN.settle_pending(path=fp2)
check("到期后结算", st2["n_settled"] == 1, st2)
check("无错误", not st2["errors"], st2["errors"])

o = JN.load_journal(fp2)[0]["outcome"]
check("含区间", "from_date" in o and "to_date" in o, o.get("from_date"))
check("含前向收益", o["forward_return"] is not None)
check("含 edge", o["edge"] is not None)
check("含方向", o["direction"] == -1)
check("edge = 涨跌 × 方向",
      abs(o["edge"] - o["forward_return"] * o["direction"]) < 1e-6,
      (o["edge"], o["forward_return"]))
check("含命中判定", isinstance(o["hit"], bool), o["hit"])
check("命中与 edge 同号",
      o["hit"] == (o["edge"] > 0), (o["hit"], o["edge"]))
check("记录了数据源", "source" in o, o.get("source"))
check("结算后不重复结算",
      JN.settle_pending(path=fp2)["n_settled"] == 0)

print("\n[5] 结算：无价格时跳过而不是瞎算")

fp3 = TMP / "noprice.jsonl"
JN.record("999999", "加投", price=None, path=fp3)
recs = JN.load_journal(fp3)
recs[0]["ts"] = time.time() - 40 * 86400
fp3.write_text(chr(10).join(json.dumps(r, ensure_ascii=False) for r in recs)
               + chr(10), encoding="utf-8")
st3 = JN.settle_pending(path=fp3)
check("缺价格不结算", st3["n_settled"] == 0)
check("给出跳过原因", any("缺" in e or "价格" in e for e in st3["errors"]),
      st3["errors"])

print("\n[6] 复盘")


def make_settled(path, items):
    lines = []
    for i, (action, fwd, direction) in enumerate(items):
        lines.append(json.dumps({
            "id": f"id{i}", "ts": time.time() - 100 * 86400,
            "as_of": "2026-01-05", "ticker": "sp500", "name": "标普500",
            "action": action, "direction": direction, "price_then": 100.0,
            "outcome": {"forward_return": fwd, "edge": fwd * direction,
                        "hit": fwd * direction > 0,
                        "from_date": "2026-01-05", "to_date": "2026-02-03",
                        "price_from": 100.0, "price_to": 100 * (1 + fwd),
                        "bars": 20},
        }, ensure_ascii=False))
    Path(path).write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


fp4 = TMP / "review.jsonl"
make_settled(fp4, [("减投", -0.05, -1), ("减投", 0.02, -1),
                   ("减投", -0.03, -1), ("减投", 0.01, -1),
                   ("加投", 0.04, 1), ("加投", 0.06, 1),
                   ("加投", -0.02, 1), ("加投", 0.03, 1)])
rv = JN.review(fp4)
check("复盘可算", rv["ok"] is True)
check("统计条数", rv["n"] == 8, rv["n"])
check("按动作分组", len(rv["by_action"]) == 2, len(rv["by_action"]))
check("总体命中率", rv["overall_hit_rate"] is not None,
      rv["overall_hit_rate"])
check("样本足够时给结论", "继续积累" not in rv["verdict"], rv["verdict"])

# 减投命中：跌的时候减投 = 对
jian = next(r for r in rv["by_action"] if r["action"] == "减投")
check("减投 8 条里 4 条方向对",
      abs(jian["hit_rate"] - 0.5) < 1e-6, jian["hit_rate"])

# 小样本必须拒绝下结论
fp5 = TMP / "small.jsonl"
make_settled(fp5, [("减投", -0.05, -1), ("加投", 0.04, 1)])
rv5 = JN.review(fp5)
check("小样本不下结论", "没有统计意义" in rv5["verdict"], rv5["verdict"])
check("记录小样本阈值", rv5["min_sample"] == JN.MIN_SAMPLE)

check("无已结算时返回 ok=False",
      JN.review(TMP / "empty.jsonl")["ok"] is False)

print("\n[7] 报告")

txt = JN.format_journal(JN.load_journal(fp4))
check("台账可渲染", "决策台账" in txt)
check("含方向列", "方向" in txt)
check("空台账可渲染", "还没有记录" in JN.format_journal([]))
check("复盘可渲染", "决策复盘" in JN.format_report(rv))
check("复盘含免责说明",
      "不等于你赚了钱" in JN.format_report(rv))

print("\n[8] 接口与 CLI")

import api_v2 as API
from fastapi.testclient import TestClient

_orig_file = JN.JOURNAL_FILE
JN.JOURNAL_FILE = fp4
try:
    cli = TestClient(API.app)
    r = cli.get("/api/journal")
    check("接口 200", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        b = r.json()
        check("返回条数", b.get("n") == 8, b.get("n"))
        check("全部已结算", b.get("pending") == 0, b.get("pending"))
        check("含文本", "决策台账" in b.get("text", ""))

    r2 = cli.get("/api/journal?review=true")
    check("review 模式", r2.status_code == 200 and "review" in r2.json())
finally:
    JN.JOURNAL_FILE = _orig_file

import quantpilot as QP
check("CLI 注册 journal", "journal" in QP.COMMANDS)
pp = QP.build_parser()
check("解析 --settle", pp.parse_args(["journal", "--settle"]).settle is True)
check("解析 --review", pp.parse_args(["journal", "--review"]).review is True)
check("advice 支持 --record",
      pp.parse_args(["advice", "--record"]).record is True)
check("advice 默认不记账",
      pp.parse_args(["advice"]).record is False)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  决策台账可用")
print("#" * 70)
