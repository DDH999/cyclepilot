# -*- coding: utf-8 -*-
"""
M3 验证：大模型诊断 + 降级模板
====================================================================
跑法:  python tests/test_m3.py      （全程不需要网络）

分组
  [1] 模板三段结构
  [2] 综合定性分档
  [3] 核心致命伤定位
  [4] 整改建议
  [5] 降级链（无 Key / 禁用 / 断网 / 超时 / 坏响应）
  [6] 模型输出解析
  [7] 提示词内容
  [8] 永不抛异常
  [9] LLM 成功路径（打桩）
====================================================================
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import ai_diagnostician as AI
import template_engine as TE

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M3 验证   AI={AI.__version__}  template={TE.__version__}")
print("#" * 70)

os.environ.pop("QP_LLM_KEY", None)
os.environ.pop("QP_LLM_DISABLE", None)

REP_GOOD = {
    "verdict": "PASS", "grade": "A", "score": 96.0,
    "radar": {k: 95.0 for k in TE.DIM_ORDER}, "dimensions": [], "notes": [],
}
REP_CAPPED = {
    "verdict": "WARNING", "grade": "C", "score": 74.0, "grade_capped": True,
    "radar": {"anti_cheat": 100.0, "anti_overfit": 0.0, "stress": 90.0,
              "purity": 100.0, "cost": 80.0, "stability": 80.0},
    "dimensions": [{"key": "anti_overfit", "label": "抗过拟合", "status": "WARN",
                    "score": 0.0, "weight": 0.2, "note": "PBO 无法计算"}],
    "notes": ["[抗过拟合] PBO 无法计算"],
}
REP_FAIL = {
    "verdict": "FAIL", "grade": "D", "score": 40.0,
    "radar": {"anti_cheat": 100.0, "anti_overfit": 20.0, "stress": 10.0,
              "purity": 100.0, "cost": 100.0, "stability": 100.0},
    "dimensions": [], "notes": [],
}
REP_CHEAT = {"verdict": "REJECTED", "grade": "REJECTED", "score": 0.0,
             "reject_kind": "cheat", "radar": {}, "dimensions": [], "notes": []}
REP_DATA = {"verdict": "REJECTED", "grade": "REJECTED", "score": 0.0,
            "reject_kind": "data", "radar": {}, "dimensions": [], "notes": []}
REP_NA = {"verdict": "PASS", "grade": "A", "score": 100.0,
          "radar": {"anti_cheat": 100.0, "anti_overfit": None, "stress": 100.0,
                    "purity": None, "cost": 100.0, "stability": None},
          "dimensions": [], "notes": ["[策略纯度] 全程空仓，本项豁免"]}

# ══════════════════════════════════════════════════════════════
print("\n[1] 模板三段结构")

for name, rep in [("优秀", REP_GOOD), ("封顶", REP_CAPPED), ("高危", REP_FAIL),
                  ("作弊", REP_CHEAT), ("数据缺失", REP_DATA)]:
    r = TE.render(rep)
    ok = (r["summary"].startswith("综合定性") and r["fatal"].startswith("核心致命伤")
          and r["advice"].startswith("整改建议"))
    check(f"{name} 三段齐全且带标签", ok)
    check(f"{name} 三段非空", all(len(r[k]) > 8 for k in ("summary", "fatal", "advice")))

check("模板来源标记", TE.render(REP_GOOD)["source"] == "template")
check("模板带引擎版本", TE.render(REP_GOOD)["engine_version"] == TE.__version__)

# ══════════════════════════════════════════════════════════════
print("\n[2] 综合定性分档")

check("A → 优秀", "优秀" in TE.summarize(REP_GOOD))
check("B → 中性偏好", "中性偏好" in TE.summarize(
    {**REP_GOOD, "grade": "B", "score": 80}))
check("C 封顶 → 中性偏谨慎", "中性偏谨慎" in TE.summarize(REP_CAPPED))
check("C 封顶注明原因", "封顶" in TE.summarize(REP_CAPPED))
check("D → 高危", "高危" in TE.summarize(REP_FAIL))
check("作弊 → 高危且点明", "高危" in TE.summarize(REP_CHEAT)
      and "未来函数" in TE.summarize(REP_CHEAT))
check("数据缺失 → 无法评估", "无法评估" in TE.summarize(REP_DATA))

# ══════════════════════════════════════════════════════════════
print("\n[3] 核心致命伤定位")

f = TE.find_fatal(REP_CAPPED)
check("定位到最低维度", "抗过拟合" in f, f[:40])
check("包含维度得分", "0 分" in f)
check("包含严重度", "严重度高" in f)
check("引用维度说明", "PBO 无法计算" in f)

f_all = TE.find_fatal(REP_GOOD)
check("全优不虚构致命伤", "未发现致命缺陷" in f_all, f_all[:40])
check("全优也报最低维度", "95 分" in f_all)

f_cheat = TE.find_fatal(REP_CHEAT)
check("作弊优先于分数", "未来函数" in f_cheat)
check("数据缺失单独表述", "未跑通" in TE.find_fatal(REP_DATA))

check("无雷达数据不崩溃", isinstance(TE.find_fatal({"verdict": "PASS"}), str))

# ══════════════════════════════════════════════════════════════
print("\n[4] 整改建议")

a = TE.build_advice(REP_CAPPED)
check("指向低分维度", "抗过拟合" in a)
check("给出具体动作", "Walk-Forward" in a)
check("无重复标点", "。。" not in a, a[-30:])
check("全优给运维建议", "定期" in TE.build_advice(REP_GOOD))
check("作弊给重写建议", "重写信号生成逻辑" in TE.build_advice(REP_CHEAT))
check("数据缺失先修管道", "数据管道" in TE.build_advice(REP_DATA))

r_issues = TE.render(REP_CAPPED)["issues"]
check("issues 只含低分维度", len(r_issues) == 1 and r_issues[0]["dimension"] == "anti_overfit")
check("issues 带严重度", r_issues[0]["severity"] == "高")
check("issues 带中文标签", r_issues[0]["label"] == "抗过拟合")

# ══════════════════════════════════════════════════════════════
print("\n[5] 降级链")

r = AI.diagnose(REP_CAPPED)
check("无 Key 走模板", r["source"] == "template")
check("标记已降级", r["degraded"] is True)
check("原因码为 no_api_key", r["fallback_reason"] == "no_api_key", r["fallback_reason"])
check("降级仍产出三段", all(r[k].startswith(p) for k, p in
                       [("summary", "综合定性"), ("fatal", "核心致命伤"),
                        ("advice", "整改建议")]))

r_dis = AI.diagnose(REP_CAPPED, use_llm=False)
check("use_llm=False 走模板", r_dis["fallback_reason"] == "llm_disabled")

os.environ["QP_LLM_DISABLE"] = "1"
r_env = AI.diagnose(REP_CAPPED, api_key="sk-any")
check("QP_LLM_DISABLE 生效", r_env["fallback_reason"] == "llm_disabled")
os.environ.pop("QP_LLM_DISABLE", None)

r_net = AI.diagnose(REP_CAPPED, api_key="sk-fake", base="http://127.0.0.1:1", timeout=2)
check("断网降级不崩溃", r_net["source"] == "template")
check("断网原因码合理",
      r_net["fallback_reason"] in ("network_error", "timeout", "request_failed"),
      r_net["fallback_reason"])

r_bad = AI.diagnose(REP_CAPPED, api_key="sk-fake", base="not-a-url", timeout=1)
check("非法 base 降级", r_bad["source"] == "template")

# ══════════════════════════════════════════════════════════════
print("\n[6] 模型输出解析")

ok = AI.parse_llm_output('{"qualitative":"中性","fatal":"抗压不足","advice":"加止损"}')
check("纯 JSON 可解析", ok["qualitative"] == "中性")

fence = AI.parse_llm_output('```json\n{"qualitative":"中性","fatal":"f","advice":"a"}\n```')
check("代码围栏可解析", fence["qualitative"] == "中性")

prose = AI.parse_llm_output('以下是结论：{"qualitative":"中性","fatal":"f","advice":"a"} 完毕')
check("前后废话可解析", prose["fatal"] == "f")

for bad, why in [("不是 JSON", "纯文本"), ("{}", "空对象"),
                 ('{"qualitative":"x"}', "缺字段"), ("[1,2,3]", "非对象")]:
    try:
        AI.parse_llm_output(bad)
        check(f"应拒绝：{why}", False)
    except AI.LLMError as ex:
        check(f"应拒绝：{why}", ex.reason == "bad_response", ex.reason)

check("空字符串被拒", _raise(lambda: AI.parse_llm_output("")) if False else True)
try:
    AI.parse_llm_output("")
    check("空字符串被拒", False)
except AI.LLMError:
    check("空字符串被拒", True)

check("标签剥离正确", AI._strip_label("综合定性：中性", "综合定性") == "中性")
check("剥离不误删正文", AI._strip_label("性格测试结论", "综合定性") == "性格测试结论")
check("无标签原样返回", AI._strip_label("中性", "综合定性") == "中性")

# ══════════════════════════════════════════════════════════════
print("\n[7] 提示词内容")

kb = AI.load_knowledge_base()
check("知识库可加载", len(kb) >= 5, len(kb))
check("含通用原则", "通用原则" in kb)
check("含六维度分节", all(k in kb for k in TE.DIM_ORDER))

ctx = AI.retrieve_context(REP_CAPPED, kb)
check("上下文命中最低维度", "anti_overfit" in ctx or "PBO" in ctx)
check("上下文长度受限", len(ctx) <= 2400, len(ctx))

prompt = AI.build_prompt(REP_CAPPED, ctx)
check("提示词含体检分数", "抗过拟合" in prompt and "0.0" in prompt)
check("提示词含三段要求", "综合定性" in prompt and "核心致命伤" in prompt
      and "整改建议" in prompt)
check("提示词含禁编造约束", "无法判断" in prompt)
check("系统提示词为人格设定", "风控合规专家" in AI.SYSTEM_PROMPT)
check("系统提示词要求 JSON", "JSON" in AI.SYSTEM_PROMPT)
check("系统提示词禁止情绪化", "禁止" in AI.SYSTEM_PROMPT)
check("不使用毒舌设定", "毒舌" not in (AI.SYSTEM_PROMPT + AI.PROMPT_PERSONA))

# ══════════════════════════════════════════════════════════════
print("\n[8] 永不抛异常")

weird = [None, {}, {"verdict": "PASS"}, {"radar": None},
         {"verdict": "REJECTED", "radar": {"x": "abc"}},
         {"radar": {"anti_cheat": float("nan")}},
         {"dimensions": None, "notes": None}]
for i, w in enumerate(weird):
    try:
        rr = AI.diagnose(w, use_llm=False)
        ok2 = isinstance(rr, dict) and rr.get("summary")
        check(f"畸形输入#{i} 不崩溃且有三段", bool(ok2))
    except Exception as ex:
        check(f"畸形输入#{i} 不崩溃且有三段", False, repr(ex))

# ══════════════════════════════════════════════════════════════
print("\n[9] LLM 成功路径（打桩）")

_orig = AI.call_llm
try:
    AI.call_llm = lambda prompt, key, **kw: json.dumps({
        "qualitative": "中性偏谨慎，抗压能力不足",
        "fatal": "极端行情下回撤显著放大",
        "advice": "增加组合级回撤制动与个股止损",
        "issues": [{"dimension": "stress", "severity": "高",
                    "evidence": "区间回撤 2.3 倍", "fix": "加止损"}],
    }, ensure_ascii=False)
    r_llm = AI.diagnose(REP_CAPPED, api_key="sk-test")
    check("成功路径 source=llm", r_llm["source"] == "llm", r_llm["source"])
    check("成功路径未降级", r_llm["degraded"] is False)
    check("三段带标签", r_llm["summary"].startswith("综合定性：")
          and r_llm["fatal"].startswith("核心致命伤：")
          and r_llm["advice"].startswith("整改建议："))
    check("不重复标签", r_llm["summary"].count("综合定性") == 1, r_llm["summary"])
    check("保留 issues", len(r_llm["issues"]) == 1)
    check("记录模型名", r_llm["model"] == AI.DEFAULT_MODEL)
    check("记录提示词版本", r_llm["prompt_version"] == AI.PROMPT_VERSION)

    AI.call_llm = lambda *a, **k: (_ for _ in ()).throw(AI.LLMError("bad_response", "坏 JSON"))
    r_badjson = AI.diagnose(REP_CAPPED, api_key="sk-test")
    check("坏响应降级", r_badjson["source"] == "template"
          and r_badjson["fallback_reason"] == "bad_response")

    AI.call_llm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("意外"))
    r_exc = AI.diagnose(REP_CAPPED, api_key="sk-test")
    check("意外异常也降级", r_exc["source"] == "template"
          and r_exc["fallback_reason"] == "unexpected_error")
finally:
    AI.call_llm = _orig

check("报告可渲染", "综合定性" in AI.format_report(r)) 
check("降级报告标注来源", "本地规则模板" in AI.format_report(r))

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
print("# 全部通过  M3 主链路与降级链均已验证")
print("#" * 70)
