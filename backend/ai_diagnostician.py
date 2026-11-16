# -*- coding: utf-8 -*-
"""
大模型诊断器（M3）
====================================================
主链路：DeepSeek API（OpenAI 兼容协议）
兜底链：本地规则模板引擎（template_engine）

**核心契约：diagnose() 永不抛异常。**
无论无 Key、断网、限流、超额、返回非法 JSON，都会返回一份结构完整的诊断，
并在 `degraded` / `fallback_reason` 中如实标注降级原因。

人格设定
--------
客观、严谨、不带情绪的专业金融风控合规专家（**不使用任何调侃或毒舌措辞**）。

输出三部分（强制）
------------------
1. summary  综合定性（一句话：优秀 / 中性 / 高危）
2. fatal    核心致命伤（扣分最严重的维度 + 原因）
3. advice   整改建议（可执行的具体方向）

环境变量
--------
QP_LLM_KEY       必填。API Key（缺失则直接走模板，不发起网络请求）
QP_LLM_BASE      默认 https://api.deepseek.com
QP_LLM_MODEL     默认 deepseek-chat
QP_LLM_TIMEOUT   默认 20（秒）
QP_LLM_DISABLE   设为 1 时强制走模板（答辩断网演练用）

依赖：标准库 + template_engine
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import template_engine as TE

__version__ = "0.1.0"
__all__ = [
    "PROMPT_VERSION", "PROMPT_PERSONA", "SYSTEM_PROMPT",
    "load_knowledge_base", "retrieve_context", "build_prompt",
    "call_llm", "parse_llm_output", "diagnose", "format_report",
]

log = logging.getLogger(__name__)

PROMPT_VERSION = "1.0"
DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 20.0
KB_PATH = Path(__file__).resolve().parent / "knowledge_base.md"

# 人格与硬性约束（提示词核心，禁止改成调侃风格）
PROMPT_PERSONA = (
    "你是一名资深金融风控合规专家，负责对量化策略进行体检结论的解读。"
    "写作风格要求：客观、严谨、专业、克制。"
    "只依据提供的数据下结论，不得推测未给出的信息；数据不足时必须明确说明"
    "“无法判断”，不得编造数字或结论。"
    "禁止使用任何情绪化、调侃、夸张或贬损性的表述。"
)

SYSTEM_PROMPT = (
    PROMPT_PERSONA
    + "你必须严格返回一个 JSON 对象，不要输出任何 JSON 之外的文字。"
    "JSON 字段为：qualitative（综合定性，一句话，说明优秀/中性/高危）、"
    "fatal（核心致命伤，指出扣分最严重的维度并解释原因）、"
    "advice（整改建议，给出可执行的具体修改方向）、"
    "issues（数组，每项含 dimension、severity、evidence、fix）。"
)


# ══════════════════════════════════════════════════════════
# 知识库（轻量 RAG：按维度关键词检索，无需向量库）
# ══════════════════════════════════════════════════════════
def load_knowledge_base(path: Path | str = KB_PATH) -> dict[str, str]:
    """
    读取 Markdown 知识库，按 `## <key>` 分节。

    返回 {section_key: text}；文件缺失或解析失败时返回空 dict（不抛异常）。
    """
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception as ex:
        log.warning("知识库读取失败：%s", ex)
        return out
    cur = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^##\s+(\S+)\s*$", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        elif cur is not None:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


def retrieve_context(report: dict, kb: dict[str, str] | None = None,
                     max_chars: int = 2400) -> str:
    """
    按「最差两个维度 + 通用原则 + 输出规范」抽取知识库片段。

    纯关键词检索：命中维度直接取整节，避免引入向量库依赖。
    """
    kb = kb if kb is not None else load_knowledge_base()
    if not kb:
        return ""

    radar = (report or {}).get("radar") or {}
    scored = []
    for k in TE.DIM_ORDER:
        try:
            v = float(radar.get(k))
        except (TypeError, ValueError):
            continue
        if v == v:
            scored.append((k, v))
    worst = [k for k, _ in sorted(scored, key=lambda x: x[1])[:2]]

    parts: list[str] = []
    for key in ("通用原则",) + tuple(worst) + ("输出规范",):
        if key in kb and kb[key]:
            parts.append(f"## {key}\n{kb[key]}")
    text = "\n\n".join(parts)
    return text[:max_chars]


# ══════════════════════════════════════════════════════════
# 提示词
# ══════════════════════════════════════════════════════════
def _fmt_report(report: dict) -> str:
    """把六维报告压成紧凑文本喂给模型。"""
    report = report or {}
    L = [
        f"裁决结论：{report.get('verdict')}",
        f"综合评级：{report.get('grade')}",
        f"加权得分：{report.get('score')}",
    ]
    if report.get("grade_capped"):
        L.append("注意：存在数学计算异常项，评级已封顶至 C")
    if report.get("reject_kind"):
        L.append(f"否决类型：{report['reject_kind']}")
    L.append("")
    L.append("六维得分（0-100，None 表示该项豁免）：")
    radar = report.get("radar") or {}
    for k in TE.DIM_ORDER:
        L.append(f"  - {TE.DIM_LABEL.get(k, k)}（{k}）：{radar.get(k)}")
    L.append("")
    L.append("维度明细：")
    for d in report.get("dimensions") or []:
        note = (d.get("note") or "").strip()
        L.append(f"  - {d.get('label')} | 状态 {d.get('status')} | "
                 f"得分 {d.get('score')} | 权重 {d.get('weight')}"
                 + (f" | 说明：{note}" if note else ""))
    if report.get("notes"):
        L.append("")
        L.append("体检说明：")
        for n in report["notes"]:
            L.append(f"  - {n}")
    return "\n".join(L)


def build_prompt(report: dict, context: str = "") -> str:
    """构造用户提示词（含三部分硬性要求）。"""
    req = (
        "请基于以上体检数据输出诊断，必须包含且仅包含以下三段内容：\n"
        "1. 综合定性：用一句话说明该策略属于优秀、中性还是高危，并给出依据。\n"
        "2. 核心致命伤：指出扣分最严重的维度，解释其成因与后果。\n"
        "3. 整改建议：给出可执行的具体修改方向，避免空泛表述。\n"
        "若数据不足以下结论，必须直接说明“无法判断”，不得臆测。"
    )
    blocks = [f"【体检数据】\n{_fmt_report(report)}"]
    if context:
        blocks.insert(0, f"【风控知识库摘录】\n{context}")
    blocks.append(f"【输出要求】\n{req}")
    return "\n\n".join(blocks)


# ══════════════════════════════════════════════════════════
# 调用
# ══════════════════════════════════════════════════════════
class LLMError(RuntimeError):
    """内部异常类型，携带降级原因码。"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def call_llm(prompt: str, api_key: str, base: str = DEFAULT_BASE,
             model: str = DEFAULT_MODEL,
             timeout: float = DEFAULT_TIMEOUT) -> str:
    """
    调用 OpenAI 兼容的 chat/completions 接口，返回原始文本。

    失败时抛 LLMError（由 diagnose 捕获并降级），不在这里吞异常。
    """
    if not api_key:
        raise LLMError("no_api_key", "未设置 QP_LLM_KEY")
    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as ex:
        code = getattr(ex, "code", 0)
        reason = ("rate_limited" if code == 429 else
                  "auth_failed" if code in (401, 403) else
                  "quota_exhausted" if code == 402 else
                  f"http_{code}")
        raise LLMError(reason, f"HTTP {code}") from None
    except urllib.error.URLError as ex:
        raise LLMError("network_error", str(getattr(ex, "reason", ex))) from None
    except TimeoutError:
        raise LLMError("timeout", f"超过 {timeout}s") from None
    except Exception as ex:  # 兜底：任何意外都不应让调用方崩溃
        raise LLMError("request_failed", repr(ex)) from None

    try:
        payload = json.loads(raw)
        return payload["choices"][0]["message"]["content"]
    except Exception as ex:
        raise LLMError("bad_response", f"响应无法解析：{ex}") from None


def parse_llm_output(text: str) -> dict:
    """
    从模型输出里提取 JSON（容忍 ```json 围栏与前后废话）。

    校验三段内容齐全；缺任何一段都抛 LLMError("bad_response")，触发降级。
    """
    if not text or not text.strip():
        raise LLMError("empty_response", "模型返回空内容")

    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)
    if fence:
        s = fence.group(1)
    else:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            s = s[start:end + 1]

    try:
        data = json.loads(s)
    except Exception as ex:
        raise LLMError("bad_response", f"不是合法 JSON：{ex}") from None
    if not isinstance(data, dict):
        raise LLMError("bad_response", "JSON 根节点不是对象")

    def _pick(*names) -> str:
        for n in names:
            v = data.get(n)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    qualitative = _pick("qualitative", "summary", "综合定性")
    fatal = _pick("fatal", "核心致命伤", "fatal_flaw")
    advice = _pick("advice", "整改建议", "suggestion")
    if not (qualitative and fatal and advice):
        raise LLMError("bad_response", "缺少三段内容之一")

    issues = data.get("issues")
    if not isinstance(issues, list):
        issues = []
    return {"qualitative": qualitative, "fatal": fatal, "advice": advice,
            "issues": issues, "raw": data}


# ══════════════════════════════════════════════════════════
# 主接口
# ══════════════════════════════════════════════════════════
def _template_result(report: dict, reason: str, detail: str = "",
                     latency_ms: float = 0.0) -> dict:
    """构造降级结果（模板引擎产出 + 降级元信息）。"""
    try:
        t = TE.render(report)
    except Exception as ex:  # 模板引擎理论上不抛，双保险
        log.exception("模板引擎异常")
        t = {"summary": "综合定性：无法生成诊断（模板引擎异常）。",
             "fatal": "核心致命伤：诊断生成失败。",
             "advice": "整改建议：请人工复核体检数据。",
             "issues": [], "source": "template", "engine_version": TE.__version__}
        detail = f"{detail}; template_error={ex!r}"
    return {
        "source": "template",
        "degraded": True,
        "fallback_reason": reason,
        "fallback_detail": detail,
        "model": None,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latency_ms": round(latency_ms, 1),
        "summary": t["summary"],
        "fatal": t["fatal"],
        "advice": t["advice"],
        "issues": t.get("issues", []),
        "engine_version": t.get("engine_version"),
    }


def diagnose(report: dict, use_llm: bool = True,
             api_key: str | None = None,
             base: str | None = None,
             model: str | None = None,
             timeout: float | None = None,
             kb_path: Path | str | None = None) -> dict:
    """
    【主接口】生成诊断。**永不抛异常。**

    参数
    ----
    report    : scoring.full_report 的输出（六维报告）
    use_llm   : False 时强制走模板（演练/离线）
    api_key   : 缺省读环境变量 QP_LLM_KEY
    base/model/timeout : 缺省读环境变量，再缺省用内置默认值

    返回 dict
    --------
    source          "llm" | "template"
    degraded        是否降级
    fallback_reason 降级原因码（None 表示未降级）
    summary/fatal/advice  三段诊断文本
    issues          结构化问题清单
    """
    t0 = time.perf_counter()
    report = report or {}

    # 0) 强制降级 / 无 Key → 直接模板，不发网络请求
    if not use_llm:
        return _template_result(report, "llm_disabled",
                                "use_llm=False", _elapsed(t0))
    if str(os.environ.get("QP_LLM_DISABLE", "")).strip() in ("1", "true", "True"):
        return _template_result(report, "llm_disabled",
                                "QP_LLM_DISABLE=1", _elapsed(t0))

    key = (api_key if api_key is not None else os.environ.get("QP_LLM_KEY", "")).strip()
    if not key:
        return _template_result(report, "no_api_key",
                                "未设置环境变量 QP_LLM_KEY", _elapsed(t0))

    base = base or os.environ.get("QP_LLM_BASE", DEFAULT_BASE)
    model = model or os.environ.get("QP_LLM_MODEL", DEFAULT_MODEL)
    try:
        timeout = float(timeout if timeout is not None
                        else os.environ.get("QP_LLM_TIMEOUT", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT

    # 1) 主链路
    try:
        ctx = retrieve_context(report,
                              load_knowledge_base(kb_path if kb_path else KB_PATH))
        prompt = build_prompt(report, ctx)
        raw = call_llm(prompt, key, base=base, model=model, timeout=timeout)
        parsed = parse_llm_output(raw)
    except LLMError as ex:
        log.warning("LLM 不可用，降级模板：%s (%s)", ex.reason, ex.detail)
        return _template_result(report, ex.reason, ex.detail, _elapsed(t0))
    except Exception as ex:  # 任何意外都降级，绝不外抛
        log.exception("LLM 调用异常")
        return _template_result(report, "unexpected_error", repr(ex), _elapsed(t0))

    # 2) 成功
    return {
        "source": "llm",
        "degraded": False,
        "fallback_reason": None,
        "fallback_detail": "",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latency_ms": round(_elapsed(t0), 1),
        "summary": "综合定性：" + _strip_label(parsed["qualitative"], "综合定性"),
        "fatal": "核心致命伤：" + _strip_label(parsed["fatal"], "核心致命伤"),
        "advice": "整改建议：" + _strip_label(parsed["advice"], "整改建议"),
        "issues": parsed["issues"],
        "engine_version": __version__,
    }


def _elapsed(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _strip_label(text: str, label: str) -> str:
    """
    去掉模型可能自带的前缀标签。

    注意：不能用 str.lstrip("综合定性：")——lstrip 是按字符集剥离，
    会误删正文里的“性”“：”等字符。这里按整串前缀判断。
    """
    t = (text or "").strip()
    for cand in (label, label + "：", label + ":", label + " "):
        if t.startswith(cand):
            return t[len(cand):].lstrip("：: ").strip()
    return t


# ══════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════
def format_report(res: dict) -> str:
    L = ["=" * 66, "策略风险诊断报告", "=" * 66]
    src = "大模型（%s）" % res.get("model") if res.get("source") == "llm" \
        else "本地规则模板（降级）"
    L.append(f"  诊断来源: {src}")
    if res.get("degraded"):
        L.append(f"  降级原因: {res.get('fallback_reason')}"
                 f"  {res.get('fallback_detail') or ''}".rstrip())
    L.append(f"  耗时: {res.get('latency_ms')} ms   "
             f"提示词版本: {res.get('prompt_version')}")
    L.append("-" * 66)
    L.append("  " + str(res.get("summary") or ""))
    L.append("")
    L.append("  " + str(res.get("fatal") or ""))
    L.append("")
    L.append("  " + str(res.get("advice") or ""))
    if res.get("issues"):
        L.append("-" * 66)
        L.append("  问题清单:")
        for i in res["issues"]:
            if isinstance(i, dict):
                dim = i.get("label") or i.get("dimension") or "-"
                sev = i.get("severity") or "-"
                fix = i.get("fix") or i.get("suggested_fix") or ""
                L.append(f"    - [{sev}] {dim}：{fix}")
    L.append("=" * 66)
    return "\n".join(L)


# ══════════════════════════════════════════════════════════
# 自测（不需要网络）
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    demo = {
        "verdict": "WARNING", "grade": "C", "score": 74.0, "grade_capped": True,
        "radar": {"anti_cheat": 100.0, "anti_overfit": 0.0, "stress": 90.0,
                  "purity": 100.0, "cost": 80.0, "stability": 80.0},
        "dimensions": [
            {"key": "anti_overfit", "label": "抗过拟合", "status": "WARN",
             "score": 0.0, "weight": 0.2, "note": "PBO 无法计算（样本内夏普≤0）"},
            {"key": "stress", "label": "抗压能力", "status": "PASS",
             "score": 90.0, "weight": 0.2, "note": ""},
        ],
        "notes": ["[抗压能力] 扣费后年化 0.00%，未达存活线"],
    }

    # 无 Key → 必然降级
    os.environ.pop("QP_LLM_KEY", None)
    r1 = diagnose(demo)
    assert r1["source"] == "template" and r1["degraded"] is True
    assert r1["fallback_reason"] == "no_api_key"
    assert r1["summary"].startswith("综合定性")
    assert r1["fatal"].startswith("核心致命伤")
    assert r1["advice"].startswith("整改建议")
    print(format_report(r1))

    # 假 Key + 不可达地址 → 仍必须降级而不崩
    r2 = diagnose(demo, api_key="sk-fake", base="http://127.0.0.1:1", timeout=2)
    assert r2["source"] == "template" and r2["degraded"] is True
    print(f"[降级路径] {r2['fallback_reason']} / {r2['fallback_detail'][:60]}")

    # 强制禁用
    r3 = diagnose(demo, use_llm=False)
    assert r3["fallback_reason"] == "llm_disabled"

    # 解析器容错
    ok = parse_llm_output('```json\n{"qualitative":"中性","fatal":"抗压不足",'
                          '"advice":"加止损"}\n```')
    assert ok["qualitative"] == "中性"
    try:
        parse_llm_output("这不是 JSON")
        raise SystemExit("应当拒绝非法输出")
    except LLMError:
        pass

    print(f"OK v{__version__}  M3 主链路与降级链均正常")
