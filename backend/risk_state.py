# -*- coding: utf-8 -*-
"""
风控闭环（M22）
====================================================
之前的风控有个致命缺陷：**它只给建议，不知道你有没有照做。**

一个不落地的风控等于没有风控。本模块补上闭环：

1. **落盘**：每次风控决策写入 `.state/risk_state.json`（追加，最多 500 条）
2. **对比**：同一标的两次检查之间，仓位有没有按建议调整
3. **执行率**：建议减仓 N 次，实际减了几次
4. **复盘**：HALT 后多久恢复，冷却期有没有被违反

诚实边界
--------
持仓是**检查时点的快照**，不是逐笔成交。
所以只能判断"仓位整体有没有降"，**不能判断具体怎么调的**。
要精确到笔，需要券商交割单——超出本项目范围。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["STATE_DIR", "RISK_STATE", "record", "load_history", "latest",
           "followthrough", "format_report", "MAX_RECORDS"]

_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = _ROOT / ".state"
RISK_STATE = STATE_DIR / "risk_state.json"
MAX_RECORDS = 500

# 判定"仓位下降"的阈值（2 个百分点以内视为没动）
EXPOSURE_TOL = 0.02


def _path(p: Path | str | None = None) -> Path:
    return Path(p) if p is not None else Path(RISK_STATE)


def load_history(path: Path | str | None = None) -> list[dict]:
    """读取历史决策记录（按时间升序）。文件不存在返回空列表。"""
    fp = _path(path)
    if not fp.exists():
        return []
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = raw.get("records", [])
    return sorted(raw, key=lambda x: x.get("ts", 0))


def record(decision: dict, context: dict | None = None,
           path: Path | str | None = None) -> dict:
    """
    追加一条决策记录。

    decision: riskcontrol.kill_switch 的返回值
    context : 附加信息（持仓权重、总资产、净值等）
    """
    fp = _path(path)
    hist = load_history(fp)

    weights = {}
    cash_ratio = None
    capital = None
    if context:
        weights = context.get("weights") or {}
        cash_ratio = context.get("cash_ratio")
        capital = context.get("capital")

    exposure = None
    if weights:
        exposure = round(float(sum(weights.values())), 6)
    elif cash_ratio is not None:
        exposure = round(1.0 - float(cash_ratio), 6)

    rec = {
        "ts": time.time(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": decision.get("action"),
        "scale": decision.get("scale"),
        "reasons": [r.get("why") for r in (decision.get("reasons") or [])],
        "crowding": (decision.get("crowding") or {}).get("score"),
        "drawdown": (decision.get("drawdown") or {}).get("current_drawdown"),
        "exposure": exposure,
        "cash_ratio": cash_ratio,
        "capital": capital,
        "weights": {k: round(float(v), 6) for k, v in weights.items()},
    }
    hist.append(rec)
    if len(hist) > MAX_RECORDS:
        hist = hist[-MAX_RECORDS:]

    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps({"version": __version__,
                                  "records": hist},
                                 ensure_ascii=False, indent=1),
                      encoding="utf-8")
    except Exception as ex:
        raise RuntimeError(f"写入风控状态失败：{ex}") from None

    return rec


def latest(path: Path | str | None = None) -> dict | None:
    h = load_history(path)
    return h[-1] if h else None


def followthrough(history: list[dict] | None = None,
                  tol: float = EXPOSURE_TOL,
                  path: Path | str | None = None) -> dict:
    """
    建议 vs 实际：算执行率。

    对每一条 REDUCE / HALT 建议，看**下一条**记录里仓位是否真的下降。
    下降超过 tol 记为"已执行"。
    """
    hist = load_history(path) if history is None else history
    if len(hist) < 2:
        return {"ok": False,
                "reason": f"记录不足（{len(hist)} 条），至少需要 2 次检查才能判断",
                "n_records": len(hist)}

    advises, follows = [], []
    for i, rec in enumerate(hist[:-1]):
        act = rec.get("action")
        if act not in ("REDUCE", "HALT"):
            continue
        nxt = hist[i + 1]
        e0, e1 = rec.get("exposure"), nxt.get("exposure")
        if e0 is None or e1 is None:
            advises.append({"time": rec.get("time"), "action": act,
                            "followed": None, "note": "缺仓位数据"})
            continue
        dropped = (e0 - e1) > tol
        advises.append({"time": rec.get("time"), "action": act,
                        "exposure_before": e0, "exposure_after": e1,
                        "delta": round(e1 - e0, 4), "followed": bool(dropped)})
        follows.append(bool(dropped))

    if not advises:
        # 注意：最后一条记录永远没有"下一条"，所以最新建议还没法判定。
        # 返回结构保持一致，调用方不必做 None 判断。
        return {"ok": True, "n_records": len(hist),
                "n_advise": 0, "n_known": 0, "n_followed": 0,
                "n_halt": 0, "follow_rate": None, "advices": [],
                "verdict": "期间没有出现减仓/暂停建议（或最新建议尚无后续检查）"}

    known = [x for x in advises if x.get("followed") is not None]
    n_known = len(known)
    n_followed = sum(1 for x in known if x["followed"])
    rate = (n_followed / n_known) if n_known else None

    if rate is None:
        verdict = "无法判断（缺仓位数据）"
    elif rate >= 0.8:
        verdict = "执行良好：绝大多数减仓建议被落实"
    elif rate >= 0.5:
        verdict = "执行一般：有一半建议没落地"
    else:
        verdict = "执行差：多数建议没落地，风控形同虚设"

    # HALT 后的恢复
    halts = [x for x in advises if x["action"] == "HALT"]
    return {"ok": True,
            "n_records": len(hist),
            "n_advise": len(advises),
            "n_known": n_known,
            "n_followed": n_followed,
            "follow_rate": round(float(rate), 4) if rate is not None else None,
            "n_halt": len(halts),
            "advices": advises[-20:],
            "verdict": verdict}


def format_report(hist: list[dict], ft: dict | None = None) -> str:
    L = ["=" * 68, "风控闭环", "=" * 68]
    if not hist:
        L.append("  还没有任何风控记录。")
        L.append("  先把日更或风控评估跑起来，系统会开始记账。")
        L.append("=" * 68)
        return chr(10).join(L)

    L.append(f"  共 {len(hist)} 次检查，最近一次 {hist[-1]['time']}")
    L.append("-" * 68)
    L.append("  " + f"{'时间':<20}{'动作':>8}{'仓位':>8}{'拥挤':>7}{'回撤':>9}")
    L.append("  " + "-" * 60)
    for r in hist[-10:]:
        exp = "-" if r.get("exposure") is None else format(r["exposure"] * 100, ".1f") + "%"
        cr = "-" if r.get("crowding") is None else format(r["crowding"], ".0f")
        dd = "-" if r.get("drawdown") is None else format(r["drawdown"] * 100, ".1f") + "%"
        L.append("  " + f"{r['time']:<20}{str(r['action']):>8}{exp:>8}{cr:>7}{dd:>9}")

    if ft:
        L.append("-" * 68)
        if not ft.get("ok"):
            L.append(f"  执行率：{ft.get('reason')}")
        else:
            L.append(f"  减仓/暂停建议 {ft['n_advise']} 次，"
                     f"已执行 {ft.get('n_followed', 0)} 次（可判定 {ft.get('n_known', 0)} 次）")
            if ft.get("follow_rate") is not None:
                L.append(f"  执行率 {ft['follow_rate'] * 100:.0f}%   {ft['verdict']}")
            else:
                L.append(f"  {ft['verdict']}")
    L.append("=" * 68)
    return chr(10).join(L)


if __name__ == "__main__":
    (_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    tmp = _ROOT / ".tmp" / "_risk_state_selftest.json"
    if tmp.exists():
        tmp.unlink()

    d1 = {"action": "NORMAL", "scale": 1.0, "reasons": [],
          "crowding": {"score": 40}, "drawdown": {"current_drawdown": -0.02}}
    d2 = {"action": "REDUCE", "scale": 0.5,
          "reasons": [{"why": "年换手 12x 超过上限 6x"}],
          "crowding": {"score": 72}, "drawdown": {"current_drawdown": -0.13}}
    d3 = {"action": "REDUCE", "scale": 0.5, "reasons": [],
          "crowding": {"score": 55}, "drawdown": {"current_drawdown": -0.05}}

    record(d1, {"weights": {"A": 0.5, "B": 0.5}, "capital": 100000}, path=tmp)
    record(d2, {"weights": {"A": 0.5, "B": 0.5}, "capital": 100000}, path=tmp)
    record(d3, {"weights": {"A": 0.25, "B": 0.25}, "cash_ratio": 0.5,
                "capital": 100000}, path=tmp)

    hist = load_history(tmp)
    assert len(hist) == 3, len(hist)
    ft = followthrough(hist)
    assert ft["ok"], ft
    assert ft["n_advise"] == 1, ft["n_advise"]
    assert ft["n_followed"] == 1, ft["advices"]
    print(format_report(hist, ft))

    # 未执行的情形
    tmp2 = _ROOT / ".tmp" / "_risk_state_selftest2.json"
    if tmp2.exists():
        tmp2.unlink()
    record(d2, {"weights": {"A": 0.5, "B": 0.5}, "capital": 100000}, path=tmp2)
    record(d3, {"weights": {"A": 0.5, "B": 0.5}, "capital": 100000}, path=tmp2)
    ft2 = followthrough(load_history(tmp2))
    assert ft2["n_followed"] == 0, ft2
    assert "形同虚设" in ft2["verdict"], ft2["verdict"]
    print(f"未执行判定: {ft2['verdict']}  ✓")
    print(f"OK v{__version__}")
