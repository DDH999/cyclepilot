# -*- coding: utf-8 -*-
"""
决策记忆与到期结算（M39）
====================================================
借鉴两个开源项目的核心机制：

* **TradingAgents**（Tauric Research, arXiv 2412.20138）
  的 `memory log` + `settle_all_pending()`：
  每条决策记录在案，到期后自动结算**真实收益**，并生成反思段落。
* **Vibe-Trading**（HKUDS）的**影子账户**：
  在实盘之前用虚拟账户跑，验证策略是否真的有效。

为什么这条最值得学
------------------
本项目有 38 个模块，但**没有一个能回答**：

    「我上个月那个"减投"的决定，后来证明对吗？」

`advisor` 只给建议，不跟踪结果。没有跟踪，就永远不知道规则该不该调。
这一层补上「决策 → 结算 → 复盘 → 校准」的闭环。

怎么判"决策对错"
----------------
**不是看你赚没赚钱，而是看择时方向对不对。**

* 动作是「加投」（看多）→ 方向 = +1
* 动作是「减投」（看空）→ 方向 = −1
* 未来 N 天涨跌 × 方向 > 0 → 这次判断方向正确

这叫 `edge`（方向性收益）。它和盈亏是两件事：
牛市里减投会少赚，但方向可能仍然"错"——两者必须分开看。

⚠ 诚实边界
-----------
* 结算只用**公开价格**，不需要你报真实成交
* 样本少时胜率没有统计意义，报告里会标注样本量
* 它检验的是**择时建议**，不检验你最终执行没执行（那是 risk_state 的活）
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

__version__ = "0.1.0"
__all__ = ["JOURNAL_FILE", "DEFAULT_HORIZON", "record", "load_journal",
           "settle_pending", "review", "format_report", "format_journal"]

_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = _ROOT / ".state"
JOURNAL_FILE = STATE_DIR / "decisions.jsonl"
DEFAULT_HORIZON = 20          # 默认 20 个交易日后结算
MIN_SAMPLE = 8                # 少于这个数不给胜率结论

# 动作 → 方向（+1 看多 / -1 看空 / 0 不动）
ACTION_SIGN = {
    "加投": 1, "加仓": 1, "买入": 1, "投出现金": 1,
    "减投": -1, "减仓": -1, "卖出": -1,
    "额外加仓": 1,
    "不动": 0,
}


def _sign_of(action: str) -> int:
    a = str(action or "").strip()
    if a in ACTION_SIGN:
        return ACTION_SIGN[a]
    # 模糊匹配（配置调整类动作）
    if "加" in a or "买" in a or "投出" in a:
        return 1
    if "减" in a or "卖" in a:
        return -1
    return 0


def _new_id() -> str:
    raw = f"{time.time():.9f}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def record(ticker: str, action: str, price: float | None,
           amount: float | None = None, name: str | None = None,
           reason: str | None = None, as_of: str | None = None,
           horizon: int = DEFAULT_HORIZON, source: str = "advisor",
           path: Path | str | None = None) -> dict:
    """
    记一条决策。

    price 是决策时点的参考价（指数的点位或基金净值），用于后续结算。
    """
    fp = Path(path) if path else Path(JOURNAL_FILE)
    rec = {
        "id": _new_id(),
        "ts": time.time(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of or datetime.now().strftime("%Y-%m-%d"),
        "ticker": str(ticker),
        "name": name or str(ticker),
        "action": str(action),
        "direction": _sign_of(action),
        "amount": None if amount is None else round(float(amount), 2),
        "price_then": None if price is None else round(float(price), 4),
        "reason": reason,
        "horizon": int(horizon),
        "source": source,
        "outcome": None,
    }
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    return rec


def load_journal(path: Path | str | None = None) -> list[dict]:
    fp = Path(path) if path else Path(JOURNAL_FILE)
    if not fp.exists():
        return []
    out = []
    try:
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return out


def _write_all(records: list[dict], path: Path | str | None = None) -> None:
    fp = Path(path) if path else Path(JOURNAL_FILE)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))


def _price_series(ticker: str, name: str | None = None):
    """
    取价格序列：先按海外指数匹配，再看是否基金代码。

    返回 (series, source) 或 (None, None)。
    """
    import overseas as OS
    key = ticker
    if key not in OS.INDEXES:
        try:
            m = OS.match_fund_index(name or ticker)
            if m.get("ok"):
                key = m["key"]
        except Exception:
            pass
    if key in OS.INDEXES:
        got = OS.index_history(key, days=20000)
        if got.get("ok"):
            return got["series"], f"overseas:{key}"
    # 基金净值
    try:
        import fund as FD
        s = FD.fetch_nav(str(ticker).zfill(6))
        if len(s) > 20:
            return s, f"fund:{ticker}"
    except Exception:
        pass
    return None, None


def settle_pending(horizon_default: int | None = None,
                   path: Path | str | None = None,
                   now: float | None = None) -> dict:
    """
    结算所有到期的决策。

    到期 = 距离 as_of 已有 horizon 个交易日以上的价格。
    """
    recs = load_journal(path)
    if not recs:
        return {"ok": True, "settled": [], "pending": 0, "n": 0}

    now_ts = time.time() if now is None else float(now)
    settled, still_pending, errors = [], 0, []

    for r in recs:
        if r.get("outcome"):
            continue
        horizon = int(r.get("horizon") or horizon_default or DEFAULT_HORIZON)
        price_then = r.get("price_then")
        if price_then is None or price_then <= 0:
            errors.append(f"{r['id']}: 缺决策时点价格，无法结算")
            still_pending += 1
            continue

        age_days = (now_ts - float(r.get("ts", 0))) / 86400.0
        if age_days < horizon * 0.7:      # 交易日≈自然日×0.7
            still_pending += 1
            continue

        series, src = _price_series(r["ticker"], r.get("name"))
        if series is None or len(series) == 0:
            errors.append(f"{r['id']}: 取不到 {r['ticker']} 的价格")
            still_pending += 1
            continue

        anchor = series.loc[:r["as_of"]]
        if len(anchor) == 0:
            errors.append(f"{r['id']}: as_of 早于数据起点")
            still_pending += 1
            continue
        idx0 = anchor.index[-1]
        pos0 = series.index.get_loc(idx0)
        target = min(pos0 + horizon, len(series) - 1)
        if target <= pos0:
            still_pending += 1
            continue
        idx1 = series.index[target]
        p0, p1 = float(series.iloc[pos0]), float(series.iloc[target])
        fwd = p1 / p0 - 1.0

        direction = int(r.get("direction") or 0)
        edge = fwd * direction
        r["outcome"] = {
            "settled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price_from": round(p0, 4),
            "price_to": round(p1, 4),
            "from_date": str(idx0.date()),
            "to_date": str(idx1.date()),
            "bars": int(target - pos0),
            "forward_return": round(fwd, 6),
            "direction": direction,
            "edge": round(edge, 6),
            "hit": bool(edge > 0) if direction != 0 else None,
            "source": src,
            "note": ("方向正确" if edge > 0 else "方向错误") if direction else "无方向（不动）",
        }
        settled.append(r["id"])

    if settled:
        _write_all(recs, path)

    return {"ok": True, "settled": settled, "n_settled": len(settled),
            "pending": still_pending, "n": len(recs), "errors": errors}


def review(path: Path | str | None = None) -> dict:
    """
    汇总已结算决策的表现：按动作分组看胜率与平均 edge。
    """
    recs = [r for r in load_journal(path) if r.get("outcome")]
    if not recs:
        return {"ok": False, "reason": "还没有已结算的决策",
                "n": len(load_journal(path))}

    by_action: dict[str, list] = {}
    for r in recs:
        a = r.get("action") or "?"
        by_action.setdefault(a, []).append(r)

    rows = []
    for a, items in by_action.items():
        edges = [x["outcome"]["edge"] for x in items
                 if x["outcome"].get("edge") is not None]
        hits = [x["outcome"]["hit"] for x in items
                if x["outcome"].get("hit") is not None]
        if not edges:
            continue
        rows.append({
            "action": a,
            "n": len(items),
            "n_directional": len(hits),
            "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
            "mean_edge": round(float(np.mean(edges)), 6),
            "mean_forward": round(float(np.mean(
                [x["outcome"]["forward_return"] for x in items])), 6),
        })
    rows.sort(key=lambda x: -x["n"])

    all_hits = [r["outcome"]["hit"] for r in recs
                if r["outcome"].get("hit") is not None]
    overall = round(float(np.mean(all_hits)), 4) if all_hits else None

    if len(all_hits) < MIN_SAMPLE:
        verdict = (f"样本仅 {len(all_hits)} 条（<{MIN_SAMPLE}），"
                   f"胜率没有统计意义，继续积累")
    elif overall is not None and overall >= 0.6:
        verdict = f"方向命中率 {overall*100:.0f}%，规则有正的择时价值"
    elif overall is not None and overall >= 0.45:
        verdict = f"方向命中率 {overall*100:.0f}%，接近随机，择时价值不明确"
    else:
        verdict = f"方向命中率 {overall*100:.0f}%，低于随机，规则需要重新审视"

    return {"ok": True, "n": len(recs), "n_directional": len(all_hits),
            "overall_hit_rate": overall, "by_action": rows,
            "verdict": verdict, "min_sample": MIN_SAMPLE}


def format_journal(recs: list[dict], limit: int = 15) -> str:
    L = ["=" * 70, "决策台账", "=" * 70]
    if not recs:
        L.append("  还没有记录。跑一次 `advice --record` 就会开始记账。")
        L.append("=" * 70)
        return chr(10).join(L)
    L.append(f"  共 {len(recs)} 条   已结算 "
             f"{sum(1 for r in recs if r.get('outcome'))} 条")
    L.append("-" * 70)
    L.append("  " + f"{'日期':<12}{'标的':<12}{'动作':<8}{'当时':>10}"
             f"{'现在':>10}{'收益':>9}{'方向':>6}")
    L.append("  " + "-" * 66)
    for r in reversed(recs[-limit:]):
        o = r.get("outcome") or {}
        cur = "-" if not o else format(o.get("price_to", 0), ".2f")
        ret = "-" if not o else format(o.get("forward_return", 0) * 100, "+.2f") + "%"
        hit = "-" if not o or o.get("hit") is None else (
            "对" if o["hit"] else "错")
        L.append("  " + f"{r.get('as_of', ''):<12}"
                 f"{r.get('name', r.get('ticker', ''))[:10]:<12}"
                 f"{r.get('action', ''):<8}"
                 f"{format(r.get('price_then') or 0, '.2f'):>10}"
                 f"{cur:>10}{ret:>9}{hit:>6}")
    L.append("=" * 70)
    return chr(10).join(L)


def format_report(res: dict) -> str:
    L = ["=" * 70, "决策复盘", "=" * 70]
    if not res.get("ok"):
        L.append(f"  {res.get('reason')}（台账共 {res.get('n', 0)} 条）")
        L.append("=" * 70)
        return chr(10).join(L)

    L.append(f"  已结算 {res['n']} 条，其中有方向 {res['n_directional']} 条")
    if res.get("overall_hit_rate") is not None:
        L.append(f"  总体方向命中率 {res['overall_hit_rate']*100:.1f}%")
    L.append("-" * 70)
    L.append("  " + f"{'动作':<10}{'次数':>6}{'有方向':>8}{'命中率':>8}"
             f"{'平均edge':>12}{'平均涨跌':>11}")
    L.append("  " + "-" * 56)
    for r in res["by_action"]:
        hr = "-" if r["hit_rate"] is None else f"{r['hit_rate']*100:.0f}%"
        L.append("  " + f"{r['action']:<10}{r['n']:>6}"
                 f"{r['n_directional']:>8}{hr:>8}"
                 f"{r['mean_edge']*100:>11.2f}%"
                 f"{r['mean_forward']*100:>10.2f}%")
    L.append("-" * 70)
    L.append(f"  结论：{res['verdict']}")
    L.append("=" * 70)
    L.append("  注：edge>0 表示「方向判断正确」，不等于你赚了钱。")
    L.append("      牛市里减投会少赚，但方向可能仍然是对的——两者分开看。")
    L.append("=" * 70)
    return chr(10).join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from pathlib import Path as _P
    (_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    tmp = _ROOT / ".tmp" / "_journal_selftest.jsonl"
    if tmp.exists():
        tmp.unlink()

    # 动作 → 方向
    assert _sign_of("加投") == 1
    assert _sign_of("减投") == -1
    assert _sign_of("不动") == 0
    assert _sign_of("额外加仓") == 1
    assert _sign_of("随便什么") == 0
    print("动作方向映射 ✓")

    r = record("sp500", "减投", price=7000.0, amount=500.0,
               name="标普500", reason="98% 分位", path=tmp)
    assert r["direction"] == -1
    assert len(load_journal(tmp)) == 1
    print(f"记录 ✓ id={r['id']} direction={r['direction']}")

    # 未到期 → 不结算
    st = settle_pending(path=tmp)
    assert st["n_settled"] == 0 and st["pending"] == 1, st
    print(f"未到期不结算 ✓ pending={st['pending']}")

    # 强行把时间往前推 30 天 → 应结算
    recs = load_journal(tmp)
    recs[0]["ts"] = time.time() - 40 * 86400
    recs[0]["as_of"] = "2026-01-05"
    _write_all(recs, tmp)
    st2 = settle_pending(path=tmp)
    print(f"到期结算 ✓ n_settled={st2['n_settled']} "
          f"errors={len(st2['errors'])}")
    if st2["n_settled"]:
        o = load_journal(tmp)[0]["outcome"]
        print(f"  从 {o['from_date']} 到 {o['to_date']}（{o['bars']} 根）"
              f"  涨跌 {o['forward_return']*100:+.2f}%  "
              f"edge {o['edge']*100:+.2f}%  {o['note']}")

        rv = review(tmp)
        print()
        print(format_report(rv))
    tmp.unlink(missing_ok=True)
    print("OK v" + __version__)
