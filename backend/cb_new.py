# -*- coding: utf-8 -*-
"""
可转债打新扫描器（M40）
====================================================
对个人投资者，这是**确定性最高**的正期望动作之一：

* 零成本（中签才缴款）
* 单个账户上限固定（顶格申购 100 万市值，实际缴款最多 1000 元/签）
* 破发有风险，但可提前量化

判断依据（全部来自公开数据）
----------------------------
1. **转股价值** = 100 / 转股价 × 正股价
   —— 决定上市首日价格的核心，>100 是赚的起点
2. **市场溢价率** = 近期已上市转债的（现价 / 转股价值 - 1）中位数
   —— A 股转债市场长期给 10%~40% 溢价，这是打新赚钱的根源
3. **预期首日价格** = 转股价值 × (1 + 市场溢价率)
   —— <100 意味着大概破发
4. **发行规模** —— 小盘（<10 亿）易被炒，涨幅更大；大盘稳但弹性小
5. **信用评级** —— AA- 以下风险明显更高

⚠ 诚实的边界
-------------
* 这是**启发式打分**，不是预测模型；历史规律会变
* 中签率很低（历史 0.001%~0.01%），**大多数时候你什么都中不到**
* 破发是真实存在的（2022 年、2024 年都出现过批量破发）
* 本模块只做「要不要申购」的判断，不保证赚钱
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["RISK_LEVELS", "load_bonds", "market_premium", "assess_bond",
           "scan", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "cb"
CACHE_FILE = CACHE_DIR / "bond_list.csv"
CACHE_MAX_AGE_HOURS = 6

# 首日溢价上限：全市场转债的 42% 中位溢价不能直接套到新债首日
# （新债首日有抛压 + 涨幅规则约束，实际多在 10%~30%）。
PREMIUM_CAP = 0.30

RISK_LEVELS = [(25, "低", "转股价值高，破发概率小"),
               (45, "中低", "有一定保护"),
               (65, "中", "接近盈亏平衡，看上市时点情绪"),
               (85, "中高", "转股价值偏低，破发风险明显"),
               (101, "高", "很可能破发，建议放弃")]


def _to_num(s) -> pd.Series:
    return pd.to_numeric(pd.Series(s).astype(str)
                         .str.replace("%", "", regex=False)
                         .str.replace(",", "", regex=False),
                         errors="coerce")


def load_bonds(refresh: bool = False) -> pd.DataFrame:
    """拉取全部可转债数据（6 小时缓存）。"""
    if not refresh and CACHE_FILE.exists():
        try:
            age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
            if age_h <= CACHE_MAX_AGE_HOURS:
                df = pd.read_csv(CACHE_FILE, dtype={"债券代码": str,
                                                    "申购代码": str,
                                                    "正股代码": str})
                if len(df) > 50:
                    return df
        except Exception:
            pass

    import akshare as ak
    raw = ak.bond_zh_cov()
    if raw is None or len(raw) == 0:
        raise RuntimeError("可转债列表返回空")
    df = raw.copy()
    for c in ("债券代码", "申购代码", "正股代码"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.zfill(6)
    for c in ("正股价", "转股价", "转股价值", "债现价", "发行规模",
              "中签率", "转股溢价率"):
        if c in df.columns:
            df[c] = _to_num(df[c])

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False, encoding="utf-8")
    except Exception:
        pass
    return df


def market_premium(df: pd.DataFrame, recent: int = 60) -> dict:
    """
    从已上市转债估算市场溢价率（现价/转股价值-1 的中位数）。

    这是打新赚钱的根基：A 股转债长期给正溢价。
    """
    if not {"债现价", "转股价值"}.issubset(df.columns):
        return {"ok": False, "reason": "缺 债现价/转股价值 字段", "premium": None}

    d = df[(df["转股价值"] > 0) & (df["债现价"] > 0)].copy()
    if len(d) == 0:
        return {"ok": False, "reason": "没有同时有现价和转股价值的样本",
                "premium": None}
    d["溢价"] = d["债现价"] / d["转股价值"] - 1.0
    recent_d = d.tail(int(recent))
    med = float(d["溢价"].median())
    med_recent = float(recent_d["溢价"].median())
    return {"ok": True, "premium": round(med, 4),
            "premium_recent": round(med_recent, 4),
            "n": int(len(d)), "n_recent": int(len(recent_d)),
            "note": "近期样本更贴近当下情绪"}


def assess_bond(row: dict, premium: float | None) -> dict:
    """
    给单只新债打分（0~100，越高越危险）。

    核心：预期首日价 = 转股价值 × (1 + 市场溢价)；距 100 多远决定风险。
    """
    code = str(row.get("债券代码", ""))
    name = str(row.get("债券简称", ""))
    price = row.get("正股价")
    conv = row.get("转股价")
    cv = row.get("转股价值")
    size = row.get("发行规模")
    rating = str(row.get("信用评级") or "").strip()

    # 转股价值优先用字段，缺失则用 正股价/转股价 现算
    if (cv is None or not np.isfinite(cv) or cv <= 0):
        if price and conv and conv > 0:
            cv = 100.0 / float(conv) * float(price)
        else:
            cv = None

    reasons: list[str] = []
    score = 50.0

    if cv is None:
        return {"code": code, "name": name, "ok": False,
                "reason": "缺转股价/正股价，无法估值", "score": None}

    cv = float(cv)
    # ① 转股价值（权重最大）
    if cv >= 105:
        score -= 30
        reasons.append(f"转股价值 {cv:.1f}，保本垫很厚")
    elif cv >= 95:
        score -= 15
        reasons.append(f"转股价值 {cv:.1f}，接近面值")
    elif cv >= 85:
        score += 10
        reasons.append(f"转股价值 {cv:.1f}，需要正溢价才不破发")
    else:
        score += 30
        reasons.append(f"转股价值仅 {cv:.1f}，破发风险高")

    # ② 预期首日价
    exp_price = None
    if premium is not None and np.isfinite(premium):
        used = min(float(premium), PREMIUM_CAP)
        exp_price = cv * (1.0 + used)
        if exp_price >= 115:
            score -= 12
            cap_note = ("（已按首日上限 %.0f%% 保守估计）" % (PREMIUM_CAP * 100)
                        if float(premium) > PREMIUM_CAP else "")
            reasons.append(f"按市场溢价 {used*100:.0f}% 估，"
                           f"首日约 {exp_price:.0f} 元{cap_note}")
        elif exp_price >= 100:
            score -= 3
            reasons.append(f"估首日约 {exp_price:.0f} 元（薄利，"
                           f"按溢价 {used*100:.0f}%）")
        else:
            score += 28
            reasons.append(f"估首日约 {exp_price:.0f} 元 → **预计破发**")

    # ③ 发行规模（小盘易炒）
    if size is not None and np.isfinite(size):
        if size <= 5:
            score -= 6
            reasons.append(f"盘小（{size:.1f} 亿），易被炒")
        elif size >= 30:
            score += 4
            reasons.append(f"盘大（{size:.1f} 亿），涨幅弹性小")

    # ④ 评级
    if rating:
        low_grade = any(g in rating for g in ("A+", "A-", "BBB", "BB", "B"))
        if low_grade:
            score += 8
            reasons.append(f"评级 {rating}，信用风险偏高")

    score = float(np.clip(score, 0, 100))
    level, note = "中", ""
    for bound, lv, nt in RISK_LEVELS:
        if score < bound:
            level, note = lv, nt
            break

    advice = ("放弃" if score >= 85 else
              "谨慎申购" if score >= 65 else "正常申购")

    return {"code": code, "name": name, "ok": True,
            "convert_value": round(cv, 2),
            "expected_price": (round(exp_price, 2)
                               if exp_price is not None else None),
            "issue_size": (round(float(size), 2)
                           if size is not None and np.isfinite(size) else None),
            "rating": rating or None,
            "score": round(score, 1), "level": level, "note": note,
            "advice": advice, "reasons": reasons}


def scan(days_ahead: int = 7, refresh: bool = False,
         today: str | None = None) -> dict:
    """
    扫描未来 N 天内可申购的新债。

    返回 {ok, date, items, premium, verdict}
    """
    df = load_bonds(refresh=refresh)
    pm = market_premium(df)
    premium = pm.get("premium_recent") if pm.get("ok") else None

    t0 = pd.Timestamp(today) if today else pd.Timestamp(date.today())
    t1 = t0 + pd.Timedelta(days=int(days_ahead))
    d = df.copy()
    d["_d"] = pd.to_datetime(d.get("申购日期"), errors="coerce")
    todo = d[(d["_d"] >= t0) & (d["_d"] <= t1)].sort_values("_d")

    items = []
    for _, row in todo.iterrows():
        r = row.to_dict()
        r["申购日期"] = (r["_d"].date().isoformat()
                         if pd.notna(r["_d"]) else None)
        a = assess_bond(r, premium)
        a["apply_date"] = r.get("申购日期")
        a["apply_code"] = row.get("申购代码")
        a["stock_name"] = row.get("正股简称")
        a["upper"] = row.get("申购上限")
        items.append(a)

    if not items:
        verdict = f"{t0.date()} 起 {days_ahead} 天内没有可申购的新债"
    else:
        go = [x for x in items if x.get("ok") and x["advice"] != "放弃"]
        verdict = (f"共 {len(items)} 只可申购，建议申 {len(go)} 只，"
                   f"放弃 {len(items)-len(go)} 只")
    return {"ok": True, "from": str(t0.date()), "to": str(t1.date()),
            "days_ahead": int(days_ahead), "market_premium": pm,
            "premium_used": premium, "items": items, "verdict": verdict}


def format_report(res: dict) -> str:
    L = ["=" * 70, "可转债打新扫描", "=" * 70]
    if not res.get("ok"):
        L.append(f"  失败：{res.get('reason')}")
        L.append("=" * 70)
        return "\n".join(L)

    pm = res.get("market_premium") or {}
    L.append(f"  扫描区间 {res['from']} ~ {res['to']}")
    if pm.get("ok"):
        L.append(f"  市场溢价率：长期中位 {pm['premium']*100:.1f}%  "
                 f"近期中位 {pm['premium_recent']*100:.1f}%"
                 f"（{pm['n']} 只样本）")
    else:
        L.append(f"  市场溢价率不可用：{pm.get('reason')}")
    L.append(f"  {res['verdict']}")
    L.append("-" * 70)

    if not res["items"]:
        L.append("  （无）")
    else:
        L.append("  " + f"{'申购日':<12}{'名称':<10}{'转股价值':>9}"
                 f"{'估首日':>8}{'规模':>7}{'风险':>7}{'建议':>10}")
        L.append("  " + "-" * 64)
        for x in res["items"]:
            if not x.get("ok"):
                L.append("  " + f"{x.get('apply_date', ''):<12}"
                         f"{x.get('name', '')[:8]:<10}  无法估值：{x.get('reason')}")
                continue
            L.append("  " + f"{x['apply_date'] or '':<12}"
                     f"{x['name'][:8]:<10}"
                     f"{x['convert_value']:>9.1f}"
                     f"{(x['expected_price'] or 0):>8.0f}"
                     f"{(x['issue_size'] or 0):>7.1f}"
                     f"{x['level']:>7}"
                     f"{x['advice']:>10}")

    L.append("-" * 70)
    for x in res["items"]:
        if x.get("ok"):
            L.append(f"  {x['name']}（{x['code']}，申购代码 {x['apply_code']}）")
            for r in x["reasons"]:
                L.append(f"    · {r}")
    L.append("=" * 70)
    L.append("  ⚠ 中签率很低（历史 0.001%~0.01%），大多数时候什么都中不到。")
    L.append("     破发真实存在（2022/2024 都出现过批量破发），本工具只辅助判断，不保证赚钱。")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # 离线自测：构造数据验证逻辑
    fake = pd.DataFrame({
        "债券代码": ["111001", "111002", "111003"],
        "债券简称": ["高价债", "平价款", "破发款"],
        "正股价": [12.0, 10.0, 8.0],
        "转股价": [10.0, 10.0, 10.0],
        "转股价值": [120.0, 100.0, 80.0],
        "发行规模": [3.0, 15.0, 40.0],
        "信用评级": ["AA", "AA-", "A+"],
        "债现价": [130.0, 115.0, 90.0],
    })
    pm = market_premium(fake)
    print("市场溢价:", pm)
    assert pm["ok"], pm
    # 三个样本溢价 0.083/0.150/0.125 → 中位数 0.125
    assert abs(pm["premium"] - 0.125) < 1e-6, pm["premium"]

    for i, row in fake.iterrows():
        a = assess_bond(row.to_dict(), pm["premium"])
        print(f"  {a['name']}: 转股价值 {a['convert_value']:.0f} "
              f"估首日 {a['expected_price']:.0f} 风险 {a['score']:.0f} "
              f"({a['level']}) → {a['advice']}")
        assert a["ok"]

    hi = assess_bond(fake.iloc[0].to_dict(), 0.15)
    lo = assess_bond(fake.iloc[2].to_dict(), 0.15)
    assert hi["score"] < lo["score"], (hi["score"], lo["score"])
    assert lo["advice"] == "放弃", lo["advice"]
    print("  结论：转股价值高的风险低、低的建议放弃 ✓")

    # 缺数据要拒绝而不是瞎算
    bad = assess_bond({"债券代码": "x", "债券简称": "x"}, 0.15)
    assert bad["ok"] is False
    print("  缺转股价/正股价 → 拒绝估值 ✓")
    print("\nOK v" + __version__)
