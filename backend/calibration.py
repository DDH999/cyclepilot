# -*- coding: utf-8 -*-
"""
评分效度校准（M21）
====================================================
**这是整个系统最该被质疑的地方。**

我们一直在用六维打分给策略评级，但从来没有验证过：
**这把尺子量得准吗？**

本模块做三件能做的事，并明确说出一件做不到的事。

能做
----
1. **敏感性**：某一个已知缺陷出现时，对应维度是否真的掉分
2. **区分度**：干净策略与缺陷策略的分数是否分得开（AUC 近似）
3. **排序一致性**：按"已知质量"排序 vs 系统打分排序，Spearman 相关

做不到（必须承认）
------------------
**预测效度**：系统打分能否预测**未来真实收益**。
这需要一批"已知 3 年样本外表现"的真实策略，我们没有。
所以本模块只证明"能识别已知缺陷"，**不证明"能选出赚钱的策略"**。

设计原则
--------
所有输入都由**真实模块计算**（validator / stress_tester / purity_tester），
不手搓百分比，否则测的就不是系统本身了。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["PROFILES", "build_scenario", "run_profile", "calibrate",
           "format_report"]

TD = 252
START, END = "2014-01-02", "2025-12-31"

# 故意的未来函数（用于作弊场景）
CHEAT_CODE = """
def signal(df):
    future = df['close'].shift(-1)      # 未来函数
    return (future > df['close']).astype(int)
"""

CLEAN_CODE = """
def signal(df):
    ma20 = df['close'].rolling(20).mean()
    return (df['close'] > ma20).astype(int)
"""


# ══════════════════════════════════════════════════════════
# 场景构造
# ══════════════════════════════════════════════════════════
def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range(START, END)


def _bench(rng, n) -> np.ndarray:
    return rng.normal(0.00028, 0.0105, n)


def build_scenario(name: str, seed: int = 7) -> dict:
    """
    构造一个已知质量的策略画像。

    返回 dict：returns / bench / weights / code / turnover / slippage_rate /
              n_trials / expected（期望结论）/ defect（缺陷类型，None 表示干净）
    """
    rng = np.random.default_rng(seed)
    idx = _calendar()
    n = len(idx)
    bench = _bench(rng, n)
    codes = [f"{i:06d}" for i in range(12)]

    base = {"returns": None, "bench": pd.Series(bench, index=idx),
            "weights": None, "code": CLEAN_CODE,
            "turnover": 2.0, "slippage_rate": 0.0005,
            "n_trials": 1, "expected": "pass", "defect": None}

    if name == "clean_alpha":
        # 真实 alpha：0.4 beta + 稳定超额 + 低噪声
        r = 0.40 * bench + 0.00045 + rng.normal(0, 0.006, n)
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": pd.Series(r, index=idx), "weights": w,
                "expected": "pass", "defect": None}

    if name == "beta_only":
        # 纯 beta：没有超额，但六个维度都"正常"
        r = bench + rng.normal(0, 0.0015, n)
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": pd.Series(r, index=idx), "weights": w,
                "expected": "blind_spot", "defect": "no_alpha"}

    if name == "overfit":
        # 样本内极好，样本外崩掉
        half = n // 2
        r = np.concatenate([
            rng.normal(0.0018, 0.008, half),          # IS：夏普很高
            rng.normal(-0.0008, 0.014, n - half),     # OOS：反转
        ])
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": pd.Series(r, index=idx), "weights": w,
                "n_trials": 30, "expected": "detect", "defect": "overfit"}

    if name == "high_turnover":
        # 高换手：同样的收益，被成本吃掉大半
        r = 0.40 * bench + 0.00040 + rng.normal(0, 0.006, n)
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": pd.Series(r, index=idx), "weights": w,
                "turnover": 24.0, "slippage_rate": 0.0020,
                "expected": "detect", "defect": "cost"}

    if name == "concentrated":
        # 单一持仓 90%，其余各 1% 左右
        wv = np.full(12, 0.0085)
        wv[0] = 0.90 - 11 * 0.0085
        w = pd.DataFrame(np.tile(wv, (n, 1)), index=idx, columns=codes)
        r = 0.40 * bench + 0.00040 + rng.normal(0, 0.008, n)
        return {**base, "returns": pd.Series(r, index=idx), "weights": w,
                "expected": "detect", "defect": "purity"}

    if name == "crash_prone":
        # 平时正常，2024 微盘股窗口巨亏
        r = 0.40 * bench + 0.00035 + rng.normal(0, 0.007, n)
        s = pd.Series(r, index=idx)
        mask = (s.index >= "2024-01-01") & (s.index <= "2024-02-08")
        s.loc[mask] = rng.normal(-0.045, 0.020, int(mask.sum()))
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": s, "weights": w,
                "expected": "detect", "defect": "stress"}

    if name == "unstable":
        # 收益在不同区间剧烈漂移 → 参数敏感性宽
        r = np.concatenate([
            rng.normal(0.0015, 0.012, n // 3),
            rng.normal(-0.0010, 0.012, n // 3),
            rng.normal(0.0002, 0.012, n - 2 * (n // 3)),
        ])
        s = pd.Series(r, index=idx).sample(frac=1.0, random_state=seed).sort_index()
        s = pd.Series(r, index=idx)
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": s, "weights": w,
                "expected": "detect", "defect": "stability"}

    if name == "cheat":
        r = 0.40 * bench + 0.00060 + rng.normal(0, 0.004, n)
        w = pd.DataFrame(np.full((n, 12), 1 / 12), index=idx, columns=codes)
        return {**base, "returns": pd.Series(r, index=idx), "weights": w,
                "code": CHEAT_CODE, "expected": "reject", "defect": "cheat"}

    raise ValueError(f"未知场景：{name}")


PROFILES = [
    ("clean_alpha", "干净 alpha", None, "应得高分"),
    ("beta_only", "纯 beta 无 alpha", "no_alpha", "六维**看不出**（盲区）"),
    ("overfit", "过拟合", "overfit", "样本外维度应降"),
    ("high_turnover", "高换手", "cost", "成本维度应降"),
    ("concentrated", "过度集中", "purity", "纯度维度应降"),
    ("crash_prone", "极端行情巨亏", "stress", "抗压维度应降"),
    ("unstable", "分段漂移", "stability", "稳健性维度应降"),
    ("cheat", "未来函数", "cheat", "应直接一票否决"),
]


# ══════════════════════════════════════════════════════════
# 跑单场景
# ══════════════════════════════════════════════════════════
def _sensitivity_grid(returns: pd.Series) -> dict:
    """
    参数敏感性：不同回看窗口下的年化夏普，取 min/max/median。
    窗口差异越大 → 越不稳健。
    """
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    sharpes = []
    for win in (20, 40, 60, 120, 250):
        if len(r) <= win + 20:
            continue
        roll = r.rolling(win).mean() / r.rolling(win).std(ddof=0)
        v = (roll * np.sqrt(TD)).dropna()
        if len(v):
            sharpes.append(float(v.median()))
    if not sharpes:
        return {}
    return {"min": round(min(sharpes), 4), "max": round(max(sharpes), 4),
            "median": round(float(np.median(sharpes)), 4)}


def run_profile(name: str, seed: int = 7, verbose: bool = False) -> dict:
    """跑一个场景，返回各维度结果 + 总分 + 等级。"""
    import purity_tester as PT
    import scoring as SC
    import stress_tester as ST
    import validator as V

    sc = build_scenario(name, seed)
    rets = sc["returns"]
    df_ret = pd.DataFrame({"ret": rets})

    # 1) 代码：未来函数
    code_ok, findings = None, None
    if sc.get("code"):
        findings = V.scan_code(sc["code"])
        code_ok = len(findings) == 0

    # 2) 成本：滑点/换手敏感性
    try:
        slip = ST.apply_slippage(df_ret, slippage_rate=sc["slippage_rate"],
                                 turnover=sc["turnover"],
                                 turnover_period="year", verbose=False)
    except Exception:
        slip = None

    # 3) 抗压：极端行情套件
    # 注意：run_crash_suite 返回 list[dict]，而 scoring.dim_stress 要 dict，
    # 直接传 list 会在 item_from_crash 里 AttributeError。
    # 这里取「最差的那个窗口」作为代表。
    crash = None
    try:
        suite = ST.run_crash_suite(df_ret, verbose=False)
        ok = [x for x in suite if isinstance(x, dict) and x.get("ok")]
        if ok:
            crash = max(ok, key=lambda x: (x.get("ratio") or 0))
        elif isinstance(suite, dict):
            crash = suite
    except Exception:
        crash = None

    # 4) 过拟合：前后半段
    overfit = None
    try:
        half = len(rets) // 2
        overfit = V.calculate_overfitting(rets.iloc[:half], rets.iloc[half:],
                                          n_trials=sc["n_trials"], verbose=False)
    except Exception:
        overfit = None

    # 5) 纯度
    purity = None
    try:
        purity = PT.purity_report(weights=sc["weights"], returns=rets,
                                  bench=sc["bench"], verbose=False)
    except Exception:
        purity = None

    # 6) 稳健性
    sens = _sensitivity_grid(rets)

    rep = SC.full_report(code_ok=code_ok, findings=findings, overfit=overfit,
                         slippage=slip, crash=crash, purity=purity,
                         sensitivity=sens or None)

    return {
        "name": name,
        "expected": sc["expected"],
        "defect": sc["defect"],
        "score": rep.get("score"),
        "grade": rep.get("grade"),
        "verdict": rep.get("verdict"),
        "status": rep.get("status") or rep.get("final_status"),
        "dims": {d["key"]: {"score": d["score"], "status": d["status"]}
                 for d in rep.get("dimensions", [])},
        "raw": {"overfit": overfit, "slippage": slip,
                "purity": purity, "sensitivity": sens,
                "n_findings": len(findings or [])},
    }


# ══════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════
def _auc(pos: list[float], neg: list[float]) -> float | None:
    """AUC：好策略得分高于坏策略的概率。"""
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def calibrate(seeds: tuple[int, ...] = (7, 11), verbose: bool = False) -> dict:
    """
    跑全部场景（多随机种子），汇总：
      * 每个场景的分数与等级
      * 区分度 AUC（干净 vs 缺陷）
      * 各维度敏感性（对应维度是否下降）
      * 盲区（未被检出的缺陷）
    """
    runs = []
    for name, label, defect, expect in PROFILES:
        for sd in seeds:
            r = run_profile(name, seed=sd, verbose=verbose)
            r.update(label=label, expect=expect)
            runs.append(r)

    clean = [r["score"] for r in runs if r["defect"] is None]
    defective = [r["score"] for r in runs
                 if r["defect"] not in (None, "no_alpha")]
    blind = [r["score"] for r in runs if r["defect"] == "no_alpha"]

    # 维度敏感性：缺陷场景中，对应维度是否低于干净场景
    dim_map = {"overfit": "anti_overfit", "cost": "cost",
               "purity": "purity", "stress": "stress",
               "stability": "stability", "cheat": "anti_cheat"}
    sensitivity = {}
    clean_dims = {}
    if clean:
        for k in ("anti_overfit", "cost", "purity", "stress", "stability"):
            vals = [r["dims"].get(k, {}).get("score") for r in runs
                    if r["defect"] is None]
            vals = [v for v in vals if v is not None]
            clean_dims[k] = float(np.mean(vals)) if vals else None

    detected, missed = [], []
    for defect, dim in dim_map.items():
        own = [r for r in runs if r["defect"] == defect]
        if not own:
            continue
        ok_all = True
        for r in own:
            base = clean_dims.get(dim)
            cur = r["dims"].get(dim, {}).get("score")
            if dim == "anti_cheat":
                hit = (r["dims"].get("anti_cheat", {}).get("status")
                       in ("REJECTED", "FAIL", "WARN")) or (r["score"] or 0) < 60
            elif base is None or cur is None:
                hit = (r["score"] or 0) < (np.mean(clean) - 5 if clean else 75)
            else:
                hit = cur < base - 1.0
            ok_all = ok_all and hit
        (detected if ok_all else missed).append(
            {"defect": defect, "dim": dim,
             "baseline": clean_dims.get(dim),
             "observed": [r["dims"].get(dim, {}).get("score") for r in own]})
        sensitivity[defect] = ok_all

    # 排序一致性（只看能排序的：干净 > 缺陷）
    order_ok = []
    if clean:
        c = float(np.mean(clean))
        for d in (defective or []):
            order_ok.append(d < c)
    rank_acc = round(sum(order_ok) / len(order_ok), 4) if order_ok else None

    return {
        "ok": True,
        "n_runs": len(runs),
        "seeds": list(seeds),
        "runs": runs,
        "clean_mean": round(float(np.mean(clean)), 2) if clean else None,
        "defect_mean": round(float(np.mean(defective)), 2) if defective else None,
        "blindspot_mean": round(float(np.mean(blind)), 2) if blind else None,
        "auc": _auc(clean, defective),
        "rank_accuracy": rank_acc,
        "detected": detected,
        "missed": missed,
        "sensitivity": sensitivity,
        "verdict": _verdict(clean, defective, missed, blind),
    }


def _verdict(clean, defective, missed, blind) -> str:
    if not clean:
        return "无法判定：干净样本缺失"
    gap = float(np.mean(clean)) - (float(np.mean(defective)) if defective else 0)
    if missed:
        names = "、".join(m["defect"] for m in missed)
        return (f"部分失效：{names} 未被检出。"
                f"干净组 {np.mean(clean):.0f} 分 vs 缺陷组 "
                f"{np.mean(defective) if defective else float('nan'):.0f} 分（差 {gap:.0f}）")
    return f"六个已知缺陷均被检出，分数差 {gap:.0f} 分"


def format_report(res: dict) -> str:
    L = ["=" * 70, "评分效度校准", "=" * 70]
    L.append(f"  跑了 {res['n_runs']} 次（种子 {res['seeds']}）")
    L.append(f"  干净组均分   {res['clean_mean']}")
    L.append(f"  缺陷组均分   {res['defect_mean']}")
    if res.get("blindspot_mean") is not None:
        L.append(f"  纯 beta 组   {res['blindspot_mean']}   ← 本该低却高 = 盲区")
    L.append(f"  区分度 AUC   {res['auc']}   （1.0=完美分开，0.5=瞎猜）")
    L.append(f"  排序正确率   {res['rank_accuracy']}")
    L.append("-" * 70)
    L.append("  " + f"{'场景':<22}{'等级':>6}{'总分':>8}" + "   期望")
    L.append("  " + "-" * 64)
    seen = set()
    for r in res["runs"]:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        sc = "—" if r["score"] is None else format(r["score"], ".1f")
        gr = "-" if not r["grade"] else str(r["grade"])
        L.append("  " + f"{r['label']:<22}{gr:>6}{sc:>8}" + "   " + str(r["expect"]))
    L.append("-" * 70)
    L.append("  缺陷检出：")
    for defect, ok in res["sensitivity"].items():
        L.append("    " + ("[检出] " if ok else "[漏检] ") + defect)
    L.append("-" * 70)
    L.append(f"  结论: {res['verdict']}")
    L.append("  ⚠ 本校准只证明「能识别已知缺陷」，")
    L.append("     不能证明「能选出未来赚钱的策略」——那需要真实策略库。")
    L.append("=" * 70)
    return chr(10).join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    res = calibrate(seeds=(7,))
    print(format_report(res))

    print()
    print("各维度明细（种子 7）：")
    for r in res["runs"]:
        parts = []
        for k, v in r["dims"].items():
            s = "-" if v["score"] is None else format(v["score"], ".0f")
            parts.append(k + "=" + s)
        sc = "-" if r["score"] is None else format(r["score"], ".1f")
        print("  " + f"{r['label']:<20} 总分 {sc:>6} {r['grade']} | " + " ".join(parts))
    print()
    print(f"OK v{__version__}")
