# -*- coding: utf-8 -*-
"""
策略与研究层（M61）：参数级 Walk-forward
====================================================
与 M11 的 `walkforward.py` 分工
-------------------------------
| 模块 | 做什么 |
|---|---|
| `walkforward.py`（M11） | 对**收益序列**做 purged 分段，检验"换一段还成立吗" |
| 本模块（M61） | 对**参数**做 rolling 选择：每折 IS 选参 → 紧邻 OOS 验证 |

两者互补：M11 看时间稳定性，本模块看**选参是否过拟合**。

为什么必须有这个
----------------
原来 `tune_oos` 允许用全区间挑最优参数再报告收益，等于自己给自己判卷。
Walk-forward 强制：**参数只能用它之前的数据选，选完立刻在没见过的未来用。**

三条防前视硬约束
----------------
1. 选参只用 `[train_start, train_end]`
2. OOS 从训练段结束的**下一个交易日**开始，不重叠
3. OOS 曲线独立起算，不与 IS 混合
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["PARAM_GRID", "load_panel", "split_folds", "run_walkforward",
           "format_report"]

_ROOT = Path(__file__).resolve().parent.parent

PARAM_GRID = {"lookback": [10, 20, 40], "top_n": [5, 10, 20]}


def load_panel(market: str = "cn", min_rows: int = 1500,
               max_codes: int | None = None) -> dict:
    """从 Parquet 库加载面板，只留历史足够长的。"""
    import store
    # 用一次性加载：循环 read_bars 在长表下等于 5000 次全量读
    return store.load_panel(market, min_rows=int(min_rows),
                            max_codes=max_codes)


def align_panel(panel: dict, min_cover: float = 0.9,
                max_missing: float = 0.05) -> tuple[dict, dict]:
    """
    面板覆盖对齐：只保留「共同覆盖窗口内数据完整」的股票。

    为什么必须做
    -----------
    库里混着 1992 起和 2023 起的股票。引擎用**日期并集**做网格，
    早期几乎无人可买 → 组合长期空仓 → 净值平滑，
    会被误读成"低波动/低回撤"（实测年化波动仅 1.3%）。

    做法（分位数法，比"逐日覆盖"稳）
    --------------------------------
    1. 取各票起始日的 90 分位作为窗口起点
    2. 取各票结束日的 10 分位作为窗口终点
    3. 所有票截到 [start, end]，剔除缺失率 > max_missing 的
    """
    if not panel:
        return {}, {"ok": False, "reason": "空面板"}
    n = len(panel)
    firsts = sorted(pd.Timestamp(d["date"].min()) for d in panel.values())
    lasts = sorted(pd.Timestamp(d["date"].max()) for d in panel.values())
    # 起点取「min_cover 分位」（90% 的票此时已上市）
    # 终点取「1-min_cover 分位」（90% 的票此时还活着）
    k_hi = min(int(float(min_cover) * (n - 1)), n - 1)
    k_lo = min(int((1 - float(min_cover)) * (n - 1)), n - 1)
    start = firsts[k_hi] if n > 1 else firsts[0]
    end = lasts[k_lo] if n > 1 else lasts[-1]
    if start >= end:
        return panel, {"ok": False, "reason": "窗口为空"}

    # 窗口内的交易日数（用全样本并集近似）
    all_dates = set()
    for d in panel.values():
        all_dates.update(pd.DatetimeIndex(d["date"]))
    window_days = len([x for x in all_dates if start <= x <= end])
    out = {}
    dropped = 0
    for c, d in panel.items():
        sub = d[(d["date"] >= start) & (d["date"] <= end)]
        if len(sub) < 30:
            dropped += 1
            continue
        # 缺失率必须相对**窗口交易日数**算，而不是相对自身行数
        if window_days > 0 and (1 - len(sub) / window_days) > float(max_missing):
            dropped += 1
            continue
        out[c] = sub.reset_index(drop=True)
    info = {"ok": len(out) >= 3, "start": str(start.date()),
            "end": str(end.date()), "n_in": n, "n_out": len(out),
            "dropped": dropped, "min_cover": min_cover}
    return out, info


def split_folds(dates, n_folds: int | None = 4, test_days: int = 250,
                min_train_days: int = 750,
                start_after: str | None = None) -> list[dict]:
    """
    **固定训练窗滚动**切分（不是"训练段无限变长"）。

    错误做法（已修）：tr_end = T - test*n → 训练段 33 年、OOS 全在最后。
    正确做法：训练窗固定长度，逐折向前滚动，OOS 紧随其后。
    """
    d = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))
    if start_after:
        d = d[d >= pd.Timestamp(start_after)]
    T = len(d)
    tr = int(min_train_days)
    te = int(test_days)
    if T < tr + te:
        return []
    folds = []
    pos = 0
    k = 0
    while pos + tr + te <= T:
        if n_folds is not None and k >= int(n_folds):
            break
        k += 1
        folds.append({
            "fold": k,
            "train": (d[pos], d[pos + tr - 1]),
            "test": (d[pos + tr], d[pos + tr + te - 1]),
            "n_train": tr, "n_test": te,
        })
        pos += te          # 每折向前滚一个测试段
    return folds


def _weights(panel: dict, factor: str, lookback: int, top_n: int,
             a, b, rebalance: int = 20,
             universe: dict | None = None) -> pd.DataFrame:
    import strategy_lab as SL
    sub = {c: d[(d["date"] >= a) & (d["date"] <= b)].reset_index(drop=True)
           for c, d in panel.items()}
    sub = {c: d for c, d in sub.items() if len(d) > int(lookback) + 5}
    if len(sub) < 3:
        return pd.DataFrame()
    sc = SL.build_scores(sub, factor, int(lookback))
    if sc is None or len(sc) == 0:
        return pd.DataFrame()
    rows, idx = [], []
    dts = list(sc.index)
    for i in range(0, len(dts), int(rebalance)):
        d = dts[i]
        row = sc.loc[d].dropna()
        if universe is not None:
            el = universe.get(str(pd.Timestamp(d).date()))
            if el:
                row = row[row.index.isin(el)]
        if len(row) < int(top_n):
            continue
        w = pd.Series(0.0, index=sc.columns)
        w[row.nlargest(int(top_n)).index] = 1.0 / int(top_n)
        rows.append(w); idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx)) if rows \
        else pd.DataFrame()


def _run(panel: dict, w: pd.DataFrame, a, b, max_positions: int = 10) -> dict:
    import backtest_strict as BS
    sub = {c: d[(d["date"] >= a) & (d["date"] <= b)].reset_index(drop=True)
           for c, d in panel.items()}
    sub = {c: d for c, d in sub.items() if len(d) > 30}
    if len(sub) < 3 or w is None or len(w) == 0:
        return {"ok": False}
    try:
        r = BS.StrictBacktest(sub, capital=1e6).run(
            w, max_positions=int(max_positions))
    except Exception:
        return {"ok": False}
    return {"ok": True, "annual": float(r["annual"]),
            "sharpe": float(r["sharpe"]), "max_dd": float(r["max_drawdown"]),
            "nav": r["nav"], "n_delist": int(r.get("n_delist_liquidations", 0))}


def run_walkforward(panel: dict, factor: str = "low_vol", n_folds: int | None = 4,
                    test_days: int = 250, min_train_days: int = 750,
                    rebalance: int = 20, grid: dict | None = None,
                    universe: dict | None = None,
                    start_after: str | None = None,
                    verbose: bool = True) -> dict:
    grid = grid or PARAM_GRID
    dates = sorted(set().union(*[set(d["date"]) for d in panel.values()]))
    folds = split_folds(dates, n_folds, test_days, min_train_days,
                        start_after)
    if not folds:
        return {"ok": False, "reason": "区间不足，无法切折",
                "factor": factor}

    rows, picks = [], []
    for f in folds:
        a, b = f["train"]
        ta, tb = f["test"]
        # ── IS 选参（只用训练段）
        best = None
        for lb in grid["lookback"]:
            for tn in grid["top_n"]:
                w = _weights(panel, factor, lb, tn, a, b, rebalance, universe)
                r = _run(panel, w, a, b, max_positions=int(tn))
                if not r.get("ok"):
                    continue
                sc = r["sharpe"] if np.isfinite(r["sharpe"]) else -9.0
                if best is None or sc > best["score"]:
                    best = {"lookback": lb, "top_n": tn, "score": sc,
                            "is_annual": r["annual"], "is_dd": r["max_dd"]}
        if best is None:
            rows.append({"fold": f["fold"],
                         "train": f"{a.date()}~{b.date()}",
                         "test": f"{ta.date()}~{tb.date()}",
                         "ok": False, "reason": "IS 无有效参数"})
            continue
        picks.append((best["lookback"], best["top_n"]))

        # ── OOS（用 IS 参数，只在测试段）
        w = _weights(panel, factor, best["lookback"], best["top_n"],
                     ta, tb, rebalance, universe)
        r = _run(panel, w, ta, tb, max_positions=int(best["top_n"]))
        rows.append({
            "fold": f["fold"],
            "train": f"{a.date()}~{b.date()}",
            "test": f"{ta.date()}~{tb.date()}",
            "lookback": best["lookback"], "top_n": best["top_n"],
            "is_sharpe": round(float(best["score"]), 3),
            "is_annual": round(float(best["is_annual"]) * 100, 2),
            "ok": bool(r.get("ok")),
            "oos_annual": (round(r["annual"] * 100, 2) if r.get("ok") else None),
            "oos_sharpe": (round(r["sharpe"], 3) if r.get("ok") else None),
            "oos_max_dd": (round(r["max_dd"] * 100, 2) if r.get("ok") else None),
            "nav": r.get("nav") if r.get("ok") else None,
        })
        if verbose:
            print(f"  折{f['fold']} IS[{a.date()}~{b.date()}] "
                  f"Sh={best['score']:.2f} lb={best['lookback']} "
                  f"n={best['top_n']} → OOS {rows[-1]['oos_annual']}%",
                  flush=True)

    # 拼接 OOS
    segs = []
    for r in rows:
        nav = r.get("nav")
        if nav is None or len(nav) < 2:
            continue
        n = nav / nav.iloc[0]
        if segs:
            n = n * segs[-1].iloc[-1]
        segs.append(n)
    oos = pd.concat(segs).sort_index() if segs else pd.Series(dtype=float)

    res = {"ok": len(oos) > 1, "factor": factor, "folds": rows,
           "params_picked": picks, "oos_nav": oos}
    if res["ok"]:
        yrs = max((oos.index[-1] - oos.index[0]).days / 365.25, 1e-9)
        res["oos_annual"] = float((oos.iloc[-1] / oos.iloc[0]) ** (1 / yrs) - 1)
        rr = oos.pct_change(fill_method=None).dropna()
        res["oos_sharpe"] = float(rr.mean() / rr.std() * np.sqrt(252)) \
            if len(rr) and rr.std() > 0 else 0.0
        res["oos_max_dd"] = float((oos / oos.cummax() - 1).min())
        uniq = len(set(picks))
        res["param_stability"] = round(1 - (uniq - 1) / max(len(picks), 1), 3)
        pairs = [(x["is_annual"], x["oos_annual"]) for x in rows
                 if x.get("oos_annual") is not None]
        res["is_oos_decay"] = (round(float(np.mean([i - o for i, o in pairs])), 2)
                               if pairs else None)
    return res


def format_report(res: dict) -> str:
    L = ["=" * 78, "参数级 Walk-forward 报告", "=" * 78]
    if not res.get("ok"):
        L.append(f"  失败：{res.get('reason')}")
        L.append("=" * 78)
        return "\n".join(L)
    L.append(f"  因子 {res['factor']}   折数 {len(res['folds'])}")
    L.append("-" * 78)
    L.append(f"  {'折':<3}{'训练段':<25}{'测试段':<25}{'参数':<11}"
             f"{'IS年化':>8}{'OOS年化':>9}")
    for f in res["folds"]:
        if not f.get("ok"):
            L.append(f"  {f['fold']:<3}{f['train']:<25}{f['test']:<25}失败")
            continue
        L.append(f"  {f['fold']:<3}{f['train']:<25}{f['test']:<25}"
                 f"lb{f['lookback']}/n{f['top_n']:<6}"
                 f"{f['is_annual']:>7.2f}%{f['oos_annual']:>8.2f}%")
    L.append("-" * 78)
    L.append(f"  拼接 OOS  年化 {res['oos_annual']*100:.2f}%   "
             f"夏普 {res['oos_sharpe']:.3f}   "
             f"最大回撤 {res['oos_max_dd']*100:.2f}%")
    L.append(f"  参数稳定度 {res['param_stability']:.2f}   "
             f"IS→OOS 平均衰减 {res.get('is_oos_decay')}pp")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse, sys, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(_ROOT / "backend"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=int, default=600)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--train", type=int, default=750)
    ap.add_argument("--lookbacks", type=str, default="20,40")
    ap.add_argument("--topns", type=str, default="10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--factor", type=str, default="low_vol")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    import numpy as np, pandas as pd, store

    print("加载数据 ...", flush=True)
    panel = store.load_panel("cn", min_rows=max(a.train + a.test, 1000))
    codes = sorted(panel)
    if not a.full:
        rng = np.random.default_rng(a.seed)
        n = min(a.codes, len(codes))
        panel = {c: panel[c] for c in rng.choice(codes, size=n, replace=False)}
    print(f"面板 {len(panel)} 只（库内 {len(codes)} 只）", flush=True)

    grid = {"lookback": [int(x) for x in a.lookbacks.split(",")],
            "top_n": [int(x) for x in a.topns.split(",")]}
    nc = len(grid["lookback"]) * len(grid["top_n"])
    print(f"网格 {nc} 组 x {a.folds} 折 = {nc * a.folds} 次回测", flush=True)

    r = run_walkforward(panel, a.factor, n_folds=a.folds,
                        test_days=a.test, min_train_days=a.train,
                        grid=grid, verbose=True)
    print()
    print(format_report(r))

    if r.get("ok"):
        CL = pd.DataFrame({c: d.set_index("date")["close"]
                           for c, d in panel.items()}).sort_index()
        a0 = str(r["oos_nav"].index[0].date())
        b0 = str(r["oos_nav"].index[-1].date())
        s2 = CL.loc[a0:b0].pct_change(fill_method=None).mean(axis=1).add(1).cumprod().dropna()
        yrs = (s2.index[-1] - s2.index[0]).days / 365.25
        ew = float((s2.iloc[-1] / s2.iloc[0]) ** (1 / yrs) - 1)
        print("")
        print("【必做】跟等权对比：")
        sa = r["oos_annual"]
        print(f"  等权年化 {ew*100:+.2f}%   策略 {sa*100:+.2f}%   超额 {(sa-ew)*100:+.2f}pp")
