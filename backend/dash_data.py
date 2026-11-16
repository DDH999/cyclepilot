# -*- coding: utf-8 -*-
"""
看板数据层（M51）
====================================================
给 Streamlit 页面提供数据，**不做任何计算之外的逻辑**。

四个数据源
----------
| 函数 | 来源 | 说明 |
|---|---|---|
| `latest_trade_list()` | `reports/trade_list_*.csv` | 最新一份清单 |
| `position_now()` | `position_manager` | 实时仓位状态 |
| `nav_history()` | `backtest_strict` + 缓存 | 策略净值 vs 沪深300，落盘缓存 |
| `clean_reports()` | `reports/clean_*.md` | 数据体检报告列表 |

缓存策略
--------
`nav_history()` 结果写到 `reports/nav_history.csv`，
下次直接读（页面秒开）；加 `force=True` 才重算。
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["REPORTS", "latest_trade_list", "position_now", "nav_history",
           "clean_reports", "paper_nav", "overview"]

_ROOT = Path(__file__).resolve().parent.parent
REPORTS = _ROOT / "reports"
NAV_CACHE = REPORTS / "nav_history.csv"

# 净值曲线默认区间：与本地缓存的面板一致，避免现场联网
NAV_START = "2019-01-02"
NAV_END = "2024-12-31"


def _read_csv_str(fp: Path, **kw) -> pd.DataFrame:
    """按字符串读 code 列，防止 000625 变成 625。"""
    df = pd.read_csv(fp, **kw)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def latest_trade_list() -> dict:
    """最新一份交易清单（按文件名日期倒序）。"""
    if not REPORTS.exists():
        return {"ok": False, "reason": "reports 目录不存在"}
    files = sorted(REPORTS.glob("trade_list_*.csv"), reverse=True)
    if not files:
        return {"ok": False, "reason": "还没有交易清单，先跑 "
                                       "`python backend/daily_scheduler.py --job trade`"}
    fp = files[0]
    try:
        df = _read_csv_str(fp)
    except Exception as ex:
        return {"ok": False, "reason": f"读取失败：{ex}"}
    date = fp.stem.replace("trade_list_", "")
    md = fp.with_suffix(".md")
    return {"ok": True, "path": str(fp), "date": date, "rows": len(df),
            "df": df,
            "md": (md.read_text(encoding="utf-8") if md.exists() else ""),
            "mtime": datetime.fromtimestamp(fp.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M")}


def position_now(band: float = 0.02, window: int = 20,
                 cap_low: float = 0.30, cap_high: float = 0.80) -> dict:
    """实时仓位状态。"""
    try:
        import position_manager as PM
        close = PM.load_index("sh000300")
        caps = PM.position_cap(close, window=window, cap_low=cap_low,
                               cap_high=cap_high, band=band)
        sm = PM.cap_summary(caps)
        tail = caps.iloc[-1]
        return {"ok": True, "summary": sm,
                "close": round(float(tail["close"]), 2),
                "ma": (round(float(tail["ma"]), 2)
                       if pd.notna(tail["ma"]) else None),
                "diff_pct": (round(float(tail["diff"]) * 100, 2)
                             if pd.notna(tail["diff"]) else None),
                "cap": round(float(tail["cap"]), 4),
                "band": band, "window": window,
                "cap_low": cap_low, "cap_high": cap_high}
    except Exception as ex:
        return {"ok": False, "reason": str(ex)[:200]}


def nav_history(factor: str = "low_vol", top_n: int = 10,
                start: str = NAV_START, end: str = NAV_END,
                use_cap: bool = True, band: float = 0.02,
                force: bool = False) -> dict:
    """
    策略净值 vs 沪深300（含仓位上限）。

    结果缓存到 `reports/nav_history.csv`。
    """
    # v2：基准口径从「沪深300价格指数」改为「同池等权」，旧缓存必须失效
    tag = f"v2|{factor}|{top_n}|{start}|{end}|{int(use_cap)}|{band}"
    meta_fp = REPORTS / "nav_history.meta.txt"
    if not force and NAV_CACHE.exists():
        try:
            if meta_fp.exists() and meta_fp.read_text(encoding="utf-8").strip() == tag:
                df = pd.read_csv(NAV_CACHE, parse_dates=["date"])
                if len(df) > 50:
                    return {"ok": True, "df": df, "cached": True,
                            "key": tag, "path": str(NAV_CACHE)}
        except Exception:
            pass

    t0 = time.time()
    try:
        import backtest_strict as BS
        import position_manager as PM
        import strategy_lab as SL

        got = BS.fetch_panel(BS.default_universe(60), start, end)
        panel = got["panel"]
        if len(panel) < 10:
            return {"ok": False, "reason": f"股票池不足（{len(panel)} 只）"}

        sc = SL.build_scores(panel, factor, 20)
        w = SL.factor_weights(sc, factor, top_n, 20)
        if w.empty:
            return {"ok": False, "reason": "因子权重为空"}

        cap_series = None
        if use_cap:
            close = PM.load_index("sh000300")
            caps = PM.position_cap(close, window=20, cap_low=0.30,
                                   cap_high=0.80, band=band)
            cap_series = caps["cap"]

        bt = BS.StrictBacktest(panel, capital=1_000_000).run(
            w, max_positions=top_n, cap=cap_series)
        nav = bt["nav"]

        # ── 基准：同池等权买入持有（与策略同票池、同前复权口径）
        # 原实现用沪深300价格指数，与策略的前复权收益口径不一致
        # （策略含分红再投，指数不含），且"精选 58 只成分股 vs 宽基指数"
        # 本身自带 17.69pp/年 的虚假超额。这里统一口径，让超额可比。
        CL = pd.DataFrame({c: d.set_index("date")["close"]
                           for c, d in panel.items()}).sort_index()
        ew = CL.pct_change(fill_method=None).mean(axis=1).add(1.0).cumprod()
        ew = ew.reindex(nav.index).ffill()
        bench_ew = ew / float(ew.dropna().iloc[0])

        # 沪深300 仅作参考线，不参与超额计算
        close = PM.load_index("sh000300")
        bench_hs = close.reindex(nav.index).ffill()
        bench_hs = bench_hs / float(bench_hs.dropna().iloc[0])

        df = pd.DataFrame({
            "date": nav.index,
            "nav": (nav / float(nav.iloc[0])).to_numpy(),
            "bench": bench_ew.to_numpy(),
            "bench_hs300": bench_hs.to_numpy(),
        })
        if cap_series is not None:
            cp = cap_series.reindex(nav.index).ffill().to_numpy()
            df["cap"] = cp
        df = df.dropna(subset=["nav"])

        REPORTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(NAV_CACHE, index=False)
        meta_fp.write_text(tag, encoding="utf-8")
        def _ann(series) -> float | None:
            s = pd.Series(np.asarray(series), index=nav.index).dropna()
            if len(s) < 2:
                return None
            yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
            return round(float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), 4)

        bench_ann = _ann(bench_ew)
        return {"ok": True, "df": df, "cached": False, "key": tag,
                "path": str(NAV_CACHE),
                "elapsed": round(time.time() - t0, 1),
                "metrics": {
                    "annual": bt["annual"], "sharpe": bt["sharpe"],
                    "max_drawdown": bt["max_drawdown"],
                    "final_nav": bt["final_nav"],
                    "bench_ew_annual": bench_ann,
                    "excess_vs_ew_annual": (
                        round(float(bt["annual"]) - bench_ann, 4)
                        if bench_ann is not None else None),
                    "bias_note": ("票池=当前沪深300成分，仍含幸存者偏差；"
                                  "基准已改为同池等权，超额不再混入口径差")}}
    except Exception as ex:
        return {"ok": False, "reason": f"{type(ex).__name__}: {str(ex)[:200]}"}


def clean_reports(n: int = 5) -> list[dict]:
    """最近 N 份数据清洗报告。"""
    if not REPORTS.exists():
        return []
    files = sorted(REPORTS.glob("clean_*.md"), reverse=True)[:int(n)]
    out = []
    for fp in files:
        try:
            out.append({"date": fp.stem.replace("clean_", ""),
                        "path": str(fp),
                        "text": fp.read_text(encoding="utf-8")})
        except Exception:
            continue
    return out


def paper_nav() -> dict:
    """纸盘净值（若有）。"""
    fp = REPORTS / "paper_nav.csv"
    if not fp.exists():
        return {"ok": False, "reason": "还没有纸盘净值"}
    try:
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        if "nav" not in df.columns or len(df) < 2:
            return {"ok": False, "reason": "纸盘净值格式不符"}
        return {"ok": True, "df": df, "path": str(fp),
                "start": str(df["date"].iloc[0].date()),
                "end": str(df["date"].iloc[-1].date())}
    except Exception as ex:
        return {"ok": False, "reason": str(ex)[:150]}


def overview() -> dict:
    """页面顶部要的全部数据。"""
    return {"trade": latest_trade_list(),
            "position": position_now(),
            "clean": clean_reports(3),
            "paper": paper_nav(),
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== 交易清单 ===")
    t = latest_trade_list()
    if t["ok"]:
        print(f"  {t['date']}  {t['rows']} 只  {t['mtime']}")
        print(t["df"].head(3).to_string(index=False))
    else:
        print("  ", t["reason"])

    print("\n=== 仓位 ===")
    p = position_now()
    if p["ok"]:
        print(f"  {p['summary']['current_state']} → "
              f"{p['cap']*100:.0f}%  沪深300 {p['close']}  均线 {p['ma']}")
    else:
        print("  ", p["reason"])

    print("\n=== 净值曲线（首次约 10 秒）===")
    n = nav_history()
    if n["ok"]:
        df = n["df"]
        print(f"  {len(df)} 天  {df['date'].iloc[0].date()} ~ "
              f"{df['date'].iloc[-1].date()}  缓存={n['cached']}")
        print(f"  策略终值 {df['nav'].iloc[-1]:.3f}  "
              f"基准终值 {df['bench'].iloc[-1]:.3f}")
    else:
        print("  ", n["reason"])

    print("\n=== 清洗报告 ===")
    for c in clean_reports(3):
        print(f"  {c['date']}  {len(c['text'])} 字符")

    print("\nOK v" + __version__)
