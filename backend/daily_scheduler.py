# -*- coding: utf-8 -*-
"""
每日任务调度器（M48）—— 基于 `schedule` 库
====================================================
⚠ 与旧 `scheduler.py` 的区别
---------------------------
* `scheduler.py`（旧）：读 `schedule.json` 配置，**用 subprocess 跑一堆命令**
* 本模块（新）：**在进程内**用 `schedule` 库跑两个具体任务，产出文件

两者可共存，别混淆。本模块不依赖 `schedule.json`。

任务表
------
| 时间 | 任务 | 产出 |
|---|---|---|
| **15:30** | 数据清洗（校验缓存缺口/空值/零成交/陈旧） | `reports/clean_YYYYMMDD.md` |
| **20:00** | 生成**次日**交易清单 | `reports/trade_list_YYYYMMDD.csv` + `.md` |

为什么是这两个时间
------------------
* **15:30**：A 股 15:00 收盘，数据源一般 15:20 后更新完毕
* **20:00**：复盘与次日计划都在晚上；也给数据源留补数时间

设计要点
--------
1. **幂等**：同一天重复跑只覆盖当天报告，不产生重复动作
2. **非交易日**：数据未更新时清单会标出信号日，不会假装有新数据
3. **失败不中断**：单个任务异常只记日志，调度器继续
4. **可测试**：任务都是纯函数，`trigger("clean")` 可立即执行

⚠ 边界
------
* 只生成清单，**不下单**
* 清单里的因子在 2022–2024 只有弱正 IC，**是候选池不是买入信号**
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"
__all__ = ["CLEAN_TIME", "TRADE_TIME", "clean_data", "make_trade_list",
           "trigger", "build_scheduler", "run_forever", "setup_logging"]

_ROOT = Path(__file__).resolve().parent.parent
REPORTS = _ROOT / "reports"
LOGS = _ROOT / "logs"
CACHE = _ROOT / ".cache"

CLEAN_TIME = "15:30"
TRADE_TIME = "20:00"

log = logging.getLogger("daily_scheduler")


def setup_logging(level: int = logging.INFO) -> Path:
    """日志同时写控制台与 logs/daily_scheduler.log。"""
    LOGS.mkdir(parents=True, exist_ok=True)
    fp = LOGS / "daily_scheduler.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(fp, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return fp


# ══════════════════════════════════════════════════════════
def _scan_cache() -> list[dict]:
    """扫描缓存日线文件，逐个体检。"""
    rows = []
    for sub in ("strict", "cb", "index", "us"):
        d = CACHE / sub
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(fp)
            except Exception:
                rows.append({"file": str(fp.relative_to(CACHE)),
                             "error": "读取失败"})
                continue
            n = len(df)
            if n == 0:
                rows.append({"file": str(fp.relative_to(CACHE)),
                             "error": "空文件"})
                continue
            issues = []
            dcol = next((c for c in ("date", "日期") if c in df.columns), None)
            last_date = None
            gap_max = 0
            if dcol:
                try:
                    ds = pd.to_datetime(df[dcol], errors="coerce").dropna()
                    if len(ds) > 1:
                        ds = ds.sort_values()
                        gaps = ds.diff().dt.days.dropna()
                        gap_max = int(gaps.max()) if len(gaps) else 0
                        if gap_max > 15:
                            issues.append(f"最长缺口 {gap_max} 天")
                        last_date = ds.iloc[-1]
                        stale = int((pd.Timestamp.now().normalize()
                                     - last_date.normalize()).days)
                        if stale > 7:
                            issues.append(f"末日期滞后 {stale} 天")
                except Exception:
                    pass
            for c in ("open", "high", "low", "close"):
                if c in df.columns:
                    bad = int(pd.to_numeric(df[c], errors="coerce").isna().sum())
                    if bad:
                        issues.append(f"{c} 缺失 {bad}")
                    break
            zero_vol = None
            if "volume" in df.columns:
                v = pd.to_numeric(df["volume"], errors="coerce")
                zero_vol = int((v.fillna(0) == 0).sum())
                if zero_vol > n * 0.1:
                    issues.append(f"零成交 {zero_vol}/{n}")
            rows.append({"file": str(fp.relative_to(CACHE)), "n": n,
                         "last": (str(last_date.date())
                                  if last_date is not None else None),
                         "gap_max": gap_max, "zero_vol": zero_vol,
                         "issues": issues, "ok": len(issues) == 0})
    return rows


def clean_data(write_report: bool = True, notify: bool = False) -> dict:
    """
    15:30 任务：扫描缓存数据，列出问题文件并写报告。

    真正的"清洗"动作只做**识别与标记**，不擅自删改数据——
    数据是资产，误删比留着脏数据更糟。
    """
    t0 = time.time()
    log.info("开始数据清洗")
    try:
        rows = _scan_cache()
    except Exception as ex:
        log.exception("扫描失败")
        return {"ok": False, "job": "clean", "error": str(ex)}

    bad = [r for r in rows if not r.get("ok", True)]
    stale = [r for r in rows
             if r.get("last") and int((pd.Timestamp.now().normalize()
                                       - pd.Timestamp(r["last"])).days) > 7]

    today = datetime.now().strftime("%Y%m%d")
    path = None
    if write_report and rows:
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = REPORTS / f"clean_{today}.md"
        L = [f"# 数据清洗报告 {today}", "",
             f"- 扫描文件：{len(rows)}",
             f"- 有问题：{len(bad)}",
             f"- 数据陈旧（>7 天）：{len(stale)}", "",
             "| 文件 | 行数 | 末日期 | 最长缺口 | 零成交 | 状态 |",
             "|---|---|---|---|---|---|"]
        for r in rows[:200]:
            if r.get("error"):
                L.append(f"| {r['file']} | — | — | — | — | ❌ {r['error']} |")
                continue
            status = "✅" if r.get("ok") else "⚠ " + "；".join(r["issues"][:2])
            L.append(f"| {r['file']} | {r.get('n')} | {r.get('last')} | "
                     f"{r.get('gap_max')} | {r.get('zero_vol')} | {status} |")
        path.write_text("\n".join(L), encoding="utf-8")
        log.info(f"清洗报告已写入 {path}")

    el = round(time.time() - t0, 2)
    log.info(f"数据清洗完成：{len(rows)} 个文件，{len(bad)} 个有问题，{el}s")

    mail_out = None
    if notify and (len(bad) or len(stale)):
        # 异常才发信：正常的清洗结果不值得占用你的收件箱
        try:
            import mailer as ML
            rep = ML.render_daily(
                trade=None,
                position=None,
                clean={"n_bad": len(bad), "n_stale": len(stale)})
            rep["subject"] = f"[QuantPilot] 数据异常告警 {today}（{len(bad)} 个文件）"
            mail_out = ML.send_mail(rep["subject"], rep["html"], rep["text"])
            log.info(f"异常告警邮件：{'已发送' if mail_out.get('ok') else mail_out.get('error')}")
        except Exception as ex:
            log.warning(f"发送告警邮件失败：{ex}")
            mail_out = {"ok": False, "error": str(ex)}

    return {"ok": True, "job": "clean", "n_files": len(rows),
            "n_bad": len(bad), "n_stale": len(stale),
            "report": str(path) if path else None, "elapsed": el,
            "mail": mail_out}


# ══════════════════════════════════════════════════════════
def make_trade_list(n: int = 60, top_n: int = 10, factor: str = "low_vol",
                    window: int = 20, cap_low: float = 0.30,
                    cap_high: float = 0.80, write: bool = True,
                    notify: bool = False) -> dict:
    """
    20:00 任务：用今日收盘数据生成**次日开盘**的交易清单。

    流程：沪深300 趋势 → 仓位上限 → 因子选股 → 按上限分配权重 → 落盘
    """
    t0 = time.time()
    log.info(f"开始生成交易清单（因子 {factor}）")
    try:
        import backtest_strict as BS
        import position_manager as PM
        import strategy_lab as SL

        close = PM.load_index("sh000300")
        caps = PM.position_cap(close, window=window, cap_low=cap_low,
                               cap_high=cap_high)
        sm = PM.cap_summary(caps)
        cap_now = float(sm["current_cap"])

        codes = BS.default_universe(n)
        got = BS.fetch_panel(codes, "2019-01-02",
                             datetime.now().strftime("%Y-%m-%d"))
        panel = got["panel"]
        if len(panel) < 10:
            raise RuntimeError(f"股票池不足（{len(panel)} 只）")

        lb = 250 if factor == "lt_reversal" else window
        sc = SL.build_scores(panel, factor, lb)
        last = sc.dropna(how="all").index[-1]
        row = sc.loc[last].dropna()
        if len(row) < top_n:
            raise RuntimeError(f"可用标的不足（{len(row)} < {top_n}）")

        picks = row.nlargest(int(top_n))
        w_each = cap_now / len(picks) if len(picks) else 0.0

        ref = {}
        for c, df in panel.items():
            try:
                ref[c] = round(float(df["close"].iloc[-1]), 3)
            except Exception:
                ref[c] = None

        rows = [{"rank": i, "code": code, "score": round(float(s), 6),
                 "target_weight": round(w_each, 4), "ref_close": ref.get(code),
                 "factor": factor, "cap": round(cap_now, 4),
                 "signal_date": str(pd.Timestamp(last).date())}
                for i, (code, s) in enumerate(picks.items(), 1)]
        df = pd.DataFrame(rows)

        today = datetime.now().strftime("%Y%m%d")
        csv_path = md_path = None
        if write and len(df):
            REPORTS.mkdir(parents=True, exist_ok=True)
            csv_path = REPORTS / f"trade_list_{today}.csv"
            md_path = REPORTS / f"trade_list_{today}.md"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            L = [f"# 次日交易清单 {today}", "",
                 f"- 信号日：{pd.Timestamp(last).date()}（收盘数据）",
                 "- 执行：**次日开盘**",
                 f"- 仓位上限：**{cap_now*100:.0f}%**"
                 f"（沪深300 {sm['current_state']}，{window} 日均线）",
                 f"- 因子：{factor}，持仓 {top_n} 只等权", "",
                 "| # | 代码 | 目标权重 | 参考收盘 | 因子分 |",
                 "|---|---|---|---|---|"]
            for _, r in df.iterrows():
                L.append(f"| {r['rank']} | {r['code']} | "
                         f"{r['target_weight']*100:.1f}% | "
                         f"{r['ref_close']} | {r['score']} |")
            L += ["", "> ⚠ 候选池 ≠ 买入信号。因子在 2022-2024 只有弱正 IC，",
                  "> 请自行复核后再决定是否执行。"]
            md_path.write_text("\n".join(L), encoding="utf-8")
            log.info(f"交易清单已写入 {csv_path}")

        el = round(time.time() - t0, 2)
        log.info(f"交易清单完成：{len(df)} 只，上限 {cap_now*100:.0f}%，{el}s")

        mail_out = None
        if notify:
            try:
                import mailer as ML
                last_row = {  # 给邮件用的仓位快照
                    "current_state": sm["current_state"],
                    "current_cap": cap_now,
                    "close": None, "ma": None, "diff_pct": None,
                }
                try:
                    tail = caps.iloc[-1]
                    last_row.update({
                        "close": round(float(tail["close"]), 2),
                        "ma": (round(float(tail["ma"]), 2)
                               if pd.notna(tail["ma"]) else None),
                        "diff_pct": (round(float(tail["diff"]) * 100, 2)
                                     if pd.notna(tail["diff"]) else None),
                    })
                except Exception:
                    pass
                mail_out = ML.send_daily_report(
                    trade={"csv": str(csv_path) if csv_path else None},
                    position=last_row,
                    trade_rows=df.to_dict("records"))
                log.info(f"每日报告邮件：{'已发送' if mail_out.get('ok') else mail_out.get('error')}")
            except Exception as ex:
                log.warning(f"发送邮件失败：{ex}")
                mail_out = {"ok": False, "error": str(ex)}

        return {"ok": True, "job": "trade", "n": len(df), "cap": cap_now,
                "state": sm["current_state"], "factor": factor,
                "csv": str(csv_path) if csv_path else None,
                "md": str(md_path) if md_path else None,
                "signal_date": str(pd.Timestamp(last).date()), "elapsed": el,
                "mail": mail_out}
    except Exception as ex:
        log.exception("生成交易清单失败")
        return {"ok": False, "job": "trade", "error": str(ex)}


# ══════════════════════════════════════════════════════════
def trigger(job: str, **kw) -> dict:
    """立即执行一次指定任务（手动补跑 / 测试用）。"""
    if job == "clean":
        return clean_data(**kw)
    if job == "trade":
        return make_trade_list(**kw)
    return {"ok": False, "job": job, "error": f"未知任务 {job!r}"}


def build_scheduler(clean_at: str = CLEAN_TIME, trade_at: str = TRADE_TIME,
                    clean_kw: dict | None = None,
                    trade_kw: dict | None = None):
    """构造 schedule 对象（不启动循环）。"""
    import schedule
    schedule.clear()
    schedule.every().day.at(clean_at).do(clean_data, **(clean_kw or {}))
    schedule.every().day.at(trade_at).do(make_trade_list, **(trade_kw or {}))
    return schedule


def run_forever(clean_at: str = CLEAN_TIME, trade_at: str = TRADE_TIME,
                clean_kw: dict | None = None, trade_kw: dict | None = None,
                tick: float = 1.0, notify: bool = False) -> None:
    """常驻循环。Ctrl+C 退出。"""
    clean_kw = dict(clean_kw or {})
    trade_kw = dict(trade_kw or {})
    if notify:
        clean_kw.setdefault("notify", True)
        trade_kw.setdefault("notify", True)
    sch = build_scheduler(clean_at, trade_at, clean_kw, trade_kw)
    log.info(f"调度器启动：数据清洗 {clean_at}，交易清单 {trade_at}")
    log.info("按 Ctrl+C 退出")
    try:
        while True:
            sch.run_pending()
            time.sleep(float(tick))
    except KeyboardInterrupt:
        log.info("收到中断，调度器退出")


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="QuantPilot 每日任务（schedule 库）")
    ap.add_argument("--demo", action="store_true", help="立即各跑一次")
    ap.add_argument("--job", choices=["clean", "trade"], default=None)
    ap.add_argument("--no-write", action="store_true", help="不落盘")
    ap.add_argument("--clean-at", default=CLEAN_TIME)
    ap.add_argument("--trade-at", default=TRADE_TIME)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--list", action="store_true", help="只看任务表")
    ap.add_argument("--mail", action="store_true", help="跑完发邮件")
    args = ap.parse_args()

    setup_logging()
    write = not args.no_write

    if args.list:
        print(f"  15:30 自动时间 {args.clean_at}  数据清洗")
        print(f"  20:00 自动时间 {args.trade_at}  生成次日交易清单")
        sys.exit(0)

    if args.job:
        kw = {"write": write}
        if args.mail:
            kw["notify"] = True
        r = trigger(args.job, **kw)
        print(r)
        sys.exit(0 if r.get("ok") else 1)

    if args.demo:
        print("\n=== 立即执行：数据清洗 ===")
        print(clean_data(write_report=write, notify=args.mail))
        print("\n=== 立即执行：交易清单 ===")
        print(make_trade_list(top_n=args.top, write=write, notify=args.mail))
        sys.exit(0)

    run_forever(args.clean_at, args.trade_at, notify=args.mail)
