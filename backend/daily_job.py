# -*- coding: utf-8 -*-
"""
日频自动体检任务（M5）
====================================================
每天收盘后跑一次：
  1. 增量拉取自选池日线（实测 0.5s/只）
  2. 更新纸盘净值（首次用历史回补，之后逐日追加）
  3. 重算六维体检 + 诊断
  4. 与上次结果比对，列出变化
  5. 落盘 + 通知（默认 dry-run，不真发）

用法
----
    python backend/daily_job.py --once                 # 立即跑一次
    python backend/daily_job.py --once --loop          # 常驻，每天 15:30 跑
    python backend/daily_job.py --init                 # 只做历史回补
    python backend/daily_job.py --once --notify        # 真正发送通知

环境变量（通知）
----------------
QP_WATCHLIST          自选池文件，默认 watchlist.txt
QP_PAPER_NAV          纸盘净值文件，默认 reports/paper_nav.csv
QP_CAPITAL            初始资金，默认 100000
QP_SMTP_HOST / QP_SMTP_PORT / QP_SMTP_USER / QP_SMTP_PASS / QP_MAIL_TO
QP_FEISHU_WEBHOOK     飞书机器人地址
QP_NOTIFY_DRYRUN      默认 1（不真发）；设为 0 才发送
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import smtplib
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_v2 as API

__version__ = "0.1.0"
__all__ = [
    "load_watchlist", "fetch_daily", "refresh_all", "build_weights_csv",
    "init_or_update_nav", "evaluate_nav", "detect_changes", "notify",
    "run_daily", "next_run_time", "run_loop",
]

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
DEFAULT_NAV = ROOT / "reports" / "paper_nav.csv"
DEFAULT_OUT = ROOT / "reports" / "daily"
DEFAULT_CAPITAL = 100_000.0
DEFAULT_HHMM = "15:30"

# 变化判定阈值
SCORE_DELTA = 5.0        # 加权得分变化超过此值 → 提醒
DIM_DELTA = 10.0         # 单维得分变化超过此值 → 提醒
MDD_DELTA = 0.05         # 回撤扩大超过 5pp → 提醒


# ══════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════
def load_watchlist(path: Path | str = DEFAULT_WATCHLIST) -> list[str]:
    """读取自选池：每行一个 6 位代码（支持 # 注释、CSV 首列）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"自选池文件不存在：{p}")
    codes: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if not s:
            continue
        tok = s.split(",")[0].strip().strip('"').strip("'")
        tok = "".join(ch for ch in tok if ch.isdigit())
        if len(tok) == 6 and tok not in codes:
            codes.append(tok)
    if not codes:
        raise ValueError(f"自选池为空或格式不对：{p}")
    return codes


def fetch_daily(code: str, days: int = 400) -> pd.DataFrame:
    """
    拉取单只前复权日线。

    返回 index=date 的 DataFrame，**包含 open/high/low/close/volume/amount**。
    （历史上只返回 close，导致 strategy_runner.build_panel 取不到 OHLC，
      真实调用时所有标的都被判"缺数据"——这是真 bug，已修。）

    内部走 datasource 多源降级（东财 → 新浪），全挂才返回空。
    失败返回空 DataFrame（不抛异常，保持原契约）。
    """
    try:
        import datasource as DS
        got = DS.get_stock_daily(code, days=days)
        df = got.get("df")
        if df is None or len(df) == 0:
            return pd.DataFrame()
        if got.get("downgraded"):
            log.info("降级取数 %s：%s", code, got.get("errors"))
        return df.set_index("date").tail(days)
    except Exception as ex:
        log.warning("拉取失败 %s：%s", code, ex)
        return pd.DataFrame()


def refresh_all(codes: list[str], days: int = 400,
                loader=None) -> dict[str, pd.DataFrame]:
    """批量拉取；返回 {code: DataFrame}，自动跳过失败项。"""
    loader = loader or fetch_daily
    out: dict[str, pd.DataFrame] = {}
    for i, c in enumerate(codes, 1):
        df = loader(c, days)
        if df is not None and len(df):
            out[c] = df
        if i % 10 == 0:
            log.info("  已拉取 %d/%d", i, len(codes))
    return out


def build_weights_csv(codes: list[str], dates) -> str:
    """构造等权持仓 CSV（宽表），供纯度评估使用。"""
    w = 1.0 / max(len(codes), 1)
    head = "date," + ",".join(codes)
    rows = [head]
    for d in dates:
        rows.append(str(pd.Timestamp(d).date()) + "," + ",".join([f"{w:.6f}"] * len(codes)))
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════
# 纸盘净值
# ══════════════════════════════════════════════════════════
def init_or_update_nav(codes: list[str], prices: dict[str, pd.DataFrame],
                       nav_path: Path | str = DEFAULT_NAV,
                       capital: float = DEFAULT_CAPITAL) -> pd.DataFrame:
    """
    更新纸盘净值（等权买入持有）。

    * 首次：用历史共同交易日回补净值曲线
    * 之后：只追加速度缺失的交易日
    """
    nav_path = Path(nav_path)
    if not prices:
        raise ValueError("没有可用价格数据")

    frames = {c: df["close"] for c, df in prices.items()}
    close = pd.DataFrame(frames).sort_index()
    close = close.loc[:, close.notna().all()]          # 只保留全区间都有数据的
    if close.shape[1] == 0:
        raise ValueError("没有标的同时具备完整价格序列")

    rets = close.pct_change(fill_method=None).fillna(0.0)
    port_ret = rets.mean(axis=1)                        # 等权
    nav = capital * (1.0 + port_ret).cumprod()

    nav_path.parent.mkdir(parents=True, exist_ok=True)
    if nav_path.exists():
        old = pd.read_csv(nav_path, parse_dates=["date"]).set_index("date")["nav"]
        merged = pd.concat([old, nav])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = nav

    df = pd.DataFrame({"nav": merged})
    df.index.name = "date"
    df.to_csv(nav_path, float_format="%.6f")
    return merged


# ══════════════════════════════════════════════════════════
# 体检
# ══════════════════════════════════════════════════════════
def evaluate_nav(nav: pd.Series, codes: list[str]) -> dict:
    """基于纸盘净值重算体检（复用 api_v2.run_checkup，保证口径一致）。"""
    rets = nav.pct_change(fill_method=None).dropna()
    if len(rets) < 60:
        raise ValueError(f"净值仅 {len(rets)} 天，不足以体检（至少 60 天）")
    csv = "date,ret\n" + "\n".join(
        f"{d.date()},{v:.8f}" for d, v in rets.items())
    weights_csv = build_weights_csv(codes, rets.index)
    return API.run_checkup(code="", returns_csv=csv, weights_csv=weights_csv,
                           bench_csv=None, use_llm=False)


def detect_changes(prev: dict | None, cur: dict) -> list[dict]:
    """
    比对两次体检结果，返回变化清单（空列表表示无实质变化）。

    重点看：结论、评级、得分、单维得分、回撤、告警。
    """
    if not prev:
        return [{"kind": "first_run", "msg": "首次体检，建立基线"}]

    changes: list[dict] = []
    pr, cr = prev.get("report", {}), cur.get("report", {})

    if pr.get("verdict") != cr.get("verdict"):
        changes.append({"kind": "verdict", "level": "high",
                        "msg": f"结论变化：{pr.get('verdict')} → {cr.get('verdict')}"})
    if pr.get("grade") != cr.get("grade"):
        changes.append({"kind": "grade", "level": "high",
                        "msg": f"评级变化：{pr.get('grade')} → {cr.get('grade')}"})

    try:
        ds = float(cr.get("score", 0)) - float(pr.get("score", 0))
        if abs(ds) >= SCORE_DELTA:
            changes.append({"kind": "score", "level": "mid",
                            "msg": f"加权得分变化 {ds:+.1f}（{pr.get('score')} → {cr.get('score')}）"})
    except (TypeError, ValueError):
        pass

    praw, craw = pr.get("radar") or {}, cr.get("radar") or {}
    for k in praw:
        a, b = praw.get(k), craw.get(k)
        if a is None or b is None:
            continue
        try:
            d = float(b) - float(a)
        except (TypeError, ValueError):
            continue
        if abs(d) >= DIM_DELTA:
            changes.append({"kind": "dim", "level": "mid", "dim": k,
                            "msg": f"{k} 维度变化 {d:+.0f}（{a:.0f} → {b:.0f}）"})

    # 回撤扩大
    nav = cur.get("_nav")
    if nav is not None and len(nav) > 5:
        dd = float((nav / nav.cummax() - 1).min())
        pnav = prev.get("_nav")
        if pnav is not None and len(pnav) > 5:
            pdd = float((pnav / pnav.cummax() - 1).min())
            if dd < pdd - MDD_DELTA:
                changes.append({"kind": "drawdown", "level": "high",
                                "msg": f"最大回撤扩大 {abs(dd - pdd) * 100:.1f}pp"
                                       f"（{pdd * 100:.1f}% → {dd * 100:.1f}%）"})

    # 注意：诊断降级（用模板）是常态而非异常，不进入变化提醒，
    # 否则每天都会报一条噪音。
    return changes


# ══════════════════════════════════════════════════════════
# 通知
# ══════════════════════════════════════════════════════════
def _send_mail(subject: str, body: str) -> str:
    host = os.environ.get("QP_SMTP_HOST")
    to = os.environ.get("QP_MAIL_TO")
    if not (host and to):
        return "skipped_no_config"
    try:
        port = int(os.environ.get("QP_SMTP_PORT", "465"))
        user = os.environ.get("QP_SMTP_USER", "")
        pwd = os.environ.get("QP_SMTP_PASS", "")
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user or to
        msg["To"] = to
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                if user:
                    s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                if user:
                    s.starttls()
                    s.login(user, pwd)
                s.send_message(msg)
        return "sent"
    except Exception as ex:
        log.warning("邮件发送失败：%s", ex)
        return f"failed:{ex}"


def _send_feishu(text: str) -> str:
    url = os.environ.get("QP_FEISHU_WEBHOOK")
    if not url:
        return "skipped_no_config"
    try:
        body = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return "sent" if r.status == 200 else f"failed_http_{r.status}"
    except Exception as ex:
        log.warning("飞书发送失败：%s", ex)
        return f"failed:{ex}"


def notify(result: dict, changes: list[dict],
           dry_run: bool | None = None) -> dict:
    """
    发送通知。默认 dry-run（只打印不发送），需显式关闭才真发。

    返回 {dry_run, mail, feishu, text}
    """
    if dry_run is None:
        dry_run = os.environ.get("QP_NOTIFY_DRYRUN", "1").strip() not in ("0", "false", "False")

    rep = result.get("report", {})
    diag = result.get("diagnosis", {})
    lines = [
        f"【策略体检日报】{datetime.now():%Y-%m-%d %H:%M}",
        f"结论：{rep.get('verdict')}  评级：{rep.get('grade')}  得分：{rep.get('score')}",
        "",
        str(diag.get("summary") or ""),
        "",
        str(diag.get("fatal") or ""),
        "",
        str(diag.get("advice") or ""),
    ]
    if changes:
        lines += ["", "【本次变化】"] + [f"· {c['msg']}" for c in changes]
    text = "\n".join(lines)

    out = {"dry_run": bool(dry_run), "text": text,
           "mail": "dry_run", "feishu": "dry_run"}
    if dry_run:
        log.info("[dry-run] 通知内容：\n%s", text)
        return out
    out["mail"] = _send_mail(f"策略体检 {datetime.now():%m-%d} "
                             f"{rep.get('grade')}", text)
    out["feishu"] = _send_feishu(text)
    return out


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def run_daily(watchlist: Path | str = DEFAULT_WATCHLIST,
              nav_path: Path | str = DEFAULT_NAV,
              out_dir: Path | str = DEFAULT_OUT,
              capital: float = DEFAULT_CAPITAL,
              days: int = 400,
              loader=None,
              do_notify: bool = False) -> dict:
    """
    执行一次日更体检，返回结果 dict。

    loader 可注入（测试用），签名 loader(code, days) -> DataFrame。
    """
    t0 = time.perf_counter()
    watchlist, nav_path, out_dir = Path(watchlist), Path(nav_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = load_watchlist(watchlist)
    log.info("自选池 %d 只：%s", len(codes), ",".join(codes[:8]) + ("…" if len(codes) > 8 else ""))

    prices = refresh_all(codes, days, loader)
    if not prices:
        raise RuntimeError("全部标的拉取失败，请检查网络或代码是否有效")
    log.info("成功拉取 %d/%d 只", len(prices), len(codes))

    nav = init_or_update_nav(list(prices), prices, nav_path, capital)
    log.info("净值序列 %d 天（%s ~ %s）", len(nav),
             nav.index.min().date(), nav.index.max().date())

    result = evaluate_nav(nav, list(prices))
    result["_nav"] = nav

    prev_path = out_dir / "latest.json"
    prev = None
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            prev = None
    changes = detect_changes(prev, result)

    stamp = datetime.now().strftime("%Y-%m-%d")
    record = {
        "date": stamp,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "watchlist": list(prices),
        "nav_days": int(len(nav)),
        "nav_latest": float(nav.iloc[-1]),
        "report": result["report"],
        "diagnosis": result["diagnosis"],
        "warnings": result.get("warnings", []),
        "changes": changes,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    (out_dir / f"{stamp}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # latest 只存比对所需字段（不含净值数组）
    slim = {k: v for k, v in record.items()}
    (out_dir / "latest.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    notif = notify(result, changes) if do_notify else {"dry_run": True, "skipped": True}
    record["notify"] = notif
    if not do_notify:
        log.info("通知未启用（加 --notify 开启）")

    log.info("完成：%s / %s / %.0f 分 | 变化 %d 条 | %.1fs",
             record["report"].get("verdict"), record["report"].get("grade"),
             record["report"].get("score", 0), len(changes),
             (time.perf_counter() - t0))
    for c in changes:
        log.info("  · %s", c["msg"])
    return record


def next_run_time(now: datetime | None = None, hhmm: str = DEFAULT_HHMM) -> datetime:
    """计算下一个运行时刻（当天已过则顺延到明天）。"""
    now = now or datetime.now()
    hh, mm = (int(x) for x in hhmm.split(":"))
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def run_loop(hhmm: str = DEFAULT_HHMM, **kwargs) -> None:
    """常驻模式：每天到点跑一次。"""
    log.info("常驻模式启动，每天 %s 执行一次", hhmm)
    while True:
        nxt = next_run_time(hhmm=hhmm)
        wait = (nxt - datetime.now()).total_seconds()
        log.info("下次运行：%s（等待 %.1f 小时）", nxt.strftime("%Y-%m-%d %H:%M"), wait / 3600)
        time.sleep(max(wait, 1))
        try:
            run_daily(**kwargs)
        except Exception:
            log.exception("本次日更失败，将于明日重试")


# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="日频自动体检")
    ap.add_argument("--watchlist", default=str(DEFAULT_WATCHLIST))
    ap.add_argument("--nav", default=str(DEFAULT_NAV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--once", action="store_true", help="立即执行一次")
    ap.add_argument("--loop", action="store_true", help="常驻，每天定时执行")
    ap.add_argument("--at", default=DEFAULT_HHMM, help="执行时刻 HH:MM")
    ap.add_argument("--notify", action="store_true", help="真正发送通知")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    kw = dict(watchlist=a.watchlist, nav_path=a.nav, out_dir=a.out,
              capital=a.capital, days=a.days, do_notify=a.notify)
    if a.loop:
        run_loop(a.at, **kw)
    else:
        run_daily(**kw)


if __name__ == "__main__":
    main()
