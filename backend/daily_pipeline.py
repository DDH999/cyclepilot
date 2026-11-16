# -*- coding: utf-8 -*-
"""
基础设施与运维层（M69）：日更流水线
====================================================
把 1→2→3→5 层串成一条自动运行的流水线：

```
ingest  →  health  →  research  →  report  →  notify
(补数据)   (体检)     (因子+WF)    (落盘)     (邮件)
```

与已有 `pipeline.py` 的区别
--------------------------
* `pipeline.py`：**研究流水线**（特征→池化模型→purged WF→PBO/DSR→图）
* 本模块：**运维流水线**（每天定时补数据、体检、跑因子、出日报、邮件）

设计原则
--------
1. **失败隔离**：某步失败只记录，不中断后续
2. **幂等**：由 `opsjobs` 的周期判断保证同一周期只跑一次
3. **可 dry-run**：只打印计划，便于验证
4. **有界**：入库有 limit、研究有 max_codes，不会跑飞
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

import opshealth as OH
import opsjobs as OJ
import opslog as OL

__version__ = "0.1.0"
__all__ = ["STEPS", "run_daily", "register_jobs", "step_ingest",
           "step_health", "step_research", "step_report", "step_notify"]

REPORT_DIR = _ROOT / "reports"
INGEST_LIMIT = 300
MAX_CODES = 200

CTX: dict = {}


def _mark(step: str, ok: bool, info=None) -> None:
    CTX[step] = {"ok": bool(ok), "info": info,
                 "ts": time.strftime("%H:%M:%S")}


# ══════════════════════════════════════════════════════════
def step_ingest(limit: int = INGEST_LIMIT) -> dict:
    """第 1 层：补数据（有界）。limit<=0 表示本次跳过（不是"无限制"）。"""
    if int(limit) <= 0:
        _mark("ingest", True, "跳过（limit<=0）")
        return {"ok": True, "skipped": True, "done": 0, "failed": 0}
    try:
        import bulk_ingest as BI
        r = BI.run_threaded("cn", limit=int(limit), workers=3, sleep_s=0.05)
    except Exception as ex:
        _mark("ingest", False, f"{type(ex).__name__}: {ex}")
        return {"ok": False, "error": str(ex)}
    info = {"done": r.get("done", 0), "failed": r.get("failed", 0)}
    _mark("ingest", bool(r.get("ok")), info)
    return {"ok": bool(r.get("ok")), **info}


def step_health(notify: bool = False) -> dict:
    """第 5 层：体检 + 告警。"""
    res = OH.run_checks()
    sent = None
    if notify and not res["ok"]:
        fn = None
        try:
            import mailer
            cfg = mailer.load_config()
            fn = (lambda s, b: mailer.send_mail(cfg, s, b, html=False))
        except Exception:
            fn = None
        sent = OH.alert(res, fn).get("sent")
    _mark("health", res["ok"], {"n_fail": res["n_fail"], "alert": sent,
                                "report": res})
    return {"ok": res["ok"], "n_fail": res["n_fail"], "report": res,
            "alert_sent": sent}


def step_research(max_codes: int = MAX_CODES, do_wf: bool = True) -> dict:
    """第 2 层：因子 IC + 参数级 Walk-forward。"""
    import param_walkforward as PW
    import strategy_lab as SL
    import store

    codes = store.list_codes("cn")
    if len(codes) < 20:
        _mark("research", False, f"样本不足（{len(codes)} 只）")
        return {"ok": False, "reason": f"样本不足 {len(codes)}"}

    raw = {}
    for c in codes[: int(max_codes)]:
        d = store.read_bars("cn", c)
        if len(d) >= 100:
            raw[c] = d
    panel, info = PW.align_panel(raw, min_cover=0.8, max_missing=0.05)
    if not info.get("ok") or len(panel) < 10:
        _mark("research", False, f"对齐失败 {info}")
        return {"ok": False, "reason": f"对齐失败 {info}"}

    out = {"ok": True, "n_codes": len(panel),
           "window": f"{info['start']}~{info['end']}", "ic_mean": None}
    try:
        sc = SL.build_scores(panel, "low_vol", 20)
        ic = SL.ic_series(panel, sc, horizon=5)
        if isinstance(ic, pd.DataFrame) and "ic" in ic.columns:
            out["ic_mean"] = round(float(ic["ic"].mean()), 4)
    except Exception as ex:
        out["ic_error"] = f"{type(ex).__name__}: {ex}"

    if do_wf:
        try:
            wf = PW.run_walkforward(panel, "low_vol", n_folds=3, test_days=120,
                                    min_train_days=250, verbose=False)
            if wf.get("ok"):
                out["wf_oos_annual"] = round(wf["oos_annual"] * 100, 2)
                out["wf_oos_sharpe"] = round(wf["oos_sharpe"], 3)
                out["wf_folds"] = len(wf["folds"])
            else:
                out["wf_error"] = wf.get("reason")
        except Exception as ex:
            out["wf_error"] = f"{type(ex).__name__}: {ex}"

    _mark("research", True, out)
    return out


def step_report() -> dict:
    """落盘日报。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fp = REPORT_DIR / f"daily_{time.strftime('%Y%m%d')}.md"
    L = [f"# QuantPilot 日更 {time.strftime('%Y-%m-%d %H:%M')}", "",
         "## 模块状态", "", "| 步骤 | 状态 | 摘要 |", "|---|---|---|"]
    for k, v in CTX.items():
        L.append(f"| {k} | {'OK' if v['ok'] else 'FAIL'} | "
                 f"{str(v.get('info'))[:120]} |")
    L.append("")
    h = CTX.get("health") or {}
    if h.get("info"):
        L += ["## 体检", "", "```",
              OH.format_report(h["info"]["report"]), "```", ""]
    r = (CTX.get("research") or {}).get("info") or {}
    if r:
        L += ["## 研究", "",
              f"- 票池 {r.get('n_codes')} 只，窗口 {r.get('window')}",
              f"- 因子 IC（5日）{r.get('ic_mean')}"]
        if "wf_oos_annual" in r:
            L.append(f"- Walk-forward OOS 年化 {r['wf_oos_annual']}%，"
                     f"夏普 {r.get('wf_oos_sharpe')}（{r.get('wf_folds')} 折）")
        elif "wf_error" in r:
            L.append(f"- Walk-forward：{r['wf_error']}")
        L.append("")
    try:
        import certain as CT
        cr = CT.summary(100_000.0, 10.0)
        L += ["## 确定性收益（不依赖选股）", "", "```",
              CT.format_report(cr), "```", ""]
    except Exception as ex:
        L += ["## 确定性收益", "", f"（生成失败：{type(ex).__name__}）", ""]
    L += ["## 审计（最近 10 条）", "", "```"]
    df = OL.query(limit=10)
    L.append(df.to_string(index=False) if len(df) else "（无记录）")
    L += ["```"]
    fp.write_text("\n".join(L), encoding="utf-8")
    _mark("report", True, str(fp))
    return {"ok": True, "path": str(fp)}


def step_notify(path: str | None = None) -> dict:
    """
    多渠推送：邮件 / 飞书 / 企业微信 / 个人微信（Server酱）。

    未配置的渠道自动跳过；全部未配置时如实返回失败。
    """
    import notify as NT

    body = "（报告文件未生成）"
    if path and Path(path).exists():
        body = Path(path).read_text(encoding="utf-8")[:20000]
    # 群消息用摘要（前 40 行），邮件发全文
    brief = chr(10).join(body.splitlines()[:40])
    subject = f"QuantPilot 日更 {time.strftime('%Y-%m-%d')}"
    try:
        r = NT.push_all(subject, body, brief=brief)
    except Exception as ex:
        _mark("notify", False, f"{type(ex).__name__}: {ex}")
        return {"ok": False, "error": str(ex)}
    if not r.get("ok"):
        _mark("notify", False, r.get("reason") or "全部渠道失败")
        return {"ok": False, **{k: v for k, v in r.items() if k != "results"},
                "channels": {k: bool(v.get("ok"))
                             for k, v in (r.get("results") or {}).items()}}
    sent = [k for k, v in (r.get("results") or {}).items() if v.get("ok")]
    _mark("notify", True, f"已发 {len(sent)} 个渠道：{','.join(sent)}"
                          + ("（干跑）" if r.get("dry_run") else ""))
    return {"ok": True, "channels": sent, "dry_run": r.get("dry_run")}


STEPS = ["ingest", "health", "research", "report", "notify"]


def run_daily(notify: bool = False, dry_run: bool = False,
              limit: int = INGEST_LIMIT, do_wf: bool = True) -> dict:
    CTX.clear()
    if dry_run:
        print("DRY-RUN：将依次执行 " + " → ".join(STEPS))
        return {"ok": True, "dry_run": True, "steps": STEPS}

    rid = OL.new_run("daily", notify=notify)
    t0 = time.time()
    out = {}
    print(f"[{time.strftime('%H:%M:%S')}] 日更开始  run={rid}")

    def _run(name, fn):
        print(f"[{time.strftime('%H:%M:%S')}] {name} …", flush=True)
        try:
            r = fn()
        except Exception as ex:
            r = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
            OL.log(name, status="fail", kind="daily", error=str(ex)[:200])
        print(f"    -> {'OK' if r.get('ok') else 'FAIL'}  {str(r)[:140]}",
              flush=True)
        out[name] = r
        return r

    _run("ingest", lambda: step_ingest(limit))
    _run("health", lambda: step_health(notify))
    _run("research", lambda: step_research(MAX_CODES, do_wf))
    rp = _run("report", step_report)
    if notify:
        _run("notify", lambda: step_notify(rp.get("path")))

    dt = round(time.time() - t0, 1)
    ok = all((out.get(s) or {}).get("ok") for s in ("ingest", "health", "report"))
    OL.log("daily_done", status="ok" if ok else "fail", kind="daily",
           seconds=dt)
    print(f"[{time.strftime('%H:%M:%S')}] 日更结束 {dt}s  "
          f"{'全部成功' if ok else '有失败项'}")
    return {"ok": ok, "run": rid, "seconds": dt, "steps": out}


def register_jobs(notify: bool = False) -> None:
    """注册日更任务（工作日 15:30）。"""
    OJ.JOBS.clear()

    @OJ.job("daily_pipeline", "weekday@15:30", retries=1)
    def _daily():
        r = run_daily(notify=notify)
        if not r.get("ok"):
            raise RuntimeError("日更存在失败步骤")
        return r.get("seconds")


def _loop(interval_s: int = 60) -> None:
    register_jobs(notify=True)
    print(f"常驻调度启动，每 {interval_s}s 检查一次（Ctrl-C 退出）", flush=True)
    while True:
        try:
            r = OJ.run_due()
            if r:
                print(f"[{time.strftime('%H:%M:%S')}] 执行 {list(r)}",
                      flush=True)
        except KeyboardInterrupt:
            print("退出")
            return
        except Exception as ex:
            print("循环异常:", type(ex).__name__, ex, flush=True)
        time.sleep(int(interval_s))


if __name__ == "__main__":
    import argparse

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=INGEST_LIMIT)
    ap.add_argument("--no-wf", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        print(run_daily(dry_run=True))
    elif a.loop:
        _loop()
    else:
        r = run_daily(notify=a.notify, limit=a.limit, do_wf=not a.no_wf)
        sys.exit(0 if r["ok"] else 1)
