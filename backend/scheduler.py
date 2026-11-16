# -*- coding: utf-8 -*-
"""
QuantPilot · 轻量定时调度器（无第三方依赖）
============================================
用法:
  python backend/scheduler.py --run-now          # 立即跑一轮（体检 + 推送）
  python backend/scheduler.py                    # 常驻，按 schedule.json 执行
  python backend/scheduler.py --list             # 查看任务
"""
from __future__ import annotations
import sys, json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
CFG = ROOT / "schedule.json"
PY = sys.executable

DEFAULT = {
    "timezone": "Asia/Shanghai",
    "jobs": [
        {"name": "盘后行情更新", "time": "15:40", "weekdays": [1, 2, 3, 4, 5],
         "cmd": ["backend/universe.py", "csi800"], "push": False},
        {"name": "策略回测", "time": "16:10", "weekdays": [1, 2, 3, 4, 5],
         "cmd": ["backend/run_strategy.py", "--top", "30", "--vt", "0.15",
                 "--stop-loss", "0.15", "--max-per-ind", "3", "--save"], "push": False},
        {"name": "诚实检验", "time": "16:25", "weekdays": [1, 2, 3, 4, 5],
         "cmd": ["backend/validate.py"], "push": False},
        {"name": "生成图表与研报", "time": "16:35", "weekdays": [1, 2, 3, 4, 5],
         "cmd": ["backend/bt_charts.py"], "push": False},
        {"name": "研报工厂", "time": "16:40", "weekdays": [1, 2, 3, 4, 5],
         "cmd": ["backend/report_factory.py"], "push": True},
    ],
}


def ensure_cfg() -> dict:
    if not CFG.exists():
        CFG.write_text(json.dumps(DEFAULT, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[scheduler] 已生成默认配置 {CFG.name}，请按需修改")
    return json.loads(CFG.read_text(encoding="utf-8"))


def run_job(job: dict) -> dict:
    script = ROOT / job["cmd"][0]
    args = [PY, str(script)] + job["cmd"][1:]
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] ▶ {job['name']} ...", flush=True)
    try:
        p = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        ok = p.returncode == 0
        tail = (p.stdout or "")[-1200:]
        print(f"[{datetime.now():%H:%M:%S}] {'✓' if ok else '✗'} {job['name']} "
              f"用时 {time.time()-t0:.0f}s")
        if not ok:
            print((p.stderr or "")[-800:])
    except subprocess.TimeoutExpired:
        ok, tail = False, "超时(30min)"
        print(f"[scheduler] ✗ {job['name']} 超时")
    return {"name": job["name"], "ok": ok, "elapsed": round(time.time() - t0, 1), "tail": tail}


def push_report(job_results: list[dict]):
    """把最新研报摘要推送到邮件/飞书/微信"""
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        import notify
    except Exception as e:
        print(f"[scheduler] 推送模块不可用: {e}")
        return
    sp = REP / "summary.json"
    bp = REP / "backtest_results.json"
    vp = REP / "validation.json"
    lines = []
    if bp.exists():
        b = json.loads(bp.read_text(encoding="utf-8"))
        S = b.get("strategies", {})
        if S:
            best = max(S, key=lambda k: S[k].get("sharpe") or -9)
            r = S[best]
            lines.append(f"最优策略 {best}：年化{(r.get('cagr') or 0)*100:.1f}% "
                         f"Sharpe{r.get('sharpe') or 0:.2f} 回撤{(r.get('max_drawdown') or 0)*100:.1f}%")
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        lines.append(f"PBO {v['pbo']['pbo']:.3f} | DSR {v['deflated_sharpe']['dsr']:.3f} "
                     f"| IS/OOS ρ {v['is_oos_rank_corr']:+.2f}")
        o = (v.get("oos") or {}).get(v.get("best_by_is", ""), {})
        if o:
            lines.append(f"干净样本外 Sharpe {o.get('sharpe', 0):.2f} 年化 {(o.get('cagr') or 0)*100:.1f}%")
    subj = f"QuantPilot 日报 {datetime.now():%m-%d}"
    brief = subj + "｜" + "；".join(lines[:3])
    html = (f"<h2>{subj}</h2><ul>" + "".join(f"<li>{x}</li>" for x in lines) + "</ul>")
    imgs = sorted((REP / "charts").glob("34_bt_equity.png")) + \
           sorted((REP / "charts").glob("42_is_vs_oos.png"))
    res = notify.push_all(subj, html, imgs, brief=brief)
    print(f"[scheduler] 推送结果: {json.dumps(res, ensure_ascii=False)[:300]}")


def due(job: dict, now: datetime, last_run: dict) -> bool:
    if now.isoweekday() not in job.get("weekdays", [1, 2, 3, 4, 5]):
        return False
    hh, mm = map(int, job["time"].split(":"))
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    key = job["name"] + now.strftime("%Y%m%d")
    if key in last_run:
        return False
    return datetime.now() >= target and (datetime.now() - target) < timedelta(minutes=20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-now", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="只跑指定任务名")
    a = ap.parse_args()
    cfg = ensure_cfg()
    jobs = cfg["jobs"]
    if a.only:
        jobs = [j for j in jobs if a.only in j["name"]]
    if a.list:
        for j in jobs:
            print(f"  {j['time']}  周{j.get('weekdays')}  {j['name']:14} "
                  f"{'[推送]' if j.get('push') else ''}")
        return
    if a.run_now:
        res = [run_job(j) for j in jobs]
        if any(j.get("push") for j in jobs):
            push_report(res)
        print(f"\n完成 {sum(1 for r in res if r['ok'])}/{len(res)}")
        return
    print(f"[scheduler] 常驻模式启动，共 {len(jobs)} 个任务。Ctrl+C 退出。")
    last_run: dict = {}
    while True:
        now = datetime.now()
        for j in jobs:
            if due(j, now, last_run):
                run_job(j)
                last_run[j["name"] + now.strftime("%Y%m%d")] = True
                if j.get("push"):
                    push_report([])
        time.sleep(30)


if __name__ == "__main__":
    main()
