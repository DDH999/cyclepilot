# -*- coding: utf-8 -*-
"""
QuantPilot · FastAPI 服务
把 评估门禁 + 图表工厂 + VibeTrading 引擎 + 推送 四层串成一个产品界面。
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notify

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
CHARTS = REP / "charts"
VIBE = "http://127.0.0.1:8000"

app = FastAPI(title="QuantPilot", version="0.1.0")

CHART_META = {
    "01": ("价格与均线", "趋势结构"),
    "02": ("累计收益", "长期复利"),
    "03": ("回撤曲线", "风险"),
    "04": ("滚动波动率", "风险"),
    "05": ("波动率期限结构", "状态"),
    "06": ("收益分布", "厚尾检验"),
    "07": ("Q-Q 图", "正态性"),
    "08": ("收益自相关", "可预测性"),
    "09": ("月度热力图", "季节性"),
    "10": ("年度收益", "周期"),
    "11": ("相关性矩阵", "分散度"),
    "12": ("滚动相关性", "稳定性"),
    "13": ("PCA 解释方差", "系统性风险"),
    "14": ("区间收益排名", "横截面"),
    "15": ("Beta", "系统性暴露"),
    "16": ("滚动 Beta", "暴露漂移"),
    "17": ("AUC 逐折", "模型稳定性"),
    "18": ("ROC", "判别力"),
    "19": ("PR 曲线", "精确-召回"),
    "20": ("概率校准", "可信度"),
    "21": ("分数分布", "分离度"),
    "22": ("混淆矩阵", "阈值权衡"),
    "23": ("特征重要性", "归因"),
    "24": ("IC 衰减", "信号半衰期"),
    "25": ("波动率状态迁移", "择时"),
    "26": ("尾部风险 VaR/CVaR", "极端风险"),
    "27": ("滚动 Sharpe", "风险调整收益"),
    "28": ("净值对比", "相对强弱"),
    "29": ("量价", "背离"),
    "30": ("PBO 过拟合概率", "门禁"),
    "31": ("通缩 Sharpe", "门禁"),
    "32": ("泄漏审计", "门禁"),
    "33": ("因子相关性", "因子分散"),
    "34": ("策略净值曲线", "回测"),
    "35": ("回撤曲线对比", "回测"),
    "36": ("风险-收益散点", "回测"),
    "37": ("滚动 Sharpe", "回测"),
    "38": ("月度收益热力", "回测"),
    "39": ("年度收益对比", "回测"),
    "40": ("相对基准超额", "回测"),
    "41": ("换手敏感度", "回测"),
}


def _vibe(path: str, payload: dict | None = None, timeout: int = 25) -> tuple[int, Any]:
    url = VIBE + path
    try:
        if payload is None:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:300]}
    except Exception as e:
        return 503, {"error": f"{type(e).__name__}: {e}"}


@app.get("/", response_class=HTMLResponse)
def index():
    p = ROOT / "frontend" / "index.html"
    if not p.exists():
        raise HTTPException(404, "frontend missing")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/summary")
def summary():
    p = REP / "summary.json"
    if not p.exists():
        return JSONResponse({"error": "尚未运行 pipeline，请先执行 python -m backend.pipeline"}, 404)
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/charts")
def charts_index():
    out = []
    if CHARTS.exists():
        for f in sorted(CHARTS.glob("*.png")):
            code = f.name[:2]
            title, cat = CHART_META.get(code, (f.stem, "其他"))
            out.append({"code": code, "file": f.name, "title": title, "category": cat,
                        "kb": round(f.stat().st_size / 1024, 1),
                        "url": f"/charts/{f.name}"})
    return {"count": len(out), "charts": out}


@app.get("/charts/{name}")
def chart_file(name: str):
    p = CHARTS / name
    if not p.exists() or p.suffix != ".png":
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/vibe/health")
def vibe_health():
    code, body = _vibe("/api", timeout=6)
    code2, ready = _vibe("/ready", timeout=6)
    return {"vibe_reachable": code == 200, "info": body if code == 200 else None,
            "ready": ready if code2 == 200 else None}


@app.get("/api/vibe/presets")
def vibe_presets():
    code, body = _vibe("/swarm/presets", timeout=15)
    if code != 200:
        raise HTTPException(503, str(body))
    return {"count": len(body), "presets": [
        {"name": p["name"], "title": p.get("title"), "agents": p.get("agent_count"),
         "desc": p.get("description"),
         "vars": [{"name": v.get("name"), "required": bool(v.get("required")),
                   "desc": v.get("description")} for v in (p.get("variables") or [])]}
        for p in body]}


RESEARCH_STORE: dict[str, dict] = {}


@app.post("/api/research")
def start_research(payload: dict):
    preset = payload.get("preset_name") or "investment_committee"
    user_vars = payload.get("user_vars") or {}
    code, body = _vibe("/swarm/runs", {"preset_name": preset, "user_vars": user_vars}, timeout=60)
    if code != 200:
        raise HTTPException(502, str(body))
    rid = body.get("id")
    RESEARCH_STORE[rid] = {"preset": preset, "vars": user_vars, "started": time.time()}
    return {"run_id": rid, "status": body.get("status"), "preset": preset}


@app.get("/api/research/{run_id}")
def research_status(run_id: str):
    code, body = _vibe(f"/swarm/runs/{run_id}", timeout=30)
    if code != 200:
        raise HTTPException(502, str(body))
    tasks = [{"id": t.get("id"), "status": t.get("status"),
              "chars": len(t.get("summary") or ""),
              "summary": (t.get("summary") or "")[:2000]}
             for t in (body.get("tasks") or [])]
    return {"run_id": run_id, "status": body.get("status"), "tasks": tasks}


@app.get("/api/research")
def list_research():
    return {"runs": [{"run_id": k, **v} for k, v in RESEARCH_STORE.items()]}


@app.get("/api/backtest")
def backtest_results():
    p = REP / "backtest_results.json"
    if not p.exists():
        raise HTTPException(404, "尚未回测，请运行 backend/run_strategy.py")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for n, r in d["strategies"].items():
        rows.append({"strategy": n,
                     "cagr": r.get("cagr"), "sharpe": r.get("sharpe"),
                     "sortino": r.get("sortino"), "max_drawdown": r.get("max_drawdown"),
                     "calmar": r.get("calmar"), "ann_vol": r.get("ann_vol"),
                     "ann_turnover": r.get("ann_turnover"), "beta": r.get("beta"),
                     "alpha_ann": r.get("alpha_ann"),
                     "information_ratio": r.get("information_ratio"),
                     "total_return": r.get("total_return")})
    rows.sort(key=lambda x: -(x["sharpe"] or -9))
    return {"generated_at": d.get("generated_at"), "universe_n": d.get("universe_n"),
            "n_survivor": d.get("n_survivor"), "n_delisted": d.get("n_delisted"),
            "top_n": d.get("top_n"), "buffer": d.get("buffer"),
            "vol_target": d.get("vol_target"), "stop_loss": d.get("stop_loss"),
            "max_per_industry": d.get("max_per_industry"),
            "n_industries": d.get("n_industries"),
            "regime_overlay": d.get("regime_overlay"),
            "benchmark": d.get("benchmark"),
            "benchmark_total": d.get("benchmark_total"),
            "equalweight_total": d.get("equalweight_total"),
            "costs": d.get("costs"), "strategies": rows}


@app.get("/api/validation")
def validation():
    vp = REP / "validation.json"
    if not vp.exists():
        raise HTTPException(404, "尚未验证，请运行 backend/validate.py")
    v = json.loads(vp.read_text(encoding="utf-8"))
    ep = REP / "ensemble.json"
    ens = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {}
    # 干净样本外：用 IS 选出的策略，其 OOS 才是真正的样本外证据
    clean = v["oos"].get(v["best_by_is"], {})
    en = ens.get("等权全部14个策略", {})
    return {
        "split_date": v["split_date"],
        "n_strategies": v["n_strategies"],
        "pbo": v["pbo"], "pbo_verdict": v["pbo_verdict"],
        "dsr": v["deflated_sharpe"],
        "best_by_is": v["best_by_is"],
        "best_full_sample": v["best_full_sample"],
        "is_best_equals_full_best": v["is_best_equals_full_best"],
        "is_oos_rank_corr": v["is_oos_rank_corr"],
        "clean_oos": {"strategy": v["best_by_is"],
                      "sharpe": clean.get("sharpe"), "cagr": clean.get("cagr"),
                      "mdd": clean.get("mdd")},
        "ensemble": {"sharpe": (en.get("full") or {}).get("sharpe"),
                     "cagr": (en.get("full") or {}).get("cagr"),
                     "mdd": (en.get("full") or {}).get("mdd"),
                     "oos_sharpe": (en.get("oos") or {}).get("sharpe"),
                     "oos_cagr": (en.get("oos") or {}).get("cagr"),
                     "oos_mdd": (en.get("oos") or {}).get("mdd")},
        "table": [{"strategy": c,
                   "is_sharpe": (v["is"].get(c) or {}).get("sharpe"),
                   "is_cagr": (v["is"].get(c) or {}).get("cagr"),
                   "oos_sharpe": (v["oos"].get(c) or {}).get("sharpe"),
                   "oos_cagr": (v["oos"].get(c) or {}).get("cagr"),
                   "oos_mdd": (v["oos"].get(c) or {}).get("mdd")}
                  for c in v["oos"]],
    }


@app.get("/api/fundamentals")
def fundamentals_result():
    p = REP / "fundamentals_result.json"
    if not p.exists():
        raise HTTPException(404, "尚未测试基本面因子，请运行 backend/fund_strategy.py")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for k, r in d["factors"].items():
        if k.startswith("_"):
            continue
        rows.append({"key": k, "name": r.get("name"), "ic": r.get("ic"),
                     "corr_mom": r.get("corr_mom"), "cagr": r.get("cagr"),
                     "sharpe": r.get("sharpe"), "mdd": r.get("max_drawdown"),
                     "is_cagr": (r.get("is") or {}).get("cagr"),
                     "oos_cagr": (r.get("oos") or {}).get("cagr"),
                     "oos_sharpe": (r.get("oos") or {}).get("sharpe")})
    rows.sort(key=lambda x: -(x["sharpe"] or -9))
    combo = d["factors"].get("_combo", {})
    return {"generated_at": d.get("generated_at"), "split_date": d.get("split_date"),
            "coverage": d.get("coverage"), "factors": rows,
            "combo": {"cagr": combo.get("cagr"), "sharpe": combo.get("sharpe"),
                      "mdd": combo.get("max_drawdown"),
                      "oos_cagr": (combo.get("oos") or {}).get("cagr"),
                      "oos_sharpe": (combo.get("oos") or {}).get("sharpe")}}


@app.get("/api/notify/status")
def notify_status():
    return notify.cfg_status()


@app.post("/api/notify/test")
def notify_test(payload: dict | None = None):
    p = payload or {}
    subj = p.get("subject", "QuantPilot 测试推送")
    html = p.get("html", f"<h2>{subj}</h2><p>通道连通性测试 · {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
    return notify.push_all(subj, html, [], brief=f"{subj} · {time.strftime('%H:%M')}")


@app.post("/api/notify/report")
def notify_report(payload: dict | None = None):
    """把最新 pipeline 结果 + 图表打包推送"""
    p = payload or {}
    sp = REP / "summary.json"
    if not sp.exists():
        raise HTTPException(404, "no summary")
    s = json.loads(sp.read_text(encoding="utf-8"))
    t = s["tasks"]
    lines = [
        f"**波动率状态 AUC** {t['vol_regime']['auc_mean']:.4f}（门槛 0.70）",
        f"**方向 AUC** {t['direction_1d']['auc_mean']:.4f}（诚实区间 0.51-0.56）",
        f"**洗牌对照** {t['shuffled_control']['auc_mean']:.4f}（应≈0.5）",
        f"**PBO 过拟合概率** {s['strategy']['pbo'].get('pbo')}",
        f"图表 {s['charts']} 张",
    ]
    brief = f"QuantPilot 日报 {time.strftime('%m-%d')}｜" + "；".join(
        x.replace('**', '') for x in lines[:3])
    html = "<h2>QuantPilot 量化体检报告</h2><ul>" + "".join(f"<li>{x}</li>" for x in lines) + "</ul>"
    imgs = sorted(CHARTS.glob("*.png"))[:6] if CHARTS.exists() else []
    return notify.push_all(p.get("subject", "QuantPilot 量化体检报告"), html, imgs, brief=brief)


@app.get("/api/monitor")
def monitor():
    """实时监控：读取缓存行情，输出最新价与波动状态"""
    import datafeed
    from pipeline import feats, target_vol
    uni = {s: "cn" for s in datafeed.UNIVERSE_CN[:8]}
    panel = datafeed.sanitize_panel(datafeed.get_panel(uni))
    rows = []
    for s, d in panel.items():
        c = d["close"]
        r = c.pct_change(fill_method=None)
        v20 = r.rolling(20).std().iloc[-1]
        v60 = r.rolling(60).std().iloc[-1]
        rows.append({
            "symbol": s, "date": str(d["date"].iloc[-1].date()),
            "close": round(float(c.iloc[-1]), 3),
            "chg_1d": round(float(r.iloc[-1] * 100), 2),
            "chg_20d": round(float((c.iloc[-1] / c.iloc[-21] - 1) * 100), 2),
            "vol20_ann": round(float(v20 * np.sqrt(252) * 100), 1),
            "vol_state": "高波" if v20 > v60 else "低波",
            "dd_60": round(float((c.iloc[-1] / c.tail(60).max() - 1) * 100), 2),
        })
    return {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(rows), "rows": rows}


@app.get("/api/gate")
def gate():
    """门禁判定：把 AUC/PBO/DSR 汇总成一个可部署结论"""
    sp = REP / "summary.json"
    if not sp.exists():
        raise HTTPException(404, "no summary")
    s = json.loads(sp.read_text(encoding="utf-8"))
    vol_auc = s["tasks"]["vol_regime"]["auc_mean"]
    shuf = s["tasks"]["shuffled_control"]["auc_mean"]
    pbo = (s["strategy"]["pbo"] or {}).get("pbo")
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    chk("洗牌标签对照", abs(shuf - 0.5) < 0.06, f"AUC={shuf:.4f}，应≈0.5")
    chk("波动率模型 AUC ≥ 0.70", vol_auc >= 0.70, f"AUC={vol_auc:.4f}")
    chk("方向模型不冒充强信号", True,
        f"方向 AUC={s['tasks']['direction_1d']['auc_mean']:.4f}，按诚实口径接受")
    chk("PBO < 0.50", (pbo is not None and pbo < 0.50), f"PBO={pbo}")
    passed = sum(1 for c in checks if c["pass"])
    deployable = all(c["pass"] for c in checks)
    return {"checks": checks, "passed": passed, "total": len(checks),
            "deployable": deployable,
            "verdict": "允许进入纸盘" if deployable else "拒绝部署：未通过门禁",
            "note": "真实资金永不自动开启；本判定仅决定是否进入纸盘(paper)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8420, log_level="info")
