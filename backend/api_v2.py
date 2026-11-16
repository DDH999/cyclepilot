# -*- coding: utf-8 -*-
"""
体检服务 API（M4）
====================================================
纯 JSON 接口（不依赖 python-multipart）：
前端用 FileReader 读取文件文本后直接 POST。

接口
----
GET  /                    体检页面
GET  /api/health          健康检查
POST /api/checkup         提交体检
     {"code": "...", "returns_csv": "...", "weights_csv": "...",
      "bench_csv": "...", "use_llm": false, "is_split": "2023-01-01"}

核心逻辑在 `run_checkup()`（纯函数），便于脱离 HTTP 单测。

运行
----
    python backend/api_v2.py          # http://127.0.0.1:8421
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adjudicator as A
import ai_diagnostician as AI
import purity_tester as P
import scoring as SC
import stress_tester as S
import validator as V

__version__ = "0.1.0"
__all__ = ["run_checkup", "parse_series_csv", "parse_weights_csv", "app"]

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "checkup.html"
DAILY_DIR = ROOT / "reports" / "daily"
NAV_CSV = ROOT / "reports" / "paper_nav.csv"

_DATE_KEYS = ("date", "日期", "time", "datetime", "trade_date", "交易日", "时间")
_RET_KEYS = ("ret", "return", "returns", "收益", "收益率", "pct_chg", "涨跌幅", "daily_return")
_NAV_KEYS = ("nav", "净值", "value", "close", "收盘", "收盘价", "price", "价格")


# ══════════════════════════════════════════════════════════
# 解析
# ══════════════════════════════════════════════════════════
def parse_series_csv(text: str, name: str = "序列") -> pd.Series:
    """
    解析「日期 + 收益」或「日期 + 净值」CSV，返回带 DatetimeIndex 的收益 Series。

    支持列名：date/日期、ret/收益/收益率、nav/净值/close/收盘
    无表头时按「第一列日期、第二列数值」处理。
    净值模式会自动转为收益率。
    """
    if not text or not text.strip():
        raise ValueError(f"{name}为空")
    raw = text.lstrip("\ufeff").strip()

    # 无表头：手工补上
    first_line = raw.splitlines()[0]
    has_header = any(k in first_line.lower() for k in
                     ("date", "日期", "ret", "收益", "nav", "净值")) or \
        not first_line.split(",")[0].replace("-", "").replace("/", "").isdigit()

    try:
        df = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
    except Exception as ex:
        raise ValueError(f"{name}无法解析为 CSV：{ex}") from None

    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = list(df.columns)

    date_col = next((c for c in cols if c in _DATE_KEYS), cols[0])
    ret_col = next((c for c in cols if c in _RET_KEYS), None)
    nav_col = next((c for c in cols if c in _NAV_KEYS), None)
    if ret_col is None and nav_col is None:
        ret_col = cols[1] if len(cols) > 1 else None
    if ret_col is None and nav_col is None:
        raise ValueError(f"{name}未找到收益或净值列")

    idx = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    if ret_col is not None:
        val = pd.to_numeric(df[ret_col], errors="coerce")
        s = pd.Series(val.to_numpy(), index=idx)
    else:
        nav = pd.to_numeric(df[nav_col], errors="coerce")
        s = pd.Series(nav.to_numpy(), index=idx).pct_change(fill_method=None)

    s = s[~s.index.isna()].dropna().sort_index()
    s = s.astype("float64")
    # 常见错误：把百分数当小数（|值| 普遍 >1）
    if len(s) and s.abs().median() > 0.5:
        s = s / 100.0
    if len(s) < 30:
        raise ValueError(f"{name}有效样本仅 {len(s)} 条，至少需要 30 条")
    return s


def parse_weights_csv(text: str) -> pd.DataFrame | None:
    """解析宽表权重（行=日期，列=代码）。空文本返回 None。"""
    if not text or not text.strip():
        return None
    raw = text.lstrip("\ufeff").strip()
    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in _DATE_KEYS), df.columns[0])
    idx = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    body = df.drop(columns=[date_col])
    body = body.apply(pd.to_numeric, errors="coerce")
    body.index = idx
    body = body[~body.index.isna()].sort_index().fillna(0.0)
    return body if body.shape[1] >= 2 else None


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def _split_is_oos(rets: pd.Series, is_split: str):
    """按日期切分样本内外；切不开时按 70/30。"""
    try:
        split = pd.Timestamp(is_split)
    except Exception:
        split = pd.Timestamp("2023-01-01")
    is_r, oos_r = rets.loc[:split], rets.loc[split:]
    if len(is_r) < 30 or len(oos_r) < 30:
        k = int(len(rets) * 0.7)
        is_r, oos_r = rets.iloc[:k], rets.iloc[k:]
    return is_r, oos_r, str(split.date())


def run_checkup(code: str = "", returns_csv: str = "",
                weights_csv: str | None = None,
                bench_csv: str | None = None,
                use_llm: bool = False,
                is_split: str = "2023-01-01") -> dict:
    """
    执行一次完整体检：解析 → M1 → M2 → 裁决 → M3 诊断。

    保证返回结构化结果；解析失败时抛 ValueError（由 API 层转成 400），
    其余环节不抛异常。
    """
    warnings: list[str] = []

    if not returns_csv or not returns_csv.strip():
        raise ValueError("必须提供收益序列 CSV")

    rets = parse_series_csv(returns_csv, "收益序列")
    bench = None
    if bench_csv and bench_csv.strip():
        try:
            bench = parse_series_csv(bench_csv, "基准序列")
        except ValueError as ex:
            warnings.append(f"基准序列解析失败，已忽略：{ex}")

    weights = parse_weights_csv(weights_csv) if weights_csv else None
    if weights_csv and weights is None and weights_csv.strip():
        warnings.append("权重文件解析失败，纯度将只能基于收益评估")

    # ── M1：代码真实性
    has_code = bool(code and code.strip())
    if has_code:
        code_ok = V.check_future_function(code, verbose=False)
        findings = V.scan_code(code)
    else:
        code_ok = None
        findings = []
        warnings.append("未提供策略代码，防作弊维度无法评估")

    # ── M1：过拟合
    is_r, oos_r, split_used = _split_is_oos(rets, is_split)
    overfit = None
    if len(is_r) >= 30 and len(oos_r) >= 30:
        overfit = V.calculate_overfitting(is_r, oos_r, n_trials=1, verbose=False)
    else:
        warnings.append(f"样本不足以切分样本内外（{len(is_r)}/{len(oos_r)} 天）")

    # ── M1：压力测试
    slip = S.apply_slippage(rets, turnover=0.1, verbose=False)
    crash = S.test_extreme_crash(rets, *S.CRASH_WINDOWS["2024微盘股流动性危机"],
                                 label="2024微盘股", verbose=False)
    if not crash.get("ok"):
        warnings.append("收益序列未覆盖 2024 微盘股危机区间，抗压项部分豁免")

    # ── M2：纯度
    purity = P.purity_report(weights, rets, bench)

    # ── M2：六维
    report = SC.full_report(code_ok=code_ok, findings=findings, overfit=overfit,
                            slippage=slip, crash=crash, purity=purity,
                            sensitivity=None)
    report["warnings"] = warnings
    report["inputs"] = {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()),
        "end": str(rets.index.max().date()),
        "has_code": has_code,
        "has_weights": weights is not None,
        "has_bench": bench is not None,
        "is_split": split_used,
    }

    # ── M3：诊断（永不抛异常）
    diagnosis = AI.diagnose(report, use_llm=use_llm)

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "report": report,
        "diagnosis": diagnosis,
        "findings": [f.to_dict() for f in findings],
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════
# HTTP 层
# ══════════════════════════════════════════════════════════
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel

    class CheckupRequest(BaseModel):
        code: str = ""
        returns_csv: str = ""
        weights_csv: str | None = None
        bench_csv: str | None = None
        use_llm: bool = False
        is_split: str = "2023-01-01"

    app = FastAPI(title="QuantPilot 策略体检", version=__version__)

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__,
                "llm_configured": bool(os.environ.get("QP_LLM_KEY")),
                "llm_disabled": os.environ.get("QP_LLM_DISABLE") == "1"}

    @app.post("/api/checkup")
    def checkup(req: CheckupRequest):
        try:
            return run_checkup(req.code, req.returns_csv, req.weights_csv,
                               req.bench_csv, req.use_llm, req.is_split)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None
        except Exception as ex:  # 兜底：绝不给前端 500 空白页
            log.exception("体检失败")
            raise HTTPException(status_code=500,
                                detail=f"内部错误：{ex}") from None

    @app.get("/api/daily/latest")
    def daily_latest():
        """最近一次日更体检结果。"""
        fp = DAILY_DIR / "latest.json"
        if not fp.exists():
            raise HTTPException(
                status_code=404,
                detail="尚无日更记录，请先运行 python backend/daily_job.py --once")
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"读取失败：{ex}") from None

    @app.get("/api/daily/history")
    def daily_history(limit: int = 30):
        """历史日更记录（按日期升序，仅摘要）。"""
        items = []
        files = sorted(DAILY_DIR.glob("20*.json"))
        for fp in files[-max(1, min(int(limit), 365)):]:
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            rep = d.get("report") or {}
            items.append({
                "date": d.get("date"),
                "verdict": rep.get("verdict"),
                "grade": rep.get("grade"),
                "score": rep.get("score"),
                "nav_latest": d.get("nav_latest"),
                "n_changes": len(d.get("changes") or []),
            })
        return {"ok": True, "count": len(items), "items": items}

    @app.get("/api/paper/nav")
    def paper_nav(limit: int = 500):
        """纸盘净值曲线。"""
        if not NAV_CSV.exists():
            raise HTTPException(status_code=404, detail="尚无净值数据")
        try:
            df = pd.read_csv(NAV_CSV, parse_dates=["date"]).tail(max(2, int(limit)))
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"净值读取失败：{ex}") from None
        return {
            "ok": True,
            "n": int(len(df)),
            "dates": [d.strftime("%Y-%m-%d") for d in df["date"]],
            "nav": [round(float(v), 2) for v in df["nav"]],
        }

    @app.post("/api/daily/run")
    def daily_run():
        """手动触发一次日更（约 5 秒）。"""
        try:
            from daily_job import run_daily
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"日更模块不可用：{ex}") from None
        try:
            rec = run_daily(do_notify=False)
        except FileNotFoundError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None
        except Exception as ex:
            log.exception("日更失败")
            raise HTTPException(status_code=500, detail=f"日更失败：{ex}") from None
        return {
            "ok": True,
            "date": rec.get("date"),
            "report": rec.get("report"),
            "diagnosis": rec.get("diagnosis"),
            "changes": rec.get("changes"),
            "nav_latest": rec.get("nav_latest"),
            "nav_days": rec.get("nav_days"),
            "elapsed_ms": rec.get("elapsed_ms"),
        }

    @app.get("/api/attribution")
    def attribution_report(capital: float = 100000.0, turnover: float = 0.0,
                           refresh_bench: bool = False):
        """纸盘收益归因 + 成本核算（对标沪深300）。"""
        nav_fp = NAV_CSV
        if not nav_fp.exists():
            raise HTTPException(status_code=404, detail="尚无纸盘净值，请先运行日更")
        try:
            nav = pd.read_csv(nav_fp, parse_dates=["date"]).set_index("date")["nav"]
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"净值读取失败：{ex}") from None

        try:
            import attribution as AT
            bench = AT.load_bench(start=str(nav.index.min().date()),
                                  end=str(nav.index.max().date()),
                                  refresh=bool(refresh_bench))
            attr = AT.performance_attribution(nav, bench)
            cost = AT.cost_analysis(float(capital), float(turnover))
            sens = AT.turnover_sensitivity(float(capital))
            return {
                "ok": True,
                "attribution": attr,
                "cost": cost,
                "turnover_table": sens.to_dict("records"),
                "text": AT.format_report(attr, cost),
            }
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None
        except RuntimeError as ex:
            raise HTTPException(status_code=503, detail=f"基准不可用：{ex}") from None
        except Exception as ex:
            log.exception("归因失败")
            raise HTTPException(status_code=500, detail=f"归因失败：{ex}") from None

    @app.get("/api/risk")
    def risk_dashboard(capital: float = 100000.0, turnover: float = 12.0,
                       adv: float = 5e8, refresh_bench: bool = False):
        """
        风控仪表盘：把数据体检 / 滚动样本外 / 成本容量 / 四道闸门聚合成一个结论。

        参数
        ----
        capital  : 组合规模（影响冲击成本与容量）
        turnover : 年换手（影响成本与换手闸门）
        adv      : 持仓标的日均成交额假设（默认 5 亿）
        """
        import blackswan as BW
        import datacheck as DC
        import execution as EX
        import riskcontrol as RC
        import walkforward as WF

        if not NAV_CSV.exists():
            raise HTTPException(status_code=404,
                                detail="尚无纸盘净值，请先运行日更")
        try:
            nav = pd.read_csv(NAV_CSV, parse_dates=["date"]).set_index("date")["nav"]
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"净值读取失败：{ex}") from None
        if len(nav) < 60:
            raise HTTPException(status_code=400,
                                detail=f"净值仅 {len(nav)} 天，不足以做风控评估")

        warn: list[str] = []

        # ── 数据体检
        dq = DC.check_series(nav, name="纸盘净值")
        # ── 滚动样本外
        wf = WF.purged_walk_forward(nav)
        # ── 尾部与脆弱度
        tail = BW.tail_statistics(nav)
        frag = BW.fragility_score(tail)
        # ── 成本与容量
        rets = nav.pct_change(fill_method=None).dropna()
        sigma = float(np.std(rets.to_numpy(), ddof=0)) if len(rets) else 0.012
        cost = EX.cost_breakdown(float(capital), float(turnover), float(adv), sigma)
        try:
            capacity = EX.max_capacity(float(turnover), float(adv), sigma, 0.03)
        except Exception:
            capacity = None

        # ── 持仓（纸盘是等权买入持有）
        codes: list[str] = []
        try:
            wl = ROOT / "watchlist.txt"
            if wl.exists():
                for line in wl.read_text(encoding="utf-8").splitlines():
                    tok = line.split("#")[0].split(",")[0].strip().strip("'\"")
                    tok = "".join(ch for ch in tok if ch.isdigit())
                    if len(tok) == 6 and tok not in codes:
                        codes.append(tok)
        except Exception:
            pass
        if not codes:
            warn.append("未找到 watchlist.txt，拥挤度与风格暴露仅按未知分组估算")
        weights = None
        weights_source = "watchlist"
        try:
            import portfolio as PF
            hp = ROOT / "holdings.csv"
            if hp.exists():
                hd = PF.load_holdings(hp)
                hc = [h.code for h in hd["holdings"]]
                if hc:
                    hp_px = PF.fetch_prices_for(hd, refresh=bool(refresh_bench))
                    mtm_ = PF.mark_to_market(hd, hp_px)
                    w = PF.weights_of(mtm_)
                    if len(w):
                        weights = pd.DataFrame([w.to_dict()],
                                               index=[nav.index[-1]])
                        codes = list(w.index)
                        weights_source = "holdings"
                        if mtm_.get("n_missing"):
                            warn.append(
                                f"持仓中有 {mtm_['n_missing']} 只取不到价格，"
                                "权重仅按可取价标的计算")
        except Exception as ex:
            warn.append(f"持仓读取失败，改用 watchlist 等权：{ex}")

        if weights is None and codes:
            weights = pd.DataFrame([{c: 1.0 / len(codes) for c in codes}],
                                   index=[nav.index[-1]])

        # ── 风格分组（复用行业缓存）
        groups: dict[str, str] = {}
        ind_fp = ROOT / ".cache" / "industries.csv"
        if ind_fp.exists():
            try:
                ind = pd.read_csv(ind_fp, dtype=str)
                ccol = next((c for c in ind.columns
                             if "code" in c.lower() or "代码" in c), ind.columns[0])
                icol = next((c for c in ind.columns
                             if "industry" in c.lower() or "行业" in c), ind.columns[-1])
                groups = {str(r[ccol]).zfill(6): str(r[icol])
                          for _, r in ind.iterrows()}
            except Exception:
                warn.append("行业缓存读取失败，风格暴露按未知分组")
        elif codes:
            warn.append("无行业缓存，风格暴露按未知分组（可先跑 backend/sector.py）")

        # 持仓里若有基金，用基金大类覆盖其分组
        try:
            import portfolio as PF2
            hp2 = ROOT / "holdings.csv"
            if hp2.exists():
                hd2 = PF2.load_holdings(hp2)
                fg = PF2.groups_for(hd2)
                for c, g in fg.items():
                    if c in (codes or []) and g != "未知":
                        groups[c] = g
        except Exception:
            pass

        # ── 与基准相关性
        bench_r2 = None
        try:
            import attribution as AT
            bench = AT.load_bench(start=str(nav.index.min().date()),
                                  end=str(nav.index.max().date()),
                                  refresh=bool(refresh_bench))
            bench_r2 = AT.performance_attribution(nav, bench)["r2"]
        except Exception as ex:
            warn.append(f"基准不可用，拥挤度同质化维度跳过：{ex}")

        # ── 风控决策
        crowd = RC.crowding_score(weights, turnover_annual=float(turnover),
                                  bench_r2=bench_r2,
                                  participation=cost["participation"])
        style = (RC.style_exposure(weights, groups) if weights is not None
                 else {"ok": False, "reason": "无持仓"})
        decision = RC.kill_switch(
            nav, weights, groups or None,
            turnover_annual=float(turnover), bench_r2=bench_r2,
            participation=cost["participation"],
            data_ok=dq.get("ok"))

        return {
            "ok": True,
            "as_of": str(nav.index[-1].date()),
            "n_days": int(len(nav)),
            "nav_latest": float(nav.iloc[-1]),
            "decision": decision,
            "drawdown": decision.get("drawdown"),
            "crowding": crowd,
            "style": style,
            "data": dq,
            "walkforward": wf,
            "tail": tail,
            "fragility": frag,
            "cost": cost,
            "capacity": capacity,
            "limits": decision.get("limits"),
            "inputs": {"capital": float(capital), "turnover": float(turnover),
                       "adv": float(adv), "n_codes": len(codes),
                       "weights_source": weights_source,
                       "sigma_daily": round(sigma, 6)},
            "warnings": warn,
        }

    @app.get("/api/portfolio")
    def portfolio_view(refresh: bool = False):
        """读取 holdings.csv，按最新价估值。"""
        import portfolio as PF

        fp = ROOT / "holdings.csv"
        if not fp.exists():
            raise HTTPException(
                status_code=404,
                detail="尚无 holdings.csv。请按 code,shares,cost 三列创建，"
                       "或先运行 python backend/portfolio.py 看示例")
        try:
            data = PF.load_holdings(fp)
        except (ValueError, FileNotFoundError) as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None

        # 注意：必须用 fetch_prices_for —— 它会按 kind 分别取
        # 股票收盘价 / 场外基金净值 / ETF 净值。用 fetch_last_prices 会把
        # 基金代码当股票代码去拉，全部取不到。
        prices = (PF.fetch_prices_for(data, refresh=bool(refresh))
                  if data["holdings"] else {})
        mtm = PF.mark_to_market(data, prices)
        return {"ok": bool(mtm.get("ok")), "portfolio": mtm,
                "text": PF.format_report(mtm)}

    @app.post("/api/checkup_code")
    def checkup_code(payload: dict):
        """
        策略代码一体化体检：代码 → 回测 → 六维。

        body: {"code": "...", "codes": ["600519"], "start": "2023-01-01",
               "end": "2025-12-31", "capital": 1000000}
        codes 省略时用 watchlist.txt。
        """
        import strategy_runner as SR

        code = str((payload or {}).get("code") or "")
        if not code.strip():
            raise HTTPException(status_code=400, detail="缺少 code（策略源码）")

        codes = (payload or {}).get("codes") or []
        if not codes:
            wl = ROOT / "watchlist.txt"
            if wl.exists():
                for line in wl.read_text(encoding="utf-8").splitlines():
                    tok = line.split("#")[0].split(",")[0].strip()
                    tok = "".join(ch for ch in tok if ch.isdigit())
                    if len(tok) == 6 and tok not in codes:
                        codes.append(tok)
        if not codes:
            raise HTTPException(status_code=400,
                                detail="没有可用标的：请传 codes 或提供 watchlist.txt")

        try:
            built = SR.build_panel(codes, (payload or {}).get("start"),
                                   (payload or {}).get("end"))
        except Exception as ex:
            raise HTTPException(status_code=400,
                                detail=f"取行情失败：{ex}") from None

        try:
            rep = SR.full_checkup(code, built["panel"],
                                  capital=float((payload or {}).get("capital",
                                                                    1_000_000.0)))
        except Exception as ex:
            raise HTTPException(status_code=500,
                                detail=f"体检失败：{ex}") from None

        def safe(obj):
            import pandas as pd
            if isinstance(obj, dict):
                return {k: safe(v) for k, v in obj.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))}
            if isinstance(obj, (list, tuple)):
                return [safe(x) for x in obj]
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return None
            return obj

        return {"ok": bool(rep.get("ok")),
                "report": safe(rep),
                "text": SR.format_report(rep),
                "missing": built.get("missing"),
                "n_symbols": len(built.get("panel") or {})}

    @app.get("/api/dca")
    def dca_view(refresh: bool = False):
        """定投台账分析（XIRR）。"""
        import dca as D
        import fund as FD

        fp = ROOT / "dca_ledger.csv"
        if not fp.exists():
            raise HTTPException(status_code=404,
                                detail="尚无 dca_ledger.csv（date,code,amount,nav）")
        try:
            led = D.load_ledger(fp)
        except (ValueError, FileNotFoundError) as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None

        navs: dict[str, float] = {}
        series: dict = {}
        warn: list[str] = []
        for c in sorted({c for c in led["code"].unique() if c}):
            try:
                sseries = FD.fetch_nav(c)
                series[c] = sseries
                navs[c] = float(sseries.iloc[-1])
            except Exception as ex:
                warn.append(f"{c} 取净值失败：{ex}")

        # 台账里 nav 留空时，按净值序列补当日净值（模板默认就是空的）
        if series:
            led = D.fill_missing_nav(led, series)

        try:
            res = D.dca_analysis(led, navs)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex)) from None
        return {"ok": bool(res.get("ok")), "result": res,
                "text": D.format_report(res), "warnings": warn}

    @app.get("/api/sources")
    def sources_view():
        """数据源健康状况：哪些源可用、有没有熔断、降级了几次。"""
        import datasource as DS
        return DS.sources_status()

    @app.get("/api/perturbation")
    def perturbation_view(noise: float = 0.30, seed: int = 7):
        """扰动测试：轻微改变输入，看结论稳不稳。"""
        import robustness as RB

        if not NAV_CSV.exists():
            raise HTTPException(status_code=404, detail="尚无净值，请先运行日更")
        nav = pd.read_csv(NAV_CSV, parse_dates=["date"]).set_index("date")["nav"]
        cfg = RB.PerturbCfg(noise_mult=float(noise), seed=int(seed))
        res = RB.perturbation_suite(nav, cfg)
        return {"ok": bool(res.get("ok")), "result": res,
                "text": RB.format_perturbation(res)}

    @app.get("/api/stress")
    def stress_view(seed: int = 11):
        """压力测试：合成极端情景，看策略扛不扛得住。"""
        import robustness as RB

        if not NAV_CSV.exists():
            raise HTTPException(status_code=404, detail="尚无净值，请先运行日更")
        nav = pd.read_csv(NAV_CSV, parse_dates=["date"]).set_index("date")["nav"]
        bench = None
        try:
            import attribution as AT
            bench = AT.load_bench(start=str(nav.index.min().date()),
                                  end=str(nav.index.max().date()))
        except Exception:
            bench = None
        res = RB.stress_suite(nav, bench=bench, cfg=RB.StressCfg(seed=int(seed)))
        return {"ok": bool(res.get("ok")), "result": res,
                "text": RB.format_stress(res),
                "has_bench": bench is not None}

    @app.get("/api/zz")
    def zz_info():
        """zzshare 是否可用。"""
        import zz_source as ZZ
        return ZZ.available()

    @app.get("/api/valuation")
    def valuation_view(date: str | None = None):
        """
        全市场估值快照（zzshare）：PE/PB/PCF 中位数与分位。

        注意：这是**等权个股中位数**，不等于指数 PE（指数是市值加权）。
        """
        import zz_source as ZZ
        if not ZZ.ZZ_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="zzshare 未安装，无法提供全市场估值")
        snap = ZZ.valuation_snapshot(date)
        return {"ok": bool(snap.get("ok")), "snapshot": snap,
                "text": ZZ.format_report(ZZ.available(), snap)}

    @app.get("/api/industry_map")
    def industry_map_view(refresh: bool = False):
        """股票 → 行业映射（首次约 1 分钟，之后走缓存）。"""
        import zz_source as ZZ
        if not ZZ.ZZ_AVAILABLE:
            raise HTTPException(status_code=503, detail="zzshare 未安装")
        got = ZZ.industry_map(refresh=bool(refresh))
        if not got.get("ok"):
            raise HTTPException(status_code=500, detail=got.get("reason"))
        m = got["mapping"]
        return {"ok": True, "n": got["n"], "source": got["source"],
                "sample": dict(list(m.items())[:20])}

    @app.get("/api/overseas")
    def overseas_view(key: str = "sp500", days: int = 500,
                      refresh: bool = False):
        """海外指数历史（yfinance → 新浪回退）。"""
        import overseas as OS
        try:
            res = OS.index_history(key, days=int(days), refresh=bool(refresh))
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None
        out = {k: v for k, v in res.items() if k != "series"}
        if res.get("ok"):
            s = res["series"]
            out["series_tail"] = [
                {"date": str(d.date()), "close": round(float(v), 4)}
                for d, v in s.tail(30).items()]
            out["latest"] = round(float(s.iloc[-1]), 4)
        return {"ok": bool(res.get("ok")), "result": out,
                "text": OS.format_report(res)}

    @app.get("/api/overseas/diagnose")
    def overseas_diagnose():
        """海外指数数据源连通性诊断（区分封锁/限流/正常）。"""
        import overseas as OS
        return OS.diagnose()

    @app.get("/api/baostock")
    def baostock_view():
        """baostock 可用性与会话状态。"""
        import baostock_source as BS
        return BS.available()

    @app.get("/api/keys")
    def keys_view():
        """凭据状态（只返回掩码，绝不返回明文）。"""
        import credentials as CR
        return CR.status()

    @app.get("/api/us")
    def us_view(ticker: str = "AAPL", what: str = "all"):
        """美股数据：profile / latest / prices（ROIC + xfinlink）。"""
        import us_source as US
        import datetime as _dt

        st = US.status()
        if what == "status":
            return {"ok": True, "status": st, "text": US.format_report(st)}

        t = str(ticker).upper()
        out = {"ok": True, "ticker": t, "profile": None, "latest": None,
               "prices": None, "errors": {}}
        if what in ("all", "profile"):
            try:
                out["profile"] = US.roic_profile(t)
            except Exception as ex:
                out["errors"]["profile"] = f"{type(ex).__name__}: {str(ex)[:150]}"
        if what in ("all", "latest"):
            try:
                out["latest"] = US.roic_latest(t)
            except Exception as ex:
                out["errors"]["latest"] = f"{type(ex).__name__}: {str(ex)[:150]}"
        if what in ("all", "prices"):
            try:
                end = _dt.date.today()
                start = end - _dt.timedelta(days=30)
                got = US.xfin_prices(t, str(start), str(end))
                df = got["df"]
                keep = [c for c in ("date", "close", "adj_close", "volume")
                        if c in df.columns]
                out["prices"] = (df[keep].tail(30).to_dict("records")
                                 if keep else [])
                out["prices_source"] = got["source"]
            except Exception as ex:
                out["errors"]["prices"] = f"{type(ex).__name__}: {str(ex)[:150]}"
        out["ok"] = not out["errors"]
        return out

    @app.get("/api/quality")
    def quality_view(ticker: str = "AAPL", refresh: bool = False):
        """基本面质量评分（美股，ROIC.ai）。"""
        import quality as QL
        try:
            rep = QL.quality_report(str(ticker).upper(), refresh=bool(refresh))
        except Exception as ex:
            raise HTTPException(status_code=500,
                                detail=f"评分失败：{ex}") from None
        return {"ok": bool(rep.get("ok")), "report": rep,
                "text": QL.format_report(rep)}

    @app.get("/api/advice")
    def advice_view(monthly: float = 1000.0):
        """
        今日决策：定投倍数 / 再平衡 / 回撤加仓。

        读 holdings.csv 拿持仓与权重，读报告净值算回撤。
        """
        import advisor as AD
        import fund as FD
        import portfolio as PF

        hp = ROOT / "holdings.csv"
        if not hp.exists():
            raise HTTPException(status_code=404,
                                detail="尚无 holdings.csv")
        try:
            data = PF.load_holdings(hp)
        except Exception as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from None
        codes = [h.code for h in data["holdings"]]
        if not codes:
            raise HTTPException(status_code=400, detail="持仓为空")

        names: dict[str, str] = {}
        try:
            cat = FD.load_catalog()
            for c in codes:
                info = FD.lookup_fund(c, cat)
                if info:
                    names[c] = info["name"]
        except Exception:
            pass

        weights = None
        try:
            prices = PF.fetch_prices_for(data)
            mtm = PF.mark_to_market(data, prices)
            if mtm.get("ok"):
                weights = dict(mtm.get("weights") or {})
                if mtm.get("cash_ratio"):
                    weights["CASH"] = mtm["cash_ratio"]
        except Exception:
            weights = None

        dd = None
        if NAV_CSV.exists():
            try:
                nav = pd.read_csv(NAV_CSV, parse_dates=["date"]
                                  ).set_index("date")["nav"]
                dd = float((nav / nav.cummax() - 1).iloc[-1])
            except Exception:
                dd = None

        res = AD.advise(codes, names, weights, monthly_amount=float(monthly),
                        portfolio_drawdown=dd)
        return {"ok": bool(res.get("ok")), "result": res,
                "text": AD.format_report(res)}

    @app.get("/api/ml_risk")
    def ml_risk_view(key: str = "sp500"):
        """回撤风险预测（含逐市场验证结果）。"""
        import ml_risk as ML
        val = ML.validate_market(key)
        if not val.get("ok"):
            raise HTTPException(status_code=400,
                                detail=str(val.get("reason"))) from None
        cur = ML.current_risk(key) if val.get("usable") else None
        return {"ok": True, "validation": val, "current": cur,
                "text": ML.format_report(val, cur)}

    @app.get("/api/optimizer")
    def optimizer_view(keys: str = "sp500,ndx,nikkei,ftse"):
        """
        马科维茨 / 凯利 实测：权重稳定性 + 凯利是否隐含杠杆 + 该用哪种配置。
        """
        import optimizer as OP
        import overseas as OS

        klist = [k.strip() for k in str(keys).split(",") if k.strip()]
        navs: dict = {}
        missing: list[str] = []
        for k in klist:
            got = OS.index_history(k, days=20000)
            if got.get("ok"):
                navs[got["name"]] = got["series"]
            else:
                missing.append(k)
        if len(navs) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"可用指数不足 2 个（缺 {missing}）")

        R = OP.to_returns(navs)
        rep = OP.optimize_report(R)
        return {"ok": bool(rep.get("ok")), "report": rep,
                "text": OP.format_report(rep), "missing": missing}

    @app.get("/api/journal")
    def journal_view(settle: bool = False, review: bool = False,
                     limit: int = 20):
        """决策台账：列出 / 结算 / 复盘。"""
        import journal as JN

        out: dict = {"ok": True}
        if settle:
            out["settle"] = JN.settle_pending()
        if review:
            rep = JN.review()
            out["review"] = rep
            out["review_text"] = JN.format_report(rep)
        recs = JN.load_journal()
        out["n"] = len(recs)
        out["pending"] = sum(1 for r in recs if not r.get("outcome"))
        out["items"] = recs[-int(limit):]
        out["text"] = JN.format_journal(recs, limit=int(limit))
        return out

    @app.get("/api/cb_new")
    def cb_new_view(days: int = 7, refresh: bool = False):
        """可转债打新扫描。"""
        import cb_new as CB
        try:
            res = CB.scan(days_ahead=int(days), refresh=bool(refresh))
        except Exception as ex:
            raise HTTPException(status_code=500,
                                detail=f"取数失败：{ex}") from None
        return {"ok": bool(res.get("ok")), "result": res,
                "text": CB.format_report(res)}

    @app.get("/api/fee")
    def fee_view(monthly: float = 4000.0, years: float = 20.0,
                 rate: float = 0.07,
                 a_purchase: float | None = None,
                 c_service: float | None = None):
        """费率核对：A 类 vs C 类。"""
        import fee_check as FE
        res = FE.compare_ac(monthly=float(monthly), years=float(years),
                            annual_return=float(rate),
                            a_purchase=a_purchase, c_service=c_service)
        imp = FE.fee_impact(monthly=float(monthly), years=float(years),
                            annual_return=float(rate),
                            fee_low=0.0060, fee_high=0.0110)
        return {"ok": bool(res.get("ok")), "result": res,
                "impact": imp, "text": FE.format_report(res, imp)}

    @app.get("/api/admit")
    def admit_view(file: str, bench_key: str = "sp500",
                   turnover: float = 0.0, trades: int | None = None):
        """主动交易准入判定（读 CSV）。"""
        import pandas as pd
        import paper_validate as PV
        import overseas as OS

        p = Path(file)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"文件不存在：{file}")
        try:
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        except Exception as ex:
            raise HTTPException(status_code=400,
                                detail=f"读取失败：{ex}") from None
        if "ret" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV 缺少 ret 列")

        bench = None
        got = OS.index_history(bench_key, days=20000)
        if got.get("ok"):
            bench = got["series"].pct_change(fill_method=None).dropna()

        res = PV.validate(df["ret"].astype(float), bench,
                          turnover_year=float(turnover), n_trades=trades)
        return {"ok": bool(res.get("ok")), "result": res,
                "text": PV.format_report(res)}

    @app.get("/api/cb_double_low")
    def cb_double_low_view(top: int = 15):
        """双低候选池 + 强赎风险名单。"""
        import cb_double_low as DL
        sc = DL.screener(top=int(top))
        rr = DL.redeem_risk()
        return {"ok": bool(sc.get("ok")), "screen": sc, "redeem": rr,
                "text": DL.format_screen(sc)}

    @app.get("/api/cb_paper")
    def cb_paper_view(step: bool = False, init: bool = False,
                      capital: float = 100000.0):
        """可转债模拟盘：查看 / 初始化 / 推进一步。"""
        import cb_paper as CP

        if init:
            st = CP.init_state(capital)
            return {"ok": True, "state": st, "text": CP.format_state(st)}
        if step:
            r = CP.advance(capital=capital, execute=True)
            return {"ok": bool(r.get("ok")), "step": {
                k: v for k, v in r.items() if k != "state"},
                "state": r.get("state"),
                "text": CP.format_state(r.get("state") or {})}
        st = CP.load_state()
        return {"ok": bool(st.get("capital")), "state": st,
                "text": CP.format_state(st)}

    @app.get("/api/position")
    def position_view(window: int = 20, cap_low: float = 0.30,
                      cap_high: float = 0.80, band: float = 0.0,
                      confirm: int = 1, days: int = 60):
        """
        仓位管理器：沪深300 趋势 → 当日最大仓位上限。

        返回当前状态、历史统计与最近 N 天的明细。
        """
        import position_manager as PM
        try:
            close = PM.load_index("sh000300")
            caps = PM.position_cap(close, window=int(window),
                                   cap_low=float(cap_low),
                                   cap_high=float(cap_high),
                                   band=float(band),
                                   confirm=int(confirm))
            sm = PM.cap_summary(caps)
            tail = caps.tail(int(max(days, 5)))
            series = []
            for d, row in tail.iterrows():
                series.append({
                    "date": str(pd.Timestamp(d).date()),
                    "close": (round(float(row["close"]), 2)
                              if pd.notna(row["close"]) else None),
                    "ma": (round(float(row["ma"]), 2)
                           if pd.notna(row["ma"]) else None),
                    "diff_pct": (round(float(row["diff"]) * 100, 2)
                                 if pd.notna(row["diff"]) else None),
                    "cap": round(float(row["cap"]), 4),
                    "above": bool(row["state"] > 0),
                })
        except Exception as ex:
            raise HTTPException(status_code=500,
                                detail=f"仓位计算失败：{ex}") from None

        params = {"window": int(window), "cap_low": float(cap_low),
                  "cap_high": float(cap_high), "band": float(band),
                  "confirm": int(confirm)}
        txt = PM.format_report(caps, sm, window=int(window),
                               cap_low=float(cap_low),
                               cap_high=float(cap_high))
        return {"ok": True, "params": params, "summary": sm,
                "series": series, "text": txt}

    @app.get("/")
    def index():
        if FRONTEND.exists():
            return FileResponse(FRONTEND, media_type="text/html")
        return JSONResponse({"ok": False, "message": "前端文件缺失"}, status_code=404)

except Exception:  # pragma: no cover - FastAPI 缺失时仍可单独用 run_checkup
    app = None


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    port = int(os.environ.get("QP_PORT", "8421"))
    print(f"体检服务启动: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
