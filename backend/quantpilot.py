# -*- coding: utf-8 -*-
"""
统一入口（M23）
====================================================
之前要跑一套流程得记住 10 条命令、15 个模块名。现在一个入口：

    python backend/quantpilot.py status      一屏看全部状态
    python backend/quantpilot.py portfolio   持仓概览（联网取价）
    python backend/quantpilot.py dca         定投分析（联网取净值）
    python backend/quantpilot.py risk        风控决策 + 记账闭环
    python backend/quantpilot.py daily       日更一次
    python backend/quantpilot.py calibrate   评分效度校准
    python backend/quantpilot.py check       全量自检
    python backend/quantpilot.py serve       启动 Web 看板

设计原则
--------
* `status` 默认**不联网**，随时可跑，不会因为网断而失败
* 每个子命令失败都不影响其他命令
* 缺文件时给出"下一步该做什么"，而不是抛一堆栈
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

__version__ = "0.1.0"

FILES = {
    "holdings.csv": "持仓（code,type,shares,cost）",
    "dca_ledger.csv": "定投台账（date,code,amount,nav）",
    "watchlist.txt": "自选股列表（日更用）",
    "reports/paper_nav.csv": "纸盘净值（日更生成）",
}

COMMANDS: dict[str, tuple[str, callable]] = {}


def command(name: str, help_text: str):
    def deco(fn):
        COMMANDS[name] = (help_text, fn)
        return fn
    return deco


# ══════════════════════════════════════════════════════════
@command("status", "一屏看全部状态（不联网）")
def cmd_status(args) -> int:
    import pandas as pd

    print("=" * 68)
    print(f"QuantPilot 状态  v{__version__}")
    print("=" * 68)

    print("\n[文件]")
    have = 0
    for rel, desc in FILES.items():
        fp = ROOT / rel
        if fp.exists():
            have += 1
            size = fp.stat().st_size
            extra = ""
            if rel.endswith(".csv"):
                try:
                    # 注释行不算数据行（模板里会写说明）
                    lines = [ln for ln in
                             fp.read_text(encoding="utf-8").splitlines()
                             if ln.strip() and not ln.strip().startswith("#")]
                    n = max(len(lines) - 1, 0)      # 减去表头
                    extra = f"  {n} 条"
                except Exception:
                    extra = "  (读取失败)"
            print(f"  [有] {rel:<24}{size:>8} 字节{extra}   {desc}")
        else:
            print(f"  [无] {rel:<24}{'':>12}        {desc}")
    if have == 0:
        print("\n  提示：先按 backend/portfolio.py 或 dca.py 里的示例建文件")

    print("\n[风控记录]")
    try:
        import risk_state as RS
        hist = RS.load_history()
        if hist:
            last = hist[-1]
            print(f"  最近 {last['time']}  动作 {last['action']}  "
                  f"仓位系数 {last['scale']}")
            ft = RS.followthrough(hist)
            if ft.get("ok") and ft.get("follow_rate") is not None:
                print(f"  执行率 {ft['follow_rate'] * 100:.0f}%  {ft['verdict']}")
            else:
                print(f"  {ft.get('verdict') or ft.get('reason')}")
        else:
            print("  还没有记录（跑一次 `risk` 就会开始记账）")
    except Exception as ex:
        print(f"  读取失败：{ex}")

    print("\n[净值]")
    nav_fp = ROOT / "reports" / "paper_nav.csv"
    if nav_fp.exists():
        try:
            nav = pd.read_csv(nav_fp, parse_dates=["date"]).set_index("date")["nav"]
            dd = float((nav / nav.cummax() - 1).iloc[-1])
            print(f"  {len(nav)} 天  {nav.index[0].date()} ~ {nav.index[-1].date()}")
            print(f"  最新 {nav.iloc[-1]:,.2f}   当前回撤 {dd * 100:.2f}%")
        except Exception as ex:
            print(f"  读取失败：{ex}")
    else:
        print("  无（跑一次 `daily` 生成）")

    print()
    print("[下一步建议]")
    tips = []
    if not (ROOT / "holdings.csv").exists():
        tips.append("建 holdings.csv 记录真实持仓")
    if not (ROOT / "dca_ledger.csv").exists():
        tips.append("建 dca_ledger.csv 记录定投")
    if not nav_fp.exists():
        tips.append("跑 daily 生成净值，风控才有数据可算")
    try:
        import risk_state as RS2
        if len(RS2.load_history()) < 2:
            tips.append("多跑几次 risk，闭环才能判断「你有没有照做」")
    except Exception:
        pass
    if tips:
        for t in tips[:4]:
            print("  - " + t)
    else:
        print("  - 一切就绪，建议每天跑一次 risk")

    print("\n[可用命令]")
    for name, (h, _) in sorted(COMMANDS.items()):
        print(f"  {name:<12}{h}")
    print("=" * 68)
    return 0


@command("portfolio", "持仓概览（联网取最新价/净值）")
def cmd_portfolio(args) -> int:
    import portfolio as PF
    fp = ROOT / "holdings.csv"
    if not fp.exists():
        print("没有 holdings.csv。请按 code,type,shares,cost 四列创建。")
        return 2
    data = PF.load_holdings(fp)
    prices = PF.fetch_prices_for(data, refresh=getattr(args, "refresh", False))
    mtm = PF.mark_to_market(data, prices)
    print(PF.format_report(mtm))
    return 0 if mtm.get("ok") else 1


@command("dca", "定投分析（XIRR）")
def cmd_dca(args) -> int:
    import dca as D
    import fund as FD
    fp = ROOT / "dca_ledger.csv"
    if not fp.exists():
        print("没有 dca_ledger.csv。请按 date,code,amount,nav 四列创建。")
        return 2
    led = D.load_ledger(fp)
    codes = sorted({c for c in led["code"].unique() if c})
    navs: dict[str, float] = {}
    series: dict = {}
    for c in codes:
        try:
            sseries = FD.fetch_nav(c)
            series[c] = sseries
            navs[c] = float(sseries.iloc[-1])
        except Exception as ex:
            print(f"  [跳过] {c} 取净值失败：{ex}")
    if series:
        led = D.fill_missing_nav(led, series)
    res = D.dca_analysis(led, navs)
    print(D.format_report(res))
    return 0 if res.get("ok") else 1


@command("risk", "风控决策 + 记账（形成闭环）")
def cmd_risk(args) -> int:
    import pandas as pd
    import riskcontrol as RC
    import risk_state as RS

    nav_fp = ROOT / "reports" / "paper_nav.csv"
    if not nav_fp.exists():
        print("没有净值数据。先跑 `python backend/quantpilot.py daily`。")
        return 2
    nav = pd.read_csv(nav_fp, parse_dates=["date"]).set_index("date")["nav"]

    weights = None
    context: dict = {"capital": getattr(args, "capital", 100000.0)}
    groups = {}
    hp = ROOT / "holdings.csv"
    if hp.exists():
        try:
            import portfolio as PF
            hd = PF.load_holdings(hp)
            prices = PF.fetch_prices_for(hd)
            mtm = PF.mark_to_market(hd, prices)
            w = PF.weights_of(mtm)
            if len(w):
                weights = pd.DataFrame([w.to_dict()], index=[nav.index[-1]])
                groups = PF.groups_for(hd)
                context.update({"weights": w.to_dict(),
                                "cash_ratio": mtm.get("cash_ratio"),
                                "capital": mtm.get("total_value")})
        except Exception as ex:
            print(f"  [警告] 持仓读取失败，改用等权：{ex}")

    turnover = float(getattr(args, "turnover", 12.0))
    decision = RC.kill_switch(nav, weights, groups or None,
                              turnover_annual=turnover,
                              data_ok=True)
    print(RC.format_report(decision))
    rec = RS.record(decision, context)
    print(f"  已记账 {rec['time']}")
    ft = RS.followthrough(RS.load_history())
    print(RS.format_report(RS.load_history(), ft))
    return 0


@command("perturb", "扰动测试（结论稳不稳）")
def cmd_perturb(args) -> int:
    import pandas as pd
    import robustness as RB
    nav_fp = ROOT / "reports" / "paper_nav.csv"
    if not nav_fp.exists():
        print("没有净值数据。先跑 `quantipilot.py daily`。")
        return 2
    nav = pd.read_csv(nav_fp, parse_dates=["date"]).set_index("date")["nav"]
    res = RB.perturbation_suite(nav, RB.PerturbCfg(seed=int(getattr(args, "seed", 7))))
    print(RB.format_perturbation(res))
    return 0 if res.get("ok") else 1


@command("stress", "压力测试（极端情景扛不扛）")
def cmd_stress(args) -> int:
    import pandas as pd
    import robustness as RB
    nav_fp = ROOT / "reports" / "paper_nav.csv"
    if not nav_fp.exists():
        print("没有净值数据。先跑 `quantipilot.py daily`。")
        return 2
    nav = pd.read_csv(nav_fp, parse_dates=["date"]).set_index("date")["nav"]
    bench = None
    try:
        import attribution as AT
        bench = AT.load_bench(start=str(nav.index.min().date()),
                              end=str(nav.index.max().date()))
    except Exception as ex:
        print(f"  [提示] 基准不可用，相关性断裂情景将跳过：{ex}")
    res = RB.stress_suite(nav, bench=bench, cfg=RB.StressCfg(seed=int(getattr(args, "seed", 11))))
    print(RB.format_stress(res))
    return 0 if res.get("ok") else 1


@command("cache", "缓存统计与清理")
def cmd_cache(args) -> int:
    import cache_manager as CM

    if getattr(args, "clean", False):
        res = CM.clean_all(dry_run=not getattr(args, "yes", False))
        if res["dry_run"]:
            print(f"[干跑] 将清空缓存：{res['files']} 个文件，"
                  f"{res['freed_human']}")
            print("  确认无误后加 --yes 真正执行")
        else:
            print(f"已清空缓存：{res['files']} 个文件，释放 {res['freed_human']}")
        return 0

    days = getattr(args, "trim", None)
    if days is not None:
        res = CM.trim(float(days), dry_run=not getattr(args, "yes", False))
        if res["dry_run"]:
            print(f"[干跑] 超过 {days} 天的缓存：{res['candidates']} 个，"
                  f"可释放 {res['freed_human']}")
            for f in res["files"][:10]:
                print("   " + f)
            print("  确认无误后加 --yes 真正执行")
        else:
            print(f"已清理 {res['removed']} 个文件，释放 {res['freed_human']}")
        return 0

    print(CM.format_report(CM.stats()))
    return 0


@command("valuation", "全市场估值快照（zzshare）")
def cmd_valuation(args) -> int:
    import zz_source as ZZ
    info = ZZ.available()
    if not info["available"]:
        print("zzshare 未安装。安装：python -m pip install zzshare")
        return 2
    snap = ZZ.valuation_snapshot(getattr(args, "date", None))
    print(ZZ.format_report(info, snap))
    return 0 if snap.get("ok") else 1


@command("overseas", "海外指数历史（标普/纳指/日经/富时）")
def cmd_overseas(args) -> int:
    import overseas as OS
    if getattr(args, "diagnose", False):
        d = OS.diagnose()
        print("代理:", d["proxy"] or "未设置")
        for c in d["checks"]:
            print(f"  {c['name']:<14} ok={str(c['ok']):<5} "
                  f"status={c['status']:<4} {c['elapsed_s']:>5}s  {c['detail']}")
        print()
        print("结论:", d["verdict"])
        return 0 if any(c["ok"] for c in d["checks"]) else 1
    keys = getattr(args, "keys", None) or ["sp500", "ndx", "nikkei", "ftse"]
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    bad = 0
    for k in keys:
        try:
            res = OS.index_history(k, days=int(getattr(args, "days", 500)))
        except ValueError as ex:
            print(f"[跳过] {ex}")
            bad += 1
            continue
        print(OS.format_report(res))
        if not res.get("ok"):
            bad += 1
    return 0 if bad < len(keys) else 1


@command("industry", "构建股票→行业映射（zzshare 或 baostock）")
def cmd_industry(args) -> int:
    src = str(getattr(args, "source", "zz") or "zz").lower()
    if src.startswith("b"):
        import baostock_source as BS
        if not BS.BS_AVAILABLE:
            print("baostock 未安装")
            return 2
        print("正在拉取证监会行业分类（约 63 秒，缓存 30 天）…")
        got = BS.industry_map(refresh=bool(getattr(args, "refresh", False)))
        BS.close()
        path = ".cache/baostock/industry.csv"
    else:
        import zz_source as ZZ
        if not ZZ.ZZ_AVAILABLE:
            print("zzshare 未安装")
            return 2
        print("正在拉取行业板块成分（约 60 秒，缓存 30 天）…")
        got = ZZ.industry_map(refresh=bool(getattr(args, "refresh", False)))
        path = ".cache/zzshare/industry.csv"

    if not got.get("ok"):
        print(f"失败：{got.get('reason')}")
        return 1
    print(f"完成：{got['n']} 只股票，来源 {got['source']}")
    print(f"缓存：{path}")
    return 0


@command("us", "美股数据（ROIC.ai / xfinlink）")
def cmd_us(args) -> int:
    import us_source as US
    st = US.status()
    if getattr(args, "status", False) or not getattr(args, "ticker", None):
        print(US.format_report(st))
        return 0

    ticker = str(args.ticker).upper()
    what = str(getattr(args, "what", "all") or "all")
    ok = 0
    if what in ("all", "profile"):
        try:
            p = US.roic_profile(ticker)
            print(f"[ROIC] {p['company_name']}  ({p['ticker']})")
            print(f"       行业 {p['sector']} / {p['industry']}   "
                  f"CEO {p.get('ceo')}   员工 {p.get('employees')}")
            print(f"       {p['description'][:150]}")
            ok += 1
        except Exception as ex:
            print(f"[ROIC] 资料失败：{type(ex).__name__}: {str(ex)[:120]}")
    if what in ("all", "latest"):
        try:
            q = US.roic_latest(ticker)
            print(f"[ROIC] 最新 {q['date']}  收盘 {q['close']}  "
                  f"成交 {q['volume']}")
            ok += 1
        except Exception as ex:
            print(f"[ROIC] 最新价失败：{type(ex).__name__}: {str(ex)[:120]}")
    if what in ("all", "prices"):
        try:
            import datetime as _dt
            end = _dt.date.today()
            start = end - _dt.timedelta(days=30)
            r = US.xfin_prices(ticker, str(start), str(end))
            df = r["df"]
            print(f"[xfin] 价格 {r['n']} 行（{r['source']}）")
            cols = [c for c in ("date", "close", "adj_close", "volume")
                    if c in df.columns]
            if cols:
                print(df[cols].tail(3).to_string(index=False))
            ok += 1
        except Exception as ex:
            print(f"[xfin] 价格失败：{type(ex).__name__}: {str(ex)[:120]}")
    if what == "search":
        try:
            r = US.xfin_search(ticker)
            print(f"[xfin] 搜索 {r.get('n')} 条")
            for row in r.get("rows", [])[:10]:
                print("   ", row)
            ok += 1
        except Exception as ex:
            print(f"[xfin] 搜索失败：{type(ex).__name__}: {str(ex)[:120]}")
    return 0 if ok else 1


@command("keys", "查看/设置 API 凭据（只显示掩码）")
def cmd_keys(args) -> int:
    import credentials as CR
    if getattr(args, "set", None):
        name, _, val = str(args.set).partition("=")
        if not val:
            print("用法：keys --set roicai=你的Key")
            return 2
        r = CR.save_key(name.strip(), val.strip())
        print(f"已保存 {name} = {r['masked']}  →  {r['path']}")
        return 0
    print(CR.format_report(CR.status()))
    return 0


@command("quality", "基本面质量评分（美股，ROIC.ai）")
def cmd_quality(args) -> int:
    import quality as QL
    ticker = str(getattr(args, "ticker", "") or "").upper()
    if not ticker:
        print("用法：quality --ticker AAPL")
        return 2
    print(f"正在拉取 {ticker} 财报（三张表，约 36 秒；缓存命中则秒回）…")
    r = QL.quality_report(ticker, refresh=bool(getattr(args, "refresh", False)))
    print(QL.format_report(r))
    return 0 if r.get("ok") else 1


@command("advice", "今日决策：该投多少 / 该不该调仓")
def cmd_advice(args) -> int:
    import advisor as AD
    import dca as D
    import fund as FD
    import portfolio as PF

    hp = ROOT / "holdings.csv"
    if not hp.exists():
        print("没有 holdings.csv。先填你的持仓（code,type,shares,cost）。")
        return 2

    data = PF.load_holdings(hp)
    codes = [h.code for h in data["holdings"]]
    if not codes:
        print("持仓为空")
        return 2

    # 基金名称（用于匹配基准指数）
    names: dict[str, str] = {}
    try:
        cat = FD.load_catalog()
        for c in codes:
            info = FD.lookup_fund(c, cat)
            if info:
                names[c] = info["name"]
    except Exception as ex:
        print(f"  [提示] 基金名录不可用，将无法匹配指数：{str(ex)[:60]}")

    # 权重（用最新净值/价格）
    weights = None
    try:
        prices = PF.fetch_prices_for(data)
        mtm = PF.mark_to_market(data, prices)
        if mtm.get("ok"):
            weights = dict(mtm.get("weights") or {})
            if mtm.get("cash_ratio"):
                weights["CASH"] = mtm["cash_ratio"]
    except Exception as ex:
        print(f"  [提示] 取价失败，跳过再平衡检查：{str(ex)[:60]}")

    # 组合回撤（来自纸盘净值）
    dd = None
    nav_fp = ROOT / "reports" / "paper_nav.csv"
    if nav_fp.exists():
        try:
            import pandas as pd
            nav = pd.read_csv(nav_fp, parse_dates=["date"]).set_index("date")["nav"]
            dd = float((nav / nav.cummax() - 1).iloc[-1])
        except Exception:
            dd = None

    monthly = float(getattr(args, "monthly", 1000.0) or 1000.0)
    if dd is None:
        dd = getattr(args, "drawdown", None)

    res = AD.advise(codes, names, weights, monthly_amount=monthly,
                    portfolio_drawdown=(None if dd is None else float(dd)),
                    use_ml=bool(getattr(args, "ml", False)))
    print(AD.format_report(res))

    # 记账：把这次的动作写入决策台账，到期后自动结算
    if getattr(args, "record", False) and res.get("ok"):
        import journal as JN
        per = (res.get("details") or {}).get("index_percentiles") or {}
        by_name = {v["name"]: k for k, v in per.items() if v.get("ok")}
        n_rec = 0
        for a in res.get("actions") or []:
            key = by_name.get(a.get("target"))
            price = None
            if key and per.get(key, {}).get("current"):
                price = per[key]["current"]
            if price is None:
                continue
            JN.record(key, a["action"], price=price,
                      amount=a.get("amount"), name=a.get("target"),
                      reason=a.get("reason"), horizon=20, source="advice")
            n_rec += 1
        if n_rec:
            print(f"\n已记账 {n_rec} 条 → .state/decisions.jsonl"
                  f"（{JN.DEFAULT_HORIZON} 个交易日后可结算）")
        else:
            print()
            print("[提示] 未记账：当前持仓里没有能匹配上海外指数（标普/纳指/日经/富时）的基金，"
                  "或动作是组合级的，缺少可结算的参考价。")
            print("       要逐条记账，请把 QDII 指数基金填进 holdings.csv。")

    return 0 if res.get("ok") else 1


@command("ml", "回撤风险预测（机器学习，逐市场验证）")
def cmd_ml(args) -> int:
    import ml_risk as ML
    keys = getattr(args, "keys", None) or ["sp500", "nikkei", "ndx", "ftse"]
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    for k in keys:
        val = ML.validate_market(k, refresh=bool(getattr(args, "refresh", False)))
        if not val.get("ok"):
            print(f"[{k}] 验证失败：{val.get('reason')}")
            continue
        cur = ML.current_risk(k) if val.get("usable") else None
        print(ML.format_report(val, cur))
    return 0


@command("optimize", "马科维茨 / 凯利 能不能用（实测）")
def cmd_optimize(args) -> int:
    import optimizer as OP
    import overseas as OS

    KEYS = {"sp500": "标普500", "ndx": "纳指100",
            "nikkei": "日经225", "ftse": "富时100"}
    keys = getattr(args, "keys", None) or list(KEYS)
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]

    navs = {}
    for k in keys:
        got = OS.index_history(k, days=20000)
        if got.get("ok"):
            navs[got["name"]] = got["series"]
        else:
            print(f"  [跳过] {k}：{got.get('reason')}")
    if len(navs) < 2:
        print("可用数据不足 2 个指数，无法做配置分析。")
        return 2

    R = OP.to_returns(navs)
    rep = OP.optimize_report(R)
    print(OP.format_report(rep))
    return 0 if rep.get("ok") else 1


@command("journal", "决策台账：记录 / 结算 / 复盘")
def cmd_journal(args) -> int:
    import journal as JN

    if getattr(args, "settle", False):
        st = JN.settle_pending()
        print(f"结算完成：本次结算 {st.get('n_settled', 0)} 条，"
              f"仍待结算 {st.get('pending', 0)} 条")
        for e in (st.get("errors") or [])[:5]:
            print(f"  [跳过] {e}")

    if getattr(args, "review", False):
        print(JN.format_report(JN.review()))
        return 0

    print(JN.format_journal(JN.load_journal(),
                            limit=int(getattr(args, "limit", 15) or 15)))
    if not getattr(args, "settle", False):
        pend = sum(1 for r in JN.load_journal() if not r.get("outcome"))
        if pend:
            print(f"  还有 {pend} 条待结算，到期后跑 "
                  f"`journal --settle` 或 `journal --review`")
    return 0


@command("cb", "可转债打新扫描")
def cmd_cb(args) -> int:
    import cb_new as CB
    try:
        res = CB.scan(days_ahead=int(getattr(args, "days", 7) or 7),
                      refresh=bool(getattr(args, "refresh", False)))
    except Exception as ex:
        print(f"取数失败：{ex}")
        return 1
    print(CB.format_report(res))
    return 0


@command("fee", "费率核对（A 类 vs C 类）")
def cmd_fee(args) -> int:
    import fee_check as FE
    res = FE.compare_ac(
        monthly=float(getattr(args, "monthly", 4000) or 4000),
        years=float(getattr(args, "years", 20) or 20),
        annual_return=float(getattr(args, "rate", 0.07) or 0.07),
        a_purchase=getattr(args, "a_purchase", None),
        c_service=getattr(args, "c_service", None))
    imp = FE.fee_impact(
        monthly=float(getattr(args, "monthly", 4000) or 4000),
        years=float(getattr(args, "years", 20) or 20),
        annual_return=float(getattr(args, "rate", 0.07) or 0.07),
        fee_low=0.0060, fee_high=0.0110)
    print(FE.format_report(res, imp))
    return 0


@command("admit", "主动交易准入判定（五道闸门）")
def cmd_admit(args) -> int:
    import pandas as pd
    import paper_validate as PV
    import overseas as OS

    path = getattr(args, "file", None)
    if not path:
        print("用法：admit --file returns.csv [--bench-key sp500] "
              "[--turnover 3.0] [--trades 60]")
        print("  returns.csv 需要两列：date,ret（ret 为日收益率小数）")
        return 2

    try:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    except Exception as ex:
        print(f"读取失败：{ex}")
        return 1
    if "ret" not in df.columns:
        print("CSV 缺少 ret 列")
        return 2
    rets = df["ret"].astype(float)

    bench = None
    key = getattr(args, "bench_key", None)
    if key:
        got = OS.index_history(key, days=20000)
        if got.get("ok"):
            bench = got["series"].pct_change(fill_method=None).dropna()
        else:
            print(f"  基准 {key} 取不到，第三道闸门将不通过")

    res = PV.validate(rets, bench,
                      turnover_year=float(getattr(args, "turnover", 0) or 0),
                      n_trades=getattr(args, "trades", None))
    print(PV.format_report(res))
    return 0 if res.get("all_pass") else 1


@command("cbdl", "可转债双低：候选池 / 强赎 / 回测")
def cmd_cbdl(args) -> int:
    import cb_double_low as DL

    if getattr(args, "backtest", False):
        import pandas as pd
        df = DL.load_current()
        price = pd.to_numeric(df["债现价"], errors="coerce")
        codes = df.loc[price > 0, "债券代码"].astype(str).str.zfill(6).tolist()
        n = int(getattr(args, "bonds", 100) or 100)
        print(f"拉取 {n} 只转债历史（首次约 1 分钟，之后走缓存）…")
        h = DL.fetch_history(codes, max_bonds=n,
                             refresh=bool(getattr(args, "refresh", False)))
        print(f"成功 {h['n_ok']} 只，失败 {h['n_fail']} 只")
        if h["n_ok"] < 10:
            print("可用样本太少，无法回测")
            return 1
        px, pm = DL.build_panel(h["history"])
        bt = DL.backtest_double_low(px, pm,
                                    top_n=int(getattr(args, "top", 10) or 10),
                                    rebalance_days=int(
                                        getattr(args, "rebalance", 20) or 20))
        print(DL.format_backtest(bt))
        print("\n  ⚠ 这份回测**不能直接当结论**：")
        print("     样本只有当前仍存续的转债，已退市/强赎的缺失。")
        return 0

    sc = DL.screener(top=int(getattr(args, "top", 15) or 15),
                     refresh=bool(getattr(args, "refresh", False)))
    print(DL.format_screen(sc))
    rr = DL.redeem_risk()
    print(f"\n强赎风险：{rr['n']} 只转股价值 ≥{DL.REDEEM_LINE:.0f}")
    for x in rr["items"][:5]:
        print(f"  {x['code']} {x['name']:<8} 价 {x['price']:>7.2f}  "
              f"转股价值 {x['convert_value']:>7.1f}")
    return 0


@command("cbpaper", "可转债模拟盘（含滑点与冲击成本）")
def cmd_cbpaper(args) -> int:
    import cb_paper as CP

    if getattr(args, "cost", False):
        print("=" * 72)
        print("转债交易成本模型")
        print("=" * 72)
        print(f"  佣金 {CP.COMMISSION_RATE*10000:.1f} 万（最低 {CP.COMMISSION_MIN} 元），无印花税")
        print(f"  半价差 {CP.HALF_SPREAD_BP:.1f} bp")
        print(f"  冲击 = {CP.IMPACT_K} × sqrt(参与率) × 10 bp")
        print("-" * 72)
        print("  " + f"{'下单额':>12}{'日成交额':>14}{'参与率':>10}{'成本bp':>9}{'标记':>12}")
        for val, amt in ((10000, 2.0e8), (100000, 2.0e8), (100000, 9e7),
                         (500000, 9e7), (2000000, 3e7)):
            c = CP.order_cost(val, amt)
            mark = "需拆单" if c["too_large"] else ""
            if c["unexecutable"]:
                mark = "不可执行"
            print(f"  {val:>12,}{amt:>14,.0f}{c['participation']*100:>9.3f}%"
                  f"{c['total_bp']:>9.1f}{mark:>12}")
        print("=" * 72)
        return 0

    if getattr(args, "init", False):
        st = CP.init_state(float(getattr(args, "capital", 100000) or 100000))
        print(f"已初始化模拟盘，本金 {st['capital']:,.0f} 元 → {CP.PAPER_FILE}")
        print(CP.format_state(st))
        return 0

    if getattr(args, "step", False):
        r = CP.advance(capital=float(getattr(args, "capital", 100000) or 100000),
                       execute=not getattr(args, "dry", False),
                       force_rebalance=bool(getattr(args, "rebalance", False)))
        if not r.get("ok"):
            print(f"推进失败：{r.get('reason')}")
            return 1
        print(f"{r['date']}  净值 {r['nav_before']:,.0f} → {r['nav_after']:,.0f}"
              f"   {'已调仓' if r['rebalanced'] else '无调仓'}"
              f"   成交 {r['n_executed']} 笔   成本 {r['cost']:.2f} 元")
        for s in r.get("skipped") or []:
            print(f"  [跳过] {s}")
        print(CP.format_state(r["state"]))
        return 0

    print(CP.format_state(CP.load_state()))
    return 0


@command("strict", "严格回测：审计未来函数 / 样本外检验")
def cmd_strict(args) -> int:
    import backtest_strict as BS

    n = int(getattr(args, "n", 60) or 60)
    codes = BS.default_universe(n)
    print(f"股票池 {len(codes)} 只，拉取 2019-2024 日线…")
    got = BS.fetch_panel(codes, "2019-01-02", "2024-12-31",
                         refresh=bool(getattr(args, "refresh", False)))
    print(f"  成功 {got['n_ok']} 只，失败 {got['n_missing']} 只")
    if got["n_ok"] < 20:
        print("样本太少，终止")
        return 1
    panel = got["panel"]

    if not getattr(args, "no_audit", False):
        print()
        print(BS.format_audit(BS.audit_panel(panel)))

    if getattr(args, "oos", False):
        print()
        res = BS.tune_oos(
            panel,
            is_range=("2019-01-02", "2021-12-31"),
            oos_range=("2022-01-04", "2024-12-31"),
            grid={"lookback": [20, 60, 120], "top_n": [10, 20],
                  "rebalance": [10, 20]},
            verbose=True)
        print()
        print(BS.format_oos(res))
        if res.get("ok"):
            import pandas as pd
            gr = pd.DataFrame(res["grid_results"]).sort_values(
                "is_sharpe", ascending=False).head(8)
            print("\n  参数组合（按 IS 夏普排序）：")
            print(gr.to_string(index=False))
    return 0


@command("lab", "多策略族样本外横评（动量/反转/低波/价值代理）")
def cmd_lab(args) -> int:
    import backtest_strict as BS
    import strategy_lab as SL

    n = int(getattr(args, "n", 60) or 60)
    codes = BS.default_universe(n)
    print(f"股票池 {len(codes)} 只，拉取 2019-2024 日线…")
    got = BS.fetch_panel(codes, "2019-01-02", "2024-12-31",
                         refresh=bool(getattr(args, "refresh", False)))
    print(f"  成功 {got['n_ok']} 只，失败 {got['n_missing']} 只")
    if got["n_ok"] < 20:
        print("样本太少，终止")
        return 1
    panel = got["panel"]

    if not getattr(args, "no_audit", False):
        print()
        print(BS.format_audit(BS.audit_panel(panel)))

    print("\n四因子横评（IS 2019-2021 调参 → OOS 2022-2024 检验）")
    res = SL.run_matrix(
        panel,
        is_range=("2019-01-02", "2021-12-31"),
        oos_range=("2022-01-04", "2024-12-31"),
        grid={"lookback": [20, 60, 120], "top_n": [10, 20],
              "rebalance": [10, 20]},
        verbose=True)
    print()
    print(SL.format_matrix(res))
    print()
    print(SL.format_ic(res))
    return 0


@command("pos", "仓位管理器（大盘趋势过滤）")
def cmd_pos(args) -> int:
    import position_manager as PM

    close = PM.load_index(str(getattr(args, "index", "sh000300") or "sh000300"),
                          refresh=bool(getattr(args, "refresh", False)))
    caps = PM.position_cap(
        close,
        window=int(getattr(args, "window", 20) or 20),
        cap_low=float(getattr(args, "cap_low", 0.30) or 0.30),
        cap_high=float(getattr(args, "cap_high", 0.80) or 0.80),
        band=float(getattr(args, "band", 0.0) or 0.0),
        confirm=int(getattr(args, "confirm", 1) or 1))
    sm = PM.cap_summary(caps)
    print(PM.format_report(caps, sm,
                           window=int(getattr(args, "window", 20) or 20),
                           cap_low=float(getattr(args, "cap_low", 0.30) or 0.30),
                           cap_high=float(getattr(args, "cap_high", 0.80) or 0.80)))
    if getattr(args, "save", False):
        import pandas as pd
        out = ROOT / "reports" / "position_cap.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        caps[["close", "ma", "state", "cap"]].to_csv(out, encoding="utf-8-sig")
        print(f"  已写入 {out}")
    return 0


@command("sched", "每日任务调度（15:30 清洗 / 20:00 交易清单）")
def cmd_sched(args) -> int:
    import daily_scheduler as DS

    DS.setup_logging()
    if getattr(args, "list", False):
        print(f"  数据清洗    每天 {DS.CLEAN_TIME}")
        print(f"  交易清单    每天 {DS.TRADE_TIME}")
        return 0
    if getattr(args, "now", False):
        job = getattr(args, "job", None)
        jobs = [job] if job else ["clean", "trade"]
        for j in jobs:
            print(f"\n=== 立即执行 {j} ===")
            print(DS.trigger(j, write=not getattr(args, "no_write", False)))
        return 0
    DS.run_forever()
    return 0


@command("mail", "邮件推送：检查配置 / 预览 / 发测试信")
def cmd_mail(args) -> int:
    import mailer as ML

    if getattr(args, "help_config", False) or not any(
            [getattr(args, "check", False), getattr(args, "preview", False),
             getattr(args, "send", False), getattr(args, "test", False)]):
        print(ML.format_config_help())
        return 0

    if getattr(args, "check", False):
        g = ML.load_config()
        print(f"  配置来源：{g['source']}")
        for k, v in (g.get("display") or {}).items():
            print(f"    {k:<12}{v}")
        print(f"  状态：{'完整，可以发信' if g['ok'] else '不完整 → ' + str(g.get('error'))}")
        return 0 if g["ok"] else 1

    if getattr(args, "preview", False):
        demo_rows = [{"rank": 1, "code": "600519", "target_weight": 0.03,
                      "ref_close": 1410.27, "score": -0.0137}]
        pos = {"current_state": "均线下方", "current_cap": 0.30,
               "close": 4016.29, "ma": 4069.52, "diff_pct": -1.31}
        r = ML.render_daily(trade={"csv": None}, position=pos,
                            clean=None, trade_rows=demo_rows)
        print("主题：", r["subject"])
        print("-" * 60)
        print(r["text"])
        out = ROOT / ".tmp" / "mail_preview.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r["html"], encoding="utf-8")
        print("-" * 60)
        print(f"HTML 预览：{out}")
        return 0

    # 真实发送
    import daily_scheduler as DS
    import position_manager as PM

    prof = getattr(args, "provider", None)
    if prof:
        if prof not in ML.PROVIDERS:
            print(f"未知服务商 {prof}，可选：{', '.join(ML.PROVIDERS)}")
            return 2
        pv = ML.PROVIDERS[prof]
        print(f"  提示：{prof} → {pv['smtp_host']}:{pv['smtp_port']} "
              f"ssl={pv['use_ssl']} tls={pv['use_tls']}")
        print(f"  {pv['note']}")

    if getattr(args, "test", False):
        r = ML.render_daily(trade={"csv": None}, position=None, clean=None,
                            trade_rows=[])
        out = ML.send_mail("[QuantPilot] 测试邮件", r["html"], r["text"])
        print(out)
        return 0 if out.get("ok") else 1

    # send：发今天的真实清单（若存在）
    import pandas as pd
    rows = []
    csv_path = None
    try:
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y%m%d")
        csv = ROOT / "reports" / f"trade_list_{today}.csv"
        if csv.exists():
            # 必须按字符串读，否则 000625 会变成 625
            rows = pd.read_csv(csv, dtype={"code": str}).to_dict("records")
            csv_path = str(csv)
            print(f"  读取今日清单 {len(rows)} 条：{csv.name}")
        else:
            print("  今日没有清单文件，将发送仓位状态 + 说明")
    except Exception as ex:
        print(f"  读取清单失败：{ex}")

    pos = None
    try:
        close = PM.load_index("sh000300")
        caps = PM.position_cap(close, band=0.02)
        sm = PM.cap_summary(caps)
        tail = caps.iloc[-1]
        pos = {"current_state": sm["current_state"],
               "current_cap": sm["current_cap"],
               "close": round(float(tail["close"]), 2),
               "ma": (round(float(tail["ma"]), 2)
                      if pd.notna(tail["ma"]) else None),
               "diff_pct": (round(float(tail["diff"]) * 100, 2)
                            if pd.notna(tail["diff"]) else None)}
    except Exception as ex:
        print(f"  仓位状态取数失败：{ex}")

    out = ML.send_daily_report(trade={"csv": csv_path}, position=pos,
                               trade_rows=rows)
    print(out)
    return 0 if out.get("ok") else 1


@command("dash", "启动 Streamlit 看板")
def cmd_dash(args) -> int:
    import subprocess
    import sys as _sys

    page = ROOT / "frontend" / "dashboard.py"
    if not page.exists():
        print(f"找不到页面：{page}")
        return 2
    port = int(getattr(args, "port", 8501) or 8501)
    print(f"启动 Streamlit → http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止")
    try:
        # fileWatcher 会持续扫描工作区，和测试/自检抢资源（实测让一套测试
        # 从 4 秒变成超时 600 秒），这里默认关掉。
        cmd = [_sys.executable, "-m", "streamlit", "run", str(page),
               "--server.port", str(port),
               "--server.fileWatcherType", "none",
               "--browser.gatherUsageStats", "false"]
        if getattr(args, "watch", False):
            cmd = [c for i, c in enumerate(cmd)
                   if not (c == "--server.fileWatcherType"
                           or (i > 0 and cmd[i - 1] == "--server.fileWatcherType"))]
            cmd += ["--server.fileWatcherType", "auto"]
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("已停止")
    return 0


@command("feed", "多市场取数（A股/港股/美股/加密货币，自动降级）")
def cmd_feed(args) -> int:
    import feed as FD

    market = str(getattr(args, "market", "") or "").lower()
    code = str(getattr(args, "code", "") or "")

    if market and code:
        r = FD.get_daily(market, code, refresh=bool(getattr(args, "refresh", False)))
        if r["ok"]:
            print(f"{market} {code}: {len(r['df'])} 行  来源 {r['source']}")
            print(r["df"].tail(3).to_string(index=False))
        else:
            print(f"{market} {code}: 失败")
            for a in r.get("attempts") or []:
                print("  ", a)
        return 0 if r["ok"] else 1

    print("四市场冒烟测试（A股 600519 / 港股 00700 / 美股 AAPL / 加密 BTC）")
    print()
    df = FD.smoke_test()
    print(df.to_string(index=False))
    st = FD.source_status()
    if st:
        print(f"  熔断状态: {st}")
    print(f"  {int(df['OK'].sum())}/{len(df)} 市场可用")
    return 0


@command("ops", "运维：体检 / 任务 / 备份 / 审计")
def cmd_ops(args) -> int:
    import opshealth as OH
    import opsjobs as OJ
    import opslog as OL

    act = str(getattr(args, "action", "health") or "health")
    if act == "health":
        res = OH.run_checks()
        print(OH.format_report(res))
        return 0 if res["ok"] else 1
    if act == "jobs":
        df = OJ.list_jobs()
        print(df.to_string(index=False) if len(df) else "（未注册任务）")
        return 0
    if act == "run":
        name = getattr(args, "name", None)
        if not name:
            print("需要 --name")
            return 2
        print(OJ.run_job(name, force=True))
        return 0
    if act == "backup":
        r = OH.backup(keep=int(getattr(args, "keep", 7) or 7))
        print(r)
        return 0 if r.get("ok") else 1
    if act == "audit":
        df = OL.query(limit=int(getattr(args, "limit", 20) or 20))
        print(df.to_string(index=False) if len(df) else "（无记录）")
        return 0
    print(f"未知动作 {act}")
    return 2


@command("daily", "跑一次日更")
def cmd_daily(args) -> int:
    import daily_job as DJ
    res = DJ.run_daily(notify=False)
    ok = bool(res.get("ok", True)) if isinstance(res, dict) else True
    print("日更完成" if ok else f"日更异常：{res}")
    return 0 if ok else 1


@command("calibrate", "评分效度校准")
def cmd_calibrate(args) -> int:
    import warnings
    warnings.filterwarnings("ignore")
    import calibration as CAL
    res = CAL.calibrate(seeds=(7,))
    print(CAL.format_report(res))
    return 0 if res.get("ok") else 1


@command("check", "全量自检")
def cmd_check(args) -> int:
    import preflight as PRE
    fn = getattr(PRE, "main", None)
    if callable(fn):
        rc = fn()
        return int(rc) if isinstance(rc, int) else 0
    print("preflight 未提供 main()")
    return 1


@command("serve", "启动 Web 看板")
def cmd_serve(args) -> int:
    import api_v2 as API
    port = int(getattr(args, "port", 8421))
    import logging
    logdir = ROOT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(logdir / "quantpilot.log",
                                      encoding="utf-8"),
                  logging.StreamHandler()])
    logging.getLogger(__name__).info("服务启动，端口 %s", port)
    print(f"启动 http://127.0.0.1:{port}   日志 logs/quantpilot.log")
    import uvicorn
    uvicorn.run(API.app, host="127.0.0.1", port=port, log_level="warning")
    return 0


# ══════════════════════════════════════════════════════════
def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="quantpilot",
        description="QuantPilot 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python backend/quantpilot.py status\n"
               "  python backend/quantpilot.py risk --turnover 12\n"
               "  python backend/quantpilot.py serve --port 8421")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("portfolio", help="持仓概览")
    s.add_argument("--refresh", action="store_true", help="强制刷新价格")

    sub.add_parser("dca", help="定投分析")

    s = sub.add_parser("risk", help="风控决策 + 记账")
    s.add_argument("--turnover", type=float, default=12.0, help="年换手倍数")
    s.add_argument("--capital", type=float, default=100000.0, help="组合规模")

    sub.add_parser("daily", help="跑一次日更")

    s = sub.add_parser("feed", help="多市场取数（自动降级）")
    s.add_argument("--market", type=str, default=None,
                   choices=["cn", "hk", "us", "crypto"])
    s.add_argument("--code", type=str, default=None)
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("ops", help="运维：体检/任务/备份/审计")
    s.add_argument("--action", default="health",
                   choices=["health", "jobs", "run", "backup", "audit"])
    s.add_argument("--name", type=str, default=None)
    s.add_argument("--keep", type=int, default=7)
    s.add_argument("--limit", type=int, default=20)

    s = sub.add_parser("dash", help="启动 Streamlit 看板")
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--watch", action="store_true",
                   help="开启文件监视（改代码自动重载，但会与测试抢资源）")

    s = sub.add_parser("mail", help="邮件推送")
    s.add_argument("--check", action="store_true")
    s.add_argument("--preview", action="store_true")
    s.add_argument("--send", action="store_true")
    s.add_argument("--test", action="store_true")
    s.add_argument("--help-config", dest="help_config",
                   action="store_true")
    s.add_argument("--provider", type=str, default=None)

    s = sub.add_parser("pos", help="仓位管理器（大盘趋势）")
    s.add_argument("--index", type=str, default="sh000300")
    s.add_argument("--window", type=int, default=20)
    s.add_argument("--cap-low", dest="cap_low", type=float, default=0.30)
    s.add_argument("--cap-high", dest="cap_high", type=float, default=0.80)
    s.add_argument("--band", type=float, default=0.0)
    s.add_argument("--confirm", type=int, default=1)
    s.add_argument("--save", action="store_true")
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("sched", help="每日任务调度")
    s.add_argument("--list", action="store_true")
    s.add_argument("--now", action="store_true")
    s.add_argument("--job", choices=["clean", "trade"], default=None)
    s.add_argument("--no-write", dest="no_write", action="store_true")

    s = sub.add_parser("lab", help="多策略族样本外横评")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--no-audit", dest="no_audit",
                   action="store_true")
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("strict", help="严格回测：审计+样本外")
    s.add_argument("--oos", action="store_true",
                   help="跑样本外检验（2019-21 调参 / 2022-24 检验）")
    s.add_argument("--no-audit", dest="no_audit",
                   action="store_true")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("cbpaper", help="可转债模拟盘")
    s.add_argument("--init", action="store_true")
    s.add_argument("--step", action="store_true")
    s.add_argument("--status", action="store_true")
    s.add_argument("--cost", action="store_true")
    s.add_argument("--dry", action="store_true",
                   help="只标记净值，不成交")
    s.add_argument("--rebalance", action="store_true",
                   help="强制调仓")
    s.add_argument("--capital", type=float, default=100000)

    s = sub.add_parser("cbdl", help="可转债双低：候选池/强赎/回测")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--backtest", action="store_true")
    s.add_argument("--bonds", type=int, default=100)
    s.add_argument("--rebalance", type=int, default=20)
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("cb", help="可转债打新扫描")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("fee", help="费率核对 A/C 类")
    s.add_argument("--monthly", type=float, default=4000)
    s.add_argument("--years", type=float, default=20)
    s.add_argument("--rate", type=float, default=0.07)
    s.add_argument("--a-purchase", dest="a_purchase",
                   type=float, default=None)
    s.add_argument("--c-service", dest="c_service",
                   type=float, default=None)

    s = sub.add_parser("admit", help="主动交易准入判定")
    s.add_argument("--file", type=str, default=None)
    s.add_argument("--bench-key", dest="bench_key",
                   type=str, default="sp500")
    s.add_argument("--turnover", type=float, default=0.0)
    s.add_argument("--trades", type=int, default=None)

    s = sub.add_parser("journal", help="决策台账：记录/结算/复盘")
    s.add_argument("--settle", action="store_true",
                   help="结算所有到期决策")
    s.add_argument("--review", action="store_true",
                   help="按动作分组复盘胜率")
    s.add_argument("--limit", type=int, default=15)

    s = sub.add_parser("optimize", help="马科维茨/凯利实测")
    s.add_argument("--keys", type=str, default=None,
                   help="逗号分隔：sp500,ndx,nikkei,ftse")

    s = sub.add_parser("ml", help="回撤风险预测（ML）")
    s.add_argument("--keys", type=str, default=None,
                   help="逗号分隔：sp500,nikkei,ndx,ftse")
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("advice", help="今日决策（实盘用）")
    s.add_argument("--monthly", type=float, default=1000.0,
                   help="月定投基数（元）")
    s.add_argument("--drawdown", type=float, default=None,
                   help="组合当前回撤，如 -0.12；缺省用纸盘净值")
    s.add_argument("--ml", action="store_true",
                   help="叠加机器学习回撤风险信号")
    s.add_argument("--record", action="store_true",
                   help="把本次决策写入台账（到期自动结算）")

    s = sub.add_parser("quality", help="基本面质量评分")
    s.add_argument("--ticker", type=str, required=True)
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("us", help="美股数据（ROIC/xfinlink）")
    s.add_argument("--ticker", type=str, default=None, help="如 AAPL")
    s.add_argument("--what", type=str, default="all",
                   choices=["all", "profile", "latest", "prices", "search"])
    s.add_argument("--status", action="store_true")

    s = sub.add_parser("keys", help="API 凭据（掩码显示）")
    s.add_argument("--set", type=str, default=None,
                   help="roicai=Key / xfinlink=Key")

    s = sub.add_parser("overseas", help="海外指数历史")
    s.add_argument("--keys", type=str, default=None,
                   help="逗号分隔：sp500,ndx,nikkei,ftse")
    s.add_argument("--days", type=int, default=500)
    s.add_argument("--diagnose", action="store_true",
                   help="诊断各数据源连通性")

    s = sub.add_parser("valuation", help="全市场估值快照")
    s.add_argument("--date", type=str, default=None)

    s = sub.add_parser("industry", help="构建行业映射")
    s.add_argument("--source", type=str, default="zz",
                   help="zz=zzshare（板块）/ bs=baostock（证监会）")
    s.add_argument("--refresh", action="store_true")

    s = sub.add_parser("cache", help="缓存统计与清理")
    s.add_argument("--trim", type=float, default=None,
                   help="删除超过 N 天未用的缓存")
    s.add_argument("--clean", action="store_true", help="清空全部缓存")
    s.add_argument("--yes", action="store_true", help="真的执行（默认干跑）")

    s = sub.add_parser("perturb", help="扰动测试")
    s.add_argument("--seed", type=int, default=7)

    s = sub.add_parser("stress", help="压力测试")
    s.add_argument("--seed", type=int, default=11)
    sub.add_parser("calibrate", help="评分效度校准")
    sub.add_parser("check", help="全量自检")
    sub.add_parser("status", help="一屏状态")

    s = sub.add_parser("serve", help="启动 Web 看板")
    s.add_argument("--port", type=int, default=8421)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as ex:
        # argparse 遇到非法子命令会直接 sys.exit()。
        # 作为可被 import 的库，main() 不该杀进程，统一转成返回码。
        return int(ex.code or 2)

    if not args.cmd:
        args.cmd = "status"

    entry = COMMANDS.get(args.cmd)
    if entry is None:
        parser.print_help()
        return 2

    try:
        return int(entry[1](args) or 0)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as ex:
        print(f"\n[失败] {args.cmd}: {ex}")
        if "--traceback" in (argv or []):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
