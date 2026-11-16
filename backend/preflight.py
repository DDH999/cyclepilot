# -*- coding: utf-8 -*-
"""
一键自检（preflight）
====================================================
给使用者 / 评委 / 答辩前用的总检查：环境、依赖、文件、测试、产物、服务。

用法
----
    python backend/preflight.py            # 全量检查
    python backend/preflight.py --quick    # 跳过测试（只看环境与文件）

退出码：0 全通过；1 有失败项。
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
TESTS = ROOT / "tests"

__version__ = "0.1.0"

OK, WARN, BAD = "OK", "WARN", "FAIL"
_rows: list[tuple[str, str, str]] = []


def add(name: str, status: str, detail: str = "") -> None:
    _rows.append((name, status, detail))


def _icon(st: str) -> str:
    return {OK: "[ OK ]", WARN: "[WARN]", BAD: "[FAIL]"}[st]


# ══════════════════════════════════════════════════════════
def check_env() -> None:
    v = sys.version_info
    add("Python 版本", OK if v >= (3, 10) else BAD,
        f"{v.major}.{v.minor}.{v.micro}（需 ≥3.10）")
    add("工作目录", OK if BACKEND.exists() else BAD, str(ROOT))


def check_deps() -> None:
    required = ["numpy", "pandas", "fastapi", "uvicorn", "pydantic"]
    optional = {"scipy": "PBO/DSR 统计", "akshare": "日更拉取行情", "httpx": "API 测试"}
    for m in required:
        try:
            mod = importlib.import_module(m)
            add(f"依赖 {m}", OK, getattr(mod, "__version__", ""))
        except Exception as ex:
            add(f"依赖 {m}", BAD, str(ex)[:60])
    for m, why in optional.items():
        try:
            importlib.import_module(m)
            add(f"可选 {m}", OK, why)
        except Exception:
            add(f"可选 {m}", WARN, f"缺失（影响：{why}）")


def check_files() -> None:
    core = ["validator.py", "stress_tester.py", "purity_tester.py",
            "adjudicator.py", "scoring.py", "schemas.py",
            "ai_diagnostician.py", "template_engine.py",
            "api_v2.py", "daily_job.py", "attribution.py",
            "compound.py", "valuation.py", "blackswan.py",
            "datacheck.py", "walkforward.py", "execution.py",
            "riskcontrol.py", "portfolio_theory.py", "allocation.py",
            "timeseries.py", "pitfalls.py", "portfolio.py", "fund.py", "dca.py", "calibration.py", "risk_state.py", "quantpilot.py", "strategy_runner.py", "datasource.py", "robustness.py", "cache_manager.py", "zz_source.py", "overseas.py", "baostock_source.py", "credentials.py", "us_source.py", "quality.py", "advisor.py", "ml_risk.py", "optimizer.py", "journal.py",
    "cb_new.py", "fee_check.py", "paper_validate.py", "cb_double_low.py", "cb_paper.py", "backtest_strict.py", "strategy_lab.py", "position_manager.py", "daily_scheduler.py", "mailer.py", "dash_data.py", "store.py", "data_quality.py", "ingest.py", "feed.py",
        "tushare_source.py", "bulk_ingest.py",
        "param_walkforward.py", "versions.py",
        "risk_budget.py", "risk_control.py", "risk_demo.py",
        "opslog.py", "opsjobs.py", "opshealth.py", "daily_pipeline.py",
        "notify.py", "certain.py",
            "knowledge_base.md"]
    missing = [f for f in core if not (BACKEND / f).exists()]
    add("后端模块", OK if not missing else BAD,
        f"{len(core) - len(missing)}/{len(core)}" + (f"，缺 {missing}" if missing else ""))

    tsuite = ["test_m1.py", "test_adjudicator.py", "test_m2.py",
              "test_contracts.py", "test_m3.py", "test_m4.py", "test_m5.py",
              "test_m6_attribution.py", "test_m7_compound.py",
              "test_m8_valuation.py", "test_m9_blackswan.py",
              "test_m10_risk.py", "test_m11_theory.py", "test_m12_portfolio.py", "test_m13_fund.py", "test_m14_dca.py", "test_m15_calibration.py", "test_m16_workflow.py", "test_m17_strategy_runner.py", "test_m18_datasource.py", "test_m19_robustness.py", "test_m20_deploy.py", "test_m21_zzshare.py", "test_m22_overseas_bs.py", "test_m23_us.py", "test_m24_quality.py", "test_m25_improvements.py", "test_m26_advisor.py", "test_m27_ml.py", "test_m28_optimizer.py", "test_m29_journal.py", "test_m30_yieldhunt.py", "test_m31_cb_double_low.py", "test_m32_cb_paper.py", "test_m33_backtest_strict.py", "test_m34_strategy_lab.py", "test_m35_position_scheduler.py", "test_m36_position_tab.py", "test_m37_mailer.py", "test_m38_dashboard.py", "test_m39_data_layer.py", "test_m40_versions_wf.py", "test_m41_risk_layer.py", "test_m42_ops.py", "test_m43_daily_pipeline.py", "test_m44_notify.py", "test_m45_certain.py"]
    miss_t = [f for f in tsuite if not (TESTS / f).exists()]
    add("测试文件", OK if not miss_t else BAD,
        f"{len(tsuite) - len(miss_t)}/{len(tsuite)}" + (f"，缺 {miss_t}" if miss_t else ""))

    fe = ROOT / "frontend" / "checkup.html"
    add("前端页面", OK if fe.exists() else BAD,
        f"{fe.stat().st_size} bytes" if fe.exists() else "缺失")

    for doc in ["STATUS.md", "DEPLOY.md", "docs/ARCHITECTURE.md"]:
        p = ROOT / doc
        add(f"文档 {doc}", OK if p.exists() else WARN,
            f"{p.stat().st_size} bytes" if p.exists() else "缺失")


def check_imports() -> None:
    sys.path.insert(0, str(BACKEND))
    mods = ["validator", "stress_tester", "purity_tester", "adjudicator",
            "scoring", "schemas", "template_engine", "ai_diagnostician",
            "api_v2", "daily_job", "attribution", "compound",
            "valuation", "blackswan", "datacheck", "walkforward",
            "execution", "riskcontrol", "portfolio_theory", "allocation",
            "timeseries", "pitfalls", "portfolio", "fund", "dca", "calibration", "risk_state", "quantpilot", "strategy_runner", "datasource", "robustness", "cache_manager", "zz_source", "overseas", "baostock_source", "credentials", "us_source", "quality", "advisor", "ml_risk", "optimizer", "journal", "cb_new", "fee_check", "paper_validate", "cb_double_low", "cb_paper", "backtest_strict", "strategy_lab", "position_manager", "daily_scheduler", "mailer", "dash_data", "store", "data_quality", "ingest", "feed"]
    bad = []
    vers = []
    for m in mods:
        try:
            mod = importlib.import_module(m)
            vers.append(f"{m} v{getattr(mod, '__version__', '?')}")
        except Exception as ex:
            bad.append(f"{m}: {ex}")
    add("模块导入", OK if not bad else BAD,
        "全部成功" if not bad else "；".join(bad)[:120])
    add("版本号", OK if all("v?" not in v for v in vers) else WARN,
        ", ".join(vers))


def check_artifacts() -> None:
    nav = ROOT / "reports" / "paper_nav.csv"
    if nav.exists():
        n = sum(1 for _ in nav.open(encoding="utf-8")) - 1
        add("纸盘净值", OK if n > 30 else WARN, f"{n} 个交易日")
    else:
        add("纸盘净值", WARN, "尚未生成（运行 daily_job.py --once）")

    latest = ROOT / "reports" / "daily" / "latest.json"
    if latest.exists():
        try:
            d = json.loads(latest.read_text(encoding="utf-8"))
            rep = d.get("report", {})
            add("最近日更", OK, f"{d.get('date')} {rep.get('verdict')}/{rep.get('grade')}")
        except Exception as ex:
            add("最近日更", BAD, f"解析失败：{ex}")
    else:
        add("最近日更", WARN, "尚未生成")


def check_tests() -> None:
    suites = ["test_m1", "test_adjudicator", "test_m2", "test_contracts",
              "test_m3", "test_m4", "test_m5", "test_m6_attribution",
              "test_m7_compound", "test_m8_valuation",
              "test_m9_blackswan", "test_m10_risk", "test_m11_theory", "test_m12_portfolio", "test_m13_fund", "test_m14_dca", "test_m15_calibration", "test_m16_workflow", "test_m17_strategy_runner", "test_m18_datasource", "test_m19_robustness", "test_m20_deploy", "test_m21_zzshare", "test_m22_overseas_bs", "test_m23_us", "test_m24_quality", "test_m25_improvements", "test_m26_advisor", "test_m27_ml", "test_m28_optimizer", "test_m29_journal", "test_m30_yieldhunt", "test_m31_cb_double_low", "test_m32_cb_paper", "test_m33_backtest_strict", "test_m34_strategy_lab", "test_m35_position_scheduler", "test_m36_position_tab", "test_m37_mailer", "test_m38_dashboard", "test_m39_data_layer", "test_m40_versions_wf", "test_m41_risk_layer", "test_m42_ops", "test_m43_daily_pipeline", "test_m44_notify", "test_m45_certain", "test_backtest"]
    total = 0
    failed_suites = []
    for name in suites:
        fp = TESTS / f"{name}.py"
        if not fp.exists():
            failed_suites.append(f"{name}(缺失)")
            continue
        try:
            # 注意：本机沙箱禁止子进程使用管道（capture_output 会 PermissionError 13），
            # 因此统一重定向到临时文件再读取。
            log_fp = ROOT / ".tmp" / f"preflight_{name}.log"
            log_fp.parent.mkdir(parents=True, exist_ok=True)
            txt = ""
            rc = -1
            try:
                with open(log_fp, "w", encoding="utf-8") as fh:
                    rc = subprocess.run([sys.executable, str(fp)], cwd=str(ROOT),
                                        stdout=fh, stderr=subprocess.STDOUT,
                                        timeout=600).returncode
                txt = log_fp.read_text(encoding="utf-8", errors="replace")
            except PermissionError:
                add(f"测试 {name}", WARN,
                    "当前沙箱禁止子进程重定向，请单独运行该测试")
                continue
            except subprocess.TimeoutExpired:
                failed_suites.append(f"{name}(超时)")
                add(f"测试 {name}", BAD, "超过 600 秒")
                continue
            if rc != 0:
                failed_suites.append(name)
                add(f"测试 {name}", BAD, txt.strip().splitlines()[-1][:80] if txt.strip() else "失败")
                continue
            num = 0
            for line in txt.splitlines():
                if line.startswith("# 通过"):
                    parts = line.replace("#", "").replace("通过", "").split("/")
                    try:
                        num = int(parts[0].strip())
                    except Exception:
                        num = 0
                elif line.startswith("通过") and "/" in line:
                    try:
                        num = int(line.split("/")[0].replace("通过", "").strip())
                    except Exception:
                        num = 0
            total += num
            add(f"测试 {name}", OK, f"{num} 项")
        except subprocess.TimeoutExpired:
            failed_suites.append(f"{name}(超时)")
            add(f"测试 {name}", BAD, "超过 600 秒")
        except Exception as ex:
            failed_suites.append(name)
            add(f"测试 {name}", BAD, repr(ex)[:60])

    add("测试合计", OK if not failed_suites else BAD,
        f"{total} 项通过" + (f"；失败 {failed_suites}" if failed_suites else ""))


def check_service() -> None:
    """离线检查接口定义是否齐全（不启动服务）。"""
    try:
        sys.path.insert(0, str(BACKEND))
        import api_v2
        routes = {r.path for r in api_v2.app.routes}
        need = {"/", "/api/health", "/api/checkup",
                "/api/daily/latest", "/api/daily/history",
                "/api/paper/nav", "/api/daily/run", "/api/attribution", "/api/portfolio",
                "/api/risk", "/api/dca", "/api/sources",
                "/api/perturbation", "/api/stress",
                "/api/zz", "/api/valuation", "/api/overseas", "/api/baostock",
                "/api/overseas/diagnose", "/api/us", "/api/keys", "/api/quality", "/api/advice", "/api/ml_risk", "/api/optimizer", "/api/journal", "/api/cb_new", "/api/fee", "/api/admit", "/api/cb_double_low", "/api/cb_paper", "/api/position"}
        miss = need - routes
        add("HTTP 路由", OK if not miss else BAD,
            f"{len(routes)} 条" + (f"，缺 {miss}" if miss else ""))
    except Exception as ex:
        add("HTTP 路由", BAD, repr(ex)[:80])


def check_config() -> None:
    import os
    key = os.environ.get("QP_LLM_KEY", "")
    add("大模型 Key", OK if key else WARN,
        "已配置" if key else "未配置（将使用本地模板，功能完整）")
    add("强制降级", OK,
        "已启用（QP_LLM_DISABLE=1）"
        if os.environ.get("QP_LLM_DISABLE") == "1"
        else "未启用（正常走主链路）")


# ══════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="一键自检")
    ap.add_argument("--quick", action="store_true", help="跳过测试")
    a = ap.parse_args()

    print("=" * 72)
    print(f"QuantPilot 自检  v{__version__}   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    check_env()
    check_deps()
    check_files()
    check_imports()
    check_artifacts()
    check_config()
    check_service()
    if not a.quick:
        print("  正在运行全部测试…")
        check_tests()

    print("-" * 72)
    for name, st, detail in _rows:
        print(f"  {_icon(st)} {name:<22} {detail}")
    print("-" * 72)

    n_bad = sum(1 for _, st, _ in _rows if st == BAD)
    n_warn = sum(1 for _, st, _ in _rows if st == WARN)
    if n_bad:
        print(f"结论：{n_bad} 项失败，{n_warn} 项警告 —— 请先处理失败项")
        return 1
    print(f"结论：全部通过（{n_warn} 项警告可忽略）")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
