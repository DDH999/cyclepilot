# -*- coding: utf-8 -*-
"""
M20 验证：缓存治理 + 本地部署产物
====================================================================
跑法:  python tests/test_m20_deploy.py   （不联网）
"""
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cache_manager as CM

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m20"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M20 验证  cache_manager v{CM.__version__} + 部署产物")
print("#" + "=" * 69)

print("\n[1] 缓存统计")

demo = TMP / "cache"
(demo / "fund").mkdir(parents=True)
(demo / "valuation").mkdir(parents=True)
(demo / "panel_abc.pkl").write_bytes(b"x" * 2048)
(demo / "123_cn_600519.csv").write_bytes(b"y" * 1024)
(demo / "fund" / "nav_050025.csv").write_text("a,b", encoding="utf-8")
(demo / "valuation" / "index_pe.csv").write_text("a,b", encoding="utf-8")

st = CM.stats(demo)
check("统计成功", st["ok"] is True)
check("文件数正确", st["files"] >= 4, st["files"])
check("总体积 > 0", st["total_bytes"] > 0, st["total_bytes"])
check("识别 panel 缓存", "panel 缓存" in st["categories"], list(st["categories"]))
check("识别行情 CSV", "行情 CSV" in st["categories"])
check("识别基金净值", "基金净值" in st["categories"])
check("识别估值缓存", "估值缓存" in st["categories"])
check("体积可读", "B" in st["total_human"] or "KB" in st["total_human"])
check("报告可渲染", "缓存统计" in CM.format_report(st))

empty = CM.stats(TMP / "nope")
check("不存在时返回 exists=False", empty["exists"] is False)
check("不存在不崩", empty["total_bytes"] == 0)

print("\n[2] 按龄清理")

old_file = demo / "old.csv"
old_file.write_bytes(b"z" * 512)
past = time.time() - 40 * 86400
import os
os.utime(old_file, (past, past))

d1 = CM.trim(30, root=demo, dry_run=True)
check("干跑找到候选", d1["candidates"] >= 1, d1["candidates"])
check("干跑不删文件", old_file.exists())
check("干跑记录 released", d1["freed_bytes"] > 0, d1["freed_bytes"])
check("干跑 removed=0", d1["removed"] == 0)

d2 = CM.trim(30, root=demo, dry_run=False)
check("实删成功", d2["removed"] >= 1, d2["removed"])
check("文件已删除", not old_file.exists())

d3 = CM.trim(365, root=demo, dry_run=True)
check("阈值过长时候选为 0", d3["candidates"] == 0, d3["candidates"])

print("\n[3] 安全边界（关键）")

check("reports 不在可删范围",
      CM._safe_under_cache(ROOT / "reports" / "paper_nav.csv") is False)
check("持仓文件不在可删范围",
      CM._safe_under_cache(ROOT / "holdings.csv") is False)
check("缓存内文件可删",
      CM._safe_under_cache(demo / "123_cn_600519.csv", demo) is True)
check("自定义 root 生效",
      CM._safe_under_cache(demo / "123_cn_600519.csv") is False)

# 注意：不能对全局 .cache 做真删除 —— 测试要有副作用就不可重复跑了。
# 这里只验证「安全谓词」+ 在隔离目录里实删。
check("reports 仍是安全区", CM._safe_under_cache(ROOT / "reports") is False)
check("持仓仍是安全区", CM._safe_under_cache(ROOT / "holdings.csv") is False)
check("定投台账仍是安全区", CM._safe_under_cache(ROOT / "dca_ledger.csv") is False)

iso = TMP / "iso"
iso.mkdir(parents=True, exist_ok=True)
(iso / "a.csv").write_text("x", encoding="utf-8")
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "_marker.txt").write_text("keep", encoding="utf-8")
CM.clean_all(root=iso, dry_run=False)
check("隔离目录被清空", not (iso / "a.csv").exists())
check("reports 未被波及",
      (ROOT / "reports" / "_marker.txt").exists())
(ROOT / "reports" / "_marker.txt").unlink(missing_ok=True)
check("清理不影响持仓", (ROOT / "holdings.csv").exists())
check("清理不影响定投台账", (ROOT / "dca_ledger.csv").exists())

print("\n[4] 全清")

c1 = CM.clean_all(root=demo, dry_run=True)
check("全清干跑统计文件", c1["files"] >= 1, c1["files"])
check("全清干跑不删", (demo / "panel_abc.pkl").exists())
c2 = CM.clean_all(root=demo, dry_run=False)
check("全清真删", not (demo / "panel_abc.pkl").exists())
check("全清返回释放量", c2["freed_bytes"] > 0, c2["freed_bytes"])

print("\n[5] 部署产物")


def noncomment_lines(fp):
    return [ln for ln in fp.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


req = ROOT / "requirements.txt"
check("requirements.txt 存在", req.exists())
lines = noncomment_lines(req)
# 数据源在增长（akshare/zzshare/baostock/yfinance/roicai/xfinlink），
# 上限放宽到 20，但仍要求全部锁定版本
check("依赖条目 6~20 个", 6 <= len(lines) <= 20, len(lines))
check("全部锁定版本（==）", all("==" in ln for ln in lines), lines[:3])
check("含 numpy/pandas/scipy",
      any("numpy" in ln for ln in lines) and any("pandas" in ln for ln in lines)
      and any("scipy" in ln for ln in lines))
check("含 fastapi/uvicorn",
      any("fastapi" in ln for ln in lines) and any("uvicorn" in ln for ln in lines))
check("含 akshare", any("akshare" in ln for ln in lines))
check("不再用 >= 模糊版本",
      not any(">=" in ln for ln in lines), [l for l in lines if ">=" in l][:3])

for bat, why in [("start.bat", "一键启动"), ("stop.bat", "停止"),
                 ("backup.bat", "备份"), ("install.bat", "安装依赖")]:
    fp = ROOT / bat
    check(f"{bat} 存在（{why}）", fp.exists())
    if fp.exists():
        raw = fp.read_bytes()
        check(f"{bat} 使用 CRLF", b"\r\n" in raw)
        txt = raw.decode("utf-8")
        check(f"{bat} 设置 UTF-8 代码页", "chcp 65001" in txt)
        check(f"{bat} 有暂停（双击不闪退）", "pause" in txt.lower())

sb = (ROOT / "start.bat").read_text(encoding="utf-8")
check("start.bat 先自检", "preflight" in sb)
check("start.bat 自检失败会中止", "exit /b 1" in sb)
check("start.bat 自动开浏览器", "http://127.0.0.1:8421" in sb)

bb = (ROOT / "backup.bat").read_text(encoding="utf-8")
check("backup.bat 备份持仓", "holdings.csv" in bb)
check("backup.bat 备份定投台账", "dca_ledger.csv" in bb)
check("backup.bat 备份风控状态", ".state" in bb)

check("部署手册存在", (ROOT / "docs" / "部署手册.md").exists())
man = (ROOT / "docs" / "部署手册.md").read_text(encoding="utf-8")
for token in ("start.bat", "cache --trim", "backup.bat", "127.0.0.1",
              "安全边界", "不要"):
    check(f"手册含「{token}」", token in man, token)

print("\n[6] CLI")

import quantpilot as QP
check("CLI 注册 cache", "cache" in QP.COMMANDS)
p = QP.build_parser()
a = p.parse_args(["cache", "--trim", "30"])
check("解析 --trim", a.trim == 30.0)
check("默认不 --yes（干跑）", a.yes is False)
b = p.parse_args(["cache", "--clean", "--yes"])
check("解析 --clean --yes", b.clean is True and b.yes is True)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  本地部署产物齐全")
print("#" * 70)
