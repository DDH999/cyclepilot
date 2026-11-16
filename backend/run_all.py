# -*- coding: utf-8 -*-
"""QuantPilot · 一键全流程 (python backend/run_all.py)"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STEPS = [
    ("基本面回测", ["backend/fund_strategy.py", "--save"]),
    ("诚实检验",   ["backend/validate.py", "--nav", "fund_nav.csv"]),
    ("回测图表",   ["backend/bt_charts.py"]),
    ("研报生成",   ["backend/report_factory.py"]),
]


def main():
    t0 = time.time()
    ok = 0
    for name, cmd in STEPS:
        t = time.time()
        p = subprocess.run([PY] + cmd, cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        good = p.returncode == 0
        ok += good
        print(f"[{'✓' if good else '✗'}] {name:10} {time.time()-t:5.1f}s")
        if not good:
            print((p.stderr or "")[-500:])
    print(f"\n完成 {ok}/{len(STEPS)}  总耗时 {time.time()-t0:.1f}s")
    print("控制台: python backend/app.py  ->  http://127.0.0.1:8420")


if __name__ == "__main__":
    main()
