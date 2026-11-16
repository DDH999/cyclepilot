# -*- coding: utf-8 -*-
"""M40 验证：版本管理 + 参数级 Walk-forward 切分（不联网）"""
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import param_walkforward as PW
import versions as V

PASS, FAIL = [], []
TMP = ROOT / ".tmp" / "test_m40"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
          (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M40 验证  versions v{V.__version__} / walkforward v{PW.__version__}")
print("#" + "=" * 69)

# ══════════════════════════════════════════
print("\n[1] 版本库：创建与保存")
V.ROOT = TMP / "versions"

c = V.create("factor", "测试因子", "假设", "momentum")
check("创建成功", c["ok"] is True, c.get("error"))
fid = c["id"]
check("id 有前缀", fid.startswith("fac_"), fid)
check("创建写入 meta", (V.ROOT / "factor" / fid / "meta.json").exists())

r1 = V.save("factor", fid, "s = -r", {"lb": 20}, "v1", parent=0)
check("保存 v1", r1["ok"] and r1["version"] == 1, r1)
check("记录 sha1", len(r1.get("sha1", "")) == 12, r1.get("sha1"))

r2 = V.save("factor", fid, "s = -r2", {"lb": 40}, "v2", parent=1)
check("保存 v2", r2["version"] == 2)

bad = V.save("factor", fid, "  ", {}, "空", parent=2)
check("空代码被拒", bad["ok"] is False, bad.get("error"))
check("错误提示明确", "code" in str(bad.get("error")), bad.get("error"))

lock = V.save("factor", fid, "x=1", {}, "锁", parent=1)
check("乐观锁拒绝旧 parent", lock["ok"] is False and "冲突" in lock["error"],
      lock.get("error"))
ok_lock = V.save("factor", fid, "x=2", {}, "正确 parent", parent=2)
check("正确 parent 通过", ok_lock["ok"] is True and ok_lock["version"] == 3)

print("\n[2] 版本库：读取与对比")
lt = V.latest("factor", fid)
check("latest 拿到 v3", lt["ok"] and lt["version"] == 3, lt.get("version"))
check("latest 带回代码", lt.get("code") == "x=2")
g = V.get("factor", fid, 1)
check("get 指定版本", g["ok"] and g["code"] == "s = -r")
check("get 不存在版本", V.get("factor", fid, 99)["ok"] is False)
check("latest 不存在条目", V.latest("factor", "nope")["ok"] is False)

d = V.diff("factor", fid, 1, 2)
check("diff 有输出", d["ok"] and d["n_lines"] > 0, d.get("n_lines"))
check("diff 含参数", d["params_a"] == {"lb": 20} and d["params_b"] == {"lb": 40})

items = V.list_items("factor")
check("清单 1 条", len(items) == 1, len(items))
check("清单带最新版本", items[0]["latest_version"] == 3)

print("\n[3] 版本库：运行归档")
V.record_run("factor", fid, 3, {"ic": 0.02}, {"rebalance": 20},
             "2020-01-01", "2020-12-31", "cn")
V.record_run("factor", fid, 3, {"ic": 0.05}, {"rebalance": 20},
             "2021-01-01", "2021-12-31", "cn")
runs = V.runs_of("factor", fid)
check("归档 2 条", len(runs) == 2, len(runs))
check("归档含指标", runs[0]["metrics"]["ic"] == 0.02)
check("归档含区间", runs[0]["data_start"] == "2020-01-01")
check("清单统计运行数", V.list_items("factor")[0]["runs"] == 2)
check("坏 kind 被拒", V.create("bad", "x")["ok"] is False)
check("空库返回空列表", V.list_items("selection") == [])

print("\n[4] Walk-forward 切分")
dates = pd.bdate_range("2016-01-04", "2026-01-02")
folds = PW.split_folds(dates, n_folds=4, test_days=250, min_train_days=750)
check("切出 4 折", len(folds) == 4, len(folds))
check("训练段在前", all(f["train"][0] < f["test"][0] for f in folds))
check("无重叠", all(f["train"][1] < f["test"][0] for f in folds))
check("OOS 紧邻训练", all(
    (dates[dates > f["train"][1]][0] == f["test"][0]) for f in folds))
check("每折训练长度固定",
      len({(f["train"][1] - f["train"][0]).days for f in folds}) <= 3,
      [(f["train"][1] - f["train"][0]).days for f in folds])
check("折间向前滚动",
      folds[1]["train"][0] > folds[0]["train"][0], folds[1]["train"][0])

f2 = PW.split_folds(dates, n_folds=4, test_days=250,
                    min_train_days=750, start_after="2020-01-01")
check("start_after 生效", f2[0]["train"][0] >= pd.Timestamp("2020-01-01"),
      f2[0]["train"][0])

short = PW.split_folds(pd.bdate_range("2024-01-01", periods=100),
                       n_folds=4, test_days=250, min_train_days=750)
check("区间不足返回空", short == [])

limited = PW.split_folds(dates, n_folds=2, test_days=250, min_train_days=750)
check("n_folds 上限生效", len(limited) == 2, len(limited))

print("\n[5] 参数网格")
check("网格含三个参数组合维度",
      set(PW.PARAM_GRID) == {"lookback", "top_n"}, set(PW.PARAM_GRID))
check("网格总组合 9",
      len(PW.PARAM_GRID["lookback"]) * len(PW.PARAM_GRID["top_n"]) == 9)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  版本管理与 WF 切分可用")
print("#" * 70)
