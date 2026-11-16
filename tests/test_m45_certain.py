# -*- coding: utf-8 -*-
"""M45 验证：确定性收益清单（口径与不撒谎）"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import certain as CT

P, F = [], []
def ck(n, c, d=""):
    (P if c else F).append(n)
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))

print("\n[1] 费率差")
r = CT.fee_saving(100000, 10)
ck("可计算", r.get("ok") is True, r.get("reason"))
ck("给出优劣方", r.get("better") in ("A 类", "C 类"), r.get("better"))
ck("年成本有值", r.get("a_annual_cost") is not None)

print("\n[2] 打新（必须是区间，不是单点承诺）")
c = CT.cb_opportunity()
ck("可计算", c.get("ok") is True, c.get("reason"))
ck("有区间下界", isinstance(c.get("range_low"), (int, float)))
ck("有区间上界", isinstance(c.get("range_high"), (int, float)))
ck("上界大于下界", c["range_high"] > c["range_low"])
ck("标注为估算", "不是承诺" in str(c.get("note")), c.get("note"))

print("\n[3] 定投（必须说明不是收益增强）")
d = CT.dca_vs_lump(3000, 5, 0.06, 0.20)
ck("可计算", d.get("ok") is True)
ck("说明不提高期望", "不是收益更高" in str(d.get("note")), d.get("note"))
ck("给出 p5", d.get("dca_p5") is not None)

print("\n[4] 逆回购（取不到要如实说）")
rr = CT.repo_rate()
ck("返回结构正确", "ok" in rr)
ck("成功时有年化", (not rr.get("ok")) or rr.get("rate_pct") is not None)
ck("失败时有原因", rr.get("ok") or bool(rr.get("reason")))

print("\n[5] 汇总")
s = CT.summary(100000, 10)
ck("四项齐全", len(s["items"]) == 4, len(s["items"]))
ck("报告可渲染", "确定性收益清单" in CT.format_report(s))
ck("含免责说明", "不构成收益承诺" in CT.format_report(s))

print(f"\n# 通过 {len(P)} / {len(P)+len(F)}")
if F:
    print("# 失败:", F); sys.exit(1)
