# -*- coding: utf-8 -*-
"""M44 验证：推送层（邮件/飞书/企业微信/个人微信），全部干跑，不联网"""
import importlib
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" +
          (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print("# M44 验证  推送层")
print("#" + "=" * 69)

# 强制干跑 + 假凭据
os.environ["QP_DRY_RUN"] = "1"
os.environ["QP_FEISHU_WEBHOOK"] = "https://open.feishu.cn/hook/FAKE123"
os.environ["QP_WECHAT_WEBHOOK"] = "https://qyapi.weixin.qq.com/send?key=FAKE456"
os.environ["QP_SERVERCHAN_KEY"] = "SCTFAKE789"
import notify as NT
importlib.reload(NT)

print("\n[1] 干跑默认与开关")
check("干跑由环境变量控制", NT.DRY is True, NT.DRY)
os.environ.pop("QP_DRY_RUN")
importlib.reload(NT)
check("默认不干跑（真发）", NT.DRY is False,
      "旧版本默认 True 导致配好也收不到")
os.environ["QP_DRY_RUN"] = "1"
importlib.reload(NT)

print("\n[2] 配置状态与脱敏")
st = NT.cfg_status()
check("返回四个渠道", set(["email", "feishu", "wechat",
                        "wechat_personal"]) <= set(st), list(st))
check("飞书识别为已配置", st["feishu"]["configured"] is True)
check("企业微信识别为已配置", st["wechat"]["configured"] is True)
check("个人微信识别为已配置", st["wechat_personal"]["configured"] is True)
check("webhook 已脱敏", "****" in st["feishu"]["webhook"],
      st["feishu"]["webhook"])
check("脱敏不泄露完整密钥",
      "FAKE123" not in json.dumps(st, ensure_ascii=False))
check("mask 短串打码", NT.mask("abc") == "abc" or "****" in NT.mask("abc"))
check("mask 长串保留前缀", NT.mask("https://x/verylong")[:8] == "https://")

print("\n[3] 单渠发送（干跑）")
for ch in ("email", "feishu", "wechat", "wechat_personal"):
    r = NT.test_channel(ch)
    check(f"{ch} 干跑成功", r.get("ok") is True and r.get("dry_run") is True, r)

print("\n[4] 未配置渠道")
saved = {k: os.environ.pop(k, None) for k in
         ("QP_FEISHU_WEBHOOK", "QP_WECHAT_WEBHOOK", "QP_SERVERCHAN_KEY")}
os.environ["QP_DRY_RUN"] = "1"
importlib.reload(NT)
# reload 会把 CONFIG_PATH 重置回真实文件，必须在 reload **之后**再指向空配置
NT.CONFIG_PATH = ROOT / ".tmp" / "notify_none.json"
r = NT.test_channel("feishu")
check("未配置时明确报错", r.get("ok") is False and r.get("reason") ==
      "not configured", r)
check("未知渠道报错", NT.test_channel("qq")["ok"] is False)
# 邮件从 config/mail.local.json 读取，始终"已配置"，因此只测未配置的渠道
pa = NT.push_all("t", "b", channels=["feishu", "wechat", "wechat_personal"])
check("未配置渠道全部失败", pa.get("ok") is False, pa.get("reason"))
pa2 = NT.push_all("t", "b", channels=["feishu"])
check("指定未配置渠道也失败", pa2["results"]["feishu"].get("ok") is False,
      pa2["results"]["feishu"].get("reason"))

print("\n[5] 恢复配置后广播")
for k, v in (("QP_FEISHU_WEBHOOK", "https://x/fake"),
             ("QP_WECHAT_WEBHOOK", "https://y/fake"),
             ("QP_SERVERCHAN_KEY", "SCTFAKE")):
    os.environ[k] = v
importlib.reload(NT)
pr = NT.push_all("QuantPilot 日更", "正文", brief="摘要")
check("广播成功", pr.get("ok") is True, pr)
check("覆盖三个 webhook 渠道（+邮件）", pr.get("n") >= 3, pr.get("n"))
check("含飞书/企业微信/个人微信",
      set(["feishu", "wechat", "wechat_personal"]) <= set(pr["results"]),
      list(pr["results"]))
check("全部干跑", all(v.get("dry_run") for v in pr["results"].values()))
pr2 = NT.push_all("t", "b", channels=["feishu"])
check("可指定渠道", pr2.get("n") == 1 and "feishu" in pr2["results"], pr2)

print("\n[6] 状态渲染")
txt = NT.format_status()
check("含干跑警示", "不会真发" in txt, txt[:60])
check("含配置指引", "群机器人" in txt)
check("含个人微信说明", "Server酱" in txt or "sct.ftqq.com" in txt)

print("\n[7] 个人微信走 Server酱（不是企业微信）")
check("serverchan 独立渠道",
      "wechat_personal" in NT.CHANNELS and "wechat" in NT.CHANNELS,
      NT.CHANNELS)
r = NT.test_channel("wechat_personal")
check("个人微信干跑", r.get("ok") is True and r.get("dry_run") is True, r)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  推送层可用")
print("#" * 70)
