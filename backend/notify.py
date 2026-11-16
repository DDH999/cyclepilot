# -*- coding: utf-8 -*-
"""
QuantPilot · 推送层（M70）
====================================================
四个渠道：邮件 / 飞书 / 企业微信 / **个人微信（Server酱）**

渠道对照
--------
| 渠道 | 机制 | 门槛 | 目标 |
|---|---|---|---|
| `email` | SMTP | 已配置 `config/mail.local.json` | 邮箱 |
| `feishu` | 群自定义机器人 Webhook | 建群→加机器人→复制 URL（1 分钟） | 飞书群 |
| `wechat` | 企业微信**群机器人** Webhook | 需有企业微信 | 企业微信群 |
| `wechat_personal` | Server酱 SendKey | 微信扫码登录 sct.ftqq.com | **个人微信** |

⚠ 关于「个人微信」
----------------
微信**不开放**个人号消息接口。合规路径只有两条：
1. **Server酱**（本模块实现）：公众号推送，免费约 5 条/天
2. 企业微信应用消息 → 已绑定微信（需企业管理员）
第三方库（itchat / wechaty）走逆向协议，**封号风险高，不提供**。

⚠ 干跑开关（重要）
------------------
`QP_DRY_RUN` 默认 **0（真发）**。
旧版本默认 1，导致"配置好了却从没收到"——这是最容易踩的坑。
显式设置 `QP_DRY_RUN=1` 才会干跑，此时返回值带 `dry_run: True`。

配置优先级：环境变量 > `config/notify.local.json`

| 渠道 | 环境变量 | 配置文件键 |
|---|---|---|
| 飞书 | `QP_FEISHU_WEBHOOK` | `feishu.webhook` |
| 企业微信 | `QP_WECHAT_WEBHOOK` | `wechat.webhook` |
| 个人微信 | `QP_SERVERCHAN_KEY` | `wechat_personal.sendkey` |
| 邮件 | `QP_SMTP_*` | 复用 `config/mail.local.json` |
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
import time
import urllib.parse
import urllib.request
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

__version__ = "0.2.0"

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "notify.local.json"
CHANNELS = ("email", "feishu", "wechat", "wechat_personal")

DRY = os.getenv("QP_DRY_RUN", "0").strip() == "1"

LIMITS = {"feishu": 3500, "wechat": 600, "wechat_personal": 30000,
          "email": 100000}


def _env(k, d=""):
    return os.getenv(k, d).strip()


def load_config() -> dict:
    """文件配置 + 环境变量覆盖。"""
    cfg: dict = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    for ch, ek, fk in (("feishu", "QP_FEISHU_WEBHOOK", "webhook"),
                       ("wechat", "QP_WECHAT_WEBHOOK", "webhook"),
                       ("wechat_personal", "QP_SERVERCHAN_KEY", "sendkey")):
        v = _env(ek)
        if v:
            cfg.setdefault(ch, {})[fk] = v
    return cfg


def _cfg() -> dict:
    return load_config()


def mask(s: str | None, keep: int = 8) -> str:
    s = str(s or "")
    return (s[:keep] + "****") if len(s) > keep else (s or "")


def _ready(ch: str, cfg: dict | None = None) -> bool:
    c = cfg if cfg is not None else _cfg()
    if ch == "feishu":
        return bool((c.get("feishu") or {}).get("webhook") or
                    _env("QP_FEISHU_WEBHOOK"))
    if ch == "wechat":
        return bool((c.get("wechat") or {}).get("webhook") or
                    _env("QP_WECHAT_WEBHOOK"))
    if ch == "wechat_personal":
        return bool((c.get("wechat_personal") or {}).get("sendkey") or
                    _env("QP_SERVERCHAN_KEY"))
    if ch == "email":
        if _env("QP_SMTP_HOST") and _env("QP_MAIL_TO"):
            return True
        try:
            import mailer
            return bool(mailer.load_config().get("ok"))
        except Exception:
            return False
    return False


def cfg_status() -> dict:
    """各渠道配置状态（不泄露密钥）。"""
    c = _cfg()
    return {
        "dry_run": DRY,
        "email": {"configured": _ready("email", c)},
        "feishu": {"configured": _ready("feishu", c),
                   "webhook": mask((c.get("feishu") or {}).get("webhook"))},
        "wechat": {"configured": _ready("wechat", c),
                   "webhook": mask((c.get("wechat") or {}).get("webhook")),
                   "note": "企业微信群机器人"},
        "wechat_personal": {
            "configured": _ready("wechat_personal", c),
            "sendkey": mask((c.get("wechat_personal") or {}).get("sendkey")),
            "note": "Server酱，推到个人微信"},
    }


# ─────────────────────────── Email
def send_email(subject: str, html: str, images: list[Path] | None = None,
               to: str | None = None) -> dict:
    # 优先复用 mailer 的配置文件（用户实际填的是 mail.local.json）
    if not (_env("QP_SMTP_HOST") and (_env("QP_MAIL_TO") or to)):
        try:
            import mailer
            lc = mailer.load_config()          # 包装字典 {ok, cfg, ...}
            if not lc.get("ok"):
                return {"ok": False, "channel": "email",
                        "error": lc.get("error") or "邮件未配置"}
            if DRY:
                return {"ok": True, "channel": "email", "dry_run": True,
                        "subject": subject, "len": len(html)}
            # send_mail(subject, html, ..., cfg=) —— 注意不是 (cfg, subject, html)
            r = mailer.send_mail(subject, html, cfg=lc.get("cfg"))
            return {"ok": bool(r.get("ok")), "channel": "email",
                    "via": "mail.local.json",
                    "error": r.get("error")}
        except Exception as e:
            return {"ok": False, "channel": "email",
                    "error": f"{type(e).__name__}: {e}"}

    host, port = _env("QP_SMTP_HOST"), int(_env("QP_SMTP_PORT", "465"))
    user, pwd = _env("QP_SMTP_USER"), _env("QP_SMTP_PASS")
    to = to or _env("QP_MAIL_TO")
    if DRY:
        return {"ok": True, "channel": "email", "dry_run": True,
                "would_send_to": to, "subject": subject}
    msg = MIMEMultipart("related")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = user or to
    msg["To"] = to
    msg.attach(MIMEText(html, "html", "utf-8"))
    for p in (images or [])[:8]:
        try:
            with open(p, "rb") as f:
                img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<{p.stem}>")
            img.add_header("Content-Disposition", "inline", filename=p.name)
            msg.attach(img)
        except Exception:
            pass
    try:
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as s:
                if user:
                    s.login(user, pwd)
                s.sendmail(msg["From"], [to], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls(context=ctx)
                if user:
                    s.login(user, pwd)
                s.sendmail(msg["From"], [to], msg.as_string())
        return {"ok": True, "channel": "email", "to": to}
    except Exception as e:
        return {"ok": False, "channel": "email",
                "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────── Webhook
def _post_json(url: str, payload: dict, timeout: int = 15) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return {"status": r.status,
                "body": r.read().decode("utf-8", "replace")[:400]}


def send_feishu(title: str, text_md: str) -> dict:
    url = (_cfg().get("feishu") or {}).get("webhook") or _env("QP_FEISHU_WEBHOOK")
    if not url:
        return {"ok": False, "channel": "feishu", "reason": "not configured"}
    payload = {"msg_type": "interactive", "card": {
        "header": {"title": {"tag": "plain_text", "content": title},
                   "template": "blue"},
        "elements": [{"tag": "div",
                      "text": {"tag": "lark_md",
                               "content": text_md[:LIMITS["feishu"]]}}]}}
    if DRY:
        return {"ok": True, "channel": "feishu", "dry_run": True,
                "bytes": len(json.dumps(payload))}
    try:
        r = _post_json(url, payload)
        ok = '"code":0' in r["body"] or '"StatusCode":0' in r["body"]
        return {"ok": ok, "channel": "feishu", **r}
    except Exception as e:
        return {"ok": False, "channel": "feishu",
                "error": f"{type(e).__name__}: {e}"}


def send_wechat_work(content: str) -> dict:
    """企业微信群机器人（文本，3 行以内避免风控）。"""
    url = (_cfg().get("wechat") or {}).get("webhook") or _env("QP_WECHAT_WEBHOOK")
    if not url:
        return {"ok": False, "channel": "wechat", "reason": "not configured"}
    payload = {"msgtype": "text",
               "text": {"content": content[:LIMITS["wechat"]]}}
    if DRY:
        return {"ok": True, "channel": "wechat", "dry_run": True,
                "bytes": len(json.dumps(payload))}
    try:
        r = _post_json(url, payload)
        return {"ok": '"errcode":0' in r["body"] or r["status"] == 200,
                "channel": "wechat", **r}
    except Exception as e:
        return {"ok": False, "channel": "wechat",
                "error": f"{type(e).__name__}: {e}"}


def send_serverchan(title: str, body: str) -> dict:
    """
    个人微信（Server酱）。免费额度约 5 条/天，超了会返回错误码。
    """
    key = ((_cfg().get("wechat_personal") or {}).get("sendkey")
           or _env("QP_SERVERCHAN_KEY"))
    if not key:
        return {"ok": False, "channel": "wechat_personal",
                "reason": "not configured"}
    if DRY:
        return {"ok": True, "channel": "wechat_personal", "dry_run": True,
                "len": len(body)}
    url = f"https://sctapi.ftqq.com/{key}.send"
    data = urllib.parse.urlencode(
        {"title": title[:100], "desp": body[:LIMITS["wechat_personal"]]}
    ).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=15) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        return {"ok": j.get("code") == 0, "channel": "wechat_personal",
                "code": j.get("code"), "msg": str(j.get("message"))[:120]}
    except Exception as e:
        return {"ok": False, "channel": "wechat_personal",
                "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────── 编排
DISPATCH = {
    "email": lambda s, h: send_email(s, h),
    "feishu": lambda s, h: send_feishu(s, h),
    "wechat": lambda s, h: send_wechat_work(h),
    "wechat_personal": lambda s, h: send_serverchan(s, h),
}


def test_channel(ch: str) -> dict:
    """给单个渠道发一条测试消息。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    body = f"QuantPilot 测试消息\n时间 {ts}\n收到说明配置正确。"
    if ch not in DISPATCH:
        return {"ok": False, "channel": ch, "error": f"未知渠道 {ch}"}
    r = DISPATCH[ch]("QuantPilot 测试", body)
    r["channel"] = ch
    return r


def push_all(subject: str, html: str, images: list[Path] | None = None,
             brief: str | None = None, channels=None) -> dict:
    """向所有已配置渠道推送（或指定渠道）。"""
    cfg = _cfg()
    chs = list(channels) if channels else [c for c in CHANNELS if _ready(c, cfg)]
    if not chs:
        return {"ok": False, "dry_run": DRY, "results": {},
                "reason": "没有任何已配置的渠道（见 notify.py --status）"}
    brief = brief or subject
    out = {}
    for ch in chs:
        try:
            r = DISPATCH[ch](subject, brief) if ch != "email" else \
                send_email(subject, html, images)
        except Exception as e:
            r = {"ok": False, "channel": ch,
                 "error": f"{type(e).__name__}: {e}"}
        r["channel"] = ch
        out[ch] = r
        try:
            import opslog as OL
            OL.log(ch, status="ok" if r.get("ok") else "fail",
                   kind="notify",
                   error=(r.get("error") or "")[:150] if not r.get("ok") else None)
        except Exception:
            pass
    return {"ok": any(v.get("ok") for v in out.values()), "dry_run": DRY,
            "results": out, "n": len(chs)}


def format_status() -> str:
    st = cfg_status()
    L = ["=" * 72, "推送渠道状态", "=" * 72]
    tips = {
        "feishu": "飞书群 → 设置 → 群机器人 → 添加自定义机器人 → 复制 Webhook",
        "wechat": "企业微信群 → 右键 → 添加群机器人 → 复制 Webhook",
        "wechat_personal": "sct.ftqq.com 微信扫码 → 复制 SendKey（约 5 条/天）",
        "email": "复用 config/mail.local.json",
    }
    for ch in CHANNELS:
        s = st.get(ch) or {}
        ok = bool(s.get("configured"))
        L.append(f"  [{'已配置' if ok else '未配置'}] {ch:<16}"
                 f"{s.get('webhook') or s.get('sendkey') or ''}")
        if not ok:
            L.append(f"                 {tips.get(ch, '')}")
        elif s.get("note"):
            L.append(f"                 {s['note']}")
    L.append("-" * 72)
    L.append(f"  干跑模式(QP_DRY_RUN)：{'开启（不会真发！）' if st['dry_run'] else '关闭（真发）'}")
    L.append(f"  配置文件：{CONFIG_PATH}")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--test", type=str, default=None,
                    choices=list(CHANNELS) + [None])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.test:
        print(json.dumps(test_channel(a.test), ensure_ascii=False, indent=1))
    elif a.all:
        print(json.dumps(push_all("QuantPilot 测试", "测试正文",
                                  brief="测试正文"), ensure_ascii=False,
                         indent=1))
    else:
        print(format_status())
