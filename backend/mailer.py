# -*- coding: utf-8 -*-
"""
邮件推送（M50）：SMTP 发信 + 每日报告渲染
====================================================
用途
----
把「次日交易清单 + 仓位状态 + 数据异常」推到你的邮箱。
调度器跑在后台时，没有推送等于没跑——你不会去看日志。

配置优先级
----------
1. 环境变量（最高）
2. `config/mail.local.json`（推荐，私有文件，不要提交）
3. 无配置 → 拒绝发送并提示怎么填

环境变量
--------
| 变量 | 说明 |
|---|---|
| `QP_SMTP_HOST` | 服务器，如 `smtp.qq.com` |
| `QP_SMTP_PORT` | 端口，SSL 用 465，STARTTLS 用 587 |
| `QP_SMTP_USER` | 完整邮箱地址 |
| `QP_SMTP_PASS` | **授权码**（不是登录密码！） |
| `QP_SMTP_SSL` | `1` / `0`，是否用 SSL（465 一般 True） |
| `QP_SMTP_TLS` | `1` / `0`，是否用 STARTTLS（587 一般 True） |
| `QP_MAIL_TO` | 收件人，多个用逗号分隔 |

安全
----
* 授权码**永不打印**，日志里只显示掩码（前 4 + **** + 后 4）
* `config/mail.local.json` 属于私有文件，备份时会排除

⚠ 常见坑
--------
* QQ / 163 的「授权码」要在邮箱设置里单独开启 SMTP 服务后生成，
  **不是你的登录密码**
* 465 端口必须 `use_ssl=True`；587 端口用 `use_tls=True`
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from datetime import datetime
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["CONFIG_PATH", "PROVIDERS", "load_config", "mask", "render_daily",
           "send_mail", "send_daily_report", "format_config_help"]

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "mail.local.json"

# 常见邮箱服务商预设
PROVIDERS = {
    "qq": {"smtp_host": "smtp.qq.com", "smtp_port": 465,
           "use_ssl": True, "use_tls": False,
           "note": "QQ 邮箱 → 设置 → 账户 → 开启 SMTP → 获取授权码"},
    "163": {"smtp_host": "smtp.163.com", "smtp_port": 465,
            "use_ssl": True, "use_tls": False,
            "note": "163 邮箱 → 设置 → POP3/SMTP/IMAP → 开启 → 授权码"},
    "126": {"smtp_host": "smtp.126.com", "smtp_port": 465,
            "use_ssl": True, "use_tls": False,
            "note": "同 163"},
    "gmail": {"smtp_host": "smtp.gmail.com", "smtp_port": 587,
              "use_ssl": False, "use_tls": True,
              "note": "Gmail 需要「应用专用密码」，且可能需代理"},
    "outlook": {"smtp_host": "smtp.office365.com", "smtp_port": 587,
                "use_ssl": False, "use_tls": True,
                "note": "Outlook/Microsoft 365"},
    "aliyun": {"smtp_host": "smtp.mxhichina.com", "smtp_port": 465,
               "use_ssl": True, "use_tls": False,
               "note": "阿里企业邮箱"},
}


def mask(s: str | None) -> str:
    """掩码：前 4 + **** + 后 4。短串全掩。"""
    if not s:
        return "(空)"
    s = str(s)
    if len(s) <= 8:
        return "****"
    return f"{s[:4]}****{s[-4:]}"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def load_config(path: Path | str | None = None) -> dict:
    """
    读取邮件配置（环境变量优先）。

    返回 {ok, cfg, source, error}；cfg 里的 password 已掩码的副本放在 display。
    """
    fp = Path(path) if path else CONFIG_PATH
    cfg: dict = {}

    if fp.exists():
        try:
            cfg = json.loads(fp.read_text(encoding="utf-8"))
            source = str(fp)
        except Exception as ex:
            return {"ok": False, "cfg": {}, "source": str(fp),
                    "error": f"配置文件解析失败：{ex}"}
    else:
        source = "env"

    # 环境变量覆盖
    env_map = {
        "smtp_host": "QP_SMTP_HOST", "smtp_port": "QP_SMTP_PORT",
        "user": "QP_SMTP_USER", "password": "QP_SMTP_PASS",
        "from_name": "QP_MAIL_FROM_NAME",
    }
    for k, env in env_map.items():
        v = os.environ.get(env)
        if v:
            cfg[k] = v
            source = "env"
    if os.environ.get("QP_SMTP_SSL") is not None:
        cfg["use_ssl"] = _env_bool("QP_SMTP_SSL")
        source = "env"
    if os.environ.get("QP_SMTP_TLS") is not None:
        cfg["use_tls"] = _env_bool("QP_SMTP_TLS")
        source = "env"
    to_env = os.environ.get("QP_MAIL_TO")
    if to_env:
        cfg["to"] = [x.strip() for x in to_env.split(",") if x.strip()]
        source = "env"

    # 规范化
    cfg.setdefault("smtp_port", 465)
    cfg.setdefault("use_ssl", True)
    cfg.setdefault("use_tls", False)
    cfg.setdefault("from_name", "QuantPilot")
    try:
        cfg["smtp_port"] = int(cfg["smtp_port"])
    except Exception:
        cfg["smtp_port"] = 465
    to = cfg.get("to")
    if isinstance(to, str):
        cfg["to"] = [x.strip() for x in to.split(",") if x.strip()]
    elif not isinstance(to, list):
        cfg["to"] = []

    missing = [k for k in ("smtp_host", "user", "password") if not cfg.get(k)]
    if not cfg.get("to"):
        missing.append("to")

    display = {k: cfg.get(k) for k in
               ("smtp_host", "smtp_port", "use_ssl", "use_tls", "user", "to")}
    display["password"] = mask(cfg.get("password"))
    return {"ok": len(missing) == 0, "cfg": cfg, "source": source,
            "missing": missing, "display": display,
            "error": (None if not missing else
                      f"缺少配置项：{', '.join(missing)}")}


# ══════════════════════════════════════════════════════════
def _fmt_code(code) -> str:
    """
    股票代码统一成 6 位。

    ⚠ 2026 修复：CSV 读回时 `pd.read_csv` 会把 "000625" 推断成整数 625，
       邮件正文就变成 "625"。代码一律补零，不做例外。
    """
    if code is None:
        return ""
    c = str(code).strip()
    if not c:
        return ""
    if c.isdigit() and len(c) <= 6:
        return c.zfill(6)
    return c


def _fmt_pct(x, nd=1) -> str:
    try:
        return f"{float(x)*100:.{nd}f}%"
    except Exception:
        return "—"


def render_daily(trade: dict | None = None, position: dict | None = None,
                 clean: dict | None = None,
                 trade_rows: list[dict] | None = None,
                 date: str | None = None) -> dict:
    """
    渲染每日报告（HTML + 纯文本）。

    三块：仓位状态 / 次日清单 / 数据异常
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    parts_txt = [f"QuantPilot 每日报告 {date}", "=" * 46]
    parts_html = [
        '<div style="font-family:-apple-system,Segoe UI,Microsoft YaHei,sans-serif;'
        'max-width:720px;margin:0 auto;color:#222">',
        f'<h2 style="margin:0 0 4px">QuantPilot 每日报告</h2>',
        f'<div style="color:#888;font-size:13px;margin-bottom:16px">{date}</div>',
    ]

    # ── 仓位
    if position:
        state = position.get("current_state") or "—"
        cap = position.get("current_cap")
        color = "#c0392b" if (cap is not None and cap <= 0.35) else "#27ae60"
        parts_txt += ["", f"【仓位】{state} → 最大仓位 {_fmt_pct(cap, 0)}",
                      f"  沪深300 {position.get('close', '—')}  "
                      f"均线 {position.get('ma', '—')}  "
                      f"偏离 {position.get('diff_pct', '—')}%"]
        parts_html.append(
            '<div style="border:1px solid #eee;border-radius:8px;padding:14px;'
            'margin-bottom:14px">'
            '<div style="font-size:13px;color:#888">仓位状态</div>'
            f'<div style="font-size:22px;font-weight:600;color:{color}">'
            f'{state} · 上限 {_fmt_pct(cap, 0)}</div>'
            f'<div style="font-size:13px;color:#666;margin-top:4px">'
            f"沪深300 {position.get('close', '—')} ｜ 均线 "
            f"{position.get('ma', '—')} ｜ 偏离 "
            f"{position.get('diff_pct', '—')}%</div></div>")

    # ── 清单
    if trade_rows:
        head = ["#", "代码", "目标权重", "参考收盘", "因子分"]
        parts_txt += ["", f"【次日清单】{len(trade_rows)} 只（次日开盘执行）"]
        loc = [4, 8, 10, 12, 10]
        parts_txt.append("  " + "".join(h.ljust(w) for h, w in zip(head, loc)))
        for r in trade_rows:
            cells = [str(r.get("rank", "")), _fmt_code(r.get("code")),
                     _fmt_pct(r.get("target_weight")), str(r.get("ref_close", "—")),
                     str(r.get("score", ""))]
            parts_txt.append("  " + "".join(c[:w].ljust(w)
                                            for c, w in zip(cells, loc)))
        rows_html = "".join(
            "<tr>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #f0f0f0'>{r.get('rank','')}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #f0f0f0'><b>{_fmt_code(r.get('code'))}</b></td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #f0f0f0;text-align:right'>{_fmt_pct(r.get('target_weight'))}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #f0f0f0;text-align:right'>{r.get('ref_close','—')}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #f0f0f0;text-align:right;color:#888'>{r.get('score','')}</td>"
            "</tr>" for r in trade_rows)
        parts_html.append(
            '<div style="border:1px solid #eee;border-radius:8px;padding:14px;'
            'margin-bottom:14px">'
            f'<div style="font-size:13px;color:#888;margin-bottom:8px">'
            f'次日清单（{len(trade_rows)} 只，**次日开盘**执行）</div>'
            '<table style="width:100%;border-collapse:collapse;font-size:14px">'
            '<tr style="background:#fafafa">'
            + "".join(f"<th style='padding:6px 10px;text-align:left;color:#555'>{h}</th>"
                      for h in head) +
            f"</tr>{rows_html}</table>"
            '<div style="font-size:12px;color:#b00;margin-top:8px">'
            '⚠ 候选池 ≠ 买入信号，因子在 2022-2024 只有弱正 IC</div></div>')
    elif trade is not None:
        parts_txt += ["", "【次日清单】今日无信号"]
        parts_html.append('<div style="border:1px solid #eee;border-radius:8px;'
                          'padding:14px;color:#888">今日无信号</div>')

    # ── 异常
    if clean and (clean.get("n_bad") or clean.get("n_stale")):
        nb, ns = clean.get("n_bad", 0), clean.get("n_stale", 0)
        parts_txt += ["", f"【数据异常】有问题 {nb} 个文件，陈旧 {ns} 个"]
        parts_html.append(
            '<div style="border:1px solid #f5c6cb;background:#fff5f5;'
            'border-radius:8px;padding:14px;color:#a00">'
            f'数据异常：{nb} 个文件有问题，{ns} 个数据陈旧'
            '<div style="font-size:12px;margin-top:4px">'
            '详见 reports/clean_*.md</div></div>')

    parts_txt += ["", "=" * 46,
                  "本报告由 QuantPilot 自动生成，仅供研究参考，不构成投资建议。"]
    parts_html += [
        '<div style="font-size:12px;color:#999;margin-top:16px;'
        'border-top:1px solid #eee;padding-top:10px">'
        '本报告由 QuantPilot 自动生成，仅供研究参考，不构成投资建议。</div>',
        "</div>",
    ]

    return {"subject": f"[QuantPilot] {date} 次日清单"
                        f"（{len(trade_rows or [])} 只"
                        + (f"，仓位上限 {_fmt_pct(position.get('current_cap'), 0)}"
                           if position else "") + "）",
            "text": "\n".join(parts_txt),
            "html": "".join(parts_html)}


def send_mail(subject: str, html: str, text: str | None = None,
              attachments: list[Path] | None = None,
              cfg: dict | None = None,
              timeout: float = 20.0) -> dict:
    """
    发一封邮件。cfg 缺省时自动读取配置。

    返回 {ok, to, masked_user, error}；**绝不回传密码**。
    """
    got = {"ok": True, "cfg": cfg} if cfg else load_config()
    if not got.get("ok"):
        return {"ok": False, "error": got.get("error"),
                "missing": got.get("missing")}
    c = got["cfg"]

    msg = MIMEMultipart("mixed")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((str(c.get("from_name") or "QuantPilot"),
                              str(c["user"])))
    msg["To"] = ", ".join(c["to"])

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text or "（无纯文本内容）", "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)

    for fp in attachments or []:
        try:
            p = Path(fp)
            if not p.exists():
                continue
            part = MIMEApplication(p.read_bytes())
            part.add_header("Content-Disposition", "attachment",
                            filename=("utf-8", "", p.name))
            msg.attach(part)
        except Exception:
            continue

    host = str(c["smtp_host"])
    port = int(c.get("smtp_port") or 465)
    try:
        if c.get("use_ssl"):
            ctx = ssl.create_default_context()
            server = smtplib.SMTP_SSL(host, port, timeout=timeout, context=ctx)
        else:
            server = smtplib.SMTP(host, port, timeout=timeout)
        with server:
            server.ehlo()
            if c.get("use_tls") and not c.get("use_ssl"):
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            server.login(str(c["user"]), str(c["password"]))
            server.sendmail(str(c["user"]), c["to"], msg.as_string())
    except Exception as ex:
        return {"ok": False, "to": c.get("to"),
                "masked_user": mask(c.get("user")),
                "error": f"{type(ex).__name__}: {str(ex)[:200]}"}

    return {"ok": True, "to": c.get("to"),
            "masked_user": mask(c.get("user")),
            "subject": subject, "attachments": [str(a) for a in (attachments or [])]}


def send_daily_report(trade: dict | None = None, position: dict | None = None,
                      clean: dict | None = None,
                      trade_rows: list[dict] | None = None,
                      attach_csv: bool = True, cfg: dict | None = None) -> dict:
    """渲染 + 发送每日报告（一步到位）。"""
    r = render_daily(trade=trade, position=position, clean=clean,
                     trade_rows=trade_rows)
    att = []
    if attach_csv and trade and trade.get("csv"):
        att.append(Path(trade["csv"]))
    out = send_mail(r["subject"], r["html"], r["text"], att, cfg=cfg)
    out["subject"] = r["subject"]
    out["preview"] = {"text": r["text"], "html_len": len(r["html"])}
    return out


def format_config_help() -> str:
    """配置说明（打印给用户看）。"""
    L = ["=" * 74, "邮件推送配置说明", "=" * 74]
    L.append("  方式一（推荐）：写配置文件")
    L.append(f"    {CONFIG_PATH}")
    L.append("")
    L.append('    {')
    L.append('      "smtp_host": "smtp.qq.com",')
    L.append('      "smtp_port": 465,')
    L.append('      "use_ssl": true,')
    L.append('      "use_tls": false,')
    L.append('      "user": "你的QQ号@qq.com",')
    L.append('      "password": "授权码（不是登录密码）",')
    L.append('      "to": ["收件邮箱@example.com"],')
    L.append('      "from_name": "QuantPilot"')
    L.append("    }")
    L.append("")
    L.append("  方式二：环境变量（优先级更高）")
    for k in ("QP_SMTP_HOST", "QP_SMTP_PORT", "QP_SMTP_USER", "QP_SMTP_PASS",
              "QP_SMTP_SSL", "QP_SMTP_TLS", "QP_MAIL_TO"):
        L.append(f"    {k}")
    L.append("")
    L.append("  常见服务商预设：")
    for k, v in PROVIDERS.items():
        L.append(f"    {k:<8}{v['smtp_host']}:{v['smtp_port']}  "
                 f"ssl={v['use_ssl']} tls={v['use_tls']}   {v['note']}")
    L.append("=" * 74)
    L.append("  ⚠ 授权码 ≠ 登录密码；QQ/163 需先在网页版邮箱设置里开启 SMTP 服务")
    L.append("  ⚠ config/mail.local.json 是私有文件，不要提交/分享")
    L.append("=" * 74)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="邮件推送")
    ap.add_argument("--help-config", action="store_true")
    ap.add_argument("--check", action="store_true", help="检查配置是否完整")
    ap.add_argument("--preview", action="store_true", help="只渲染预览，不发信")
    ap.add_argument("--test", action="store_true", help="发一封测试邮件")
    args = ap.parse_args()

    if args.help_config:
        print(format_config_help())
        raise SystemExit(0)

    if args.check:
        g = load_config()
        print(f"  配置来源：{g['source']}")
        for k, v in (g.get("display") or {}).items():
            print(f"    {k:<12}{v}")
        print(f"  状态：{'完整' if g['ok'] else '不完整 → ' + (g.get('error') or '')}")
        raise SystemExit(0 if g["ok"] else 1)

    demo_rows = [
        {"rank": 1, "code": "600519", "target_weight": 0.0300,
         "ref_close": 1410.27, "score": -0.0137},
        {"rank": 2, "code": "600036", "target_weight": 0.0300,
         "ref_close": 42.15, "score": -0.0151},
    ]
    pos = {"current_state": "均线下方", "current_cap": 0.30, "close": 4016.29,
           "ma": 4069.52, "diff_pct": -1.31}
    r = render_daily(trade={"csv": None}, position=pos, clean=None,
                     trade_rows=demo_rows)
    if args.preview:
        print("主题：", r["subject"])
        print("-" * 60)
        print(r["text"])
        out = _ROOT / ".tmp" / "mail_preview.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r["html"], encoding="utf-8")
        print("-" * 60)
        print(f"HTML 预览已写入 {out}")
        raise SystemExit(0)

    if args.test:
        out = send_mail("[QuantPilot] 测试邮件", r["html"], r["text"])
        print(out)
        raise SystemExit(0 if out.get("ok") else 1)

    print(format_config_help())
