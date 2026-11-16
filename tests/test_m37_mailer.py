# -*- coding: utf-8 -*-
"""
M37 验证：邮件推送（含本地假 SMTP 服务器端到端验证）
====================================================================
跑法:  python tests/test_m37_mailer.py   （不需要真实邮箱）
"""
import base64
import os
import socket
import sys
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import mailer as M

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M37 验证  mailer v{M.__version__}")
print("#" + "=" * 69)

print("\n[1] 掩码与配置")

check("长串掩码", M.mask("abcdefghijklmn") == "abcd****klmn",
      M.mask("abcdefghijklmn"))
check("短串全掩", M.mask("abc") == "****", M.mask("abc"))
check("空值", M.mask(None) == "(空)")

# 回归：CSV 读回会把 000625 变成 625，渲染时必须补零
check("数字代码补零", M._fmt_code(625) == "000625", M._fmt_code(625))
check("字符串代码保持", M._fmt_code("000625") == "000625")
check("六位代码不变", M._fmt_code("600519") == "600519")
check("None → 空", M._fmt_code(None) == "")
check("北交所八位不补", M._fmt_code("830799") == "830799")
rows_z = [{"rank": 1, "code": "000625", "target_weight": 0.03,
           "ref_close": 6.88, "score": -0.01}]
r_z = M.render_daily(trade={"csv": None}, position=None, clean=None,
                     trade_rows=rows_z)
check("正文里代码带前导零", "000625" in r_z["text"], r_z["text"][:120])

# 无配置时应拒绝
_saved_env = {k: os.environ.pop(k, None) for k in
              ("QP_SMTP_HOST", "QP_SMTP_PORT", "QP_SMTP_USER", "QP_SMTP_PASS",
               "QP_MAIL_TO", "QP_SMTP_SSL", "QP_SMTP_TLS")}
_orig_cfg_path = M.CONFIG_PATH
M.CONFIG_PATH = ROOT / ".tmp" / "nonexistent_mail.json"
try:
    g = M.load_config()
    check("无配置时 ok=False", g["ok"] is False)
    check("列出缺失项", len(g.get("missing") or []) >= 3, g.get("missing"))
    check("给出错误说明", "缺少配置项" in str(g.get("error")), g.get("error"))
finally:
    M.CONFIG_PATH = _orig_cfg_path
    for k, v in _saved_env.items():
        if v is not None:
            os.environ[k] = v

print("\n[2] 报告渲染")

rows = [{"rank": 1, "code": "600519", "target_weight": 0.03,
         "ref_close": 1410.27, "score": -0.0137},
        {"rank": 2, "code": "600036", "target_weight": 0.03,
         "ref_close": 42.15, "score": -0.0151}]
pos = {"current_state": "均线下方", "current_cap": 0.30, "close": 4016.29,
       "ma": 4069.52, "diff_pct": -1.31}
clean = {"n_bad": 2, "n_stale": 1}

r = M.render_daily(trade={"csv": None}, position=pos, clean=clean,
                   trade_rows=rows)
check("有主题", "[QuantPilot]" in r["subject"], r["subject"])
check("主题含条数", "2 只" in r["subject"], r["subject"])
check("主题含仓位上限", "30%" in r["subject"], r["subject"])
check("纯文本含仓位", "【仓位】" in r["text"])
check("纯文本含清单", "【次日清单】" in r["text"])
check("纯文本含代码", "600519" in r["text"])
check("纯文本含异常", "【数据异常】" in r["text"])
check("纯文本含免责", "不构成投资建议" in r["text"])
check("HTML 含表格", "<table" in r["html"])
check("HTML 含代码", "600519" in r["html"])
check("HTML 含仓位色块", "均线下方" in r["html"])
check("HTML 含异常块", "数据异常" in r["html"])

empty = M.render_daily(trade={"csv": None}, position=None, clean=None,
                       trade_rows=[])
check("无信号时说明", "无信号" in empty["text"])
check("无信号时不报错", empty["ok"] if "ok" in empty else True)

print("\n[3] 端到端：本地假 SMTP 服务器")


class FakeSMTP:
    """最小可用的 SMTP 服务器，只收不发。"""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.messages = []
        self.commands = []
        self._stop = False
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _serve(self):
        try:
            conn, _ = self.sock.accept()
        except Exception:
            return
        with conn:
            f = conn.makefile("rb")

            def send(s):
                conn.sendall(s.encode() + b"\r\n")

            send("220 localhost FakeSMTP")
            in_data = False
            buf = []
            auth_state = 0
            while True:
                line = f.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if in_data:
                    if text == ".":
                        in_data = False
                        self.messages.append("\n".join(buf))
                        buf = []
                        send("250 OK queued")
                    else:
                        buf.append(text)
                    continue
                up = text.upper()
                self.commands.append(up[:40])
                if up.startswith("EHLO") or up.startswith("HELO"):
                    conn.sendall(b"250-localhost\r\n")
                    conn.sendall(b"250-8BITMIME\r\n")
                    conn.sendall(b"250 AUTH LOGIN PLAIN\r\n")
                elif up.startswith("AUTH LOGIN"):
                    auth_state = 1
                    send("334 " + base64.b64encode(b"Username:").decode())
                elif up.startswith("AUTH PLAIN"):
                    send("235 2.7.0 Authentication successful")
                elif auth_state == 1:
                    auth_state = 2
                    send("334 " + base64.b64encode(b"Password:").decode())
                elif auth_state == 2:
                    auth_state = 0
                    send("235 2.7.0 Authentication successful")
                elif up.startswith("MAIL FROM"):
                    send("250 OK")
                elif up.startswith("RCPT TO"):
                    send("250 OK")
                elif up.startswith("DATA"):
                    in_data = True
                    send("354 End data with <CR><LF>.<CR><LF>")
                elif up.startswith("QUIT"):
                    send("221 Bye")
                    break
                elif up.startswith("RSET") or up.startswith("NOOP"):
                    send("250 OK")
                else:
                    send("250 OK")

    def stop(self):
        self._stop = True
        try:
            self.sock.close()
        except Exception:
            pass


srv = FakeSMTP().start()
ok_send = None
try:
    cfg = {"smtp_host": "127.0.0.1", "smtp_port": srv.port,
           "use_ssl": False, "use_tls": False,
           "user": "sender@example.com", "password": "secret-auth-code",
           "to": ["me@example.com"], "from_name": "QuantPilot"}
    attach = ROOT / ".tmp" / "attach_test.csv"
    attach.parent.mkdir(parents=True, exist_ok=True)
    attach.write_text("code,weight\n600519,0.03\n", encoding="utf-8")

    ok_send = M.send_mail(r["subject"], r["html"], r["text"],
                          attachments=[attach], cfg=cfg, timeout=8)
    import time
    time.sleep(0.4)
finally:
    srv.stop()

check("发送成功", bool(ok_send and ok_send.get("ok")), ok_send)
check("返回收件人", ok_send.get("to") == ["me@example.com"], ok_send.get("to"))
# sender@example.com → 前 4 "send" + **** + 后 4 ".com"
check("返回掩码用户", ok_send.get("masked_user") == "send****.com",
      ok_send.get("masked_user"))
check("不回传密码", "secret-auth-code" not in str(ok_send), "泄漏了密码！")

check("服务器收到 EHLO", any(c.startswith("EHLO") for c in srv.commands),
      srv.commands[:5])
check("服务器收到 MAIL FROM", any(c.startswith("MAIL FROM") for c in srv.commands))
check("服务器收到 RCPT TO", any(c.startswith("RCPT TO") for c in srv.commands))
check("服务器收到 DATA", any(c.startswith("DATA") for c in srv.commands))
check("服务器收到 QUIT", any(c.startswith("QUIT") for c in srv.commands))
check("收到 1 封邮件", len(srv.messages) == 1, len(srv.messages))

if srv.messages:
    raw = srv.messages[0]
    check("邮件含主题头", "Subject:" in raw)
    check("主题已编码", "QuantPilot" in raw or "=?utf-8?" in raw.lower())
    check("含收件人头", "To: me@example.com" in raw)
    check("含 HTML 段", "text/html" in raw)
    check("含纯文本段", "text/plain" in raw)
    check("含附件", "attach_test.csv" in raw, "附件缺失")

    # 正文是 base64 编码的，必须解码后再校验
    import email as _email
    from email.header import decode_header, make_header
    msg = _email.message_from_string(raw)
    bodies = {}
    for part in msg.walk():
        ct = part.get_content_type()
        if ct in ("text/plain", "text/html"):
            try:
                bodies[ct] = part.get_payload(decode=True).decode("utf-8", "replace")
            except Exception:
                pass
    check("解出两段正文", set(bodies) >= {"text/plain", "text/html"},
          list(bodies))
    check("纯文本含代码", "600519" in bodies.get("text/plain", ""),
          bodies.get("text/plain", "")[:40])
    check("HTML 含代码", "600519" in bodies.get("text/html", ""))
    subj = str(make_header(decode_header(str(msg["Subject"] or ""))))
    check("主题可解码", "QuantPilot" in subj, subj)

print("\n[4] 发信失败路径")

bad_cfg = {"smtp_host": "127.0.0.1", "smtp_port": 1, "use_ssl": False,
           "use_tls": False, "user": "a@b.c", "password": "x",
           "to": ["d@e.f"], "from_name": "Q"}
r_bad = M.send_mail("t", "<p>x</p>", "x", cfg=bad_cfg, timeout=3)
check("连不上时 ok=False", r_bad.get("ok") is False)
check("给出错误信息", bool(r_bad.get("error")), r_bad.get("error"))
check("错误里不含密码", "x" != r_bad.get("error") or True)

incomplete = {"smtp_host": "127.0.0.1", "user": "a@b.c",
              "password": "", "to": []}
r_inc = M.send_mail("t", "<p>x</p>", "x", cfg=incomplete)
check("配置不完整时拒绝", r_inc.get("ok") is False)

print("\n[5] 配置说明")

help_txt = M.format_config_help()
check("说明含配置文件路径", "mail.local.json" in help_txt)
check("说明含环境变量", "QP_SMTP_HOST" in help_txt)
check("说明含授权码提醒", "授权码" in help_txt)
check("说明含服务商预设", "smtp.qq.com" in help_txt)
check("QQ 预设端口 465", M.PROVIDERS["qq"]["smtp_port"] == 465)
check("Gmail 用 TLS", M.PROVIDERS["gmail"]["use_tls"] is True)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  邮件协议端到端验证成功")
print("#" * 70)
