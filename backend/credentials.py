# -*- coding: utf-8 -*-
"""
凭据管理（M33）
====================================================
原则
----
1. **Key 只从环境变量或本地配置文件读取**，绝不写进代码、绝不打印完整值
2. 显示一律掩码：前 4 位 + `****` + 后 4 位
3. 配置文件放在 `config/keys.local.json`，属于本机私有，不随项目分发
4. 环境变量优先于文件（方便临时覆盖/CI）

支持的 Key
----------
* `ROICAI_API_KEY`   —— ROIC.ai（美股基本面 + 价格）
* `XFINLINK_API_KEY` —— xfinlink（美股价格 / 基本面 / 持仓）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["CONFIG_PATH", "KNOWN_KEYS", "load_key", "save_key", "mask",
           "status", "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = _ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "keys.local.json"

KNOWN_KEYS = {
    "roicai": {"env": "ROICAI_API_KEY", "label": "ROIC.ai"},
    "xfinlink": {"env": "XFINLINK_API_KEY", "label": "xfinlink"},
}


def mask(key: str | None) -> str:
    """掩码显示，永不明文。"""
    if not key:
        return "(未设置)"
    k = str(key)
    if len(k) <= 8:
        return k[:2] + "****"
    return f"{k[:4]}****{k[-4:]}"


def _read_file() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def load_key(name: str) -> str | None:
    """
    读取指定服务的 Key。环境变量优先。

    name: "roicai" / "xfinlink"
    """
    if name not in KNOWN_KEYS:
        raise ValueError(f"未知服务 {name!r}，可选：{list(KNOWN_KEYS)}")
    env_name = KNOWN_KEYS[name]["env"]
    v = (os.environ.get(env_name) or "").strip()
    if v:
        return v
    v = (_read_file().get(name) or "").strip()
    return v or None


def save_key(name: str, key: str) -> dict:
    """写入本地配置文件（保留其它已有键）。"""
    if name not in KNOWN_KEYS:
        raise ValueError(f"未知服务 {name!r}")
    data = _read_file()
    data[name] = str(key).strip()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass
    return {"ok": True, "path": str(CONFIG_PATH), "masked": mask(data[name])}


def status() -> dict:
    """各 Key 是否就绪（只返回掩码）。"""
    out = {"config_path": str(CONFIG_PATH), "exists": CONFIG_PATH.exists(),
           "keys": {}}
    for name, meta in KNOWN_KEYS.items():
        val = load_key(name)
        out["keys"][name] = {
            "label": meta["label"],
            "env": meta["env"],
            "set": bool(val),
            "masked": mask(val),
            "source": ("env" if (os.environ.get(meta["env"]) or "").strip()
                       else ("file" if val else None)),
        }
    return out


def format_report(st: dict) -> str:
    L = ["=" * 66, "API 凭据", "=" * 66]
    L.append(f"  配置文件：{st['config_path']}"
             + ("（存在）" if st.get("exists") else "（不存在）"))
    L.append("-" * 66)
    for name, info in st["keys"].items():
        flag = "[已设置]" if info["set"] else "[未设置]"
        src = f"  来源 {info['source']}" if info["source"] else ""
        L.append(f"  {info['label']:<12}{flag:<10}{info['masked']}{src}")
    L.append("-" * 66)
    L.append("  优先读环境变量，其次读本地配置文件")
    L.append("=" * 66)
    return "\n".join(L)


if __name__ == "__main__":
    print(format_report(status()))
