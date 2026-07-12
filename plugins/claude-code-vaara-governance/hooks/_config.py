"""Plugin settings from ``~/.vaara/claude-code/config.json``.

Written by ``/vaara-setup`` and hand-editable. Environment variables win
over the file, so existing env-based setups keep working unchanged. A
missing or malformed file means defaults; a settings file must never be
able to break the hooks.

Keys:
  mode           "protect" (default) | "watch" (record, never block) | "off"
  protection     policy preset: "eco" | "balanced" | "performance" | "strict"
  notifications  true (default) | false, desktop popups on block/escalate
  agent_id       agent id written to the audit chain (default "claude-code")
  audit_db       audit DB path (default ~/.vaara/claude-code/audit.db)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_PATH = Path.home() / ".vaara" / "claude-code" / "config.json"


def load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def plugin_disabled(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_DISABLE") == "1":
        return True
    return cfg.get("mode") == "off"


def shadow_mode(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_SHADOW") == "1":
        return True
    return cfg.get("mode") == "watch"


def agent_id(cfg: dict) -> str:
    return os.environ.get("VAARA_PLUGIN_AGENT_ID") or cfg.get("agent_id") or "claude-code"


def audit_db_path(cfg: dict) -> Path:
    override = os.environ.get("VAARA_PLUGIN_AUDIT_DB") or cfg.get("audit_db")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "claude-code" / "audit.db"


def notifications_enabled(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_NOTIFY") == "0":
        return False
    return cfg.get("notifications", True) is not False


def protection_preset(cfg: dict) -> str | None:
    preset = os.environ.get("VAARA_PLUGIN_PROTECTION") or cfg.get("protection")
    return preset if isinstance(preset, str) and preset else None
