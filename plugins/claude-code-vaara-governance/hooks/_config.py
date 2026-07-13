"""Plugin settings from ``~/.vaara/claude-code/config.json``.

Written by ``/vaara-setup`` and hand-editable. Environment variables win
over the file, so existing env-based setups keep working unchanged. A
missing or malformed file means defaults; a settings file must never be
able to break the hooks.

Keys:
  mode           "protect" (default) | "watch" (record, never block) | "off"
  protection     policy preset: "eco" | "balanced" | "performance" | "strict"
  thresholds     optional {"escalate": E, "deny": D} overriding the preset's
                 default thresholds (0 <= E <= D <= 1)
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


def fail_open(cfg: dict) -> bool:
    """Whether a missing vaara package passes MCP calls through unscored.

    Default False: in protect mode the gate fails closed when its engine
    is not importable. ``"fail_open": true`` in config.json opts back
    into availability-over-enforcement.
    """
    if os.environ.get("VAARA_PLUGIN_FAIL_OPEN") == "1":
        return True
    return cfg.get("fail_open") is True


def protection_preset(cfg: dict) -> str | None:
    preset = os.environ.get("VAARA_PLUGIN_PROTECTION") or cfg.get("protection")
    return preset if isinstance(preset, str) and preset else None


def custom_thresholds(cfg: dict) -> tuple[float, float] | None:
    """Optional custom decision thresholds from config.json.

    Shape: ``"thresholds": {"escalate": 0.5, "deny": 0.8}``. Both keys
    required, numeric, with 0 <= escalate <= deny <= 1. Anything else is
    ignored (never break the hook over a bad settings file). When present,
    these override the preset's default thresholds; the preset still
    provides the rest of the policy shape.
    """
    raw = cfg.get("thresholds")
    if not isinstance(raw, dict):
        return None
    escalate, deny = raw.get("escalate"), raw.get("deny")
    if not isinstance(escalate, (int, float)) or not isinstance(deny, (int, float)):
        return None
    if isinstance(escalate, bool) or isinstance(deny, bool):
        return None
    if not (0 <= escalate <= deny <= 1):
        return None
    return float(escalate), float(deny)
