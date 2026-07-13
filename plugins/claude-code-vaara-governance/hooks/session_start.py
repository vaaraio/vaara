#!/usr/bin/env python3
"""SessionStart hook: validate the Vaara install, print a one-line
governance summary, and (when configured) record the EU AI Act
Article 50(1) disclosure for this session into the audit trail.

Exits 0 in all cases; missing or broken install just prints a warning,
and the disclosure step can never break the hook.

Env vars match the rest of the plugin (VAARA_PLUGIN_DISABLE,
VAARA_PLUGIN_AUDIT_DB, VAARA_PLUGIN_SHADOW,
VAARA_PLUGIN_ARTICLE50_STATEMENT).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config  # noqa: E402

CFG = _config.load_config()


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _audit_db_path() -> Path:
    return _config.audit_db_path(CFG)


def _plugin_version() -> str:
    try:
        manifest = Path(__file__).parent.parent / ".claude-plugin" / "plugin.json"
        return json.loads(manifest.read_text()).get("version", "?")
    except Exception:
        return "?"


def _record_session_disclosure(statement: str, session_id: str) -> str:
    """Record the 50(1) disclosure for this session. Returns a status word."""
    try:
        from vaara.audit.article50 import record_disclosure
        from vaara.audit.sqlite_backend import SQLiteAuditBackend
    except ImportError:
        return "skipped (needs vaara>=1.27)"
    try:
        db_path = _audit_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        backend = SQLiteAuditBackend(db_path)
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        record_disclosure(
            trail,
            paragraph="50(1)",
            statement=statement,
            agent_id=_config.agent_id(CFG),
            session_id=session_id,
            channel="claude-code-session",
        )
        return "recorded"
    except Exception as exc:
        return f"failed ({exc!r})"


def main() -> int:
    if _config.plugin_disabled(CFG):
        _emit("vaara-governance: off (config.json mode or VAARA_PLUGIN_DISABLE=1).")
        return 0

    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        event = {}
    session_id = event.get("session_id", "") if isinstance(event, dict) else ""

    try:
        import vaara
    except ImportError:
        _emit(
            "vaara-governance: vaara package not importable. "
            "Run `pip install vaara>=0.40.1` to enable runtime governance. "
            "Tool calls will pass through unchecked until then."
        )
        return 0

    mode = "watch (nothing blocked, all recorded)" if _config.shadow_mode(CFG) else "protect"
    preset = _config.protection_preset(CFG) or "balanced"
    notif = "on" if _config.notifications_enabled(CFG) else "off"
    db_path = _audit_db_path()
    existed = db_path.exists()
    try:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend

        db_path.parent.mkdir(parents=True, exist_ok=True)
        SQLiteAuditBackend(db_path)
        db_state = "existing" if existed else "created"
    except Exception as exc:
        db_state = f"unavailable ({exc!r})"

    disclosure = ""
    statement = _config.article50_statement(CFG)
    if statement:
        status = _record_session_disclosure(statement, session_id)
        disclosure = f", article50_disclosure={status}"

    _emit(
        f"vaara-governance v{_plugin_version()} loaded "
        f"(vaara {vaara.__version__}, mode={mode}, protection={preset}, "
        f"notifications={notif}, audit_db={db_path} [{db_state}]{disclosure}). "
        f"Settings: /vaara-setup"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
