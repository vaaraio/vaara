#!/usr/bin/env python3
"""SessionStart hook: validate the Vaara install and print a one-line
governance summary so the operator sees the plugin loaded.

Exits 0 in all cases; missing or broken install just prints a warning.

Env vars match the rest of the plugin (VAARA_PLUGIN_DISABLE,
VAARA_PLUGIN_AUDIT_DB, VAARA_PLUGIN_SHADOW).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config  # noqa: E402

CFG = _config.load_config()


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _audit_db_path() -> Path:
    return _config.audit_db_path(CFG)


def main() -> int:
    if _config.plugin_disabled(CFG):
        _emit("vaara-governance: off (config.json mode or VAARA_PLUGIN_DISABLE=1).")
        return 0

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

    _emit(
        f"vaara-governance v0.2.0 loaded "
        f"(vaara {vaara.__version__}, mode={mode}, protection={preset}, "
        f"notifications={notif}, audit_db={db_path} [{db_state}]). "
        f"Settings: /vaara-setup"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
