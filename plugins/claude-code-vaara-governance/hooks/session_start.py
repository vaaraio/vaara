#!/usr/bin/env python3
"""SessionStart hook: validate the Vaara install and print a one-line
governance summary so the operator sees the plugin loaded.

Exits 0 in all cases; missing or broken install just prints a warning.

Env vars match the rest of the plugin (VAARA_PLUGIN_DISABLE,
VAARA_PLUGIN_AUDIT_DB, VAARA_PLUGIN_SHADOW).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _audit_db_path() -> Path:
    override = os.environ.get("VAARA_PLUGIN_AUDIT_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "claude-code" / "audit.db"


def main() -> int:
    if os.environ.get("VAARA_PLUGIN_DISABLE") == "1":
        _emit("vaara-governance: disabled via VAARA_PLUGIN_DISABLE=1.")
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

    mode = "shadow" if os.environ.get("VAARA_PLUGIN_SHADOW") == "1" else "enforce"
    db_path = _audit_db_path()
    db_state = "new" if not db_path.exists() else "existing"

    _emit(
        f"vaara-governance v0.1.0 loaded "
        f"(vaara {vaara.__version__}, mode={mode}, audit_db={db_path} [{db_state}])."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
