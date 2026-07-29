# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara menu``: a numbered list of the most useful commands.

The full CLI is dozens of subcommands. The menu shows the ones a person
needs day to day — status, setup, export, verify — without walls of text
or knowledge-level gatekeeping.

Every menu item delegates to the same ``vaara.cli`` entry points the flags
reach, so the menu adds no second code path: it only asks the questions
the flags would have demanded up front.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

CONFIG_PATH = Path(
    os.environ.get("VAARA_PLUGIN_CONFIG")
    or Path.home() / ".vaara" / "claude-code" / "config.json"
)
DEFAULT_DB = Path.home() / ".vaara" / "trail" / "audit.db"


def _load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def _cli(args: list[str]) -> int:
    from vaara.cli import main

    return main(args)


# ---------------------------------------------------------------- actions


def _status() -> None:
    """Show what Vaara is governing."""
    import sqlite3
    from datetime import datetime, timezone

    db = Path(_ask("Audit DB", str(DEFAULT_DB))).expanduser()
    if not db.exists():
        print("No trail yet. It appears once an agent runs governed.")
        return
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0]
        blocked = conn.execute(
            "SELECT COUNT(*) FROM audit_records WHERE event_type = 'action_blocked'"
        ).fetchone()[0]
        escalated = conn.execute(
            "SELECT COUNT(*) FROM audit_records WHERE event_type = 'escalation_sent'"
        ).fetchone()[0]
        agents = conn.execute(
            "SELECT COUNT(DISTINCT agent_id) FROM audit_records"
        ).fetchone()[0]
        tools = conn.execute(
            "SELECT COUNT(DISTINCT tool_name) FROM audit_records"
        ).fetchone()[0]
        last = conn.execute(
            "SELECT MAX(timestamp) FROM audit_records"
        ).fetchone()[0]
    except sqlite3.Error as exc:
        print(f"Could not read the trail: {exc}")
        return
    finally:
        conn.close()
    when = (
        datetime.fromtimestamp(last, tz=timezone.utc).isoformat(timespec="seconds")
        if last else "never"
    )
    print(f"  records:          {total}")
    print(f"  agents seen:      {agents}")
    print(f"  distinct tools:   {tools}")
    print(f"  actions blocked:  {blocked}")
    print(f"  escalations:      {escalated}")
    print(f"  last activity:    {when} UTC")


def _shadow_report() -> None:
    """Show what would have been blocked in shadow mode."""
    db = _ask("Audit DB", str(DEFAULT_DB))
    _cli(["trail", "shadow-report", "--db", db])


def _export() -> None:
    """Export evidence from the trail."""
    db = _ask("Audit DB", str(DEFAULT_DB))
    key = _ask("Signing key (PEM, blank = auto-generate dev key)")
    if not key:
        key = str(Path.home() / ".vaara" / "signing.pem")
        if not Path(key).exists():
            _cli(["keygen", "--dev", "--out", key])
    out = _ask("Output file", "evidence.zip")
    _cli(["trail", "export-bundle", "--db", db, "--key", key, "--out", out])


def _verify() -> None:
    """Verify a signed evidence package."""
    zip_path = _ask("Evidence package to verify")
    if not zip_path:
        return
    pubkey = _ask("Trusted public key (blank = key inside package)")
    args = ["trail", "verify", "--zip", zip_path]
    if pubkey:
        args += ["--pubkey", pubkey]
    _cli(args)


def _settings() -> None:
    """Settings: gate mode, protection preset."""
    from vaara.policy.modes import available_modes

    cfg = _load_config()
    print(f"  1) Gate mode           now: {cfg.get('mode', 'protect')}")
    print(f"  2) Protection preset   now: {cfg.get('protection', 'balanced')}")
    choice = _ask("Change which (blank = back)")
    if choice == "1":
        mode = _ask("protect (blocks) / watch (records only)",
                    cfg.get("mode", "protect"))
        if mode in ("protect", "watch"):
            cfg["mode"] = mode
    elif choice == "2":
        modes = list(available_modes())
        preset = _ask(" / ".join(modes), cfg.get("protection", "balanced"))
        if preset in modes:
            cfg["protection"] = preset
    else:
        return
    _save_config(cfg)
    print("Saved.")


def _check_updates() -> None:
    """Check whether a newer version of Vaara is available."""
    import urllib.request

    import vaara

    installed = vaara.__version__
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/vaara/json", timeout=10
        ) as resp:
            latest = json.load(resp)["info"]["version"]
    except Exception:
        print(f"Installed {installed}. Could not reach pypi.org to compare.")
        return
    if latest == installed:
        print(f"Up to date ({installed}).")
    else:
        print(
            f"{latest} is available (installed {installed}). "
            "Update: pip install -U vaara  (or: brew upgrade vaara)"
        )


# ---------------------------------------------------------------- menu

ITEMS: list[tuple[str, Callable[[], None]]] = [
    ("Status: what Vaara is governing", _status),
    ("Shadow report: what would have been blocked", _shadow_report),
    ("Export evidence package from the trail", _export),
    ("Verify a signed evidence package", _verify),
    ("Settings", _settings),
    ("Check for updates", _check_updates),
]


def run_menu(once: bool = False) -> int:
    """Interactive loop. ``once`` renders and exits after one choice."""
    while True:
        print()
        print("vaara")
        for i, (label, _) in enumerate(ITEMS, 1):
            print(f"  {i}) {label}")
        print("  q) quit")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return 0
        if choice in ("q", "quit", "exit", ""):
            return 0
        try:
            index = int(choice) - 1
            _, action = ITEMS[index]
        except (ValueError, IndexError):
            print("Pick a number from the list.")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print()
        if once:
            return 0
