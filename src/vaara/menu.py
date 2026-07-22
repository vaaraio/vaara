# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara menu``: the CLI as a menu instead of a wall of subcommands.

The full CLI surface is dozens of subcommands; a person who does not
live in a terminal needs five. This module renders a numbered menu over
the existing commands, gated by a settings depth the user picks once:

- ``basic``: status, record a disclosure, export the evidence package,
  settings, update check.
- ``professional``: adds verify and the shadow report.
- ``enterprise``: adds the Article 12 export and receipt anchoring.

Every menu item delegates to the same ``vaara.cli`` entry points the
flags reach, so the menu adds no second code path: it only asks the
questions the flags would have demanded up front. The chosen level is
stored as ``user_level`` in the plugin config
(``~/.vaara/claude-code/config.json``), next to the keys the macOS app
and the Claude Code hook already read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

LEVELS = ("basic", "professional", "enterprise")

CONFIG_PATH = Path(
    os.environ.get("VAARA_PLUGIN_CONFIG")
    or Path.home() / ".vaara" / "claude-code" / "config.json"
)
DEFAULT_DB = Path.home() / ".vaara" / "claude-code" / "audit.db"


def _load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def user_level() -> str:
    level = _load_config().get("user_level", "")
    return level if level in LEVELS else "basic"


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def _cli(args: list[str]) -> int:
    from vaara.cli import main

    return main(args)


# ---------------------------------------------------------------- actions

def _status() -> None:
    import sqlite3
    from datetime import datetime, timezone

    db = Path(_ask("Audit DB", str(DEFAULT_DB))).expanduser()
    if not db.exists():
        print(f"No trail yet at {db}. It appears once an agent runs governed.")
        return
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0]
        blocked = conn.execute(
            "SELECT COUNT(*) FROM audit_records WHERE event_type = 'action_blocked'"
        ).fetchone()[0]
        disclosures = conn.execute(
            "SELECT COUNT(*) FROM audit_records WHERE tool_name = "
            "'vaara.article50.disclosure' AND event_type = 'action_requested'"
        ).fetchone()[0]
        agents = conn.execute(
            "SELECT COUNT(DISTINCT agent_id) FROM audit_records"
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
    print(f"Trail: {db}")
    print(f"  records:               {total}")
    print(f"  agents seen:           {agents}")
    print(f"  actions blocked:       {blocked}")
    print(f"  Article 50 disclosures: {disclosures}")
    print(f"  last activity (UTC):   {when}")


def _record_disclosure() -> None:
    db = _ask("Audit DB", str(DEFAULT_DB))
    statement = _ask("What was disclosed (the notice text)")
    if not statement:
        print("A disclosure needs the statement.")
        return
    principal = _ask("On whose behalf does the agent act (blank = generic record)")
    args = ["trail", "record-disclosure", "--db", db, "--statement", statement]
    if principal:
        step = _ask(
            "Step (first_interaction / authorisation / reporting / "
            "validation / new_interaction / ai_output_received / on_inquiry)",
            "first_interaction",
        )
        args += ["--on-behalf-of", principal, "--step", step]
    _cli(args)


def _export_article50() -> None:
    db = _ask("Audit DB", str(DEFAULT_DB))
    key = _ask("Signing key (PEM). Blank makes a dev key at ~/.vaara/signing.pem")
    if not key:
        key = str(Path.home() / ".vaara" / "signing.pem")
        if not Path(key).exists():
            _cli(["keygen", "--dev", "--out", key])
    out = _ask("Write the package to", "article50.zip")
    _cli(["trail", "export-article50", "--db", db, "--key", key, "--out", out])


def _verify() -> None:
    zip_path = _ask("Package zip to verify")
    if not zip_path:
        return
    pubkey = _ask("Trusted public key (blank = key inside the zip)")
    args = ["trail", "verify", "--zip", zip_path]
    if pubkey:
        args += ["--pubkey", pubkey]
    _cli(args)


def _shadow_report() -> None:
    db = _ask("Audit DB", str(DEFAULT_DB))
    _cli(["trail", "shadow-report", "--db", db])


def _export_article12() -> None:
    db = _ask("Audit DB", str(DEFAULT_DB))
    key = _ask("Signing key (PEM)")
    if not key:
        print("The Article 12 package must be signed; a key is required.")
        return
    out = _ask("Write the package to", "article12.zip")
    _cli(["trail", "export-article12", "--db", db, "--key", key, "--out", out])


def _check_updates() -> None:
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


def _settings() -> None:
    cfg = _load_config()
    print(f"  1) Settings depth      now: {cfg.get('user_level', 'basic')}")
    print(f"  2) Gate mode           now: {cfg.get('mode', 'protect')}")
    print(f"  3) Protection preset   now: {cfg.get('protection', 'balanced')}")
    print(f"  4) Article 50 principal now: "
          f"{cfg.get('article50_on_behalf_of', '(unset)')}")
    choice = _ask("Change which (blank = back)")
    if choice == "1":
        level = _ask("basic / professional / enterprise", user_level())
        if level in LEVELS:
            cfg["user_level"] = level
    elif choice == "2":
        mode = _ask("protect (blocks) / watch (records only)",
                    cfg.get("mode", "protect"))
        if mode in ("protect", "watch"):
            cfg["mode"] = mode
    elif choice == "3":
        preset = _ask("eco / balanced / performance / strict",
                      cfg.get("protection", "balanced"))
        if preset in ("eco", "balanced", "performance", "strict"):
            cfg["protection"] = preset
    elif choice == "4":
        principal = _ask("Person or company the agent acts for (blank clears)")
        if principal:
            cfg["article50_on_behalf_of"] = principal
        else:
            cfg.pop("article50_on_behalf_of", None)
    else:
        return
    _save_config(cfg)
    print("Saved.")


# ---------------------------------------------------------------- menu

#: (label, minimum level, action)
ITEMS: list[tuple[str, str, Callable[[], None]]] = [
    ("Status: what the trail holds", "basic", _status),
    ("Record an Article 50 disclosure", "basic", _record_disclosure),
    ("Export the Article 50 evidence package", "basic", _export_article50),
    ("Verify an evidence package", "professional", _verify),
    ("Shadow report: what would have been blocked", "professional",
     _shadow_report),
    ("Export the Article 12 record-keeping package", "enterprise",
     _export_article12),
    ("Settings", "basic", _settings),
    ("Check for updates", "basic", _check_updates),
]


def visible_items(level: str) -> list[tuple[str, Callable[[], None]]]:
    rank = {name: i for i, name in enumerate(LEVELS)}
    threshold = rank.get(level, 0)
    return [
        (label, action) for label, minimum, action in ITEMS
        if rank[minimum] <= threshold
    ]


def run_menu(once: bool = False) -> int:
    """Interactive loop. ``once`` renders and handles a single choice."""
    while True:
        level = user_level()
        items = visible_items(level)
        print()
        print(f"vaara · settings depth: {level} "
              "(change under Settings)")
        for i, (label, _) in enumerate(items, 1):
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
            _, action = items[index]
        except (ValueError, IndexError):
            print("Pick a number from the list.")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print()
        if once:
            return 0
