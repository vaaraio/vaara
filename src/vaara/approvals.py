# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""File-based approval handshake between the gate and a human surface.

The pipeline (or a hook driving it) writes ``<action_id>.request.json``
into the approvals directory and polls for ``<action_id>.decision.json``;
whoever fronts the human — the ``vaara`` CLI or any
script — lists pending requests and writes the decision. Plain files with
no daemon and no dependencies, so every surface can adopt the protocol.
The request schema is exactly what a watcher reads: ``action_id``,
``tool_name``, ``reason``, ``requested_at`` (unix seconds; watchers skip
requests older than 10 minutes). The decision file carries ``decision``
("approve" or "deny") and ``decided_at``.

The gate owns both files' lifecycle: they are removed on approve, deny,
and timeout alike, so an unattended machine never accumulates stale
requests.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

APPROVALS_DIR = Path.home() / ".vaara" / "approvals"

__all__ = ["APPROVALS_DIR", "request_approval"]


def request_approval(
    action_id: str,
    tool_name: str,
    reason: str,
    *,
    approvals_dir: Path = APPROVALS_DIR,
    timeout: float = 60.0,
    poll_interval: float = 0.2,
) -> str:
    """Ask a human to approve ``action_id``; block until answered or timeout.

    Returns ``"approve"``, ``"deny"``, or ``"timeout"``. Any unreadable or
    unexpected decision value is ignored and polling continues, so a
    corrupt file can never be mistaken for consent. Cleans up its own
    request and decision files in every outcome.
    """
    approvals_dir = Path(approvals_dir)
    approvals_dir.mkdir(parents=True, exist_ok=True)
    request_file = approvals_dir / f"{action_id}.request.json"
    decision_file = approvals_dir / f"{action_id}.decision.json"
    request_file.write_text(json.dumps({
        "action_id": action_id,
        "tool_name": tool_name,
        "reason": reason,
        "requested_at": time.time(),
    }))
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if decision_file.exists():
                try:
                    decision = json.loads(decision_file.read_text()).get("decision", "")
                except (ValueError, OSError):
                    decision = ""
                if decision in ("approve", "deny"):
                    return decision
            time.sleep(poll_interval)
        return "timeout"
    finally:
        for file in (request_file, decision_file):
            try:
                file.unlink()
            except OSError:
                pass
