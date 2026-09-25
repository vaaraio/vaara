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
requests older than 10 minutes) and ``nonce``. The decision file carries
``decision`` ("approve" or "deny"), ``decided_at`` and ``mac``.

A decision counts only with a valid ``mac``: HMAC-SHA256 under the approval
key (``keys/approval-hmac.key`` beside the approvals directory, hex, created
by the gate on first use with mode 0600) over
``vaara-approval/v1``, the action id, the request's nonce and the decision,
one per line. Any process of the user could write a decision file; only one
that can read the key can sign it, and the deny rules refuse the governed
agent both the key and the directory. The nonce ties a decision to one
request, so an old signed decision cannot answer a new one.

The gate owns both files' lifecycle: they are removed on approve, deny,
and timeout alike, so an unattended machine never accumulates stale
requests.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

APPROVALS_DIR = Path.home() / ".vaara" / "approvals"
KEY_NAME = "approval-hmac.key"
MAC_CONTEXT = "vaara-approval/v1"

__all__ = ["APPROVALS_DIR", "approval_key_path", "decision_mac", "request_approval",
           "write_decision"]


def approval_key_path(approvals_dir: Path = APPROVALS_DIR) -> Path:
    """The key file for ``approvals_dir``: ``keys/`` in its parent."""
    return Path(approvals_dir).parent / "keys" / KEY_NAME


def _approval_key(approvals_dir: Path, create: bool) -> Optional[bytes]:
    """The approval key, created with mode 0600 if missing and ``create``."""
    path = approval_key_path(approvals_dir)
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass  # another gate made it first; read theirs
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_hex(32))
    try:
        return bytes.fromhex(path.read_text().strip())
    except (OSError, ValueError):
        return None


def decision_mac(key: bytes, action_id: str, nonce: str, decision: str) -> str:
    """Hex HMAC-SHA256 of one decision, as the app and the gate compute it."""
    message = "\n".join((MAC_CONTEXT, action_id, nonce, decision)).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def write_decision(action_id: str, decision: str, *,
                   approvals_dir: Path = APPROVALS_DIR) -> bool:
    """Answer a pending request as a human surface does: signed, atomically.

    Returns False when there is no such request or no key to sign with.
    """
    approvals_dir = Path(approvals_dir)
    key = _approval_key(approvals_dir, create=False)
    try:
        request = json.loads((approvals_dir / f"{action_id}.request.json").read_text())
    except (OSError, ValueError):
        return False
    nonce = request.get("nonce")
    if key is None or not isinstance(nonce, str) or decision not in ("approve", "deny"):
        return False
    _write_atomic(approvals_dir / f"{action_id}.decision.json", json.dumps({
        "decision": decision,
        "decided_at": time.time(),
        "mac": decision_mac(key, action_id, nonce, decision),
    }))
    return True


def _write_atomic(path: Path, text: str) -> None:
    """Publish ``text`` at ``path`` in one step, or not at all.

    ``Path.write_text`` opens with O_TRUNC and then writes, so between those
    two calls the file exists and is empty. A watcher polling the directory
    sees it and reads nothing: measured at 66 of 300 requests on a warm
    filesystem, which is not a rare race. A watcher that calls json.loads on
    what it finds raises there, and if that kills its poll loop the request is
    never answered and the gate blocks for its whole timeout.

    Writing under a temporary name in the same directory and renaming keeps
    the request invisible until it is complete, since os.replace is atomic
    within a filesystem.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


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

    Returns ``"approve"``, ``"deny"``, or ``"timeout"``. Any unreadable,
    unexpected or unsigned decision is ignored and polling continues, so a
    corrupt or forged file can never be mistaken for consent. Without a
    readable key no decision can verify, and the request times out. Cleans
    up its own request and decision files in every outcome.
    """
    approvals_dir = Path(approvals_dir)
    approvals_dir.mkdir(parents=True, exist_ok=True)
    key = _approval_key(approvals_dir, create=True)
    nonce = secrets.token_hex(16)
    request_file = approvals_dir / f"{action_id}.request.json"
    decision_file = approvals_dir / f"{action_id}.decision.json"
    _write_atomic(request_file, json.dumps({
        "action_id": action_id,
        "tool_name": tool_name,
        "reason": reason,
        "requested_at": time.time(),
        "nonce": nonce,
    }))
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if decision_file.exists():
                try:
                    answer = json.loads(decision_file.read_text())
                except (ValueError, OSError):
                    answer = {}
                decision = answer.get("decision", "") if isinstance(answer, dict) else ""
                mac = answer.get("mac", "") if isinstance(answer, dict) else ""
                if (decision in ("approve", "deny") and key is not None
                        and isinstance(mac, str)
                        and hmac.compare_digest(
                            mac, decision_mac(key, action_id, nonce, decision))):
                    return decision
            time.sleep(poll_interval)
        return "timeout"
    finally:
        for file in (request_file, decision_file):
            try:
                file.unlink()
            except OSError:
                pass
