# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A local dashboard that runs wherever the CLI runs.

The macOS menu-bar client is native because a menu bar is native. Everything
else it shows (the trail, the anchors, the published heads, what is configured)
is data, and data does not need three implementations. This serves that data as
one page from the machine it already lives on.

Deliberate constraints, each of them the reason this exists:

- Standard library only. No FastAPI, no uvicorn, no build step. A bare install
  can run it.
- Binds 127.0.0.1 by default. The trail is the most sensitive thing on the box
  and nothing here should be reachable from the network by accident.
- Settings are editable, because a Linux or Windows user should not be reading
  a window while a macOS user changes the same values. Writes go through the
  same vaara.menu helpers the macOS client uses, so the two cannot drift, and
  only declared keys with declared values are accepted.

What stays native: the always-on traffic light and the approval prompt, because
both have to exist when no browser is open.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

_PAGE = Path(__file__).with_name("dashboard.html")
_ASSET_DIR = Path(__file__).with_name("assets")
_ASSETS = {
    "/vaara-wordmark-light.png": _ASSET_DIR / "vaara-wordmark-light.png",
    "/vaara-wordmark-dark.png": _ASSET_DIR / "vaara-wordmark-dark.png",
}


def _load_trail(db_path: Optional[Path], trail_path: Optional[Path]) -> Any:
    from vaara.audit.trail import AuditRecord, AuditTrail

    if db_path:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend

        return SQLiteAuditBackend(db_path).load_trail()

    trail = AuditTrail()
    if trail_path and trail_path.exists():
        with open(trail_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trail._records.append(AuditRecord.from_dict(json.loads(line)))
    return trail


def _summarize(trail: Any) -> dict:
    records = list(trail._records)
    decisions = [r for r in records if (r.data or {}).get("decision")]
    counts = {"allow": 0, "escalate": 0, "deny": 0}
    for r in decisions:
        d = str((r.data or {}).get("decision", "")).lower()
        if d in counts:
            counts[d] += 1
    gaps = [r for r in records if getattr(r.event_type, "value", "") == "anchor_gap"]
    latest = decisions[-1] if decisions else None
    return {
        "records": len(records),
        "decisions": counts,
        # A gap is not an error to hide. It is the period nobody witnessed, and
        # it belongs on the front of the dashboard rather than in a log file.
        "gaps": len(gaps),
        "latest": {
            "decision": (latest.data or {}).get("decision") if latest else None,
            "tool": latest.tool_name if latest else None,
            "agent": latest.agent_id if latest else None,
            "reason": (latest.data or {}).get("reason") if latest else None,
            "at": latest.timestamp if latest else None,
        },
        "anchors": [a.to_dict() for a in getattr(trail, "_anchors", [])],
        "publications": trail.publications() if hasattr(trail, "publications") else [],
    }


def _history(trail: Any, limit: int) -> list[dict]:
    out = []
    for r in list(trail._records)[-limit:][::-1]:
        data = r.data or {}
        out.append({
            "at": r.timestamp,
            "event": getattr(r.event_type, "value", str(r.event_type)),
            "agent": r.agent_id,
            "tool": r.tool_name,
            "decision": data.get("decision"),
            "reason": data.get("reason"),
            "hash": r.record_hash,
        })
    return out


# Settings a non-macOS user was previously locked out of. Same config.json the
# macOS client writes, via the same helpers, so the two cannot drift.
# macOS-only keys (menubar_graph, webkitGovernance) are deliberately absent
# rather than shown and ignored.
_SETTINGS: dict[str, dict] = {
    "user_level": {
        "label": "Detail level",
        "options": ["basic", "professional", "enterprise"],
        "help": "How much a notification explains before you decide.",
    },
    "notify_on": {
        "label": "Notify on",
        "options": ["off", "deny", "escalate", "all"],
        "help": "Which decisions raise a notification. Allowed moves pass in silence.",
    },
    "approval_style": {
        "label": "Approval style",
        "options": ["blocking", "timeout"],
        "help": "Whether an escalation waits for you, or denies when it times out.",
    },
    "alert_window_minutes": {
        "label": "Alert window (minutes)",
        "options": ["5", "15", "60", "1440"],
        "help": "How far back the dashboard counts recent interventions.",
    },
}


class _Handler(BaseHTTPRequestHandler):
    db_path: Optional[Path] = None
    trail_path: Optional[Path] = None
    # A page in another tab can POST to 127.0.0.1 without ever reading the
    # response. It cannot read this token, which is only in the page we serve,
    # so requiring it on writes closes that door.
    token: str = ""

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Local-only tool, but there is no reason for any of it to be framed,
        # sniffed or sent anywhere.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(json.dumps(payload).encode(), "application/json", code)

    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler API)
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._send(_PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if path in _ASSETS:
                # Served from the installed package, not fetched. The real
                # wordmark, and it still renders with the network off.
                self._send(_ASSETS[path].read_bytes(), "image/png")
                return
            if path == "/api/summary":
                trail = _load_trail(self.db_path, self.trail_path)
                self._json(_summarize(trail))
                return
            if path == "/api/history":
                trail = _load_trail(self.db_path, self.trail_path)
                self._json(_history(trail, 200))
                return
            if path == "/api/config":
                from vaara.menu import CONFIG_PATH, _load_config

                self._json({
                    "path": str(CONFIG_PATH),
                    "values": _load_config(),
                    "fields": _SETTINGS,
                    "token": self.token,
                })
                return
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # a dashboard must not take the process down
            self._json({"error": repr(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler API)
        path = self.path.split("?", 1)[0]
        if path != "/api/config":
            self._json({"error": "not found"}, 404)
            return
        if self.headers.get("X-Vaara-Token", "") != self.token:
            self._json({"error": "bad or missing token"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._json({"error": f"unreadable body: {exc!r}"}, 400)
            return

        from vaara.menu import CONFIG_PATH, _load_config, _save_config

        cfg = _load_config()
        changed = {}
        for key, spec in _SETTINGS.items():
            if key not in payload:
                continue
            value = str(payload[key])
            # Only known keys, only declared values. A settings page should not
            # be a way to write arbitrary JSON into the config the gate reads.
            if value not in spec["options"]:
                self._json({"error": f"{key}: {value!r} is not one of "
                                     f"{spec['options']}"}, 400)
                return
            cfg[key] = value
            changed[key] = value
        try:
            _save_config(cfg)
        except Exception as exc:
            self._json({"error": f"could not write {CONFIG_PATH}: {exc!r}"}, 500)
            return
        self._json({"saved": changed, "path": str(CONFIG_PATH)})

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr access log; this is a desktop tool."""


def serve(
    *,
    db: Optional[str] = None,
    trail: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 7517,
    open_browser: bool = True,
) -> int:
    if not _PAGE.exists():
        print(f"dashboard page missing: {_PAGE}")
        return 2

    _Handler.db_path = Path(db).expanduser() if db else None
    _Handler.trail_path = Path(trail).expanduser() if trail else None
    _Handler.token = secrets.token_urlsafe(24)

    try:
        httpd = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        print(f"cannot bind {host}:{port}: {exc}")
        return 1

    url = f"http://{host}:{port}/"
    source = db or trail or "(none: pass --db or --trail)"
    print(f"Vaara dashboard on {url}")
    print(f"  reading   {source}")
    print(f"  bound to  {host} only, not reachable from the network")
    print("  read-only, Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def free_port(preferred: int = 7517) -> int:
    """The preferred port if it is free, otherwise one the OS picks."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
