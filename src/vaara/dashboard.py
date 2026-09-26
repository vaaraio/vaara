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
  a window while a macOS user changes settings. Writes go through vaara.menu
  into the hook's config.json, and only declared keys with declared values are
  accepted. The macOS app keeps its own display settings in menubar.json.

What stays native: the always-on traffic light and the approval prompt, because
both have to exist when no browser is open.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

_PAGE = Path(__file__).with_name("dashboard.html")
_ASSET_DIR = Path(__file__).with_name("assets")
_ASSETS = {
    "/vaara-wordmark-light.png": (_ASSET_DIR / "vaara-wordmark-light.png", "image/png"),
    "/vaara-wordmark-dark.png": (_ASSET_DIR / "vaara-wordmark-dark.png", "image/png"),
    "/favicon.svg": (_ASSET_DIR / "favicon.svg", "image/svg+xml"),
}


def _policy_state(path: Optional[Path]) -> dict:
    """Read the active policy and validate it, without importing the pipeline.

    Rules are the sharpest thing on this page: they decide what an agent is
    allowed to do. So the dashboard reports what is loaded, what the validator
    says about it, and where it came from, rather than showing thresholds with
    no indication of whether the file is even sound.
    """
    if not path or not path.exists():
        return {"path": str(path) if path else "", "loaded": False,
                "reason": "no policy file configured (--policy)"}
    try:
        from vaara.policy.validate import validate_source

        policy, report = validate_source(path)
        if policy is None:
            return {"path": str(path), "loaded": False,
                    "reason": "policy did not parse",
                    "issues": [str(i) for i in report.issues][:20]}
        th = policy.thresholds_default
        return {
            "path": str(path),
            "loaded": True,
            "escalate": th.escalate,
            "deny": th.deny,
            "action_classes": sorted(policy.action_classes)[:40],
            "action_class_count": len(policy.action_classes),
            "issues": [str(i) for i in report.issues][:20],
        }
    except Exception as exc:
        return {"path": str(path), "loaded": False, "reason": repr(exc)}


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


def _alert_window(cfg: dict) -> int:
    """Minutes the summary looks back for recent interventions (default 5)."""
    try:
        minutes = int(cfg.get("alert_window_minutes", 5))
    except (TypeError, ValueError):
        return 5
    return minutes if minutes > 0 else 5


def _summarize(trail: Any, window_minutes: int = 5) -> dict:
    records = list(trail._records)
    decisions = [r for r in records if (r.data or {}).get("decision")]
    counts = {"allow": 0, "escalate": 0, "deny": 0}
    since = time.time() - window_minutes * 60
    recent = {"minutes": window_minutes, "deny": 0, "escalate": 0}
    for r in decisions:
        d = str((r.data or {}).get("decision", "")).lower()
        if d in counts:
            counts[d] += 1
        if d in ("deny", "escalate") and (r.timestamp or 0) >= since:
            recent[d] += 1
    gaps = [r for r in records if getattr(r.event_type, "value", "") == "anchor_gap"]
    latest = decisions[-1] if decisions else None
    return {
        "records": len(records),
        "decisions": counts,
        # Interventions inside the operator's alert window (alert_window_minutes).
        "recent": recent,
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


def _oslayer_state() -> dict:
    """The Linux OS layer as the operator sees it: guard, picks, open questions."""
    import sys

    if not sys.platform.startswith("linux"):
        return {"available": False}
    from vaara.approvals import APPROVALS_DIR
    from vaara.oslayer import floor, manage, selection

    return {
        "available": True,
        "apparmor": floor.apparmor_enabled(),
        "guard": manage.guard_status(),
        "selection": selection.load(str(Path.home())).to_json(),
        "modes": list(selection.MODES),
        "pending": manage.pending(APPROVALS_DIR),
    }


def _oslayer_change(payload: dict) -> tuple[dict, int]:
    """Apply one change from the page. Only these three actions exist."""
    from vaara.oslayer import manage, selection

    action = payload.get("action")
    home = str(Path.home())
    try:
        if action == "folder":
            mode = payload.get("mode")
            if mode not in selection.MODES + ("off",):
                return {"error": f"mode must be one of {', '.join(selection.MODES)} or off"}, 400
            sel = selection.set_folder(selection.load(home), str(payload.get("path", "")),
                                       None if mode == "off" else mode)
        elif action == "app":
            sel = selection.set_app(selection.load(home), str(payload.get("path", "")),
                                    bool(payload.get("attach", True)))
        elif action == "decide":
            from vaara.approvals import write_decision

            decision = payload.get("decision")
            if decision not in ("approve", "deny"):
                return {"error": "decision must be approve or deny"}, 400
            if not write_decision(str(payload.get("action_id", "")), decision):
                return {"error": "that request is no longer waiting"}, 409
            return {"saved": decision}, 200
        else:
            return {"error": "unknown action"}, 400
    except selection.SelectionError as exc:
        return {"error": str(exc)}, 400
    path, told = manage.save(sel, home)
    return {"saved": action, "path": str(path), "guard_told": told}, 200


# Settings a non-macOS user was previously locked out of, written to the hook's
# config.json. notify_on is read by the hook's notify(); alert_window_minutes
# by _summarize here. The macOS app reads its own copies from menubar.json.
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
    policy_path: Optional[Path] = None
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
                # assets, and they still render with the network off.
                asset, ctype = _ASSETS[path]
                self._send(asset.read_bytes(), ctype)
                return
            if path == "/api/summary":
                from vaara.menu import _load_config

                trail = _load_trail(self.db_path, self.trail_path)
                self._json(_summarize(trail, _alert_window(_load_config())))
                return
            if path == "/api/history":
                trail = _load_trail(self.db_path, self.trail_path)
                self._json(_history(trail, 200))
                return
            if path == "/api/policy":
                self._json(_policy_state(self.policy_path))
                return
            if path == "/api/oslayer":
                self._json(_oslayer_state())
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
        if path not in ("/api/config", "/api/policy", "/api/oslayer"):
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

        if path == "/api/policy":
            self._write_thresholds(payload)
            return
        if path == "/api/oslayer":
            try:
                body, code = _oslayer_change(payload if isinstance(payload, dict) else {})
            except Exception as exc:
                body, code = {"error": repr(exc)}, 500
            self._json(body, code)
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

    def _write_thresholds(self, payload: dict) -> None:
        """Write escalate/deny back to the policy file, validator-gated.

        A threshold decides whether an agent is stopped, so this refuses
        anything the policy validator would reject rather than writing a file
        the pipeline will then fail to load.
        """
        import json as _json

        path = self.policy_path
        if not path or not path.exists():
            self._json({"error": "no policy file configured (--policy)"}, 400)
            return
        try:
            esc = float(payload["escalate"]); deny = float(payload["deny"])
        except Exception:
            self._json({"error": "escalate and deny must be numbers"}, 400)
            return
        if not (0.0 <= esc <= 1.0 and 0.0 <= deny <= 1.0):
            self._json({"error": "thresholds must be in [0,1]"}, 400)
            return
        if esc >= deny:
            self._json({"error": f"escalate ({esc}) must be below deny ({deny})"}, 400)
            return

        text = path.read_text()
        is_yaml = path.suffix in (".yaml", ".yml")
        if is_yaml:
            try:
                import yaml
            except ImportError:
                self._json({"error": "editing a YAML policy needs the yaml extra"}, 400)
                return
            data = yaml.safe_load(text) or {}
        else:
            data = _json.loads(text or "{}")

        # The document key is thresholds.default. thresholds_default is the
        # dataclass field name, and writing that instead produced a block the
        # loader ignored while the write reported success.
        th = data.setdefault("thresholds", {}).setdefault("default", {})
        th["escalate"], th["deny"] = esc, deny

        # Validate the whole edited document before it replaces anything.
        from vaara.policy.validate import validate_source

        policy, report = validate_source(data)
        if policy is None:
            self._json({"error": "the edit would not validate",
                        "issues": [str(i) for i in report.issues][:10]}, 400)
            return

        new_text = (yaml.safe_dump(data, sort_keys=False) if is_yaml
                    else _json.dumps(data, indent=2) + "\n")
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(text)
        path.write_text(new_text)
        self._json({"saved": {"escalate": esc, "deny": deny},
                    "path": str(path), "backup": str(backup)})

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr access log; this is a desktop tool."""


def serve(
    *,
    db: Optional[str] = None,
    trail: Optional[str] = None,
    policy: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 7517,
    open_browser: bool = True,
) -> int:
    if not _PAGE.exists():
        print(f"dashboard page missing: {_PAGE}")
        return 2

    _Handler.db_path = Path(db).expanduser() if db else None
    _Handler.trail_path = Path(trail).expanduser() if trail else None
    _Handler.policy_path = Path(policy).expanduser() if policy else None
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
    print("  settings and policy thresholds are writable from the page")
    print("  writes need a token served only in that page, Ctrl-C to stop")
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
