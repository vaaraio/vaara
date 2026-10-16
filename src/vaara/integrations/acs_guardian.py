# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara as a Guardian for the OWASP Agent Control Standard (ACS) v0.1.0.

    Observed Agent                         this module
    --------------                         -----------
    handshake/hello  ClientHello  ------>  ServerHello
    steps/toolCallRequest  ------------->  InterceptionPipeline.intercept
                                           allow -> allow, deny -> deny,
                                           escalate -> deny (approver_unavailable)
    steps/toolCallResult  -------------->  report_outcome on the same action
    other content steps  --------------->  allow
    every content step                     a ContextEntry on the session chain,
                                           its head returned as chain_hash
    system/ping                            allow, outside the chain

The ACS chain (spec 8.2) is per session and lives in this process. The
decision behind each tool step also goes onto Vaara's own trail, which is
hash chained and signed, and the response names it: ``policy_data.vaara``
carries the action id and the trail record hash, ``policy_references`` the
decision's policy digest. A party who holds no ACS key can check that record
offline; ACS's HMAC baseline proves integrity only to a key holder.

Every content step's entry carries ``request_hash``, so the chain commits to
what was asked (the ACS-Audit profile). With a deployment secret every request
must carry an HMAC-SHA256 signature over its RFC 8785 form without the
signature, under a per-session key, and every response is signed the same
way. ACS v0.1.0 leaves the HKDF salt and info open; this Guardian uses no
salt and ``info = b"acs/v0.1 hmac " + session_id``. Without a secret the
Guardian declares no signature algorithm and does not claim ``acs-core``.

Requests are refused outside a five-minute clock window, and a repeated
``request_id`` or ``nonce`` in a session is refused (spec 10.3).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import rfc8785

logger = logging.getLogger(__name__)

ACS_VERSION = "0.1.0"
DEFAULT_SKEW_WINDOW_MS = 300_000
DEFAULT_TIMEOUT_MS = 5_000
#: Requests accepted on one HTTP call.
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_BATCH = 64
#: Sessions kept in memory; the oldest is dropped first.
MAX_SESSIONS = 1024

# ACS error codes (spec 17.1) and the JSON-RPC ones used here.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNSUPPORTED_VERSION = -32001
CAPABILITY_NOT_NEGOTIATED = -32003
SIGNATURE_INVALID = -32004
REPLAY_DETECTED = -32005
TIMESTAMP_OUT_OF_WINDOW = -32006

#: Content-bearing steps this Guardian evaluates. Each writes a ContextEntry.
CONTENT_STEPS = (
    "steps/sessionStart", "steps/agentTrigger", "steps/turnStart",
    "steps/userMessage", "steps/agentResponse", "steps/toolCallRequest",
    "steps/toolCallResult", "steps/subagentStart", "steps/subagentStop",
    "steps/turnEnd", "steps/sessionEnd",
)
METHODS_EVALUATED = ("handshake/hello", "system/ping") + CONTENT_STEPS

#: The trail events that carry a decision: allow, deny and escalate.
_DECISION_EVENTS = ("decision_made", "action_blocked", "escalation_sent")

#: How a tool's exit status reads as an outcome severity on Vaara's trail.
_EXIT_SEVERITY = {"success": 0.0, "blocked": 0.0, "timeout": 0.5, "failure": 0.5}


def jcs(value: Any) -> bytes:
    """RFC 8785 bytes of ``value``."""
    return rfc8785.dumps(value)


def entry_hash(entry: dict, previous_hash: Optional[str]) -> str:
    """Spec 8.2: SHA-256 over JCS(entry without its two hashes) || previous."""
    content = {k: v for k, v in entry.items() if k not in ("entry_hash", "previous_hash")}
    prev = bytes.fromhex(previous_hash) if previous_hash else b""
    return hashlib.sha256(jcs(content) + prev).hexdigest()


def session_key(secret: bytes, session_id: str) -> bytes:
    """The per-session HMAC key: HKDF-SHA256, no salt, info names the session."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"acs/v0.1 hmac " + session_id.encode("utf-8")).derive(secret)


def _without_signature(envelope: dict) -> dict:
    out = dict(envelope)
    for slot in ("params", "result"):
        inner = out.get(slot)
        if isinstance(inner, dict) and "signature" in inner:
            out[slot] = {k: v for k, v in inner.items() if k != "signature"}
    return out


def sign_envelope(envelope: dict, key: bytes, key_id: str) -> dict:
    """``envelope`` with an HMAC-SHA256 signature in params or result."""
    mac = hmac.new(key, jcs(_without_signature(envelope)), hashlib.sha256).digest()
    sig = {"algorithm": "HMAC-SHA256", "value": base64.b64encode(mac).decode(), "key_id": key_id}
    slot = "params" if "params" in envelope else "result"
    signed = dict(envelope)
    signed[slot] = {**envelope[slot], "signature": sig}
    return signed


def verify_envelope(envelope: dict, key: bytes) -> bool:
    slot = "params" if "params" in envelope else "result"
    sig = (envelope.get(slot) or {}).get("signature")
    if not isinstance(sig, dict) or sig.get("algorithm") != "HMAC-SHA256":
        return False
    try:
        given = base64.b64decode(str(sig.get("value", "")), validate=True)
    except (ValueError, TypeError):
        return False
    want = hmac.new(key, jcs(_without_signature(envelope)), hashlib.sha256).digest()
    return hmac.compare_digest(given, want)


def _now_iso(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock(), timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _major(version: Any) -> Optional[int]:
    if not isinstance(version, str):
        return None
    head = version.split(".")
    if len(head) != 3 or not all(p.isdigit() for p in head):
        return None
    return int(head[0])


class _Refused(Exception):
    def __init__(self, code: int, message: str, data: Optional[dict] = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


@dataclass
class _Session:
    entries: list[dict] = field(default_factory=list)
    head: Optional[str] = None
    request_ids: set[str] = field(default_factory=set)
    nonces: set[str] = field(default_factory=set)
    #: ACS request_id of a toolCallRequest -> Vaara action id.
    actions: dict[str, str] = field(default_factory=dict)


class AcsGuardian:
    """Answers ACS envelopes with Vaara's decisions. Transport-agnostic."""

    def __init__(self, pipeline: Any = None, *, secret: Optional[bytes] = None,
                 key_id: str = "vaara-acs", skew_window_ms: int = DEFAULT_SKEW_WINDOW_MS,
                 deny_rules: Optional[list[dict]] = None,
                 clock: Callable[[], float] = time.time) -> None:
        # Layer-1 deny rules, the same file the host hook and the MCP proxy
        # apply; an empty list switches the layer off.
        if deny_rules is None:
            from vaara.deny_rules import load_deny_rules
            deny_rules = load_deny_rules()
        self._deny_rules = list(deny_rules)
        if pipeline is None:
            from vaara.pipeline import InterceptionPipeline
            pipeline = InterceptionPipeline()
        self._pipeline = pipeline
        self._secret = secret or None
        self._key_id = key_id
        self._skew_s = skew_window_ms / 1000.0
        self._skew_window_ms = skew_window_ms
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # -- public ---------------------------------------------------------

    def handle(self, message: Any) -> Any:
        """One JSON-RPC request or a batch; returns the response(s)."""
        if isinstance(message, list):
            if not message or len(message) > MAX_BATCH:
                return self._error(None, INVALID_REQUEST, "empty or oversized batch")
            return [self._handle_one(m) for m in message]
        return self._handle_one(message)

    def chain(self, session_id: str) -> list[dict]:
        """The session's ContextEntries, oldest first."""
        with self._lock:
            s = self._sessions.get(session_id)
            return [dict(e) for e in s.entries] if s else []

    # -- dispatch -------------------------------------------------------

    def _handle_one(self, msg: Any) -> dict:
        rid = msg.get("id") if isinstance(msg, dict) else None
        try:
            if (not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0"
                    or not isinstance(msg.get("method"), str) or "id" not in msg):
                raise _Refused(INVALID_REQUEST, "not a JSON-RPC 2.0 request")
            method = msg["method"]
            params = msg.get("params")
            if not isinstance(params, dict):
                raise _Refused(INVALID_PARAMS, "params must be an object")
            if method == "system/ping":
                return self._ping(msg, params)
            meta = params.get("metadata")
            for name in ("acs_version", "request_id", "timestamp", "payload"):
                if name not in params:
                    raise _Refused(INVALID_PARAMS, f"params.{name} is required")
            if (not isinstance(meta, dict) or not isinstance(meta.get("session_id"), str)
                    or not meta.get("session_id") or not isinstance(meta.get("agent_id"), str)):
                raise _Refused(INVALID_PARAMS, "params.metadata needs agent_id and session_id")
            if not isinstance(params["payload"], dict):
                raise _Refused(INVALID_PARAMS, "params.payload must be an object")
            if _major(params["acs_version"]) != 0:
                raise _Refused(UNSUPPORTED_VERSION, "unsupported acs_version",
                               {"supported_versions": [ACS_VERSION]})
            session_id = meta["session_id"]
            key = session_key(self._secret, session_id) if self._secret else None
            if key is not None and not verify_envelope(msg, key):
                raise _Refused(SIGNATURE_INVALID, "missing or invalid HMAC-SHA256 signature")
            if method == "handshake/hello":
                return self._sign(self._hello(msg, params, session_id), key)
            if method not in CONTENT_STEPS:
                raise _Refused(METHOD_NOT_FOUND, f"method {method} is not evaluated by this Guardian")
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    raise _Refused(CAPABILITY_NOT_NEGOTIATED, "no handshake/hello for this session",
                                   {"method": "handshake/hello"})
                self._check_replay(session, params)
            result = self._decide(method, params, meta, session)
            with self._lock:
                result["chain_hash"] = self._append(session, method, params)
            return self._sign({"jsonrpc": "2.0", "id": rid, "result": result}, key)
        except _Refused as r:
            return self._error(rid, r.code, r.message, r.data)
        except Exception:  # never answer with a traceback
            logger.exception("ACS request failed")
            return self._error(rid, INTERNAL_ERROR, "internal error")

    def _ping(self, msg: dict, params: dict) -> dict:
        payload = params.get("payload") if isinstance(params.get("payload"), dict) else {}
        out = {"status": "ok", "server_timestamp": _now_iso(self._clock)}
        if isinstance(payload.get("echo"), str):
            out["echo"] = payload["echo"]
        request_id = params.get("request_id")
        return {"jsonrpc": "2.0", "id": msg.get("id"), "result": {
            "type": "final", "acs_version": ACS_VERSION,
            "request_id": request_id if isinstance(request_id, str) else str(uuid.uuid4()),
            "decision": "allow", "payload": out}}

    def _hello(self, msg: dict, params: dict, session_id: str) -> dict:
        hello = params["payload"]
        versions = hello.get("acs_versions_supported")
        if not isinstance(versions, list) or not any(_major(v) == 0 for v in versions):
            raise _Refused(UNSUPPORTED_VERSION, "no common acs_version",
                           {"supported_versions": [ACS_VERSION]})
        transports = hello.get("transports_supported")
        transport = next((t for t in ("https", "http", "stdio")
                          if isinstance(transports, list) and t in transports), "http")
        offered = hello.get("profiles_supported")
        offered = offered if isinstance(offered, list) else []
        ours = {"acs-audit"} | ({"acs-core"} if self._secret else set())
        with self._lock:
            if session_id not in self._sessions:
                while len(self._sessions) >= MAX_SESSIONS:
                    self._sessions.pop(next(iter(self._sessions)))
                self._sessions[session_id] = _Session()
        return {"jsonrpc": "2.0", "id": msg.get("id"), "result": {
            "negotiated_version": ACS_VERSION,
            "methods_evaluated": list(METHODS_EVALUATED),
            "selected_transport": transport,
            "signature_algorithms_supported": ["HMAC-SHA256"] if self._secret else [],
            "timeout_config": {"default_ms": DEFAULT_TIMEOUT_MS},
            "skew_window_ms": self._skew_window_ms,
            "on_decision_failure": "deny",
            "approver_types_supported": [],
            "policy_requires_provenance": False,
            "profiles_accepted": sorted(p for p in offered if p in ours),
        }}

    def _check_replay(self, session: _Session, params: dict) -> None:
        sent = _parse_time(params.get("timestamp"))
        if sent is None or abs(self._clock() - sent) > self._skew_s:
            raise _Refused(TIMESTAMP_OUT_OF_WINDOW, "timestamp outside the skew window",
                           {"skew_window_ms": self._skew_window_ms})
        request_id = params.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise _Refused(INVALID_PARAMS, "params.request_id must be a string")
        if request_id in session.request_ids:
            raise _Refused(REPLAY_DETECTED, "request_id already seen in this session")
        nonce = params.get("nonce")
        if isinstance(nonce, str) and nonce in session.nonces:
            raise _Refused(REPLAY_DETECTED, "nonce already seen in this session")
        session.request_ids.add(request_id)
        if isinstance(nonce, str):
            session.nonces.add(nonce)

    def _append(self, session: _Session, method: str, params: dict) -> str:
        entry = {
            "entry_id": f"{len(session.entries) + 1}",
            "step_id": params["request_id"],
            "step_type": method,
            "request_hash": hashlib.sha256(jcs(params)).hexdigest(),
            "timestamp": _now_iso(self._clock),
        }
        entry["previous_hash"] = session.head
        entry["entry_hash"] = entry_hash(entry, session.head)
        session.entries.append(entry)
        session.head = entry["entry_hash"]
        return session.head

    # -- decisions ------------------------------------------------------

    def _decide(self, method: str, params: dict, meta: dict, session: _Session) -> dict:
        base = {"type": "final", "acs_version": ACS_VERSION, "request_id": params["request_id"],
                "metadata": {"evaluator": "deterministic", "evaluator_version": _vaara_version()}}
        payload = params["payload"]
        if method == "steps/toolCallRequest":
            return {**base, **self._tool_request(payload, meta, params, session)}
        if method == "steps/toolCallResult":
            self._tool_result(payload, session)
        return {**base, "decision": "allow"}

    def _tool_request(self, payload: dict, meta: dict, params: dict, session: _Session) -> dict:
        tool = payload.get("tool")
        name = tool.get("name") if isinstance(tool, dict) else None
        if not isinstance(name, str) or not name:
            raise _Refused(INVALID_PARAMS, "payload.tool.name is required")
        raw_args = payload.get("arguments")
        if not isinstance(raw_args, dict):
            raise _Refused(INVALID_PARAMS, "payload.arguments must be an object")
        arguments = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in raw_args.items()}
        context: dict[str, Any] = {"acs_request_id": params["request_id"]}
        if isinstance(payload.get("raw_command"), str):
            context["raw_command"] = payload["raw_command"]
        tenant = params.get("tenant_id")
        rule = self._deny_rule(name, arguments)
        ruled: dict[str, Any] = {}
        if rule is not None and getattr(self._pipeline, "_enforce", True):
            ruled = {"policy_decision": "deny", "policy_reason": f"Deny rule {rule[0]}: {rule[1]}",
                     "policy_id": f"deny_rule:{rule[0]}"}
        result = self._pipeline.intercept(
            agent_id=meta["agent_id"], tool_name=name, parameters=arguments,
            context=context, session_id=meta["session_id"],
            tenant_id=tenant if isinstance(tenant, str) else "", **ruled,
        )
        with self._lock:
            session.actions[params["request_id"]] = result.action_id
        record = self._decision_record(result.action_id)
        vaara: dict[str, Any] = {"action_id": result.action_id, "decision": result.decision,
                                 "risk_score": round(float(result.risk_score), 4)}
        out: dict[str, Any] = {"policy_data": {"vaara": vaara}}
        if record is not None:
            vaara["record_hash"] = record.record_hash
            digest = record.data.get("policy_digest")
            if isinstance(digest, str) and digest:
                out["policy_references"] = [{"policy_id": "vaara", "policy_version": digest}]
        if result.decision == "allow":
            return {**out, "decision": "allow", "reasoning": result.reason}
        if result.decision == "escalate":
            # Spec 9.2: this Guardian has no approver route yet, so a call
            # Vaara would hold for a human is refused and says why.
            return {**out, "decision": "deny", "reason_codes": ["approver_unavailable", "vaara_escalate"],
                    "reasoning": f"held for a human approver, none reachable over ACS: {result.reason}"}
        codes = [f"deny_rule:{rule[0]}"] if ruled and rule is not None else ["vaara_deny"]
        return {**out, "decision": "deny", "reason_codes": codes, "reasoning": result.reason}

    def _deny_rule(self, tool_name: str, arguments: dict) -> Optional[tuple[str, str]]:
        if not self._deny_rules:
            return None
        from vaara.deny_rules import match_deny_rule, match_deny_rule_any_field

        try:
            return (match_deny_rule(self._deny_rules, tool_name, arguments)
                    or match_deny_rule_any_field(self._deny_rules, arguments))
        except Exception as exc:  # a broken rule must not take the Guardian down
            logger.warning("deny rules failed (%r); skipping layer 1", exc)
            return None

    def _tool_result(self, payload: dict, session: _Session) -> None:
        ref = payload.get("request_id_ref")
        with self._lock:
            action_id = session.actions.pop(ref, None) if isinstance(ref, str) else None
        if action_id is None:
            return
        severity = _EXIT_SEVERITY.get(str(payload.get("exit_status")), 0.5)
        try:
            self._pipeline.report_outcome(action_id=action_id, outcome_severity=severity,
                                          description=f"ACS toolCallResult {payload.get('exit_status')}")
        except Exception:
            logger.exception("report_outcome failed for action_id=%s", action_id)

    def _decision_record(self, action_id: str) -> Any:
        records = getattr(getattr(self._pipeline, "trail", None), "_records", None) or []
        for rec in reversed(records[-50:]):
            if rec.action_id == action_id and getattr(rec.event_type, "value", "") in _DECISION_EVENTS:
                return rec
        return None

    # -- envelopes ------------------------------------------------------

    def _sign(self, response: dict, key: Optional[bytes]) -> dict:
        return sign_envelope(response, key, self._key_id) if key is not None else response

    @staticmethod
    def _error(rid: Any, code: int, message: str, data: Optional[dict] = None) -> dict:
        err: dict[str, Any] = {"code": code, "message": message}
        if data:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": rid, "error": err}


def _vaara_version() -> str:
    try:
        from vaara import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def make_handler(guardian: AcsGuardian) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "vaara-acs"

        def do_POST(self) -> None:  # noqa: N802 (http.server naming)
            if self.path.rstrip("/") != "/acs":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_error(413 if length > MAX_BODY_BYTES else 400)
                return
            try:
                message = json.loads(self.rfile.read(length))
            except ValueError:
                body = AcsGuardian._error(None, PARSE_ERROR, "parse error")
            else:
                body = guardian.handle(message)
            data = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("acs %s", fmt % args)

    return Handler


def serve(guardian: AcsGuardian, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    """A threaded HTTP server answering POST /acs. The caller runs serve_forever."""
    return ThreadingHTTPServer((host, port), make_handler(guardian))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="vaara-acs-guardian",
        description="Vaara as an OWASP ACS v0.1.0 Guardian over HTTP (POST /acs).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db", default=os.environ.get("VAARA_DB", "vaara_audit.db"),
                        help="Vaara trail database the decisions are written to.")
    parser.add_argument("--secret-file", default=os.environ.get("VAARA_ACS_SECRET_FILE"),
                        help="File holding the deployment secret for HMAC-SHA256 signatures. "
                             "Without one the Guardian signs nothing and does not claim acs-core.")
    parser.add_argument("--shadow", action="store_true", help="Record decisions, allow every call.")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.secret_file:
        parser.error("a non-loopback --host needs --secret-file: unsigned requests could be forged")
    secret = Path(args.secret_file).read_bytes().strip() if args.secret_file else None
    if secret is not None and len(secret) < 32:
        parser.error("the secret must be at least 32 bytes")

    from vaara.audit.decision_receipts import default_sink
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.audit.trail import AuditTrail
    from vaara.pipeline import InterceptionPipeline

    backend = SQLiteAuditBackend(Path(args.db))
    trail = AuditTrail(on_record=backend.write_record)
    trail._receipt_sink = default_sink(Path(args.db))
    pipeline = InterceptionPipeline(trail=trail, enforce=not args.shadow)
    server = serve(AcsGuardian(pipeline, secret=secret), args.host, args.port)
    logging.basicConfig(level=logging.INFO)
    logger.info("Vaara ACS Guardian listening at http://%s:%d/acs", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
