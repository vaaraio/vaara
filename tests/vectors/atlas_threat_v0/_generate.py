"""Regenerate atlas_threat_v0 conformance vectors.

Five vectors ground vaara.receipt/v1 against MITRE ATLAS threat patterns
(https://atlas.mitre.org/). Each fixture is self-contained JSON. The sibling
_check_independent.py reproduces every verdict with no Vaara import — only
hmac, hashlib, json, rfc8785, and stdlib — so a passing check is a property
of the bytes, not of this script.

Cases and threat mappings:

  pos_clean_execution    — authorized args, tool, scope, fresh timestamp       → ok
  neg_injected_args      — args mutated after authorization (Prompt Injection)  → args_tampered
  neg_tool_substitution  — actionType changed at runtime (Unauthorized Access)  → tool_mismatch
  neg_replay             — receipt presented outside freshness window            → stale
  neg_scope_escalation   — runtime scope exceeds authorized boundary            → scope_exceeded

Run from repo root: python3 tests/vectors/atlas_threat_v0/_generate.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import rfc8785

from vaara.attestation._attest_canonical import make_args_digest

HERE = Path(__file__).resolve().parent

KEY = b"y" * 32
SECRET_VERSION = "atlas-corpus-key-v0"

AGENT_ID = "agent:file-reader"
ACTION_TYPE_AUTH = "file.read"
ACTION_TYPE_ALT = "file.write"
SCOPE_AUTH = "path:/tmp/vaara/"
SCOPE_ESCALATED = "path:/etc/"

ARGS_AUTH = {"path": "/tmp/vaara/report.json"}
ARGS_INJECTED = {"path": "/tmp/vaara/report.json\x00; exfil /etc/passwd"}

# Pinned timestamps (milliseconds).
# IAT_MS  = 1779200000000  →  2026-05-19T14:13:20Z
# NOW_MS  = IAT_MS + 30_000   (30 s later, within the 60 s freshness window)
# NOW_STALE_MS = IAT_MS + 120_000  (120 s later, outside the window → replay)
IAT_MS = 1779200000000
NOW_MS = IAT_MS + 30_000
NOW_STALE_MS = IAT_MS + 120_000
FRESHNESS_WINDOW_MS = 60_000


def _jcs(obj: object) -> bytes:
    return rfc8785.dumps(obj)


def _args_commitment(args: dict) -> str:
    return make_args_digest(args).projection_digest


def _receipt_fields(
    *,
    action_type: str = ACTION_TYPE_AUTH,
    scope: str = SCOPE_AUTH,
    args: dict = ARGS_AUTH,
    timestamp_ms: int = IAT_MS,
    seq: int = 0,
) -> dict:
    return {
        "agentId": AGENT_ID,
        "actionType": action_type,
        "argsCommitment": _args_commitment(args),
        "iss": "vaara-test-proxy",
        "schema": "vaara.receipt/v1",
        "scope": scope,
        "seq": seq,
        "sub": "corpus/atlas-threat",
        "timestampMs": timestamp_ms,
        "version": 1,
    }


def _sign(fields: dict) -> str:
    return hmac.new(KEY, _jcs(fields), hashlib.sha256).hexdigest()


def _signed_receipt(**kwargs) -> dict:
    fields = _receipt_fields(**kwargs)
    return {**fields, "signature": _sign(fields)}


def _authorization() -> dict:
    """The pre-committed authorization declaration — what the agent was allowed to do."""
    return {
        "actionType": ACTION_TYPE_AUTH,
        "agentId": AGENT_ID,
        "argsCommitment": _args_commitment(ARGS_AUTH),
        "scope": SCOPE_AUTH,
    }


def _case(*, receipt: dict, now_ms: int = NOW_MS, expected_verdict: str) -> dict:
    return {
        "authorization": _authorization(),
        "expected_verdict": expected_verdict,
        "freshness_window_ms": FRESHNESS_WINDOW_MS,
        "now_ms": now_ms,
        "receipt": receipt,
    }


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cases_dir = HERE / "cases"

    # pos_clean_execution — all fields match authorization, timestamp within window
    _write(
        cases_dir / "pos_clean_execution.json",
        _case(receipt=_signed_receipt(), expected_verdict="ok"),
    )

    # neg_injected_args — args mutated; argsCommitment diverges from authorization
    # MITRE ATLAS: Prompt Injection (attacker injects payload into agent args)
    _write(
        cases_dir / "neg_injected_args.json",
        _case(
            receipt=_signed_receipt(args=ARGS_INJECTED),
            expected_verdict="args_tampered",
        ),
    )

    # neg_tool_substitution — actionType changed at runtime; authorized read, got write
    # MITRE ATLAS: Unauthorized Access via tool misuse
    _write(
        cases_dir / "neg_tool_substitution.json",
        _case(
            receipt=_signed_receipt(action_type=ACTION_TYPE_ALT),
            expected_verdict="tool_mismatch",
        ),
    )

    # neg_replay — receipt timestampMs is 120 s behind now_ms; outside freshness window
    # MITRE ATLAS: Replay Attack (presenting an old valid receipt)
    _write(
        cases_dir / "neg_replay.json",
        _case(
            receipt=_signed_receipt(timestamp_ms=IAT_MS),
            now_ms=NOW_STALE_MS,
            expected_verdict="stale",
        ),
    )

    # neg_scope_escalation — runtime scope exceeds authorized boundary
    # MITRE ATLAS: Privilege Escalation via scope widening
    _write(
        cases_dir / "neg_scope_escalation.json",
        _case(
            receipt=_signed_receipt(scope=SCOPE_ESCALATED),
            expected_verdict="scope_exceeded",
        ),
    )

    expected_cases: dict = {}
    for path in sorted(cases_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        expected_cases[path.stem] = {"expected_verdict": obj["expected_verdict"]}
    _write(HERE / "expected.json", {"cases": expected_cases})
    print(f"wrote {len(expected_cases)} atlas_threat_v0 vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
