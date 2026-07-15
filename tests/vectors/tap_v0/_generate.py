#!/usr/bin/env python3
"""Regenerate the tap_v0 conformance vectors (Visa TAP <-> Vaara binding profile).

Imports Vaara to MINT the vaara.receipt/v1 side; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES. See README.md for
the full mapping. A Visa Trusted Agent Protocol (TAP) request becomes the
evidence a ``vaara.receipt/v1`` decision receipt names, across one action
lifecycle: a pre-action receipt (``terminal: false``) and the terminal one
(``terminal: true``) bind to the same logical TAP request but carry distinct
``actionRef`` join keys, so a mid-action receipt cannot be presented where the
final one is required.

The TAP request here is representative; the binding rests on the
content-addressing discipline (JCS / RFC 8785 over the request, the action tuple
as the join key), not on the request's exact field names.

Run: tests/vectors/tap_v0/_generate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation.decision import (
    DecisionDerived,
    EvidenceRef,
    emit_decision_record,
    make_back_link,
)
from vaara.attestation.tool_call_attestation import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HERE = Path(__file__).resolve().parent

_SCALAR = 0x42C0FFEE_1337_0BADBEEF_CAFEBABE_0DDF00D_1234567890ABCDEF_42424242  # test key
IAT = "2026-06-22T10:00:00Z"

# The authorized action. The lifecycle position lives in the tuple, so the
# in-progress and terminal steps have different action tuples and join keys.
BASE_ACTION = {
    "agentId": "agent:checkout-bot",
    "actionType": "purchase.authorize",
    "scope": "merchant:acme/order:INV-1/amount:42.00USD",
    "timestampMs": 1779271200000,
}
_ACTION_KEYS = ("agentId", "actionType", "scope", "timestampMs", "seq", "terminal")

# Representative TAP request envelope (Visa Trusted Agent Protocol). The trusted
# agent presents this to the relying party; the binding rests on the
# content-addressing discipline, not on these field names.
TAP_REQUEST = {
    "relyingParty": "merchant:acme",
    "trustedAgentId": "agent:checkout-bot",
    "consumerRef": "card:tokenized-pan-ref-9f2c",
    "intent": "purchase.authorize",
    "requestId": "tap-req-7f3a",
}


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _action_tuple(seq: int, terminal: bool) -> dict:
    return {**BASE_ACTION, "seq": seq, "terminal": terminal}


def _tap_request(seq: int, terminal: bool) -> dict:
    return {
        "schema": "tap.request/v0",
        **BASE_ACTION,
        "seq": seq,
        "terminal": terminal,
        **TAP_REQUEST,
        "actionRef": _sha256_hex(_jcs(_action_tuple(seq, terminal))),
    }


def _attestation(priv, nonce: str) -> object:
    payload = PayloadDerived(
        tool_calls=(
            ToolCallBinding(
                name="purchase.authorize",
                server_fingerprint="sha256:" + "1" * 64,
                args=make_args_digest({"order": "INV-1", "amount": "42.00USD"}),
            ),
        )
    )
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="authorize the trusted-agent purchase"),
        payload_derived=payload,
        iss="issuer://merchant-gateway",
        sub="agent:checkout-bot",
        secret_version="v1",
        alg="ES256",
        signing_material=priv,
        nonce=nonce,
        iat=IAT,
    )


def _emit_receipt(priv, request: dict, seq: int) -> dict:
    att = _attestation(priv, nonce=f"tap-{seq}")
    ref = EvidenceRef(
        digest=_sha256_hex(_jcs(request)),
        canonicalization="JCS",
        schema=request["schema"],
        ref="tap:request/" + request["actionRef"],
    )
    dd = DecisionDerived(
        decision="allow",
        decided_at=IAT,
        reason="trusted agent acted within the authorized scope of the TAP request",
        risk_score="0.10",
        threshold_allow="0.30",
        threshold_block="0.80",
        policy_id="policy:tap-authorize/1",
        evidence_ref=ref,
    )
    rec = emit_decision_record(
        back_link=make_back_link(att),
        decision_derived=dd,
        iss="issuer://merchant-gateway",
        sub="agent:checkout-bot",
        secret_version="v1",
        alg="ES256",
        signing_material=priv,
        nonce=f"d-{seq}",
        iat=IAT,
    )
    return rec.to_dict()


_STEP_OK = {
    "action_ref_recomputes": True,
    "request_binding_resolves": True,
    "receipt_signature_ok": True,
}


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    (HERE / "keys").mkdir(parents=True, exist_ok=True)
    (HERE / "keys" / "es256_public.pem").write_bytes(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    expected = {"lifecycle_distinguishes_terminal": True}
    for seq, terminal in ((0, False), (1, True)):
        request = _tap_request(seq, terminal)
        receipt = _emit_receipt(priv, request, seq)
        step = f"step{seq}"
        _write(HERE / step / "request.json", request)
        _write(HERE / step / "receipt.json", receipt)
        expected[step] = dict(_STEP_OK)
        print(f"{step} action_ref:", request["actionRef"])

    _write(HERE / "expected.json", expected)
    print("wrote tap_v0 vectors: step0, step1, expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
