"""Audience binding on execution receipts, and the three-valued check.

The case, from audit@ietf.org 2026-09-19 (a payments authorization layer):
signed verdicts with no claim naming the party they were issued for, so a
verdict for one counterparty verifies perfectly when replayed at another.
Integrity intact, association unsupported.

Two failures that a boolean cannot separate: the record names a different
audience (a positively identified conflict) and the record names no audience
(missing linkage evidence, nothing to fail on). The check returns three values
and decides the unsupported one before any comparison runs.
"""

from __future__ import annotations

import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation._receipt_types import (  # noqa: E402
    ReceiptAsserted,
    receipt_asserted_from_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation.receipt import (  # noqa: E402
    AudienceResult,
    OutcomeDerived,
    emit_receipt,
    make_back_link,
    parse_receipt,
    verify_receipt_audience,
    verify_receipt_signature,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HS_SECRET = b"audience-test-secret"


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="pay",
        server_fingerprint="sha256:" + "2" * 64,
        args=make_args_digest({"amount": "340", "currency": "USD"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:payer",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _emit(att, **overrides):
    kwargs = dict(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at="2026-09-19T10:00:00Z"
        ),
        iss="issuer://test",
        sub="agent:payer",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_receipt(**kwargs)


def _asserted(**overrides):
    kwargs = dict(
        iss="i", sub="s", iat="2026-09-19T00:00:00Z", nonce="n",
        secret_version="v1", alg="HS256",
    )
    kwargs.update(overrides)
    return ReceiptAsserted(**kwargs)


# ------------------------------------------------------------- three verdicts

def test_bound_when_aud_matches():
    receipt = _emit(_attestation(), aud="merchant:acme")
    result = verify_receipt_audience(receipt, "merchant:acme")
    assert result.verdict == "bound"
    assert bool(result) is True


def test_conflict_when_aud_differs():
    receipt = _emit(_attestation(), aud="merchant:acme")
    result = verify_receipt_audience(receipt, "merchant:other")
    assert result.verdict == "conflict"
    assert bool(result) is False
    assert "merchant:acme" in result.reason and "merchant:other" in result.reason


def test_unsupported_when_aud_absent():
    receipt = _emit(_attestation())
    result = verify_receipt_audience(receipt, "merchant:acme")
    assert result.verdict == "unsupported"
    assert result.receipt_aud is None
    assert bool(result) is False


def test_unsupported_and_conflict_are_distinct_values():
    absent = verify_receipt_audience(_emit(_attestation()), "merchant:acme")
    wrong = verify_receipt_audience(_emit(_attestation(), aud="x"), "merchant:acme")
    assert absent.verdict != wrong.verdict
    assert not absent and not wrong


def test_empty_expectation_is_the_callers_error_not_a_conflict():
    receipt = _emit(_attestation(), aud="merchant:acme")
    with pytest.raises(ValueError):
        verify_receipt_audience(receipt, "")


# ------------------------------------------------------- inside the signature

def test_aud_rides_inside_the_signed_envelope():
    receipt = _emit(_attestation(), aud="merchant:acme")
    assert verify_receipt_signature(receipt, verifying_material=HS_SECRET)
    d = receipt.to_dict()
    assert d["receiptAsserted"]["aud"] == "merchant:acme"
    d["receiptAsserted"]["aud"] = "merchant:other"
    tampered = parse_receipt(d)
    assert not verify_receipt_signature(tampered, verifying_material=HS_SECRET)


def test_replay_at_another_audience_is_a_conflict_with_signature_intact():
    # The Saifuro shape: the bytes verify, the association is wrong.
    receipt = _emit(_attestation(), aud="merchant:acme")
    assert verify_receipt_signature(receipt, verifying_material=HS_SECRET)
    assert verify_receipt_audience(receipt, "merchant:other").verdict == "conflict"


# ------------------------------------------------------ wire shape unchanged

def test_absent_aud_leaves_the_envelope_byte_identical():
    ra = _asserted()
    assert "aud" not in receipt_asserted_to_dict(ra)


def test_aud_round_trips_through_the_wire():
    ra = _asserted(aud="merchant:acme")
    d = receipt_asserted_to_dict(ra)
    assert d["aud"] == "merchant:acme"
    assert receipt_asserted_from_dict(d).aud == "merchant:acme"


@pytest.mark.parametrize("bad", ["", 7, {"a": 1}])
def test_malformed_aud_is_refused_on_read(bad):
    d = receipt_asserted_to_dict(_asserted())
    d["aud"] = bad
    with pytest.raises(AttestationError):
        receipt_asserted_from_dict(d)


def test_empty_aud_is_refused_on_emit():
    with pytest.raises(AttestationError):
        _emit(_attestation(), aud="")


def test_result_is_frozen_and_named():
    r = AudienceResult("bound", "a", "a", "ok")
    with pytest.raises(Exception):
        r.verdict = "conflict"  # type: ignore[misc]
