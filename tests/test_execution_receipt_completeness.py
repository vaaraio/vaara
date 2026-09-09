"""Execution receipts carry a completeness sequence, so a dropped one is a gap.

Authorization receipts have carried a per-boundary ``seq`` and ``runningCount``
since the held-set work. Execution receipts did not, so the held-set proof
covered one half of the pair.

What a holder could already do: every execution receipt back-links to its
attestation and the authorization side is provably contiguous, so the set
difference names every allow decision with no execution receipt. What they could
not do: separate "the action never ran" from "the receipt was dropped or never
persisted", because the execution side had no contiguity to break and therefore
no gap to find.

THE LIMIT, PINNED HERE SO IT IS NOT OVERCLAIMED LATER. Contiguity closes
dropped-in-the-middle. A pure tail truncation is not detectable by sequence
contiguity alone and needs a timestamp anchor over the running count. The same
limit is already recorded on the authorization side and this change does not
close it.
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
    OutcomeDerived,
    emit_receipt,
    make_back_link,
    parse_receipt,
    verify_receipt_signature,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HS_SECRET = b"completeness-test-secret"


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"path": "/archive/2024-Q3.md"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="archive obsolete report"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _emit(att, **overrides):
    kwargs = dict(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at="2026-09-09T10:00:00Z"
        ),
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_receipt(**kwargs)


def _block(seq=0, boundary="vaara-mcp-proxy#execution"):
    return {"boundaryId": boundary, "seq": seq, "runningCount": seq + 1}


def _asserted(**overrides):
    kwargs = dict(
        iss="i", sub="s", iat="2026-09-09T00:00:00Z", nonce="n",
        secret_version="v1", alg="HS256",
    )
    kwargs.update(overrides)
    return ReceiptAsserted(**kwargs)


# --------------------------------------------------------------- the happy path

def test_completeness_rides_inside_the_signed_envelope():
    att = _attestation()
    receipt = _emit(att, completeness=_block(7))
    assert receipt.receipt_asserted.completeness == _block(7)
    # Signed, not merely attached: verification passes over the whole envelope.
    assert verify_receipt_signature(receipt, verifying_material=HS_SECRET)


def test_tampering_with_the_sequence_breaks_the_signature():
    """The property the whole change rests on.

    A completeness block a holder could edit after the fact would let anyone
    renumber a gap out of existence.
    """
    att = _attestation()
    receipt = _emit(att, completeness=_block(3))
    wire = receipt.to_dict()
    wire["receiptAsserted"]["completeness"]["seq"] = 4
    wire["receiptAsserted"]["completeness"]["runningCount"] = 5
    forged = parse_receipt(wire)
    assert not verify_receipt_signature(forged, verifying_material=HS_SECRET)


def test_wire_round_trip_preserves_the_block():
    att = _attestation()
    receipt = _emit(att, completeness=_block(12))
    back = parse_receipt(receipt.to_dict())
    assert back.receipt_asserted.completeness == _block(12)
    assert verify_receipt_signature(back, verifying_material=HS_SECRET)


# ------------------------------------------------- absent means byte-identical

def test_absent_completeness_leaves_the_envelope_unchanged():
    """A receipt minted without the block is what it was before the field.

    This is the additive-optional contract the same file already keeps for
    sigSuite and cryptoPosture. Breaking it would invalidate every receipt
    issued before today.
    """
    att = _attestation()
    receipt = _emit(att, iat="2026-09-09T10:00:00Z", nonce="fixed-nonce")
    wire = receipt.to_dict()
    assert "completeness" not in wire["receiptAsserted"]
    assert verify_receipt_signature(receipt, verifying_material=HS_SECRET)


def test_two_receipts_differ_only_by_the_block():
    att = _attestation()
    shared = dict(iat="2026-09-09T10:00:00Z", nonce="fixed-nonce")
    without = _emit(att, **shared).to_dict()
    with_block = _emit(att, completeness=_block(0), **shared).to_dict()
    del with_block["receiptAsserted"]["completeness"]
    # Signatures differ because the preimage differs, which is the point.
    without.pop("signature"), with_block.pop("signature")
    assert without == with_block


# ------------------------------------------------------------------ validation

def test_running_count_must_equal_seq_plus_one():
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = {"boundaryId": "b", "seq": 4, "runningCount": 9}
    with pytest.raises(AttestationError, match="runningCount must equal seq"):
        receipt_asserted_from_dict(d)


def test_negative_sequence_is_refused():
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = {"boundaryId": "b", "seq": -1, "runningCount": 0}
    with pytest.raises(AttestationError, match="must not be negative"):
        receipt_asserted_from_dict(d)


def test_boolean_is_not_an_integer_sequence():
    """bool subclasses int, so True would otherwise pass as seq 1."""
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = {"boundaryId": "b", "seq": True, "runningCount": 2}
    with pytest.raises(AttestationError, match="seq must be an integer"):
        receipt_asserted_from_dict(d)


def test_empty_boundary_id_is_refused():
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = {"boundaryId": "", "seq": 0, "runningCount": 1}
    with pytest.raises(AttestationError, match="boundaryId must be a non-empty"):
        receipt_asserted_from_dict(d)


def test_partial_block_is_refused_rather_than_half_read():
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = {"boundaryId": "b", "seq": 0}
    with pytest.raises(AttestationError, match="missing required field"):
        receipt_asserted_from_dict(d)


def test_unknown_field_in_the_block_is_refused():
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = {"boundaryId": "b", "seq": 0, "runningCount": 1, "x": 1}
    with pytest.raises(AttestationError):
        receipt_asserted_from_dict(d)


def test_non_object_completeness_is_refused():
    d = receipt_asserted_to_dict(_asserted())
    d["completeness"] = "0/1"
    with pytest.raises(AttestationError, match="must be an object or absent"):
        receipt_asserted_from_dict(d)


# ------------------------------------------------- the two streams stay apart

def test_execution_boundary_is_distinct_from_the_authorization_one():
    """A shared counter would read every denied call as a gap.

    A deny mints an authorization receipt and no execution receipt, by design.
    Counting both halves on one sequence would turn every legitimate deny into
    a missing number.
    """
    from vaara.integrations._mcp_attest import _EXEC_BOUNDARY, _ISS

    assert _EXEC_BOUNDARY != _ISS
    assert _EXEC_BOUNDARY.startswith(_ISS)
    assert "execution" in _EXEC_BOUNDARY


def test_a_gap_in_the_middle_is_visible_from_the_receipts_alone():
    """The end-to-end property, stated as a holder would check it."""
    att = _attestation()
    kept = [_emit(att, completeness=_block(i)) for i in (0, 1, 3, 4)]
    seqs = sorted(r.receipt_asserted.completeness["seq"] for r in kept)
    highest = seqs[-1]
    missing = [n for n in range(highest + 1) if n not in seqs]
    assert missing == [2]
    # And every surviving receipt still verifies, so the gap is the finding
    # rather than evidence that the set was mishandled.
    assert all(verify_receipt_signature(r, verifying_material=HS_SECRET) for r in kept)


# --------------------------------------------- the checker reads both halves

def test_contiguity_checker_finds_the_gap_in_execution_receipts():
    """The end of the block's ask: a decidable answer, not a list to chase."""
    from vaara.credential._contiguity import (
        completeness_from_execution_receipts,
        verify_contiguity,
    )

    att = _attestation()
    held = [_emit(att, completeness=_block(i)).to_dict() for i in (0, 1, 3)]
    report = verify_contiguity(completeness_from_execution_receipts(held))
    assert report.boundary_id == "vaara-mcp-proxy#execution"
    assert report.missing_seqs == [2]
    assert report.ok is False


def test_a_complete_execution_run_reports_ok():
    from vaara.credential._contiguity import (
        completeness_from_execution_receipts,
        verify_contiguity,
    )

    att = _attestation()
    held = [_emit(att, completeness=_block(i)).to_dict() for i in (0, 1, 2)]
    report = verify_contiguity(completeness_from_execution_receipts(held))
    assert report.ok is True
    assert report.present == report.expected == 3


def test_receipts_without_the_block_are_skipped_not_counted_as_gaps():
    """A receipt minted before the field existed is not a missing receipt."""
    from vaara.credential._contiguity import completeness_from_execution_receipts

    att = _attestation()
    old = _emit(att).to_dict()
    new = _emit(att, completeness=_block(0)).to_dict()
    projected = completeness_from_execution_receipts([old, new])
    assert len(projected) == 1
    assert projected[0]["completeness"] == _block(0)


def test_mixing_both_streams_refuses_to_guess():
    """Authorization and execution blocks in one call must name a boundary."""
    from vaara.credential._contiguity import verify_contiguity

    mixed = [
        {"completeness": {"boundaryId": "vaara-mcp-proxy", "seq": 0,
                          "runningCount": 1}},
        {"completeness": {"boundaryId": "vaara-mcp-proxy#execution", "seq": 0,
                          "runningCount": 1}},
    ]
    with pytest.raises(ValueError, match="span multiple boundaries"):
        verify_contiguity(mixed)
