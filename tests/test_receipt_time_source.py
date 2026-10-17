"""Per-receipt clock trust: the optional ``timeSource`` and its check.

``iat`` is the issuer's wall clock. On its own a verifier cannot tell an
NTP-synced host from a board that booted at 1970 and never synced, so the
receipt may declare where its time came from: ``ntp``, ``rtc``, ``unsynced``
or ``tsa``. The declaration rides inside the signed ``receiptAsserted`` block.
Absent means unknown, which is how every receipt minted before the field reads.

The check never reports an absent or ``unsynced`` source as a trusted time. A
receipt that claims ``tsa`` has to carry a timestamp token the verifier can
check (the ``existenceProof``); a claim with no token, or with a token that
does not verify over the record, is a finding.
"""

from __future__ import annotations

import base64
import importlib.util
import json

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
    TIME_SOURCES,
    OutcomeDerived,
    TimeSourceResult,
    emit_receipt,
    make_back_link,
    parse_receipt,
    verify_receipt_signature,
    verify_receipt_time_source,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.cli import main  # noqa: E402

HS_SECRET = b"time-source-test-secret"


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


def _emit(**overrides):
    kwargs = dict(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at="2026-10-18T10:00:00Z"
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
        iss="i", sub="s", iat="2026-10-18T00:00:00Z", nonce="n",
        secret_version="v1", alg="HS256",
    )
    kwargs.update(overrides)
    return ReceiptAsserted(**kwargs)


# ------------------------------------------------------------- the field


def test_the_four_sources():
    assert TIME_SOURCES == frozenset({"ntp", "rtc", "unsynced", "tsa"})


def test_absent_leaves_the_envelope_unchanged():
    d = _emit().to_dict()
    assert "timeSource" not in d["receiptAsserted"]
    assert "timeSource" not in receipt_asserted_to_dict(_asserted())


@pytest.mark.parametrize("source", sorted(TIME_SOURCES))
def test_declared_source_is_signed(source):
    r = _emit(time_source=source)
    d = r.to_dict()
    assert d["receiptAsserted"]["timeSource"] == source
    assert verify_receipt_signature(r, verifying_material=HS_SECRET)


def test_tampered_source_breaks_the_signature():
    d = _emit(time_source="unsynced").to_dict()
    d["receiptAsserted"]["timeSource"] = "ntp"
    assert not verify_receipt_signature(parse_receipt(d), verifying_material=HS_SECRET)


def test_round_trips_through_the_wire_form():
    d = _emit(time_source="rtc").to_dict()
    again = parse_receipt(json.loads(json.dumps(d)))
    assert again.receipt_asserted.time_source == "rtc"
    assert again.to_dict() == d


@pytest.mark.parametrize("bad", ["", "gps", "NTP", 1])
def test_emit_refuses_an_unknown_source(bad):
    with pytest.raises(AttestationError):
        _emit(time_source=bad)


@pytest.mark.parametrize("bad", ["", "gps", "NTP", 1, {}, None])
def test_parse_refuses_an_unknown_source(bad):
    # null is refused too: absent is spelled by leaving the key out.
    d = receipt_asserted_to_dict(_asserted())
    d["timeSource"] = bad
    with pytest.raises(AttestationError):
        receipt_asserted_from_dict(d)


# ------------------------------------------------------------- the check


def test_absent_is_unknown_and_never_trusted():
    res = verify_receipt_time_source(_emit().to_dict())
    assert isinstance(res, TimeSourceResult)
    assert res.verdict == "unknown"
    assert res.declared is None
    assert res.trusted is False and bool(res) is False
    assert res.finding is False


def test_unsynced_is_reported_and_never_trusted():
    res = verify_receipt_time_source(_emit(time_source="unsynced").to_dict())
    assert res.verdict == "unsynced"
    assert res.trusted is False and res.finding is False


@pytest.mark.parametrize("source", ["ntp", "rtc"])
def test_clock_sources_are_issuer_declared_not_trusted(source):
    res = verify_receipt_time_source(_emit(time_source=source).to_dict())
    assert res.verdict == "declared"
    assert res.declared == source
    assert res.trusted is False and res.finding is False


def test_tsa_without_a_token_is_a_finding():
    res = verify_receipt_time_source(_emit(time_source="tsa").to_dict())
    assert res.verdict == "unverifiable"
    assert res.finding is True and res.trusted is False
    assert "existenceProof" in res.reason


def test_unknown_value_on_the_wire_is_a_finding():
    d = _emit().to_dict()
    d["receiptAsserted"]["timeSource"] = "gps"
    res = verify_receipt_time_source(d)
    assert res.verdict == "invalid"
    assert res.finding is True and res.trusted is False


def test_not_a_receipt_is_a_finding():
    res = verify_receipt_time_source({"receiptAsserted": "nope"})
    assert res.verdict == "invalid" and res.finding is True


# ------------------------------------------- tsa with a real RFC 3161 token

_HAS_TIMEANCHOR = importlib.util.find_spec("asn1crypto") is not None
requires_timeanchor = pytest.mark.skipif(
    not _HAS_TIMEANCHOR, reason="tsa check needs the timeanchor extra"
)


def _with_token(record, *, over=None):
    from tests.test_receipt_existence import (
        _attach_proof,
        _self_signed_ca,
        _token_over_record,
        _tsa_issued_by,
    )

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)
    token = _token_over_record(over or record, tsa_key, tsa_der)
    return _attach_proof(record, token)


@requires_timeanchor
def test_tsa_with_a_valid_token_is_trusted():
    from tests.test_receipt_existence import _GEN_TIME

    d = _with_token(_emit(time_source="tsa").to_dict())
    res = verify_receipt_time_source(d)
    assert res.verdict == "attested"
    assert res.trusted is True and bool(res) is True
    assert res.finding is False
    assert res.attested_time == _GEN_TIME.isoformat()


@requires_timeanchor
def test_tsa_with_a_token_over_other_bytes_is_a_finding():
    record = _emit(time_source="tsa").to_dict()
    other = _emit(time_source="tsa").to_dict()
    d = _with_token(record, over=other)
    res = verify_receipt_time_source(d)
    assert res.verdict == "unverifiable"
    assert res.finding is True and res.trusted is False


def test_tsa_with_a_garbled_token_is_a_finding():
    d = _emit(time_source="tsa").to_dict()
    from vaara.attestation.receipt import existence_record_digest

    d["existenceProof"] = {
        "backend": "rfc3161-eidas-qualified",
        "hashAlgorithm": "sha256",
        "recordDigest": existence_record_digest(d),
        "token": base64.b64encode(b"not a token").decode("ascii"),
    }
    res = verify_receipt_time_source(d)
    assert res.finding is True and res.trusted is False


@requires_timeanchor
def test_a_token_does_not_upgrade_a_clock_claim():
    d = _with_token(_emit(time_source="ntp").to_dict())
    res = verify_receipt_time_source(d)
    assert res.verdict == "declared"
    assert res.trusted is False


def test_to_dict_names_every_field():
    res = verify_receipt_time_source(_emit(time_source="unsynced").to_dict())
    assert res.to_dict() == {
        "verdict": "unsynced",
        "declared": "unsynced",
        "trusted": False,
        "finding": False,
        "attestedTime": None,
        "reason": res.reason,
    }


# ------------------------------------------------------------- verify-record


def _write(tmp_path, record):
    p = tmp_path / "record.json"
    p.write_text(json.dumps(record), encoding="utf-8")
    return str(p)


def test_cli_reports_absent_as_unknown(tmp_path, capsys):
    rc = main(["verify-record", _write(tmp_path, _emit().to_dict()), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["timeSource"]["verdict"] == "unknown"
    assert out["timeSource"]["trusted"] is False


def test_cli_text_names_unsynced(tmp_path, capsys):
    rc = main(["verify-record", _write(tmp_path, _emit(time_source="unsynced").to_dict())])
    out = capsys.readouterr().out
    assert rc == 0
    assert "time source: unsynced" in out


def test_cli_fails_a_tsa_claim_with_no_token(tmp_path, capsys):
    rc = main(["verify-record", _write(tmp_path, _emit(time_source="tsa").to_dict())])
    out = capsys.readouterr().out
    assert rc == 1
    assert "time source: FAIL" in out
