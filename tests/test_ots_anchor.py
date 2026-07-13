"""OpenTimestamps receipt anchors: pending round-trip, pinning, upgrade."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("opentimestamps")  # the 'ots' extra; skip when absent

from opentimestamps.core.notary import (  # noqa: E402
    BitcoinBlockHeaderAttestation,
    PendingAttestation,
)
from opentimestamps.core.op import OpSHA256  # noqa: E402
from opentimestamps.core.serialize import (  # noqa: E402
    BytesSerializationContext,
)
from opentimestamps.core.timestamp import Timestamp  # noqa: E402

from vaara.audit.ots_anchor import (  # noqa: E402
    ots_anchor_receipt,
    parse_ots_proof,
    signed_payload_bytes,
    upgrade_ots_anchor,
    verify_ots_anchor,
)
from vaara.audit.receipt_anchor import anchored_digest  # noqa: E402
from vaara.audit.timeanchor import TimeAnchorError  # noqa: E402

VECTOR = (Path(__file__).resolve().parents[1]
          / "tests/vectors/x402_settlement_v0/generic/step1/receipt.json")

CAL = "https://alice.btc.calendar.opentimestamps.org"


@pytest.fixture
def receipt() -> dict:
    return json.loads(VECTOR.read_text())


def _pending_calendar_transport(url: str, data, timeout: float) -> bytes:
    """Fake a calendar POST /digest reply: a pending attestation over the msg."""
    assert url.endswith("/digest") and data is not None
    ts = Timestamp(data)
    ts.attestations.add(PendingAttestation(url[: -len("/digest")]))
    ctx = BytesSerializationContext()
    ts.serialize(ctx)
    return ctx.getbytes()


def _bitcoin_upgrade_transport(url: str, data, timeout: float) -> bytes:
    """Fake a calendar GET /timestamp/<hex> reply: Bitcoin-final attestation."""
    assert data is None and "/timestamp/" in url
    msg = bytes.fromhex(url.rsplit("/", 1)[1])
    ts = Timestamp(msg)
    ts.attestations.add(BitcoinBlockHeaderAttestation(812345))
    ctx = BytesSerializationContext()
    ts.serialize(ctx)
    return ctx.getbytes()


def test_pending_anchor_shape_and_round_trip(receipt: dict) -> None:
    anchor = ots_anchor_receipt(
        receipt, calendars=[CAL], transport=_pending_calendar_transport)
    assert anchor["method"] == "opentimestamps"
    assert anchor["status"] == "pending"
    assert anchor["calendars"] == [CAL]
    # anchoredDigest pins the JCS signed payload, same rule as rfc3161 anchors.
    assert anchor["anchoredDigest"] == anchored_digest(receipt)
    # The proof is a real .ots detached timestamp committing that digest, so
    # the standard `ots` client can read it.
    detached = parse_ots_proof(base64.b64decode(anchor["proof"]))
    assert isinstance(detached.file_hash_op, OpSHA256)
    expected_raw = bytes.fromhex(anchor["anchoredDigest"].split(":", 1)[1])
    assert detached.timestamp.msg == expected_raw
    result = verify_ots_anchor(receipt, anchor)
    assert result["status"] == "pending"
    assert result["pending_calendars"] == [CAL]
    assert result["bitcoin_block_heights"] == []


def test_verify_rejects_tampered_receipt(receipt: dict) -> None:
    anchor = ots_anchor_receipt(
        receipt, calendars=[CAL], transport=_pending_calendar_transport)
    tampered = dict(receipt, version=receipt["version"] + 1)
    with pytest.raises(TimeAnchorError):
        verify_ots_anchor(tampered, anchor)


def test_verify_rejects_proof_over_wrong_digest(receipt: dict) -> None:
    anchor = ots_anchor_receipt(
        receipt, calendars=[CAL], transport=_pending_calendar_transport)
    # Swap in a proof committing a different digest; anchoredDigest still
    # matches the receipt, so only the proof-commitment check can catch it.
    other = ots_anchor_receipt(
        dict(receipt, version=receipt["version"] + 1),
        calendars=[CAL], transport=_pending_calendar_transport)
    forged = dict(anchor, proof=other["proof"])
    with pytest.raises(TimeAnchorError):
        verify_ots_anchor(receipt, forged)


def test_wrong_method_rejected(receipt: dict) -> None:
    anchor = ots_anchor_receipt(
        receipt, calendars=[CAL], transport=_pending_calendar_transport)
    with pytest.raises(TimeAnchorError):
        verify_ots_anchor(receipt, dict(anchor, method="rfc3161"))


def test_upgrade_folds_bitcoin_attestation_and_is_idempotent(receipt: dict) -> None:
    anchor = ots_anchor_receipt(
        receipt, calendars=[CAL], transport=_pending_calendar_transport)
    upgraded = upgrade_ots_anchor(anchor, transport=_bitcoin_upgrade_transport)
    assert upgraded["status"] == "confirmed"
    assert upgraded["anchoredDigest"] == anchor["anchoredDigest"]
    result = verify_ots_anchor(receipt, upgraded)
    assert result["status"] == "confirmed"
    assert result["bitcoin_block_heights"] == [812345]

    # Idempotent: a confirmed anchor is returned as-is with no network call.
    calls: list[str] = []

    def counting(url, data, timeout):
        calls.append(url)
        return _bitcoin_upgrade_transport(url, data, timeout)

    again = upgrade_ots_anchor(upgraded, transport=counting)
    assert again == upgraded
    assert calls == []


def test_upgrade_leaves_pending_untouched_on_network_failure(receipt: dict) -> None:
    anchor = ots_anchor_receipt(
        receipt, calendars=[CAL], transport=_pending_calendar_transport)

    def failing(url, data, timeout):
        raise OSError("network down")

    same = upgrade_ots_anchor(anchor, transport=failing)
    assert same == anchor
    assert same["status"] == "pending"


def test_signed_payload_bytes_hash_matches_anchored_digest(receipt: dict) -> None:
    # What you feed the standard `ots` tool: sha256(payload) == anchoredDigest.
    import hashlib
    raw = signed_payload_bytes(receipt)
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == anchored_digest(receipt)


def test_missing_extra_reports_install_hint(receipt: dict, monkeypatch) -> None:
    import vaara.audit.ots_anchor as mod
    monkeypatch.setattr(mod, "_HAS_DEPS", False)
    with pytest.raises(TimeAnchorError, match=r"vaara\[ots\]"):
        ots_anchor_receipt(receipt)


def test_cli_anchor_ots_and_upgrade_ots(receipt: dict, tmp_path, monkeypatch, capsys) -> None:
    import vaara.audit.ots_anchor as mod
    from vaara.cli import main

    monkeypatch.setattr(mod, "_default_transport", _pending_calendar_transport)
    rpath = tmp_path / "receipt.json"
    rpath.write_text(json.dumps(receipt))
    out = tmp_path / "anchored.json"
    assert main(["receipt", "anchor-ots", str(rpath), "--calendar", CAL,
                 "--out", str(out)]) == 0
    anchored = json.loads(out.read_text())
    entries = anchored["timestampAnchors"]
    assert entries[-1]["method"] == "opentimestamps"
    assert entries[-1]["status"] == "pending"

    monkeypatch.setattr(mod, "_default_transport", _bitcoin_upgrade_transport)
    out2 = tmp_path / "upgraded.json"
    assert main(["receipt", "upgrade-ots", str(out), "--out", str(out2)]) == 0
    upgraded = json.loads(out2.read_text())
    assert upgraded["timestampAnchors"][-1]["status"] == "confirmed"
    assert "confirmed" in capsys.readouterr().out
