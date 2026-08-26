# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mandate block's shape and digest binding, with no extras installed.

Deliberately free of ``rfc8785``. Minting a grant needs JCS and therefore the
attestation extra, but validating a mandate and checking that the carried
attestation matches its commitment need a hash function and nothing else. Those
are the checks a relying party runs, and a relying party is exactly the party
least likely to have the producer's optional dependencies installed, so they are
tested where a base install can reach them.

The tests that do need to mint and sign are in ``test_grant_mandate.py``.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from vaara.attestation._attest_types import AttestationError
from vaara.credential import (
    EAA_Q_SERVICE_TYPE,
    GrantMandate,
    encode_attestation,
    mandate_digest_of,
    mandate_from_dict,
    verify_mandate_binding,
)
from vaara.credential._grant_mandate import mandate_to_dict

ATTESTATION_BYTES = b"eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJvcmc6ZmkifQ.sig"


def _mandate(**overrides) -> GrantMandate:
    kwargs = dict(
        attestation_digest=mandate_digest_of(ATTESTATION_BYTES),
        attestation=encode_attestation(ATTESTATION_BYTES),
        service_type_identifier=EAA_Q_SERVICE_TYPE,
        territory="HU",
        trust_list_ref="https://example.test/TL-HU.xml",
        provider_name="Microsec Micro Software Engineering & Consulting",
        legal_person_identifier="VATHU-12345678",
        attribute_set=("organizationIdentifier", "organizationName"),
        agent_id="agent:filer-7",
    )
    kwargs.update(overrides)
    return GrantMandate(**kwargs)


# ── The digest binding ───────────────────────────────────────────────────

def test_the_digest_binds_the_carried_attestation():
    assert verify_mandate_binding(_mandate()) is True


def test_the_digest_covers_the_issued_bytes_not_the_base64():
    """Two verifiers reading this differently would disagree on a valid record."""
    mandate = _mandate()
    over_raw = "sha256:" + hashlib.sha256(ATTESTATION_BYTES).hexdigest()
    over_b64 = "sha256:" + hashlib.sha256(
        base64.b64encode(ATTESTATION_BYTES)
    ).hexdigest()

    assert mandate.attestation_digest == over_raw
    assert over_raw != over_b64


def test_altered_attestation_bytes_fail_the_binding():
    tampered = _mandate(attestation=encode_attestation(ATTESTATION_BYTES + b"x"))
    assert verify_mandate_binding(tampered) is False


def test_a_withheld_attestation_still_verifies():
    """Digest-only is a supported path: the commitment can travel alone."""
    assert verify_mandate_binding(_mandate(attestation=None)) is True


def test_undecodable_attestation_fails_rather_than_raising():
    """A malformed attestation is a verdict, not a crash on the verify path."""
    assert verify_mandate_binding(_mandate(attestation="not base64 !!")) is False


# ── The closed schema ────────────────────────────────────────────────────

def test_a_mandate_round_trips_through_its_wire_form():
    mandate = _mandate()
    assert mandate_from_dict(mandate_to_dict(mandate)) == mandate


def test_a_withheld_attestation_omits_the_key_rather_than_nulling_it():
    wire = mandate_to_dict(_mandate(attestation=None))
    assert "attestation" not in wire
    assert mandate_from_dict(wire).attestation is None


def test_an_unrecognised_mandate_key_is_rejected():
    wire = mandate_to_dict(_mandate())
    wire["issuedFor"] = "whoever"
    with pytest.raises(AttestationError, match="closed"):
        mandate_from_dict(wire)


@pytest.mark.parametrize("block", ["issuer", "subject", "agentBinding"])
def test_an_unrecognised_key_in_any_sub_block_is_rejected(block):
    wire = mandate_to_dict(_mandate())
    wire[block]["surprise"] = "value"
    with pytest.raises(AttestationError, match="closed"):
        mandate_from_dict(wire)


def test_a_non_eaaq_service_type_is_rejected():
    """A different trust service is not this instrument, whatever it says."""
    with pytest.raises(AttestationError, match="EAA/Q"):
        _mandate(service_type_identifier="http://uri.etsi.org/TrstSvc/Svctype/TSA/QTST")


def test_a_reserved_bound_via_value_is_rejected():
    wire = mandate_to_dict(_mandate())
    wire["agentBinding"]["boundVia"] = "trustMeBro"
    with pytest.raises(AttestationError, match="boundVia"):
        mandate_from_dict(wire)


def test_an_unknown_format_is_rejected():
    with pytest.raises(AttestationError, match="format"):
        _mandate(format="eaa-q-yaml")


def test_a_bad_digest_prefix_is_rejected():
    with pytest.raises(AttestationError, match="sha256:"):
        _mandate(attestation_digest="md5:" + "a" * 32)


def test_a_short_digest_is_rejected():
    with pytest.raises(AttestationError, match="sha256:"):
        _mandate(attestation_digest="sha256:" + "a" * 63)


def test_an_empty_attribute_set_is_rejected():
    """An attestation that attests nothing is not evidence of anything."""
    with pytest.raises(AttestationError, match="attributeSet"):
        _mandate(attribute_set=())


def test_a_blank_attribute_name_is_rejected():
    with pytest.raises(AttestationError, match="attributeSet"):
        _mandate(attribute_set=("organizationName", ""))


def test_the_attribute_set_keeps_its_order():
    """Order is signed, so it cannot be normalised on the way through."""
    mandate = _mandate(attribute_set=("organizationName", "organizationIdentifier"))
    parsed = mandate_from_dict(mandate_to_dict(mandate))
    assert parsed.attribute_set == ("organizationName", "organizationIdentifier")
