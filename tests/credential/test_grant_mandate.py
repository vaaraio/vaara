# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The optional mandate block: a qualified attestation carried on a grant.

Every claim a grant makes about who the issuer is is self-asserted. A verifier
can check the signature and recompute the args commitment; it cannot check that
the issuer is the organisation it says it is. The mandate block carries a
qualified electronic attestation of attributes, issued by a supervised provider
an organisation cannot be, so that one claim rests on a public register instead
of on the issuer's word.

These tests cover what is checkable offline: the block's shape, its closed
schema, and the digest that binds the carried bytes to the commitment. Resolving
the provider against the EU trusted lists is the next piece and is not here.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from vaara.attestation._attest_types import AttestationError
from vaara.credential import (
    BrokeredCredential,
    GrantBinding,
    GrantMandate,
    GrantScope,
    emit_grant,
    grant_from_dict,
    signing_payload,
    verify_grant_signature,
    verify_mandate_binding,
)

SECRET = b"0" * 32
EAAQ = "http://uri.etsi.org/TrstSvc/Svctype/EAA/Q"
ATTESTATION_BYTES = b"eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJvcmc6ZmkifQ.sig"


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _mandate(**overrides) -> GrantMandate:
    kwargs = dict(
        attestation_digest=_digest(ATTESTATION_BYTES),
        attestation=base64.b64encode(ATTESTATION_BYTES).decode("ascii"),
        service_type_identifier=EAAQ,
        territory="HU",
        trust_list_ref="https://example.test/TL-HU.xml",
        provider_name="Microsec Micro Software Engineering & Consulting",
        legal_person_identifier="VATHU-12345678",
        attribute_set=("organizationIdentifier", "organizationName"),
        agent_id="agent:filer-7",
    )
    kwargs.update(overrides)
    return GrantMandate(**kwargs)


def _scope() -> GrantScope:
    return GrantScope(
        tool_name="tx.transfer", args_commitment="sha256:" + "a" * 64, tenant_id="t",
    )


def _binding() -> GrantBinding:
    return GrantBinding(
        attestation_digest="sha256:" + "b" * 64, attestation_nonce="n-1",
    )


def _emit(**kwargs) -> BrokeredCredential:
    return emit_grant(
        scope=_scope(), binding=_binding(), iss="issuer://a", sub="agent:filer-7",
        secret_version="v1", alg="HS256", signing_material=SECRET,
        iat="2026-08-26T10:00:00Z", nonce="g-1", **kwargs,
    )


# ── The absent case, which must not move ─────────────────────────────────

def test_a_grant_without_a_mandate_is_byte_identical():
    """The whole point of an optional block: nothing published changes."""
    before = signing_payload(_emit())
    after = signing_payload(_emit(mandate=None))
    assert before == after
    assert b"mandate" not in before


def test_a_grant_without_a_mandate_has_no_mandate_key():
    assert "mandate" not in _emit().to_dict()


# ── Presence ─────────────────────────────────────────────────────────────

def test_a_mandate_is_covered_by_the_grant_signature():
    """Carried but unsigned would be a claim anyone could swap."""
    with_mandate = _emit(mandate=_mandate())
    assert b"mandate" in signing_payload(with_mandate)
    assert verify_grant_signature(with_mandate, verifying_material=SECRET)


def test_swapping_the_mandate_breaks_the_signature():
    original = _emit(mandate=_mandate())
    swapped = BrokeredCredential(
        version=original.version, alg=original.alg, scope=original.scope,
        binding=original.binding, asserted=original.asserted,
        signature=original.signature,
        mandate=_mandate(territory="SE", provider_name="IDnow Trust Services AB"),
    )
    assert not verify_grant_signature(swapped, verifying_material=SECRET)


def test_a_mandate_round_trips_through_the_wire():
    original = _emit(mandate=_mandate())
    parsed = grant_from_dict(original.to_dict())
    assert parsed.mandate == original.mandate
    assert verify_grant_signature(parsed, verifying_material=SECRET)


# ── The digest binding ───────────────────────────────────────────────────

def test_the_digest_binds_the_carried_attestation():
    assert verify_mandate_binding(_mandate()) is True


def test_altered_attestation_bytes_fail_the_binding():
    """The evidence is the bytes the provider issued, not a normalised copy."""
    tampered = _mandate(
        attestation=base64.b64encode(ATTESTATION_BYTES + b"x").decode("ascii"),
    )
    assert verify_mandate_binding(tampered) is False


def test_a_withheld_attestation_still_verifies():
    """Digest-only is a supported path: the commitment can travel alone."""
    assert verify_mandate_binding(_mandate(attestation=None)) is True


def test_a_withheld_attestation_omits_the_key_rather_than_nulling_it():
    grant = _emit(mandate=_mandate(attestation=None))
    assert "attestation" not in grant.to_dict()["mandate"]
    assert grant_from_dict(grant.to_dict()).mandate.attestation is None


def test_undecodable_attestation_fails_the_binding_rather_than_raising():
    assert verify_mandate_binding(_mandate(attestation="not base64 !!")) is False


# ── The closed schema ────────────────────────────────────────────────────

def test_an_unrecognised_mandate_key_is_rejected():
    wire = _emit(mandate=_mandate()).to_dict()
    wire["mandate"]["issuedFor"] = "whoever"
    with pytest.raises(AttestationError, match="closed"):
        grant_from_dict(wire)


def test_an_unrecognised_issuer_key_is_rejected():
    wire = _emit(mandate=_mandate()).to_dict()
    wire["mandate"]["issuer"]["supervisor"] = "someone"
    with pytest.raises(AttestationError, match="closed"):
        grant_from_dict(wire)


def test_a_non_eaaq_service_type_is_rejected():
    """A different trust service is not this instrument, whatever it says."""
    with pytest.raises(AttestationError, match="EAA/Q"):
        _mandate(service_type_identifier="http://uri.etsi.org/TrstSvc/Svctype/TSA/QTST")


def test_a_reserved_bound_via_value_is_rejected():
    wire = _emit(mandate=_mandate()).to_dict()
    wire["mandate"]["agentBinding"]["boundVia"] = "trustMeBro"
    with pytest.raises(AttestationError, match="boundVia"):
        grant_from_dict(wire)


def test_a_bad_digest_prefix_is_rejected():
    with pytest.raises(AttestationError, match="sha256:"):
        _mandate(attestation_digest="md5:" + "a" * 32)


def test_an_empty_attribute_set_is_rejected():
    """An attestation that attests nothing is not evidence of anything."""
    with pytest.raises(AttestationError, match="attributeSet"):
        _mandate(attribute_set=())


def test_the_attribute_set_survives_the_wire_in_order():
    """Order is signed, so it cannot be normalised on the way through."""
    mandate = _mandate(attribute_set=("organizationName", "organizationIdentifier"))
    parsed = grant_from_dict(_emit(mandate=mandate).to_dict())
    assert parsed.mandate.attribute_set == (
        "organizationName", "organizationIdentifier",
    )
