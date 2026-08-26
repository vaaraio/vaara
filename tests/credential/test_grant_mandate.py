# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A mandate on a minted grant: signed, swappable-proof, and byte-neutral when absent.

Every claim a grant makes about who its issuer is is self-asserted. A verifier
can check the signature and recompute the args commitment; it cannot check that
the issuer is the organisation it names. The mandate block carries a qualified
electronic attestation of attributes, issued by a supervised provider an
organisation cannot be, so that one claim rests on a public register instead of
on the issuer's word.

These tests mint and sign, so they need the attestation extra. The shape and
digest checks a relying party runs need no extras at all and live in
``test_grant_mandate_offline.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rfc8785")

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.credential import (  # noqa: E402
    EAA_Q_SERVICE_TYPE,
    BrokeredCredential,
    GrantBinding,
    GrantMandate,
    GrantScope,
    emit_grant,
    encode_attestation,
    grant_from_dict,
    mandate_digest_of,
    signing_payload,
    verify_grant_signature,
)

SECRET = b"0" * 32
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


def _emit(**kwargs) -> BrokeredCredential:
    return emit_grant(
        scope=GrantScope(
            tool_name="tx.transfer",
            args_commitment="sha256:" + "a" * 64,
            tenant_id="t",
        ),
        binding=GrantBinding(
            attestation_digest="sha256:" + "b" * 64, attestation_nonce="n-1",
        ),
        iss="issuer://a", sub="agent:filer-7", secret_version="v1",
        alg="HS256", signing_material=SECRET,
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


def test_withholding_the_attestation_after_signing_breaks_the_signature():
    """The digest-only path is a minting choice, not something done in transit."""
    original = _emit(mandate=_mandate())
    stripped = BrokeredCredential(
        version=original.version, alg=original.alg, scope=original.scope,
        binding=original.binding, asserted=original.asserted,
        signature=original.signature, mandate=_mandate(attestation=None),
    )
    assert not verify_grant_signature(stripped, verifying_material=SECRET)


def test_a_mandate_round_trips_through_the_grant_wire():
    original = _emit(mandate=_mandate())
    parsed = grant_from_dict(original.to_dict())
    assert parsed.mandate == original.mandate
    assert verify_grant_signature(parsed, verifying_material=SECRET)


def test_a_digest_only_mandate_round_trips_and_verifies():
    original = _emit(mandate=_mandate(attestation=None))
    parsed = grant_from_dict(original.to_dict())
    assert parsed.mandate.attestation is None
    assert verify_grant_signature(parsed, verifying_material=SECRET)


def test_an_unrecognised_mandate_key_is_rejected_on_the_grant_wire():
    wire = _emit(mandate=_mandate()).to_dict()
    wire["mandate"]["issuedFor"] = "whoever"
    with pytest.raises(AttestationError, match="closed"):
        grant_from_dict(wire)


def test_a_mandate_and_capabilities_coexist():
    """Two optional blocks, both signed, neither disturbing the other."""
    from vaara.credential import Capability

    grant = _emit(
        capabilities=[Capability(arg="amount", op="le", value="5000")],
        mandate=_mandate(),
    )
    payload = signing_payload(grant)
    assert b"mandate" in payload and b"capabilities" in payload
    assert verify_grant_signature(grant, verifying_material=SECRET)

    parsed = grant_from_dict(grant.to_dict())
    assert parsed.mandate == grant.mandate
    assert parsed.capabilities == grant.capabilities
