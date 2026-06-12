"""TPM evidence chain: a continuous-attestation loop bound to a SEP-2828 record.

Covers ``bind_record_to_chain_extra_data`` / ``link_digest`` (the per-link binding
primitives), ``verify_tpm_chain`` (the tiers unverified / linked / continuous, and
each cross-link invariant: hash-linkage, monotonic clock, reboot detection,
append-only IMA, stable AK, IMA-PCR pin), the ``vaara.tpm-evidence-chain/v0``
round-trip, and the structural rejection of a malformed chain bundle. The whole
wire path is exercised end to end through ``MockTPMQuoter``, which marshals and
ECDSA-signs real ``TPMS_ATTEST`` quotes, so no TPM is needed.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._tpm import (  # noqa: E402
    IMA_PCR,
    TPM_ALG_SHA256,
    MockTPMQuoter,
    replay_ima_pcr,
)
from vaara.attestation._tpm_binding import (  # noqa: E402
    bind_record_to_chain_extra_data,
)
from vaara.attestation._tpm_chain import (  # noqa: E402
    GENESIS_PREV_DIGEST,
    TPM_CHAIN_SCHEMA,
    TPMChainLink,
    link_digest,
    verify_tpm_chain,
)
from vaara.attestation.receipt import (  # noqa: E402
    build_tpm_chain_document,
    verify_tpm_chain_bundle,
)


@pytest.fixture
def ak_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def ak_pem(ak_key):
    return ak_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _record():
    return {"schema": "sep2828/v0", "decisionId": "d1", "signature": "ZmFrZQ=="}


def _line(template: str, filedata: str, path: str) -> str:
    """One ascii IMA entry with sha256-width (64 hex) template and file hashes."""
    return f"10 {template * 64} ima-ng sha256:{filedata * 64} {path}\n"


# A growing, append-only IMA log: each tick adds one measurement.
_LOGS = [
    _line("a", "b", "/usr/bin/a"),
]
_LOGS.append(_LOGS[0] + _line("c", "d", "/usr/bin/b"))
_LOGS.append(_LOGS[1] + _line("e", "f", "/usr/bin/c"))


def _build_links(
    record,
    ak_key,
    ak_pem,
    *,
    logs=None,
    clocks=(1000, 2000, 3000),
    resets=(3, 3, 3),
    restarts=(0, 0, 0),
    ak_pems=None,
):
    """Build an ordered list of TPMChainLink with each quote correctly bound.

    ``ak_pems`` optionally overrides the AK public PEM per link (to exercise an
    AK change mid-chain); the quote is still signed by ``ak_key`` so a single-AK
    chain verifies, and a per-link override only flips ``ak_stable``.
    """
    logs = logs if logs is not None else _LOGS
    q = MockTPMQuoter(ak_key)
    links = []
    prev = GENESIS_PREV_DIGEST
    for seq, log in enumerate(logs):
        pcr10 = replay_ima_pcr(log, TPM_ALG_SHA256)
        extra = bind_record_to_chain_extra_data(record, prev, seq)
        attest, sig = q.quote(
            extra,
            {IMA_PCR: pcr10},
            clock=clocks[seq],
            reset_count=resets[seq],
            restart_count=restarts[seq],
        )
        pem = ak_pems[seq] if ak_pems is not None else ak_pem
        links.append(TPMChainLink(attest, sig, pem, {IMA_PCR: pcr10}, log))
        prev = link_digest(attest)
    return links


# --- binding primitives ---------------------------------------------------


def test_chain_extra_data_is_64_bytes_and_deterministic():
    rec = _record()
    a = bind_record_to_chain_extra_data(rec, GENESIS_PREV_DIGEST, 0)
    assert len(a) == 64
    assert a == bind_record_to_chain_extra_data(rec, GENESIS_PREV_DIGEST, 0)


def test_chain_extra_data_differs_by_seq_and_prev():
    rec = _record()
    base = bind_record_to_chain_extra_data(rec, GENESIS_PREV_DIGEST, 0)
    assert base != bind_record_to_chain_extra_data(rec, GENESIS_PREV_DIGEST, 1)
    assert base != bind_record_to_chain_extra_data(rec, b"\x01" * 32, 0)


def test_negative_seq_rejected():
    with pytest.raises(ValueError):
        bind_record_to_chain_extra_data(_record(), GENESIS_PREV_DIGEST, -1)


def test_genesis_prev_digest_and_link_digest():
    assert GENESIS_PREV_DIGEST == bytes(32)
    import hashlib

    assert link_digest(b"abc") == hashlib.sha256(b"abc").digest()


# --- happy path -----------------------------------------------------------


def test_clean_chain_is_continuous(ak_key, ak_pem):
    rec = _record()
    v = verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem))
    assert v.tier == "continuous" and v.ok is True
    assert v.clock_monotonic and v.reboot_free and v.ima_append_only
    assert v.ak_stable and v.links_bound and v.n_links == 3
    assert v.freshness_basis == "chain_continuity"
    assert v.window["ima_entries_first"] == 1
    assert v.window["ima_entries_last"] == 3


def test_two_link_chain_is_continuous(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem, logs=_LOGS[:2], clocks=(1, 2),
                         resets=(3, 3), restarts=(0, 0))
    v = verify_tpm_chain(rec, links)
    assert v.tier == "continuous" and v.ok is True


def test_bundle_round_trip(ak_key, ak_pem):
    rec = _record()
    doc = build_tpm_chain_document(rec, _build_links(rec, ak_key, ak_pem))
    assert doc["schema"] == TPM_CHAIN_SCHEMA
    v = verify_tpm_chain_bundle(doc)
    assert v.tier == "continuous" and v.ok is True


def test_verdict_is_json_serializable(ak_key, ak_pem):
    rec = _record()
    json.dumps(verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem)).to_dict())


# --- linked-but-not-continuous (each link valid, a window invariant fails) -


def test_single_link_is_linked_not_continuous(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem, logs=_LOGS[:1], clocks=(1,),
                         resets=(3,), restarts=(0,))
    v = verify_tpm_chain(rec, links)
    assert v.tier == "linked" and v.ok is False


def test_clock_not_strictly_increasing_breaks_continuity(ak_key, ak_pem):
    rec = _record()
    v = verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem,
                                           clocks=(1000, 500, 3000)))
    assert v.tier == "linked" and v.ok is False
    assert v.clock_monotonic is False and v.links_bound is True


def test_clock_equal_is_not_monotonic(ak_key, ak_pem):
    rec = _record()
    v = verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem,
                                           clocks=(1000, 1000, 3000)))
    assert v.tier == "linked" and v.clock_monotonic is False


def test_reboot_resetcount_change_breaks_continuity(ak_key, ak_pem):
    rec = _record()
    v = verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem,
                                           resets=(3, 3, 9)))
    assert v.tier == "linked" and v.ok is False
    assert v.reboot_free is False and v.links_bound is True


def test_restartcount_change_breaks_continuity(ak_key, ak_pem):
    rec = _record()
    v = verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem,
                                           restarts=(0, 1, 1)))
    assert v.tier == "linked" and v.reboot_free is False


def test_ak_change_midchain_breaks_continuity(ak_key, ak_pem):
    rec = _record()
    other_pem = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Link 2 advertises a different AK; its quote is still signed by ak_key, so
    # the signature verifies against the advertised key only if it is ak_pem.
    # Use the same ak_key but flip the advertised pem on the last link's copy,
    # which both flips ak_stable and fails that link's signature check.
    links = _build_links(rec, ak_key, ak_pem)
    links[2] = replace(links[2], ak_pub_pem=other_pem)
    v = verify_tpm_chain(rec, links)
    assert v.ak_stable is False


def test_ima_not_append_only_breaks_continuity(ak_key, ak_pem):
    rec = _record()
    # An independent third log that still self-verifies but is NOT a prefix
    # extension of link 1: a rewrite of the measured state.
    fork = _line("9", "9", "/usr/bin/rogue")
    links = _build_links(rec, ak_key, ak_pem,
                         logs=[_LOGS[0], _LOGS[1], fork])
    v = verify_tpm_chain(rec, links)
    assert v.ima_append_only is False
    assert v.tier == "linked" and v.ok is False
    # every link still self-verifies (its own quote binds and replays)
    assert v.links_bound is True


# --- unverified (a link fails, including a broken hash-linkage) ------------


def test_reordered_links_fail_binding(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem)
    links[1], links[2] = links[2], links[1]
    v = verify_tpm_chain(rec, links)
    assert v.tier == "unverified" and v.links_bound is False


def test_dropped_middle_link_fails_binding(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem)
    del links[1]
    v = verify_tpm_chain(rec, links)
    assert v.tier == "unverified" and v.links_bound is False


def test_tampered_signature_in_a_link_fails(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem)
    bad = bytearray(links[1].signature)
    bad[-1] ^= 0x01
    links[1] = replace(links[1], signature=bytes(bad))
    v = verify_tpm_chain(rec, links)
    assert v.tier == "unverified" and v.links_bound is False
    assert v.links[1]["signature_valid"] is False


def test_tampered_attest_in_a_link_fails(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem)
    bad = bytearray(links[0].attest)
    bad[-1] ^= 0x01
    links[0] = replace(links[0], attest=bytes(bad))
    v = verify_tpm_chain(rec, links)
    assert v.tier == "unverified" and v.links_bound is False


def test_truncated_attest_does_not_traceback(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem)
    links[0] = replace(links[0], attest=links[0].attest[:12])
    v = verify_tpm_chain(rec, links)
    assert v.tier == "unverified"
    assert v.links[0]["parsed"] is False


# --- IMA-PCR pin ----------------------------------------------------------


def test_quiet_system_chain_pins(ak_key, ak_pem):
    # A system that loads nothing new: the IMA log is identical at each tick, so
    # PCR 10 is constant and can be pinned, while the clock still advances.
    rec = _record()
    same = [_LOGS[0], _LOGS[0], _LOGS[0]]
    links = _build_links(rec, ak_key, ak_pem, logs=same)
    pin = replay_ima_pcr(_LOGS[0], TPM_ALG_SHA256).hex()
    v = verify_tpm_chain(rec, links, expected_ima_pcr=pin)
    assert v.tier == "continuous" and v.ok is True
    assert v.pcr_pin_basis == "pinned"


def test_pin_mismatch_makes_chain_not_ok(ak_key, ak_pem):
    rec = _record()
    links = _build_links(rec, ak_key, ak_pem)
    wrong = ("11" * 32)
    v = verify_tpm_chain(rec, links, expected_ima_pcr=wrong)
    assert v.pcr_pin_basis == "pin_mismatch" and v.ok is False


# --- strict + structural --------------------------------------------------


def test_strict_is_unreachable_in_v0(ak_key, ak_pem):
    rec = _record()
    v = verify_tpm_chain(rec, _build_links(rec, ak_key, ak_pem), strict=True)
    assert v.ok is False


def test_empty_chain_raises(ak_key, ak_pem):
    with pytest.raises(ValueError):
        verify_tpm_chain(_record(), [])


def test_non_dict_record_raises(ak_key, ak_pem):
    with pytest.raises(ValueError):
        verify_tpm_chain(["not", "a", "dict"], _build_links(_record(), ak_key,
                                                            ak_pem))


def test_bundle_empty_links_raises(ak_key, ak_pem):
    doc = build_tpm_chain_document(_record(), _build_links(_record(), ak_key,
                                                           ak_pem))
    doc["links"] = []
    with pytest.raises(ValueError):
        verify_tpm_chain_bundle(doc)


def test_bundle_wrong_schema_raises(ak_key, ak_pem):
    doc = build_tpm_chain_document(_record(), _build_links(_record(), ak_key,
                                                           ak_pem))
    doc["schema"] = "something/else"
    with pytest.raises(ValueError):
        verify_tpm_chain_bundle(doc)


def test_bundle_non_base64_quote_raises(ak_key, ak_pem):
    doc = build_tpm_chain_document(_record(), _build_links(_record(), ak_key,
                                                           ak_pem))
    doc["links"][0]["quote"]["attest_b64"] = "!!!not base64!!!"
    with pytest.raises(ValueError):
        verify_tpm_chain_bundle(doc)
