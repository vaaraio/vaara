"""RFC 9942 Receipts for Inclusion over the RFC 6962 transparency log.

Checks the wire form RFC 9942 Section 5 defines (tagged COSE_Sign1, vds 395
in the protected header, vdp 396 with inclusion proofs under -1, detached
Merkle root as the payload) and, where pycose is installed, that an
independent COSE implementation verifies the signature over that root.
"""

from __future__ import annotations

import pytest

cbor2 = pytest.importorskip("cbor2")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation.cose_receipt import (  # noqa: E402
    COSE_SIGN1_TAG,
    rfc9942_receipt,
    verify_rfc9942_receipt,
)
from vaara.attestation.transparency_log import (  # noqa: E402
    InclusionProof,
    InProcessTransparencyLog,
)


def _log(n: int) -> InProcessTransparencyLog:
    log = InProcessTransparencyLog()
    for i in range(n):
        log.append(f"leaf-{i}".encode())
    return log


@pytest.fixture
def key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _receipt(log: InProcessTransparencyLog, idx: int, key: ec.EllipticCurvePrivateKey) -> bytes:
    return rfc9942_receipt(proof=log.inclusion_proof(idx), tree_root=log.root_hash,
                           private_key=key)


def test_wire_form_matches_rfc9942(key: ec.EllipticCurvePrivateKey) -> None:
    log = _log(7)
    msg = cbor2.loads(_receipt(log, 3, key))
    assert isinstance(msg, cbor2.CBORTag) and msg.tag == COSE_SIGN1_TAG == 18
    protected, unprotected, payload, signature = msg.value
    assert cbor2.loads(protected) == {1: -7, 395: 1}
    assert payload is None
    assert len(signature) == 64
    proofs = unprotected[396][-1]
    assert len(proofs) == 1
    size, index, path = cbor2.loads(proofs[0])
    assert (size, index) == (7, 3)
    assert [bytes(p) for p in path] == list(log.inclusion_proof(3).siblings)


@pytest.mark.parametrize("n,idx", [(1, 0), (2, 1), (5, 4), (8, 0), (13, 6)])
def test_receipt_verifies_for_its_entry(key: ec.EllipticCurvePrivateKey, n: int, idx: int) -> None:
    log = _log(n)
    assert verify_rfc9942_receipt(_receipt(log, idx, key),
                                  leaf_data=f"leaf-{idx}".encode(),
                                  public_key=key.public_key())


def test_wrong_entry_fails(key: ec.EllipticCurvePrivateKey) -> None:
    log = _log(5)
    assert not verify_rfc9942_receipt(_receipt(log, 2, key), leaf_data=b"leaf-3",
                                      public_key=key.public_key())


def test_other_operator_key_fails(key: ec.EllipticCurvePrivateKey) -> None:
    log = _log(5)
    other = ec.generate_private_key(ec.SECP256R1())
    assert not verify_rfc9942_receipt(_receipt(log, 2, key), leaf_data=b"leaf-2",
                                      public_key=other.public_key())


def test_signature_over_a_different_root_fails(key: ec.EllipticCurvePrivateKey) -> None:
    """The operator signs the root; a proof into another tree cannot borrow it."""
    log = _log(5)
    forged = rfc9942_receipt(proof=log.inclusion_proof(2), tree_root=bytes(32),
                             private_key=key)
    assert not verify_rfc9942_receipt(forged, leaf_data=b"leaf-2",
                                      public_key=key.public_key())


def test_leaf_index_outside_tree_fails(key: ec.EllipticCurvePrivateKey) -> None:
    log = _log(4)
    bad = InclusionProof(log_index=4, tree_size=4, siblings=())
    receipt = rfc9942_receipt(proof=bad, tree_root=log.root_hash, private_key=key)
    assert not verify_rfc9942_receipt(receipt, leaf_data=b"leaf-3",
                                      public_key=key.public_key())


def test_missing_vds_or_untagged_fails(key: ec.EllipticCurvePrivateKey) -> None:
    log = _log(3)
    protected, unprotected, payload, sig = cbor2.loads(_receipt(log, 1, key)).value
    untagged = cbor2.dumps([protected, unprotected, payload, sig])
    assert not verify_rfc9942_receipt(untagged, leaf_data=b"leaf-1",
                                      public_key=key.public_key())
    no_vds = cbor2.dumps(cbor2.CBORTag(18, [cbor2.dumps({1: -7}), unprotected, payload, sig]))
    assert not verify_rfc9942_receipt(no_vds, leaf_data=b"leaf-1",
                                      public_key=key.public_key())


def test_attached_payload_is_refused(key: ec.EllipticCurvePrivateKey) -> None:
    log = _log(3)
    protected, unprotected, _, sig = cbor2.loads(_receipt(log, 1, key)).value
    attached = cbor2.dumps(cbor2.CBORTag(18, [protected, unprotected, log.root_hash, sig]))
    assert not verify_rfc9942_receipt(attached, leaf_data=b"leaf-1",
                                      public_key=key.public_key())


def test_garbage_is_refused(key: ec.EllipticCurvePrivateKey) -> None:
    for junk in (b"", b"\xff\xff", cbor2.dumps({"inclusion": {}}), cbor2.dumps(cbor2.CBORTag(18, [1, 2]))):
        assert not verify_rfc9942_receipt(junk, leaf_data=b"x", public_key=key.public_key())


def _thaw(value: object) -> object:
    """Mutable copies of cbor2 6's tuple and frozendict, which pycose 1.1 rejects."""
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    if hasattr(value, "items"):
        return {k: _thaw(v) for k, v in value.items()}  # type: ignore[attr-defined]
    return value


def test_independent_cose_library_verifies_the_signature(key: ec.EllipticCurvePrivateKey) -> None:
    """pycose shares no code with Vaara; it checks the COSE_Sign1 over the root."""
    pytest.importorskip("pycose")
    from pycose.keys import EC2Key
    from pycose.messages import Sign1Message

    log = _log(9)
    receipt = _receipt(log, 5, key)
    # pycose 1.1 expects a list and dicts inside the tag, cbor2 6 returns immutable ones,
    # so Sign1Message.decode refuses every message. from_cose_obj takes the
    # decoded structure; the signature check below is still pycose's own code.
    tagged = cbor2.loads(receipt)
    assert tagged.tag == 18
    msg = Sign1Message.from_cose_obj(_thaw(tagged.value), True)
    numbers = key.public_key().public_numbers()
    msg.key = EC2Key(crv="P_256", x=numbers.x.to_bytes(32, "big"),
                     y=numbers.y.to_bytes(32, "big"))
    assert msg.verify_signature(detached_payload=log.root_hash)
    assert not msg.verify_signature(detached_payload=bytes(32))
