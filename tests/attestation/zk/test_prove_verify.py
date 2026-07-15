import pytest

from vaara.attestation.zk._circuit import to_fixed
from vaara.attestation.zk._prove import prove
from vaara.attestation.zk._verify import verify

BD = "sha256:" + "ab" * 32
DENY, ESC = 0.8, 0.5


@pytest.mark.parametrize(
    "score,verdict", [(0.2, "allow"), (0.6, "escalate"), (0.9, "block")]
)
def test_roundtrip(score, verdict):
    p = prove(verdict, to_fixed(score), to_fixed(DENY), to_fixed(ESC), BD)
    assert verify(p, verdict, BD) is True


def test_wrong_verdict_rejected():
    p = prove("allow", to_fixed(0.2), to_fixed(DENY), to_fixed(ESC), BD)
    assert verify(p, "block", BD) is False


def test_wrong_binding_rejected():
    p = prove("allow", to_fixed(0.2), to_fixed(DENY), to_fixed(ESC), BD)
    assert verify(p, "allow", "sha256:" + "cd" * 32) is False


def test_mutated_byte_rejected():
    p = bytearray(prove("allow", to_fixed(0.2), to_fixed(DENY), to_fixed(ESC), BD))
    p[80] ^= 0x01
    assert verify(bytes(p), "allow", BD) is False


def test_forged_false_statement_unprovable():
    # truly block; the prover cannot even build a proof claiming allow
    with pytest.raises(ValueError):
        prove("allow", to_fixed(0.9), to_fixed(DENY), to_fixed(ESC), BD)


def test_truncated_proof_rejected():
    p = prove("escalate", to_fixed(0.6), to_fixed(DENY), to_fixed(ESC), BD)
    assert verify(p[:-40], "escalate", BD) is False
