import pytest

from vaara.attestation.zk._circuit import range_witnesses, to_fixed


def test_the_branch_boundaries_are_inclusive():
    """escalate <= deny: a score at a threshold takes the branch above it."""
    d, e = to_fixed(0.8), to_fixed(0.5)
    assert all(w >= 0 for w in range_witnesses("block", d, d, e))
    assert all(w >= 0 for w in range_witnesses("escalate", e, d, e))
    with pytest.raises(ValueError):
        range_witnesses("escalate", d, d, e)
    with pytest.raises(ValueError):
        range_witnesses("allow", e, d, e)


def test_witnesses_nonneg_for_true_verdict():
    s, d, e = to_fixed(0.2), to_fixed(0.8), to_fixed(0.5)
    assert all(w >= 0 for w in range_witnesses("allow", s, d, e))


def test_witnesses_reject_false_verdict():
    s, d, e = to_fixed(0.9), to_fixed(0.8), to_fixed(0.5)  # truly block
    with pytest.raises(ValueError):
        range_witnesses("allow", s, d, e)


def test_witnesses_in_range_for_all_true_verdicts():
    cases = [(0.2, "allow"), (0.6, "escalate"), (0.9, "block")]
    d, e = to_fixed(0.8), to_fixed(0.5)
    for score, verdict in cases:
        for w in range_witnesses(verdict, to_fixed(score), d, e):
            assert 0 <= w < (1 << 32)
