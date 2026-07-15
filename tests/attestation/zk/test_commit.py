from vaara.attestation.zk._group import G, N, scalar_mul
from vaara.attestation.zk._commit import H, commit, gens, random_scalar


def test_commit_homomorphic():
    v1, b1, v2, b2 = 3, 7, 5, 11
    assert commit(v1, b1) + commit(v2, b2) == commit(v1 + v2, b1 + b2)


def test_commit_binding_generator_independent():
    # H must not be a known small multiple of G.
    h = commit(0, 1)
    assert all(h != scalar_mul(k, G) for k in range(1, 50))


def test_gens_distinct_and_deterministic():
    g1, h1 = gens(8)
    g2, h2 = gens(8)
    assert [p.to_bytes() for p in g1] == [p.to_bytes() for p in g2]
    allpts = [p.to_bytes() for p in g1 + h1]
    assert len(set(allpts)) == 16


def test_random_scalar_range():
    s = random_scalar()
    assert 0 <= s < N
