from vaara.attestation.zk._group import (
    P,
    N,
    G,
    Point,
    scalar_mul,
    hash_to_point,
    hash_to_scalar,
)

B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B


def _on_curve(pt):
    return (pt.y * pt.y - (pt.x**3 - 3 * pt.x + B)) % P == 0


def test_base_point_on_curve():
    assert _on_curve(G)


def test_scalar_mul_identity_and_order():
    assert scalar_mul(1, G) == G
    assert scalar_mul(N, G).is_infinity()


def test_known_double():
    two_g = G.double()
    assert scalar_mul(2, G) == two_g


def test_add_commutes():
    a = scalar_mul(3, G)
    b = scalar_mul(5, G)
    assert a + b == b + a
    assert (a + b) == scalar_mul(8, G)


def test_compress_roundtrip():
    p5 = scalar_mul(5, G)
    assert Point.from_bytes(p5.to_bytes()) == p5


def test_hash_to_point_on_curve_and_deterministic():
    h1 = hash_to_point(b"vaara/pedersen/H")
    h2 = hash_to_point(b"vaara/pedersen/H")
    assert h1 == h2
    assert _on_curve(h1)


def test_hash_to_scalar_range():
    s = hash_to_scalar(b"x", b"y")
    assert 0 <= s < N
