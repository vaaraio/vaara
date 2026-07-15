from vaara.attestation.zk._params import PROOF_SYSTEM, RANGE_BITS, SCALE, params_digest


def test_params_digest_stable_and_shaped():
    d = params_digest()
    assert d.startswith("sha256:") and len(d) == 71
    assert d == params_digest()


def test_constants():
    assert PROOF_SYSTEM == "vaara-p256-cap-v0"
    assert RANGE_BITS == 32 and SCALE == 10**6
