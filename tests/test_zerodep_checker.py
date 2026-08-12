"""The zero-install checker must agree with the reference checker, byte for byte.

``tests/vectors/governance_decision_v0/`` ships two checkers. ``_check_independent.py``
is the reference: it imports no Vaara, and downstream specifications pin its bytes.
``_check_zerodep.py`` reaches the same verdicts with nothing installed, implementing
RFC 8785 canonicalization, ES256 verification and SPKI parsing from the standard
library alone.

Two implementations are only worth having if a disagreement between them is loud, so
that is what these tests are: the canonicalizer is compared against ``rfc8785`` over
the corpus and over generated data, the verifier against ``cryptography``, and both
checkers must exit 0 on the committed vectors.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

VECTORS = Path(__file__).resolve().parent / "vectors" / "governance_decision_v0"
ZERODEP = VECTORS / "_check_zerodep.py"
REFERENCE = VECTORS / "_check_independent.py"


def _rfc8785():
    """The reference canonicalizer, skipped when absent.

    ``rfc8785`` and ``cryptography`` live in the ``attestation`` extra, so the base test
    matrix does not have them. Importing either at module scope would make the tests for a
    zero-install checker depend on the packages that checker exists to avoid, which is the
    wrong shape as well as a collection error. Every test below that needs one asks for it
    here; the tests that need nothing keep running everywhere.
    """
    return pytest.importorskip("rfc8785")


def _load_zerodep():
    spec = importlib.util.spec_from_file_location("_check_zerodep", ZERODEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zd = _load_zerodep()


# --- both checkers pass, and the zero-dep one passes without the dependencies ---


@pytest.mark.parametrize("checker", [REFERENCE, ZERODEP], ids=["reference", "zerodep"])
def test_checker_exits_zero(checker: Path) -> None:
    if checker is REFERENCE:
        _rfc8785()
        pytest.importorskip("cryptography")
    proc = subprocess.run([sys.executable, str(checker)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all verdicts matched expected" in proc.stdout


def test_zerodep_passes_with_no_third_party_packages_reachable() -> None:
    """``-S`` drops site-packages, so rfc8785 and cryptography cannot be imported at all.

    This is the property the file exists for: a reviewer with a bare Python reproduces
    every verdict. If an import of either package ever creeps in, this goes red.
    """
    _rfc8785()  # with no rfc8785 installed anywhere, the -S probe below proves nothing
    probe = subprocess.run(
        [sys.executable, "-S", "-c", "import rfc8785"], capture_output=True, text=True)
    assert probe.returncode != 0, "site-packages still reachable under -S, test is not proving anything"

    proc = subprocess.run([sys.executable, "-S", str(ZERODEP)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "standard library only" in proc.stdout


# --- canonicalization agrees with the reference implementation ------------------


def _corpus_objects():
    for path in sorted(VECTORS.rglob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def test_jcs_matches_rfc8785_over_the_whole_corpus() -> None:
    rfc8785 = _rfc8785()
    seen = 0
    for path, obj in _corpus_objects():
        assert zd.jcs(obj) == rfc8785.dumps(obj), f"canonicalization differs on {path.name}"
        seen += 1
    assert seen >= 20, f"expected the full corpus, only walked {seen} files"


def test_jcs_emits_utf8_not_escapes_for_the_unicode_vector() -> None:
    """The case that separates real RFC 8785 from ``json.dumps(sort_keys=True)``."""
    record = json.loads((VECTORS / "cases" / "unicode_scope.json").read_text(encoding="utf-8"))["record"]
    out = zd.jcs(record)
    assert "café".encode() in out
    assert b"\\u00e9" not in out
    assert out != json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


_JSON = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**53) + 1, max_value=2**53 - 1),
        st.text(max_size=40),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=20), children, max_size=5),
    ),
    max_leaves=25,
)


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(_JSON)
def test_jcs_matches_rfc8785_on_generated_data(value) -> None:
    """Covers what the corpus cannot: control characters, astral-plane keys, key ordering."""
    assert zd.jcs(value) == _rfc8785().dumps(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"￿": 1, "\U00010000": 2}, id="utf16-order-differs-from-codepoint-order"),
        pytest.param({"\U0001f600": 1, "z": 2}, id="astral-key"),
        pytest.param({"k": "line\nbreak\ttab\x00nul\x1f"}, id="control-characters"),
        pytest.param({"b": 1, "a": 2, "A": 3, "": 4}, id="key-ordering-including-empty"),
        pytest.param({"k": "quote\" backslash\\ slash/"}, id="escapes"),
        pytest.param({"k": "é€Ж"}, id="bmp-non-ascii"),
    ],
)
def test_jcs_matches_rfc8785_on_the_subtle_cases(value) -> None:
    """The corpus cannot reach these, and the sort key is where a JCS gets it wrong.

    ``"\\U00010000"`` encodes as the surrogate pair D800 DC00, so under UTF-16 code unit
    ordering it sorts BEFORE ``"\\uffff"`` even though its code point is higher. Sorting by
    code point instead would reverse these two keys and change the digest.
    """
    assert zd.jcs(value) == _rfc8785().dumps(value)


def test_utf16_ordering_is_not_codepoint_ordering() -> None:
    """Guard the reasoning above: if these two orders agreed, the test would prove nothing."""
    keys = ["￿", "\U00010000"]
    assert sorted(keys) != sorted(keys, key=lambda k: k.encode("utf-16-be"))


@pytest.mark.parametrize("value", [1.5, float("nan"), 2**53, -(2**53) - 1])
def test_jcs_refuses_what_it_cannot_guarantee(value) -> None:
    """An unsupported number raises rather than emitting bytes that might differ."""
    with pytest.raises(ValueError):
        zd.jcs({"n": value})


# --- ES256 verification agrees with cryptography --------------------------------


def test_es256_verifies_every_committed_signature() -> None:
    pub = zd.load_p256_pem((VECTORS / "keys" / "es256_public.pem").read_text(encoding="utf-8"))
    checked = 0
    for _path, obj in _corpus_objects():
        for wrapped in _wrapped_records(obj):
            assert zd.es256_verify(zd.jcs(wrapped["record"]), wrapped["signature"], pub) is True
            checked += 1
    assert checked >= 10, f"expected the corpus signatures, only checked {checked}"


def test_es256_rejects_a_tampered_record() -> None:
    pub = zd.load_p256_pem((VECTORS / "keys" / "es256_public.pem").read_text(encoding="utf-8"))
    wrapped = json.loads((VECTORS / "cases" / "unicode_scope.json").read_text(encoding="utf-8"))
    record = dict(wrapped["record"])
    record["agent_id"] = record["agent_id"] + "x"
    assert zd.es256_verify(zd.jcs(record), wrapped["signature"], pub) is False


def test_es256_rejects_a_tampered_signature() -> None:
    pub = zd.load_p256_pem((VECTORS / "keys" / "es256_public.pem").read_text(encoding="utf-8"))
    wrapped = json.loads((VECTORS / "cases" / "unicode_scope.json").read_text(encoding="utf-8"))
    sig = wrapped["signature"]
    flipped = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    assert zd.es256_verify(zd.jcs(wrapped["record"]), flipped, pub) is False


@pytest.mark.parametrize("bad", ["", "zz", "ab" * 63, "00" * 64])
def test_es256_rejects_malformed_signatures(bad: str) -> None:
    pub = zd.load_p256_pem((VECTORS / "keys" / "es256_public.pem").read_text(encoding="utf-8"))
    assert zd.es256_verify(b"anything", bad, pub) is False


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.binary(min_size=1, max_size=200))
def test_es256_agrees_with_cryptography_on_fresh_keys(message: bytes) -> None:
    """Independent accept and reject agreement against OpenSSL's implementation."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    key = ec.generate_private_key(ec.SECP256R1())
    r, s = decode_dss_signature(key.sign(message, ec.ECDSA(hashes.SHA256())))
    sig_hex = r.to_bytes(32, "big").hex() + s.to_bytes(32, "big").hex()
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    pub = zd.load_p256_pem(pem)
    assert zd.es256_verify(message, sig_hex, pub) is True
    assert zd.es256_verify(message + b"\x00", sig_hex, pub) is False


def test_spki_parser_rejects_an_off_curve_point() -> None:
    import base64

    der = base64.b64decode("".join(
        (VECTORS / "keys" / "es256_public.pem").read_text(encoding="utf-8")
        .replace("-----BEGIN PUBLIC KEY-----", "").replace("-----END PUBLIC KEY-----", "").split()))
    broken = bytearray(der)
    broken[-1] ^= 0xFF  # move the point off the curve
    pem = ("-----BEGIN PUBLIC KEY-----\n"
           + base64.encodebytes(bytes(broken)).decode()
           + "-----END PUBLIC KEY-----\n")
    with pytest.raises(ValueError, match="not a point on P-256"):
        zd.load_p256_pem(pem)


def _wrapped_records(obj):
    """Yield every ``{record, signature}`` pair anywhere in a corpus file."""
    if isinstance(obj, dict):
        if isinstance(obj.get("record"), dict) and isinstance(obj.get("signature"), str):
            yield obj
        for value in obj.values():
            yield from _wrapped_records(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _wrapped_records(value)
