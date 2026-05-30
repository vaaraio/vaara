"""Guard: the committed v0 SEP-2787 attestation vectors are honest.

Two layers. First, the stdlib-only ``_check_independent.py`` walker
(cryptography + rfc8785, no Vaara import) must verify every committed
fixture: if the format or a fixture drifts, the second-implementation
checker fails here in CI rather than silently shipping broken vectors.
Second, the same fixtures are re-verified through the Vaara library
(``verify_attestation`` + ``verify_args_commitment``) so the published
``expected.json`` verdicts match the reference implementation, not just
the independent walker.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives import serialization  # noqa: E402

from vaara.attestation.sep2787 import (  # noqa: E402
    parse_attestation,
    verify_args_commitment,
    verify_attestation,
)

VECTORS = Path(__file__).parent / "vectors" / "sep2787_attestation_v0"
KEYS = VECTORS / "keys"
CHECKER = VECTORS / "_check_independent.py"

# Matches EVAL_NOW in _check_independent.py and the generator.
EVAL_NOW = datetime(2026, 5, 29, 10, 0, 30, tzinfo=timezone.utc).timestamp()


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_attestation_vector_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifying_material(alg: str):
    if alg == "HS256":
        return (KEYS / "hs256_secret.bin").read_bytes()
    if alg == "ES256":
        return serialization.load_pem_public_key(
            (KEYS / "es256_public.pem").read_bytes())
    if alg == "RS256":
        return serialization.load_pem_public_key(
            (KEYS / "rs256_public.pem").read_bytes())
    raise AssertionError(f"unexpected alg {alg!r}")


def _cases():
    return sorted(p for p in (VECTORS / "normative").iterdir() if p.is_dir())


def test_independent_walker_passes_all_cases():
    assert _load_checker().main() == 0


def test_at_least_six_cases_present():
    assert len(_cases()) >= 6


@pytest.mark.parametrize("case", _cases(), ids=lambda p: p.name)
def test_library_verdicts_match_expected(case):
    raw = json.loads((case / "attestation.json").read_text())
    expected = json.loads((case / "expected.json").read_text())
    att = parse_attestation(raw)
    material = _verifying_material(att.alg)

    # now=0 makes the TTL deadline trivially pass, isolating the signature.
    sig_ok = verify_attestation(att, verifying_material=material, now=0.0)
    assert sig_ok == expected["signature_ok"]

    # At EVAL_NOW the verdict is signature AND TTL.
    combined = verify_attestation(att, verifying_material=material, now=EVAL_NOW)
    assert combined == (expected["signature_ok"] and expected["ttl_ok"])

    runtime_args_path = case / "runtime_args.json"
    if runtime_args_path.exists():
        runtime_args = json.loads(runtime_args_path.read_text())
        result = verify_args_commitment(
            att.payload_derived.tool_calls[0].args,
            runtime_arguments=runtime_args,
        )
        assert result.ok == expected["args_commitment_ok"]
        assert result.projection_match == expected["projection_match"]
    else:
        assert expected["args_commitment_ok"] is None
        assert expected["projection_match"] is None
