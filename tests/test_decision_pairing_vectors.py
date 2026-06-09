"""Guard: the committed SEP-2828 decision/outcome vectors verify independently.

Runs ``_check_independent.py`` (stdlib + cryptography + rfc8785, no Vaara
imports) over the committed fixtures. If the format or a fixture drifts,
the second-implementation checker fails here in CI rather than silently
shipping broken vectors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

VECTORS = Path(__file__).parent / "vectors" / "decision_pairing_v0"
CHECKER = VECTORS / "_check_independent.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_decision_pairing_vector_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_walker_passes_all_cases():
    assert _load_checker().main() == 0


def test_at_least_six_cases_present():
    cases = [p for p in (VECTORS / "normative").iterdir() if p.is_dir()]
    assert len(cases) >= 6


def test_vaara_verifier_reproduces_fallback_binding():
    """The Vaara reference verifier reproduces the no-2787 named-projection
    fallback binding over the committed envelopes, covering the projection
    rule's obligations: a provider view binds; a gateway view with different
    transport-local _meta binds to the same digest; changed bound params or a
    changed binding block break the back-link; an absent binding block fails
    closed instead of digesting the whole _meta."""
    import json

    from vaara.attestation.decision import (
        fallback_projection,
        parse_decision_record,
        request_envelope_digest,
        verify_decision_fallback_binding,
    )
    from vaara.attestation._receipt_verifier import (
        FALLBACK_BINDING_MALFORMED,
    )

    case = VECTORS / "normative" / "fallback_envelope_binding"
    decision = parse_decision_record(
        json.loads((case / "decision.json").read_text()))

    def env(name: str) -> dict:
        return json.loads((case / name).read_text())

    provider = env("request_envelope.json")
    gateway = env("request_envelope_gateway_view.json")

    # Provider view binds; gateway view binds to the same digest despite
    # carrying different non-binding _meta (rpelevin test 1).
    assert verify_decision_fallback_binding(
        decision, request_envelope=provider).ok is True
    assert verify_decision_fallback_binding(
        decision, request_envelope=gateway).ok is True
    assert request_envelope_digest(provider) == request_envelope_digest(gateway)
    # The digest comes from the named projection, not the whole _meta.
    assert fallback_projection(provider) == fallback_projection(gateway)

    # Changed bound params (test 3) and a changed binding block (test 2) each
    # break the back-link.
    replayed = verify_decision_fallback_binding(
        decision, request_envelope=env("request_envelope_replayed.json"))
    assert replayed.ok is False
    tampered = verify_decision_fallback_binding(
        decision, request_envelope=env("request_envelope_tampered_binding.json"))
    assert tampered.ok is False

    # An absent binding block fails closed with a distinct reason, rather than
    # widening the preimage to the whole _meta (test 4).
    malformed = verify_decision_fallback_binding(
        decision, request_envelope=env("request_envelope_no_binding.json"))
    assert malformed.ok is False
    assert malformed.reason == FALLBACK_BINDING_MALFORMED

    # The signed record names the projection version it used, so reconstruction
    # keys off trusted data; a record naming an unsupported or absent projection
    # fails closed rather than guessing the rule.
    import dataclasses

    assert decision.back_link.fallback_projection == "sep2828-fallback/1"
    bad_version = dataclasses.replace(
        decision,
        back_link=dataclasses.replace(
            decision.back_link, fallback_projection="sep2828-fallback/99"))
    assert verify_decision_fallback_binding(
        bad_version, request_envelope=provider).ok is False
    no_version = dataclasses.replace(
        decision,
        back_link=dataclasses.replace(
            decision.back_link, fallback_projection=None))
    res = verify_decision_fallback_binding(no_version, request_envelope=provider)
    assert res.ok is False
    assert res.reason == FALLBACK_BINDING_MALFORMED


def test_fallback_projection_excludes_transport_local_meta():
    """The projection digest is invariant to non-binding _meta a gateway can
    add or strip, and raises on an unreconstructable binding block."""
    from vaara.attestation.decision import (
        MalformedFallbackBindingError,
        request_envelope_digest,
    )

    base = {
        "name": "query_table",
        "arguments": {"table": "employees", "limit": 10},
        "_meta": {
            "authorization_binding": {"nonce": "n-1"},
        },
    }
    with_noise = {
        "name": "query_table",
        "arguments": {"table": "employees", "limit": 10},
        "_meta": {
            "authorization_binding": {"nonce": "n-1"},
            "io.modelcontextprotocol/progressToken": "p-9",
            "trace": {"spanId": "abc"},
        },
    }
    assert request_envelope_digest(base) == request_envelope_digest(with_noise)

    # An unsupported version and a missing block both fail closed.
    with pytest.raises(MalformedFallbackBindingError):
        request_envelope_digest(base, version="sep2828-fallback/99")
    no_block = {"name": "query_table", "arguments": {}, "_meta": {}}
    with pytest.raises(MalformedFallbackBindingError):
        request_envelope_digest(no_block)
