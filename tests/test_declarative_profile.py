"""Tests for the declarative (data-only) source-profile engine.

The engine compiles a JSON spec into the same SourceProfile the registry holds,
so a field-mapping format is a dropped file, not new dispatch code.
"""
from __future__ import annotations

import pytest

from vaara.attestation._declarative import (
    ProfileSpecError,
    compile_profile,
    load_builtin_declarative_profiles,
    resolve_path,
)
from vaara.attestation.receipt import detect_format, normalize


# ── path resolution ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("a.b", 1),
    ("list[0]", "x"),
    ("list[1]", "y"),
    ("nested[0].k", "v"),
    ("missing", None),
    ("a.b.c", None),          # walk past a scalar
    ("list[9]", None),        # index out of range
    ("nested[0].absent", None),
])
def test_resolve_path(path, expected):
    doc = {"a": {"b": 1}, "list": ["x", "y"], "nested": [{"k": "v"}]}
    assert resolve_path(doc, path) == expected


def test_resolve_bad_index_raises():
    with pytest.raises(ProfileSpecError):
        resolve_path({"x": [0]}, "x[abc]")


# ── detect operators ─────────────────────────────────────────────────────────

def _profile(detect, **extra):
    spec = {"sourceFormat": "t", "sourceTitle": "T", "detect": detect, **extra}
    return compile_profile(spec)


def test_detect_all_equals_and_startswith():
    p = _profile({"all": [
        {"path": "kind", "equals": "k"},
        {"path": "uri", "startsWith": "https://"},
    ]})
    assert p.detector({"kind": "k", "uri": "https://x"})
    assert not p.detector({"kind": "k", "uri": "ftp://x"})
    assert not p.detector({"kind": "other", "uri": "https://x"})


def test_detect_any_and_in_and_exists():
    p = _profile({"any": [
        {"path": "t", "in": ["a", "b"]},
        {"path": "flag", "exists": True},
    ]})
    assert p.detector({"t": "a"})
    assert p.detector({"flag": 0})           # present, even if falsey
    assert not p.detector({"t": "z"})
    assert not p.detector({})


def test_detect_non_dict_is_false():
    p = _profile({"all": [{"path": "x", "exists": True}]})
    assert not p.detector(["not", "a", "dict"])


# ── normalizer field mapping ──────────────────────────────────────────────────

def test_advisory_skips_absent_and_keeps_const():
    p = _profile(
        {"all": [{"path": "kind", "equals": "k"}]},
        advisory={"present": "kind", "absent": "nope", "lit": {"const": 7}},
    )
    ev = p.normalizer({"kind": "k"}).to_dict()
    assert ev["advisory"] == {"present": "k", "lit": 7}


def test_sep2828_nested_and_populated_sorted():
    p = _profile(
        {"all": [{"path": "kind", "equals": "k"}]},
        sep2828={"outcomeDerived.status": {"const": "refused"},
                 "alg": {"const": "HS256"}},
        evidencePlane="outcome",
    )
    ev = p.normalizer({"kind": "k"}).to_dict()
    assert ev["recognized"] is True
    assert ev["evidencePlane"] == "outcome"
    assert ev["sep2828"] == {"outcomeDerived": {"status": "refused"}, "alg": "HS256"}
    assert ev["populated"] == ["alg", "outcomeDerived.status"]   # sorted


def test_forced_normalizer_on_mismatch_is_unrecognized():
    p = _profile({"all": [{"path": "kind", "equals": "k"}]})
    ev = p.normalizer({"kind": "other"}).to_dict()
    assert ev["recognized"] is False


# ── malformed specs ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", [
    {"sourceTitle": "T", "detect": {"all": [{"path": "x", "exists": True}]}},  # no format
    {"sourceFormat": "t", "sourceTitle": "T"},                                  # no detect
    {"sourceFormat": "t", "sourceTitle": "T", "detect": {}},                    # empty groups
    {"sourceFormat": "t", "sourceTitle": "T",
     "detect": {"all": [{"path": "x", "nope": 1}]}},                            # unknown op
    {"sourceFormat": "t", "sourceTitle": "T", "detect": {"all": "x"}},          # not a list
    {"sourceFormat": "t", "sourceTitle": "T", "priority": "high",
     "detect": {"all": [{"path": "x", "exists": True}]}},                       # bad priority
])
def test_malformed_spec_raises(spec):
    with pytest.raises(ProfileSpecError):
        compile_profile(spec)


def test_unknown_op_raised_only_when_evaluated():
    # bad rule under 'any' still raises at evaluation time
    p = _profile({"any": [{"path": "x", "exists": True}]})
    assert p.detector({"x": 1})


# ── shipped profiles + registry integration ──────────────────────────────────

def test_builtin_profiles_registered_and_recognized():
    ids = load_builtin_declarative_profiles()
    assert "slsa-provenance" in ids
    assert "c2pa-manifest" in ids
    assert "acp-checkout" in ids
    assert "x402-settlement" in ids
    assert "ap2-payment-receipt" in ids


def test_slsa_recognized_end_to_end():
    doc = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "pkg:pypi/vaara@1.10.0", "digest": {"sha256": "9f86"}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"runDetails": {"builder": {"id": "https://builder"}}},
    }
    assert detect_format(doc) == "slsa-provenance"
    ev = normalize(doc).to_dict()
    assert ev["recognized"] is True
    assert ev["advisory"]["builderId"] == "https://builder"
    assert "signature" in ev["missing"]      # honest gap: no signing event copied


def test_c2pa_recognized_end_to_end():
    doc = {
        "claim_generator": "vaara/1.10.0",
        "title": "photo.jpg",
        "format": "image/jpeg",
        "assertions": [{"label": "c2pa.actions", "data": {}}],
    }
    assert detect_format(doc) == "c2pa-manifest"
    ev = normalize(doc).to_dict()
    assert ev["recognized"] is True
    assert ev["advisory"]["firstAssertion"] == "c2pa.actions"
    assert ev["sep2828"] == {}               # provenance, asserts no signed record field


def test_acp_checkout_recognized_end_to_end():
    doc = {
        "id": "checkout_session_7Yx",
        "status": "completed",
        "currency": "usd",
        "line_items": [{"id": "li_laptop"}],
        "totals": [{"type": "total", "amount": 271766}],
        "capabilities": {"payment": {"handlers": [{"id": "card_tokenized",
                                                    "psp": "stripe"}]}},
        "order": {"id": "ord_456", "status": "completed"},
    }
    assert detect_format(doc) == "acp-checkout"
    ev = normalize(doc).to_dict()
    assert ev["recognized"] is True
    assert ev["evidencePlane"] == "outcome"      # the commercial-outcome face
    assert ev["advisory"]["paymentPsp"] == "stripe"
    assert ev["advisory"]["orderStatus"] == "completed"
    assert ev["sep2828"] == {}                   # unsigned: asserts no signed record field
    assert "signature" in ev["missing"]          # honest gap: ACP signs nothing here


def test_acp_does_not_match_a_bare_order():
    # An ACP Order carries status + line_items but no currency/capabilities, so
    # the checkout-session profile must not claim it.
    order = {
        "type": "order",
        "id": "ord_456",
        "checkout_session_id": "cs_1",
        "permalink_url": "https://m.example/o/ord_456",
        "status": "completed",
        "line_items": [{"id": "li_1"}],
        "totals": [{"type": "total", "amount": 100}],
    }
    assert detect_format(order) != "acp-checkout"


def test_x402_settlement_recognized_end_to_end():
    # A successful x402 v2 settlement response: the settled-transaction face of
    # an agent payment. transaction + network anchor it on-chain, but the
    # response itself is unsigned, so signature stays in the honest gap.
    doc = {
        "success": True,
        "payer": "0xabc0000000000000000000000000000000000001",
        "transaction": "0xdeadbeef",
        "network": "eip155:8453",
        "amount": "1000000",
    }
    assert detect_format(doc) == "x402-settlement"
    ev = normalize(doc).to_dict()
    assert ev["recognized"] is True
    assert ev["evidencePlane"] == "outcome"      # a settled-transaction outcome
    assert ev["advisory"]["txHash"] == "0xdeadbeef"
    assert ev["advisory"]["network"] == "eip155:8453"
    assert ev["sep2828"] == {}                   # x402 settle carries no signed record field
    assert "signature" in ev["missing"]          # honest gap: the response is unsigned
    assert "backLink" in ev["missing"]           # no link to what authorized the spend


def test_x402_does_not_match_a_payment_payload():
    # An x402 PaymentPayload (client-signed authorization) carries x402Version
    # and a nested payload, but no settled success/transaction/network at top
    # level, so the settlement profile must not claim it.
    payload = {
        "x402Version": 2,
        "accepted": {"scheme": "exact", "network": "eip155:8453"},
        "payload": {
            "signature": "0xsig",
            "authorization": {
                "from": "0xabc", "to": "0xdef", "value": "1000000",
                "validAfter": "0", "validBefore": "99", "nonce": "0x01",
            },
        },
    }
    assert detect_format(payload) != "x402-settlement"


def test_ap2_payment_receipt_recognized_end_to_end():
    # AP2 post-settlement receipt (decoded SD-JWT claims). Its `reference` is a
    # real back-link to the authorizing mandate, lifted to advisory; the JWS
    # signature lives in the envelope, so signature stays an honest gap.
    doc = {
        "status": "Success",
        "iss": "https://psp.example",
        "iat": 1782000000,
        "reference": "sha256:mandatehash",
        "payment_id": "pay_123",
        "psp_confirmation_id": "psp_abc",
        "network_confirmation_id": "net_xyz",
    }
    assert detect_format(doc) == "ap2-payment-receipt"
    ev = normalize(doc).to_dict()
    assert ev["recognized"] is True
    assert ev["evidencePlane"] == "outcome"      # the post-settlement final state
    assert ev["advisory"]["mandateRef"] == "sha256:mandatehash"
    assert ev["advisory"]["pspConfirmationId"] == "psp_abc"
    assert ev["sep2828"] == {}                   # claims-only: no signed field promoted
    assert "signature" in ev["missing"]          # JWS lives in the envelope, not the claims


def test_ap2_checkout_receipt_not_claimed_as_payment_receipt():
    # An AP2 checkout receipt carries reference + status but order_id, not
    # payment_id, so the payment-receipt profile must not claim it.
    checkout = {
        "status": "Success",
        "iss": "https://merchant.example",
        "iat": 1782000000,
        "reference": "sha256:mandatehash",
        "order_id": "ord_789",
    }
    assert detect_format(checkout) != "ap2-payment-receipt"
