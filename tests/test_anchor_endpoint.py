# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the paid x402-gated /v1/anchor coin slot.

Covers the two halves that were added without direct coverage: the X402 gate's
admit/challenge decision, and the anchor endpoint wiring. The qualified anchorer
is stubbed so the suite stays offline (no QTSP network call).
"""

from __future__ import annotations

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vaara.audit.timeanchor import TimeAnchorError
    from vaara.server.routes import register
    from vaara.server.state import ServerState
    from vaara.server.x402 import X402Config, X402Gate
except ImportError:
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )


_ANCHOR = {
    "method": "rfc3161-eidas-qualified",
    "anchoredDigest": "sha256:" + "ab" * 32,
    "token": "Zm9v",
    "authority": "Sectigo Qualified Time Stamping CA",
    "tsaUrl": "http://timestamp.sectigo.com/qualified",
}


class StubAnchorer:
    """Offline stand-in for the qualified anchorer."""

    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises

    def anchor(self, receipt: dict) -> dict:
        if self._raises is not None:
            raise self._raises
        return dict(_ANCHOR)

    def attested_time(self, receipt: dict, anchor: dict) -> str:
        return "2026-07-21T12:00:00+00:00"


def make_client(gate: X402Gate | None = None, anchorer=None) -> TestClient:
    app = FastAPI()
    state = ServerState(x402_gate=gate, anchorer=anchorer or StubAnchorer())
    register(app, state)
    return TestClient(app)


def _paid_gate(facilitator: str | None = None) -> X402Gate:
    return X402Gate(X402Config(
        enabled=True, pay_to="0xabc", network="base",
        asset="0xusdc", price="1000", facilitator=facilitator,
    ))


# --- free mode (gate off) ---------------------------------------------------

def test_free_mode_anchors_the_receipt():
    """No gate enabled -> the coin slot is free: full anchor comes back."""
    client = make_client()  # default gate is disabled (env unset)
    r = client.post("/v1/anchor", json={"receipt": {"id": "r1"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["anchor"]["method"] == "rfc3161-eidas-qualified"
    assert body["anchor"]["authority"].startswith("Sectigo")
    assert body["attested"] == "2026-07-21T12:00:00+00:00"
    assert "DSS" in body["dss_hint"] or "dss" in body["dss_hint"]


def test_extra_fields_rejected():
    client = make_client()
    r = client.post("/v1/anchor", json={"receipt": {}, "sneaky": 1})
    assert r.status_code == 422


def test_upstream_qtsp_failure_is_502():
    """A QTSP condition (refusal/timeout/pin mismatch) surfaces as 502, not 500."""
    client = make_client(anchorer=StubAnchorer(raises=TimeAnchorError("QTSP down")))
    r = client.post("/v1/anchor", json={"receipt": {"id": "r1"}})
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "anchor_failed"


# --- paid mode (gate on) ----------------------------------------------------

def test_enabled_gate_challenges_without_payment():
    """Enabled gate + no X-PAYMENT -> 402 with x402 requirements; anchor untouched."""
    client = make_client(gate=_paid_gate(facilitator="https://facilitator.example"))
    r = client.post("/v1/anchor", json={"receipt": {"id": "r1"}})
    assert r.status_code == 402, r.text
    body = r.json()
    assert body["x402Version"] == 1
    accept = body["accepts"][0]
    assert accept["payTo"] == "0xabc"
    assert accept["maxAmountRequired"] == "1000"
    assert accept["resource"] == "/v1/anchor"


def test_no_provider_configured_returns_503(monkeypatch):
    """No provider is baked in: with no operator-chosen QTSP, anchoring refuses
    with 503 instead of routing traffic to a default provider nobody picked."""
    monkeypatch.delenv("VAARA_ANCHOR_TSA_URL", raising=False)
    from vaara.server.anchor import Anchorer

    app = FastAPI()
    state = ServerState(anchorer=Anchorer())  # real anchorer, no provider set
    register(app, state)
    client = TestClient(app)
    r = client.post("/v1/anchor", json={"receipt": {"id": "r1"}})
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "anchor_not_configured"


def test_enabled_gate_without_facilitator_refuses_payment():
    """A presented payment with no facilitator can't be settled -> refused, never
    trusted. Guards the invariant: an enabled gate never admits unverified pay."""
    client = make_client(gate=_paid_gate(facilitator=None))
    r = client.post(
        "/v1/anchor",
        json={"receipt": {"id": "r1"}},
        headers={"X-PAYMENT": "eyJzaWciOiJmYWtlIn0="},
    )
    assert r.status_code == 402, r.text
    assert r.json()["error"] == "payment invalid or unsettled"
