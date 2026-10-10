#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate fallback_projection_v0: the SEP-2828 fallback projection, version
tools_call_params_plus_meta_authorization_binding_v1.

Writes envelopes/<name>.json (raw tools/call request envelopes, _meta sidecars
included) and expected.json. The expected digests come from the library's own
projection, the one the signed receipts in decision_pairing_v0 were bound
under; the independent checker recomputes them without importing Vaara.

    .venv/bin/python tests/vectors/fallback_projection_v0/_generate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from vaara.attestation._decision_verifier import (  # noqa: E402
    FALLBACK_PROJECTION_V1,
    MalformedFallbackBindingError,
    request_envelope_digest,
)
from vaara.attestation._attest_canonical import canonical_json  # noqa: E402
from vaara.attestation._decision_verifier import fallback_projection  # noqa: E402

BINDING = {"nonce": "srv-nonce-7f3a", "scope": "read"}
CALL = {"name": "document_fetch", "arguments": {"file": "invoice.pdf"}}

ENVELOPES: dict[str, dict] = {
    # The same call seen by the provider and by a gateway: the sidecars differ,
    # the projection does not.
    "provider_view": {**CALL, "_meta": {"progressToken": "pt-aaa",
                                        "authorization_binding": BINDING}},
    "gateway_view": {**CALL, "_meta": {"traceparent": "00-4bf92f3577b34da6-00f067aa0ba902b7-01",
                                       "x-injected-id": "inj-999",
                                       "authorization_binding": BINDING}},
    "binding_with_policy": {"name": "gcs_read",
                            "arguments": {"bucket": "prod-data", "object": "reports/q1.csv"},
                            "_meta": {"authorization_binding": {
                                "nonce": "srv-nonce-0001", "policyId": "pol-2026-001",
                                "scope": "storage"}}},
    "non_ascii_arguments": {"name": "write_note",
                            "arguments": {"text": "Päätös € é"},
                            "_meta": {"authorization_binding": {"nonce": "srv-nonce-0002"}}},
    # Each changes one bound field of provider_view, so the digest must change.
    "different_tool": {**CALL, "name": "document_fetch_v2",
                       "_meta": {"authorization_binding": BINDING}},
    "different_arguments": {**CALL, "arguments": {"file": "statement.pdf"},
                            "_meta": {"authorization_binding": BINDING}},
    "replayed_binding": {**CALL, "_meta": {"authorization_binding": {
        "nonce": "srv-nonce-other", "scope": "read"}}},
    # No projection exists; a verifier fails closed instead of widening the preimage.
    "no_binding": {**CALL, "_meta": {"progressToken": "pt-aaa"}},
    "binding_without_nonce": {**CALL, "_meta": {"authorization_binding": {"scope": "read"}}},
    "binding_not_object": {**CALL, "_meta": {"authorization_binding": "srv-nonce-7f3a"}},
    "missing_arguments": {"name": "document_fetch", "_meta": {"authorization_binding": BINDING}},
    "unsupported_version": {**CALL, "_meta": {"authorization_binding": BINDING}},
}
VERSION = {"unsupported_version": "tools_call_params_v0"}


def main() -> int:
    out = HERE / "envelopes"
    out.mkdir(exist_ok=True)
    expected: dict[str, dict] = {}
    for name, env in ENVELOPES.items():
        (out / f"{name}.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        version = VERSION.get(name, FALLBACK_PROJECTION_V1)
        try:
            proj = fallback_projection(env, version=version)
            expected[name] = {"version": version,
                              "projectionBytes": canonical_json(proj).decode("utf-8"),
                              "attestationDigest": request_envelope_digest(env, version=version)}
        except MalformedFallbackBindingError:
            expected[name] = {"version": version, "malformed": True}
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(expected)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
