#!/usr/bin/env python3
"""Generate the credential_grant_v0 conformance vectors.

Run once with Vaara installed; emits, under
``conformance/sep2828/credential_grant_v0/sets/<name>/``, a real bound grant
plus the attestation it pins, the HS256 key, and the runtime call a gateway
would see. The committed ``_check_independent.py`` then re-derives every
verdict from those bytes with NO ``import vaara`` (rfc8785 + hashlib + hmac),
which is the whole point: a neutral party reproduces the format.

Usage: ``python scripts/build_credential_grant_vectors.py``
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vaara.attestation._attest_canonical import iso8601_to_epoch
from vaara.attestation.receipt import attestation_digest
from vaara.credential import GrantBinding, GrantScope, emit_grant
from vaara.integrations._mcp_attest import build_attest_emitter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "conformance" / "sep2828" / "credential_grant_v0"
KEY = b"conformance-credential-grant-v0-shared-secret!!"
IAT = "2026-06-18T12:00:00Z"
IAT_EPOCH = iso8601_to_epoch(IAT)
TOOL = "files.read"
ARGS = {"path": "/data/report.txt"}
TENANT = "tenant-alpha"


def _attestation(emitter):
    att, _counter = emitter.emit_attestation(
        tool_name=TOOL, arguments=ARGS, upstream_name="default", tenant_id=TENANT
    )
    return att


def _grant(att, *, nonce="grant-nonce-fixed", exp=60):
    scope = GrantScope(
        tool_name=TOOL,
        args_commitment=att.payload_derived.tool_calls[0].args.projection_digest,
        tenant_id=TENANT,
    )
    binding = GrantBinding(
        attestation_digest=attestation_digest(att),
        attestation_nonce=att.issuer_asserted.nonce,
    )
    return emit_grant(
        scope=scope, binding=binding, iss="vaara-mcp-proxy",
        sub=f"{TENANT}/default", secret_version="key-v1", alg="HS256",
        signing_material=KEY, exp_seconds=exp, nonce=nonce, iat=IAT,
    )


def _write(name, *, grant, attestation, runtime):
    d = OUT / "sets" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "grant.json").write_text(json.dumps(grant.to_dict(), indent=2) + "\n")
    (d / "attestation.json").write_text(
        json.dumps(attestation.to_dict(), indent=2) + "\n"
    )
    (d / "inputs.json").write_text(json.dumps(runtime, indent=2) + "\n")


def main() -> int:
    with TemporaryDirectory() as tmp:
        key_file = Path(tmp) / "k"
        key_file.write_bytes(KEY)
        emitter = build_attest_emitter(
            signing_key_path=key_file, receipts_dir=Path(tmp) / "r",
            upstream_commands={"default": ["echo"]},
        )
        att = _attestation(emitter)
        other = _attestation(emitter)  # a different attestation, for binding_unknown
        grant = _grant(att)

        base_runtime = {
            "toolName": TOOL, "args": ARGS, "tenantId": TENANT,
            "now": IAT_EPOCH + 5, "keyHex": KEY.hex(),
        }
        expected = {}

        _write("ok", grant=grant, attestation=att, runtime=base_runtime)
        expected["ok"] = {"ok": True, "reason": "ok"}

        forged = grant.to_dict()
        forged["signature"] = ("00" if forged["signature"][:2] != "00" else "11") + forged["signature"][2:]
        d = OUT / "sets" / "bad_signature"
        d.mkdir(parents=True, exist_ok=True)
        (d / "grant.json").write_text(json.dumps(forged, indent=2) + "\n")
        (d / "attestation.json").write_text(json.dumps(att.to_dict(), indent=2) + "\n")
        (d / "inputs.json").write_text(json.dumps(base_runtime, indent=2) + "\n")
        expected["bad_signature"] = {"ok": False, "reason": "bad_signature"}

        _write("scope_mismatch", grant=grant, attestation=att,
               runtime={**base_runtime, "args": {"path": "/etc/shadow"}})
        expected["scope_mismatch"] = {"ok": False, "reason": "scope_mismatch"}

        _write("expired", grant=grant, attestation=att,
               runtime={**base_runtime, "now": IAT_EPOCH + 5000})
        expected["expired"] = {"ok": False, "reason": "expired"}

        _write("binding_unknown", grant=grant, attestation=other, runtime=base_runtime)
        expected["binding_unknown"] = {"ok": False, "reason": "binding_unknown"}

        (OUT / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(expected)} sets to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
