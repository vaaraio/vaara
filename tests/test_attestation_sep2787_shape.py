"""SEP-2787 v2 envelope shape and validation tests.

Covers wire-format invariants (toolCalls under payloadDerived, no
``kind`` field, JSON-stringified projection) and emit-side input
validation (empty intent, empty tool calls, unsupported alg, non-bytes
HS256 secret).
"""

from __future__ import annotations

import importlib.util
import json

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from vaara.attestation.tool_call_attestation import (  # noqa: E402
    ArgsRef,
    Attestation,
    AttestationError,
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
    make_args_projection,
)

HS_SECRET = b"\x42" * 32


def _planner() -> PlannerDeclared:
    return PlannerDeclared(
        intent="archive obsolete report per Q4 retention policy",
        requested_capability="filesystem.delete",
    )


def _payload(args=None) -> PayloadDerived:
    args = args or make_args_digest({"path": "/archive/2024-Q3.md"})
    return PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=args,
    ),))


def _emit(**overrides) -> Attestation:
    kwargs = dict(
        planner_declared=_planner(),
        payload_derived=_payload(),
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_attestation(**kwargs)


def test_make_args_digest_is_deterministic():
    one = make_args_digest({"x": 1, "y": [1, 2]})
    two = make_args_digest({"y": [1, 2], "x": 1})
    assert one.projection == two.projection
    assert one.projection_digest == two.projection_digest


def test_make_args_digest_is_hash_only_identity():
    parsed = json.loads(make_args_digest({"path": "/archive/2024-Q3.md"}).projection)
    assert set(parsed) == {"digest"}
    assert parsed["digest"].startswith("sha256:")


def test_emit_rejects_empty_intent():
    with pytest.raises(AttestationError, match="intent"):
        _emit(planner_declared=PlannerDeclared(intent="   "))


def test_emit_rejects_empty_tool_calls():
    with pytest.raises(AttestationError, match="toolCalls"):
        _emit(payload_derived=PayloadDerived(tool_calls=()))


def test_emit_rejects_unsupported_alg():
    with pytest.raises(AttestationError, match="unsupported alg"):
        _emit(alg="HS512")


def test_hs256_rejects_non_bytes_secret():
    with pytest.raises(AttestationError, match="bytes shared_secret"):
        _emit(signing_material="not-bytes")


def test_to_dict_top_level_shape():
    d = _emit().to_dict()
    assert set(d) == {
        "version", "alg", "plannerDeclared",
        "issuerAsserted", "payloadDerived", "signature",
    }
    assert d["plannerDeclared"]["intent"]
    assert d["plannerDeclared"]["requestedCapability"] == "filesystem.delete"
    assert "toolCalls" not in d["plannerDeclared"]
    assert d["issuerAsserted"]["expSeconds"] == 300
    assert d["issuerAsserted"]["secretVersion"] == "v1"


def test_to_dict_tool_calls_under_payload_derived():
    d = _emit().to_dict()
    tc = d["payloadDerived"]["toolCalls"][0]
    assert tc["name"] == "delete_file"
    assert tc["serverFingerprint"].startswith("sha256:")
    assert set(tc["args"]) == {"projection", "projectionDigest"}
    assert "kind" not in tc["args"]


def test_to_dict_ref_has_no_kind():
    args = ArgsRef(ref="ipfs://Qm...", digest="sha256:" + "0" * 64)
    a = _emit(payload_derived=_payload(args=args)).to_dict()
    arg_dict = a["payloadDerived"]["toolCalls"][0]["args"]
    assert set(arg_dict) == {"ref", "digest", "canonicalization"}
    assert "kind" not in arg_dict


def test_to_dict_projection_is_json_stringified():
    args = make_args_projection({"redacted_user_id": "u-001"})
    d = _emit(payload_derived=_payload(args=args)).to_dict()
    field = d["payloadDerived"]["toolCalls"][0]["args"]["projection"]
    assert isinstance(field, str)
    assert json.loads(field) == {"redacted_user_id": "u-001"}
