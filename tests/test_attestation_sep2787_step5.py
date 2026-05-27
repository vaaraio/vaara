"""SEP-2787 verifier step 5 tests (argument commitment verification)."""

from __future__ import annotations

import hashlib
import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from vaara.attestation.sep2787 import (  # noqa: E402
    ArgsProjection,
    ArgsRef,
    canonical_json,
    make_args_digest,
    make_args_projection,
    verify_args_commitment,
)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def test_hash_only_matching_runtime_ok():
    runtime = {"path": "/archive/2024-Q3.md"}
    result = verify_args_commitment(make_args_digest(runtime), runtime_arguments=runtime)
    assert result.ok is True
    assert result.reason is None
    assert result.projection_match is True


def test_hash_only_mismatch_rejects():
    args = make_args_digest({"path": "/archive/2024-Q3.md"})
    result = verify_args_commitment(args, runtime_arguments={"path": "/etc/passwd"})
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_hash_only_key_reorder_still_matches():
    args = make_args_digest({"a": 1, "b": [1, 2, 3]})
    result = verify_args_commitment(args, runtime_arguments={"b": [1, 2, 3], "a": 1})
    assert result.ok is True
    assert result.projection_match is True


def test_ref_no_resolver_rejects():
    args = ArgsRef(ref="ipfs://Qm...", digest="sha256:" + "0" * 64)
    result = verify_args_commitment(args, runtime_arguments={"x": 1})
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_ref_resolver_content_matches():
    runtime = {"path": "/archive/2024-Q3.md"}
    canonical = canonical_json(runtime)
    args = ArgsRef(ref="memory://q3", digest=_sha256_hex(canonical))
    result = verify_args_commitment(
        args, runtime_arguments=runtime, ref_resolver=lambda _ref: canonical,
    )
    assert result.ok is True


def test_ref_digest_mismatch_rejects():
    runtime = {"path": "/archive/2024-Q3.md"}
    args = ArgsRef(ref="memory://q3", digest="sha256:" + "0" * 64)
    result = verify_args_commitment(
        args, runtime_arguments=runtime,
        ref_resolver=lambda _ref: canonical_json(runtime),
    )
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_ref_content_does_not_match_runtime_rejects():
    referenced = {"path": "/archive/2024-Q3.md"}
    runtime = {"path": "/etc/passwd"}
    canonical = canonical_json(referenced)
    args = ArgsRef(ref="memory://other", digest=_sha256_hex(canonical))
    result = verify_args_commitment(
        args, runtime_arguments=runtime, ref_resolver=lambda _ref: canonical,
    )
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_ref_resolver_raising_rejects():
    args = ArgsRef(ref="memory://oops", digest="sha256:" + "0" * 64)
    def _broken(_ref):
        raise RuntimeError("offline")
    result = verify_args_commitment(args, runtime_arguments={"x": 1}, ref_resolver=_broken)
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_identity_projection_marked_match():
    runtime = {"path": "/archive/2024-Q3.md"}
    result = verify_args_commitment(make_args_projection(runtime), runtime_arguments=runtime)
    assert result.ok is True
    assert result.projection_match is True


def test_redacted_projection_ok_but_not_identity():
    args = make_args_projection({"redacted_user_id": "u-001"})
    result = verify_args_commitment(
        args,
        runtime_arguments={"path": "/archive/2024-Q3.md", "user_id": "u-001"},
    )
    assert result.ok is True
    assert result.projection_match is False


def test_projection_with_tampered_digest_rejects():
    payload = canonical_json({"redacted_user_id": "u-001"})
    args = ArgsProjection(
        projection=payload.decode("utf-8"),
        projection_digest="sha256:" + "0" * 64,
    )
    result = verify_args_commitment(args, runtime_arguments={"x": 1})
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"
