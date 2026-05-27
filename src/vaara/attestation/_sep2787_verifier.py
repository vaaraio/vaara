"""SEP-2787 verifier step 5: argument commitment verification.

Internal module. Public surface is in ``vaara.attestation.sep2787``.

Implements step 5 of the verification rules in the SEP-2787 draft:

    If the toolCalls entry uses argsRef, resolve the URI, compute
    SHA-256 over the fetched content, and compare against the stored
    digest. Confirm the resolved content corresponds to the arguments
    being executed. If the entry uses argsProjection, recompute the
    projection digest from the projection bytes and compare against
    the stored projectionDigest. If the projection is an identity
    projection (the canonical runtime arguments themselves, or a
    hash-only-identity projection ``{"digest": "sha256:..."}`` whose
    embedded digest matches sha256(JCS(runtime arguments))), confirm
    the binding. Redacted projections are verified only to be signed;
    the verifier makes no completeness claim. Reject hash-only-identity
    projections whose embedded digest does not match the runtime args
    with args_commitment_mismatch.

Step 5 is composed by the caller after steps 1-4 (signature, nonce
replay, TTL, tool call match). It does not perform network IO.
ArgsRef resolution is delegated to a caller-supplied resolver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation._sep2787_types import (
    ArgsCommitment,
    ArgsProjection,
    ArgsRef,
)

ARGS_COMMITMENT_MISMATCH: Literal["args_commitment_mismatch"] = (
    "args_commitment_mismatch"
)


@dataclass(frozen=True)
class ArgsCommitmentResult:
    """Outcome of step 5 (argument commitment verification).

    ``ok`` is True iff the commitment binds the runtime arguments
    under the rules for the specific commitment kind.

    ``reason`` is None on success, or ``"args_commitment_mismatch"``
    on failure -- matching the spec's error-reason enum.

    ``projection_match`` is meaningful only for ArgsProjection: True
    when the projection is an identity projection of the runtime
    arguments (either the canonical arguments themselves, or a
    hash-only-identity projection whose embedded digest matches);
    False for redacted / transformed projections (signed-only,
    verifier makes no completeness claim); None for non-projection
    commitments.
    """

    ok: bool
    reason: Optional[Literal["args_commitment_mismatch"]] = None
    projection_match: Optional[bool] = None


def _sha256_hex(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _parse_hash_only_identity(projection_str: str) -> Optional[str]:
    """If projection_str is a hash-only-identity carrier, return its digest.

    A hash-only-identity projection is the JCS-canonical encoding of
    a single-key object ``{"digest": "sha256:<hex>"}``. The verifier
    treats this as an identity projection of a hash-only object: it
    binds the underlying arguments' digest without revealing the
    payload. Returns the embedded ``sha256:<hex>`` digest string, or
    None if the projection has any other shape.
    """
    try:
        obj = json.loads(projection_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"digest"}:
        return None
    digest = obj["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None
    return digest


def verify_args_commitment(
    args: ArgsCommitment,
    *,
    runtime_arguments: Any,
    ref_resolver: Optional[Callable[[str], bytes]] = None,
) -> ArgsCommitmentResult:
    """Verify the args commitment against the runtime arguments.

    ``runtime_arguments`` is the JSON-serialisable arguments object
    the server is about to execute against (i.e. the ``arguments``
    field of the incoming ``tools/call`` request).

    ``ref_resolver`` is a callable taking the ArgsRef URI and
    returning the resolved content as bytes. Required for ArgsRef
    commitments, ignored otherwise. The verifier intentionally
    does not perform network IO. The deployment chooses resolver
    policy (allowed schemes, timeouts, caching, trust).
    """
    if isinstance(args, ArgsRef):
        if ref_resolver is None:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        try:
            content = ref_resolver(args.ref)
        except Exception:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        if not isinstance(content, (bytes, bytearray)):
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        if _sha256_hex(bytes(content)) != args.digest:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        runtime_canonical = canonical_json(runtime_arguments)
        if bytes(content) != runtime_canonical:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        return ArgsCommitmentResult(ok=True)

    if isinstance(args, ArgsProjection):
        projection_bytes = args.projection.encode("utf-8")
        if _sha256_hex(projection_bytes) != args.projection_digest:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        runtime_canonical = canonical_json(runtime_arguments)
        hash_only = _parse_hash_only_identity(args.projection)
        if hash_only is not None:
            if hash_only != _sha256_hex(runtime_canonical):
                return ArgsCommitmentResult(
                    ok=False, reason=ARGS_COMMITMENT_MISMATCH,
                )
            return ArgsCommitmentResult(ok=True, projection_match=True)
        identity = projection_bytes == runtime_canonical
        return ArgsCommitmentResult(ok=True, projection_match=identity)

    return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
