"""SEP-2787 verifier step 5: argument commitment verification.

Internal module. Public surface is in ``vaara.attestation.sep2787``.

Implements step 5 of the verification rules in the SEP-2787 draft:

    If the toolCalls entry uses argsRef, resolve the URI, compute
    SHA-256 over the fetched content, and compare against the stored
    digest. Confirm the resolved content corresponds to the arguments
    being executed. If the entry uses argsProjection, compare it
    against the canonicalized runtime arguments (RFC 8785). Identity
    projections MUST match exactly; redacted projections are verified
    only to be signed -- the verifier makes no claim about
    completeness relative to the runtime arguments. If neither field
    is present, or if the content cannot be resolved and matched,
    reject with args_commitment_mismatch.

Vaara's three-way args shape (ArgsDigest / ArgsRef / ArgsProjection)
extends the spec's two-way (argsRef / argsProjection) with a
commitment-only ArgsDigest where the payload never crosses the
verifier. For ArgsDigest the verifier recomputes the JCS-canonical
hash of the runtime arguments and compares against the bound digest.

Step 5 is composed by the caller after steps 1-4 (signature, nonce
replay, TTL, tool call match). It does not perform network IO.
ArgsRef resolution is delegated to a caller-supplied resolver.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation._sep2787_types import (
    ArgsCommitment,
    ArgsDigest,
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
    when the projection equals the canonicalized runtime arguments
    (identity projection per spec), False when it differs (redacted
    projection -- verified only to be signed), None otherwise.
    """

    ok: bool
    reason: Optional[Literal["args_commitment_mismatch"]] = None
    projection_match: Optional[bool] = None


def _sha256_hex(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


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
    if isinstance(args, ArgsDigest):
        observed = _sha256_hex(canonical_json(runtime_arguments))
        if observed != args.digest:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        return ArgsCommitmentResult(ok=True)

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
        recomputed = _sha256_hex(canonical_json(args.projection))
        if recomputed != args.projection_digest:
            return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
        runtime_canonical = canonical_json(runtime_arguments)
        projection_canonical = canonical_json(args.projection)
        identity = runtime_canonical == projection_canonical
        return ArgsCommitmentResult(ok=True, projection_match=identity)

    return ArgsCommitmentResult(ok=False, reason=ARGS_COMMITMENT_MISMATCH)
