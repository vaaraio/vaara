"""Closed-schema parsers for brokered-credential wire objects.

Internal module. Public surface is in ``vaara.credential``.

Each ``*_from_dict`` validates one signed block against a closed key set and
fails closed on any unrecognized field, mirroring
``vaara.attestation._receipt_types``. Signature verification is separate
(``_grant_emit.verify_grant_signature``); these only model-validate.
"""

from __future__ import annotations

from typing import Any

from vaara.attestation._sep2787_types import VALID_ALGS, AttestationError
from vaara.credential._grant_capability import Capability, capability_from_dict
from vaara.credential._grant_types import (
    ASSERTED_KEYS,
    BINDING_KEYS,
    GRANT_KEYS,
    SCOPE_KEYS,
    BrokeredCredential,
    GrantAsserted,
    GrantBinding,
    GrantScope,
)


def _reject_unknown_keys(
    d: dict[str, Any], allowed: frozenset[str], where: str
) -> None:
    """Fail closed on any key not in the closed schema for a signed block.

    A silently dropped key would let a model-deriving verifier and a
    byte-exact verifier disagree over the signed preimage. Extending the
    schema is an explicit version bump, not a tolerated field.
    """
    extra = set(d) - allowed
    if extra:
        raise AttestationError(
            f"{where} carries unrecognized field(s) {sorted(extra)!r}; "
            "the signed schema is closed"
        )


def _require_str(d: dict[str, Any], key: str, where: str) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value:
        raise AttestationError(f"{where}.{key} must be a non-empty string")
    return value


def scope_from_dict(d: dict[str, Any]) -> GrantScope:
    if not isinstance(d, dict):
        raise AttestationError("scope must be an object")
    _reject_unknown_keys(d, SCOPE_KEYS, "scope")
    tenant = d.get("tenantId")
    if not isinstance(tenant, str):
        raise AttestationError("scope.tenantId must be a string")
    return GrantScope(
        tool_name=_require_str(d, "toolName", "scope"),
        args_commitment=_require_str(d, "argsCommitment", "scope"),
        tenant_id=tenant,
    )


def binding_from_dict(d: dict[str, Any]) -> GrantBinding:
    if not isinstance(d, dict):
        raise AttestationError("binding must be an object")
    _reject_unknown_keys(d, BINDING_KEYS, "binding")
    digest = _require_str(d, "attestationDigest", "binding")
    if not digest.startswith("sha256:"):
        raise AttestationError("binding.attestationDigest MUST be 'sha256:'")
    return GrantBinding(
        attestation_digest=digest,
        attestation_nonce=_require_str(d, "attestationNonce", "binding"),
    )


def asserted_from_dict(d: dict[str, Any]) -> GrantAsserted:
    if not isinstance(d, dict):
        raise AttestationError("asserted must be an object")
    _reject_unknown_keys(d, ASSERTED_KEYS, "asserted")
    exp = d.get("expSeconds")
    if not isinstance(exp, int) or isinstance(exp, bool) or exp <= 0:
        raise AttestationError("asserted.expSeconds must be a positive integer")
    return GrantAsserted(
        iss=_require_str(d, "iss", "asserted"),
        sub=_require_str(d, "sub", "asserted"),
        iat=_require_str(d, "iat", "asserted"),
        exp_seconds=exp,
        nonce=_require_str(d, "nonce", "asserted"),
        secret_version=_require_str(d, "secretVersion", "asserted"),
    )


def grant_from_dict(d: dict[str, Any]) -> BrokeredCredential:
    """Parse and validate a credential wire object (signature unchecked here)."""
    if not isinstance(d, dict):
        raise AttestationError("credential must be an object")
    _reject_unknown_keys(d, GRANT_KEYS, "credential")
    for req in ("version", "alg", "scope", "binding", "asserted", "signature"):
        if req not in d:
            raise AttestationError(f"credential missing required field {req!r}")
    if not isinstance(d["version"], int) or isinstance(d["version"], bool):
        raise AttestationError("credential.version must be an integer")
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    caps_raw = d.get("capabilities")
    capabilities: tuple[Capability, ...] = ()
    if caps_raw is not None:
        if not isinstance(caps_raw, list) or not caps_raw:
            raise AttestationError("credential.capabilities must be a non-empty list")
        capabilities = tuple(capability_from_dict(c) for c in caps_raw)
    return BrokeredCredential(
        version=d["version"],
        alg=d["alg"],
        scope=scope_from_dict(d["scope"]),
        binding=binding_from_dict(d["binding"]),
        asserted=asserted_from_dict(d["asserted"]),
        signature=_require_str(d, "signature", "credential"),
        capabilities=capabilities,
    )
