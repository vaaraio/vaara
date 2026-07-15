"""Reference gateway shim: refuse a tool call lacking a valid grant.

Public surface for ``vaara.credential``.

A ``CredentialGateway`` sits in front of a protected tool and answers one
question: may this ``tools/call`` execute? It reads the brokered credential
from ``params._meta["vaara/credential"]``, parses it under the closed schema,
and runs ``verify_grant`` against the runtime tool name, arguments, and tenant,
plus the set of attestation digests Vaara actually minted (read from the
proxy's receipts directory).

Honest scope: completeness holds only for tools placed behind this gateway. A
tool reachable by some other path (operator root, an alternate credential
source, a purely-local action with no chokepoint) is not covered here; that
residual is caught after the fact by reconciliation (credential fingerprint
joined to receipt digest), described in ``docs/design/credential-broker-spec.md``.
A stripped ``_meta`` degrades to a ``missing_credential`` refusal, which is
fail-closed and acceptable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from vaara.attestation._attest_types import AttestationError
from vaara.attestation.receipt import attestation_digest
from vaara.attestation.tool_call_attestation import parse_attestation
from vaara.credential._authorization_receipt import (
    AuthorizationReceipt,
    ReceiptSigner,
    mint_for_signer,
)
from vaara.credential._grant_parse import grant_from_dict
from vaara.credential._grant_verify import GrantVerdict, verify_grant


class CredentialGateway:
    """Authorize gateway-protected tool calls against brokered credentials."""

    def __init__(
        self,
        *,
        verifying_material: Any,
        receipts_dir: Path,
        expected_tenant: Optional[str] = None,
        revocation: Any = None,
        clock_skew_seconds: int = 30,
        signer: Optional[ReceiptSigner] = None,
    ) -> None:
        self._vm = verifying_material
        self._receipts_dir = Path(receipts_dir)
        self._expected_tenant = expected_tenant
        self._rev = revocation
        self._clock_skew_seconds = clock_skew_seconds
        # Opt-in: when a signer is supplied, authorize_and_receipt mints a signed
        # proof of every grant-bound decision. Absent it, the gateway only
        # verifies and the authority layer stays observation-free (off by default).
        self._signer = signer

    def _load_known_digests(self) -> frozenset[str]:
        """Recompute the attestation digest of every ``*-attest.json`` on disk.

        The attestation is written before the upstream forward, so by the time
        a credential reaches a protected tool its digest is present. Each
        digest equals the grant's ``binding.attestationDigest`` and the
        receipt's ``backLink.attestationDigest``. A file that fails to parse is
        skipped: it cannot match a digest, so skipping it only ever refuses,
        never wrongly admits.
        """
        digests: set[str] = set()
        if not self._receipts_dir.is_dir():
            return frozenset()
        for path in self._receipts_dir.glob("*-attest.json"):
            try:
                att = parse_attestation(json.loads(path.read_text(encoding="utf-8")))
                digests.add(attestation_digest(att))
            except (OSError, ValueError, AttestationError):
                continue
        return frozenset(digests)

    def _decide(
        self,
        params: Optional[dict[str, Any]],
        *,
        tool_name: str,
        arguments: Any,
    ) -> tuple[GrantVerdict, Optional[Any]]:
        """Parse and verify; return ``(verdict, credential)``.

        ``credential`` is None when no grant could be parsed at all (missing or
        malformed): there is nothing to bind a receipt to, so the refusal stands
        without one.
        """
        meta = (params or {}).get("_meta") or {}
        cred_dict = meta.get("vaara/credential")
        if cred_dict is None:
            return GrantVerdict(False, "missing_credential"), None
        try:
            cred = grant_from_dict(cred_dict)
        except AttestationError:
            return GrantVerdict(False, "malformed"), None
        verdict = verify_grant(
            cred,
            verifying_material=self._vm,
            runtime_tool_name=tool_name,
            runtime_args=arguments,
            runtime_tenant_id=self._expected_tenant or "",
            revocation=self._rev,
            known_attestation_digests=self._load_known_digests(),
            clock_skew_seconds=self._clock_skew_seconds,
        )
        return verdict, cred

    def authorize(
        self,
        params: Optional[dict[str, Any]],
        *,
        tool_name: str,
        arguments: Any,
    ) -> GrantVerdict:
        """Return the verdict for a ``tools/call``; fail closed on any defect."""
        verdict, _ = self._decide(params, tool_name=tool_name, arguments=arguments)
        return verdict

    def authorize_and_receipt(
        self,
        params: Optional[dict[str, Any]],
        *,
        tool_name: str,
        arguments: Any,
    ) -> tuple[GrantVerdict, Optional[AuthorizationReceipt]]:
        """Authorize and, when a signer is configured, mint a signed receipt.

        Allowed and refused calls both mint, so a denial leaves a portable proof
        of the non-action. The receipt is returned for the caller to persist; the
        gateway does not write it. A missing or malformed credential refuses with
        no receipt, since there is no grant to bind a proof to, and a gateway with
        no signer returns no receipt at all (off by default).
        """
        verdict, cred = self._decide(params, tool_name=tool_name, arguments=arguments)
        if self._signer is None or cred is None:
            return verdict, None
        auth = mint_for_signer(
            self._signer, credential=cred, runtime_args=arguments, verdict=verdict
        )
        return verdict, auth
