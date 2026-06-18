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

from vaara.attestation._sep2787_types import AttestationError
from vaara.attestation.receipt import attestation_digest
from vaara.attestation.sep2787 import parse_attestation
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
    ) -> None:
        self._vm = verifying_material
        self._receipts_dir = Path(receipts_dir)
        self._expected_tenant = expected_tenant
        self._rev = revocation
        self._clock_skew_seconds = clock_skew_seconds

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

    def authorize(
        self,
        params: Optional[dict[str, Any]],
        *,
        tool_name: str,
        arguments: Any,
    ) -> GrantVerdict:
        """Return the verdict for a ``tools/call``; fail closed on any defect."""
        meta = (params or {}).get("_meta") or {}
        cred_dict = meta.get("vaara/credential")
        if cred_dict is None:
            return GrantVerdict(False, "missing_credential")
        try:
            cred = grant_from_dict(cred_dict)
        except AttestationError:
            return GrantVerdict(False, "malformed")
        return verify_grant(
            cred,
            verifying_material=self._vm,
            runtime_tool_name=tool_name,
            runtime_args=arguments,
            runtime_tenant_id=self._expected_tenant or "",
            revocation=self._rev,
            known_attestation_digests=self._load_known_digests(),
            clock_skew_seconds=self._clock_skew_seconds,
        )
