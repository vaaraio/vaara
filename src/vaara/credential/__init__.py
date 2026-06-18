"""Receipt-bound, scoped, short-lived credential broker (OAuth-for-agents).

Vaara's authority layer: the proxy mints a signed, short-lived credential
bound to a specific attestation digest (transitively the mediation receipt)
and scoped to one tool + args-commitment + tenant. A gateway in front of a
protected tool refuses any call lacking a valid, unexpired, non-revoked,
attestation-bound grant. This turns bypass from "silent and succeeds" into
"requires defeating the broker"; it is detection, not a mathematical
completeness claim (see ``docs/design/credential-broker-spec.md``).

The grant reuses the SEP-2787 signing stack (HS256 / ES256 / RS256 over RFC
8785 JCS) so the grant key matches the attestation/receipt key.
"""

from __future__ import annotations

from vaara.credential._grant_emit import emit_grant, verify_grant_signature
from vaara.credential._grant_parse import (
    asserted_from_dict,
    binding_from_dict,
    grant_from_dict,
    scope_from_dict,
)
from vaara.credential._grant_types import (
    BrokeredCredential,
    GrantAlgorithm,
    GrantAsserted,
    GrantBinding,
    GrantScope,
)
from vaara.credential._grant_verify import GrantVerdict, verify_grant
from vaara.credential.gateway import CredentialGateway

__all__ = [
    "BrokeredCredential",
    "CredentialGateway",
    "GrantAlgorithm",
    "GrantAsserted",
    "GrantBinding",
    "GrantScope",
    "GrantVerdict",
    "asserted_from_dict",
    "binding_from_dict",
    "emit_grant",
    "grant_from_dict",
    "scope_from_dict",
    "verify_grant",
    "verify_grant_signature",
]
