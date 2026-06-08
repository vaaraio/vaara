# Design spec: verify a retained record under a rotated or retired key

Status: draft for v0.64. Companion to
`docs/design/resolvable-agent-identity-spec.md` (binding) and
`docs/design/cross-stack-revocation-spec.md` (revocation in time). Builds on
the eIDAS RFC 3161 time anchor in `src/vaara/audit/timeanchor.py`.

## The problem

EU AI Act Article 12 logs must stay verifiable across a retention window
measured in years. Signing keys do not last that long: an operator rotates
keys on a schedule, and a key that signed a record in 2026 is retired well
before a 2031 audit. By audit time the issuer's live DID document lists the
current key, not the one that signed the record. Two failure modes follow,
and both are wrong:

- The live document no longer lists the old key, so the signature binds to
  nothing and a naive identity check reports "not signed by this identity".
- The document keeps the old key but marks it end-of-life, and a check that
  treats any end-of-life marker as disqualifying rejects a signature that was
  perfectly valid when it was made.

A signature a key made while the key was valid stays valid forever. The
record does not become unverifiable because the signer later changed keys, any
more than a paper contract becomes void when the signatory later changes pens.
The verifier needs the key's *history*, not just its current state.

Revocation (`cross-stack-revocation-spec.md`) is the adjacent but distinct
case: a revoked key is retroactively distrusted from the revocation instant,
because revocation means compromise. Retirement is graceful end-of-life:
signatures made before it stand. This spec adds retirement; it composes the
existing revocation rule unchanged.

## Data model: a key validity window

Each key gains an optional validity window, expressed on the DID document's
verification method:

- `validFrom` (alias `notBefore`): the instant the key became valid for
  signing. Absent means unbounded below.
- `validUntil` (alias `notAfter`): the instant the key stopped being valid for
  new signatures (retirement). Absent means still active, unbounded above.

`validFrom` / `validUntil` is the W3C Verifiable Credentials Data Model 2.0
spelling and is canonical; `notBefore` / `notAfter` is accepted as an alias for
documents that borrow the X.509 vocabulary. A key with neither marker is
unbounded, so every existing document keeps verifying unchanged: this is a
purely additive convention.

The window is half-open: a signature is in window iff it was made at or after
`validFrom` and strictly before `validUntil`. The lifecycle-change instant
belongs to the state that follows it (the key is retired *at* `validUntil`),
the same convention the revocation rule uses, where a revocation at the
issuance instant binds.

A `KeyHistory` collects these windows. It is source-agnostic, mirroring
`RevocationRegistry`: built from a DID document's per-method markers, from an
operator's out-of-band key-history list, or directly. It carries no key
material, so it runs in the base install, and it exposes a canonical
`digest()` so a signed export can pin the exact windows the verifier used.

## What the verifier archives

The verifier checks against the DID document it *archived* at or near record
time, not the live document. The archived document retains the retired keys,
marked with their windows, so the rotation judgment is reproducible offline
years later. This is the regulator-grade discipline: keep the key history for
the length of the retention window. Resolution stays offline, the same
property level-2 pinned-resolvable identity already has.

## The two-tier verdict

`verify_receipt_retained` settles two questions.

**Verifiable** (the offline verdict) holds when all three are true:

1. the signature binds to a key the archived document lists (level-2,
   offline);
2. the record's claimed `iat` falls inside that key's validity window; and
3. the key was not revoked at or before `iat`.

The claimed `iat` answers "does the record claim a time when the key was
valid?". But `iat` is self-asserted. An attacker who later steals a *retired*
key could forge a record and backdate `iat` into the old window, and the
verifiable check alone cannot tell that forgery from a genuine old record.

**Corroborated** (the stronger tier) closes that gap with the eIDAS RFC 3161
time anchor. The anchor is a trusted timestamp over the record, proving it
existed no later than the attested time, signed by an authority outside the
signer's trust boundary. When the anchor predates the key's retirement and any
revocation, the in-window claim cannot be a later forgery: a stolen retired key
cannot produce a timestamp token dated before it was retired. A record is
corroborated when it is verifiable and a verified anchor lands before the key's
end of life.

`time_basis` names which the verdict rests on: `anchored` when an anchor was
supplied, `self_asserted` otherwise. A genuine old record carries an anchor
from near its signing time and is corroborated; the backdated forgery has no
such anchor and is at most verifiable-on-self-asserted-time, which the verdict
states plainly. A late anchor (taken after retirement) does not weaken the
verdict; it simply does not strengthen it.

## Why not just trust the current document

Trusting the live document at audit time fails closed in the wrong direction:
it rejects genuine old records whenever a key has rotated, which over a 7-year
window is every key. Re-resolving the historical document is also not
generally possible, since did:web serves only the current state. Archiving the
document and recording key windows is the only construction that stays both
offline and correct across the window.

## Scope and non-goals

- No new envelope field and no canonicalization change. The window lives on
  the DID document; existing receipts and every conformance vector verify
  exactly as before.
- The window check and revocation check are pure standard library. Binding
  needs the `cryptography` of the attestation extra; anchor verification needs
  the `timeanchor` extra. A verifier without the anchor extra passes an
  `anchored_time` it verified separately.
- Fail-closed throughout: an unparseable window bound, issuance instant, or
  anchor time yields "not in window" / "not corroborated", never the benefit of
  the doubt.

## Conformance vectors

`tests/vectors/key_rotation_v0/` carries the cases (retired-but-valid,
signed-after-retirement, signed-before-activation, revoked, late anchor,
unbounded, wrong key) and a Vaara-free checker that reproduces every verdict
with only `cryptography` and `rfc8785`.
