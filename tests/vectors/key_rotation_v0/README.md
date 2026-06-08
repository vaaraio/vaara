# key_rotation_v0 conformance vectors

Verify a retained record under a key that was later rotated out. See
`docs/design/key-rotation-retention-spec.md`.

EU AI Act Article 12 records must stay verifiable across a retention window
measured in years, but signing keys rotate inside that window. A record signed
in 2026 by `#key-2026` is audited years later, after the issuer rotated to a
new key and retired the old one. The record must still verify, against the DID
document the regulator archived, under the key that was valid when the record
was signed.

A record is **verifiable** when three things hold offline:

- the ES256 signature binds to a key the archived document lists (level-2),
- the claimed `iat` is inside that key's validity window
  `[not_before, not_after)`, read from the method's `validFrom` / `validUntil`
  (or `notBefore` / `notAfter`) markers, and
- the key was not revoked at or before `iat` (revocation overrides a graceful
  retirement).

A record is **corroborated** when, in addition, a verified time anchor proves
it existed before the key's retirement and any revocation, so the in-window
claim cannot be a later forgery made with a stolen retired key. Without an
anchor the verdict rests on the record's self-asserted clock, and `time_basis`
says so.

## Files

- `cases.json`: one shared ES256 receipt (`did:web` issuer, signed by the key
  that is retired later), and for each case the archived DID document and an
  optional attested `anchoredTime`.
- `expected.json`: per case, the verdict fields a conforming verifier must
  produce. The non-normative `reason` string is not compared.

## Cases

- `retired_key_no_anchor`: the key is valid at `iat`, retired later, no anchor.
  Verifiable on the self-asserted time, not corroborated.
- `retired_key_anchored`: the same, with an anchor predating retirement.
  Verifiable and corroborated.
- `signed_after_retirement`: `validUntil` is before `iat` (the leaked-retired-
  key forgery shape). Bound but out of window, not verifiable.
- `signed_before_activation`: `validFrom` is after `iat`. Out of window.
- `revoked_before_issuance`: in window, but the key is revoked before `iat`.
  Not verifiable: revocation overrides retirement.
- `anchor_after_retirement`: valid at `iat`, but the anchor was taken after
  retirement. Verifiable, not corroborated (a late anchor does not weaken the
  verdict, it just does not strengthen it).
- `unbounded_key`: the document records no window markers. Verifiable and
  unbounded (`window_recorded` is false): existing documents work unchanged.
- `wrong_key`: the document lists only a different key. Not bound.

## Reproduce

Independent checker (standard library plus `cryptography` and `rfc8785`, no
Vaara import):

```
python tests/vectors/key_rotation_v0/_check_independent.py
```

Exit code 0 means every case matched its expected verdict. Regenerate the
cases (ECDSA signatures are randomized, so signatures change but verdicts do
not) with:

```
python tests/vectors/key_rotation_v0/_generate.py
```
