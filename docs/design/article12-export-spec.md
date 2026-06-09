# Design spec: Article-12 one-command regulator export

Status: draft for v0.5. Priority 2 of 3. Companion to
`docs/signing-keys.md` and the existing export surfaces.

## Goal

One command turns a live trail into a regulator-facing Article-12
record-keeping package: the signed evidence plus a human-readable report
that maps the trail to the AI Act Article 12 logging obligations. The
institutional-anchoring artifact (SFS / SR 315), not a new evidence format.

Today an operator hands a regulator `signed.zip` (trail + manifest +
signature) and has to explain, out of band, how it satisfies Article 12.
This makes the package self-explaining.

## What already exists (reuse, do not rebuild)

- `vaara trail export` -> `export_signed()`: signed zip (trail.jsonl,
  manifest.json, trail.sig, pubkey).
- `vaara trail export-incident` -> `incident_export.py`: Article 73
  serious-incident report from a trail.
- `vaara trail export-prov` -> `prov_export.py`: W3C PROV-JSON lineage.
- `vaara trail prune` (retention): Article 12(2) deletion.
- `AuditRecord.regulatory_articles`: per-record article tags already
  carried in the chain.

The Article-12 export composes these, it does not duplicate them.

## Scope of Article 12

Article 12 (record-keeping) requires high-risk AI systems to
automatically record events ("logs") over their lifetime, to a degree
appropriate to the intended purpose, enabling: traceability, post-market
monitoring, and the Article 26(5) deployer-side monitoring duty. The
export answers "show me your Article 12 logs and prove they are intact"
in one artifact.

## Command

```
vaara trail export-article12 \
  --trail trail.jsonl \
  --key signer.pem \
  --out art12_package.zip \
  [--system-meta system.json] \
  [--period 2026-01-01:2026-06-30] \
  [--format md|html] \
  [--anchor-tsa https://tsa.example/tsr | --anchor-file anchor.json]
```

`--system-meta` is a small operator-supplied JSON (system name, provider,
intended purpose, deployer, AI Act risk classification) that the report's
cover section needs and the trail does not carry. Optional; absent fields
render as "not provided" rather than failing.

`--anchor-tsa` / `--anchor-file` (added v0.59.0, mutually exclusive) fold an
external RFC 3161 time anchor over the signed trail head into the package as
Article 19 existence-in-time evidence. See "External time anchor" below.

## Package contents (zip)

Everything `export_signed` writes, plus:

```
trail.jsonl              # unchanged signed evidence
manifest.json            # unchanged
trail.sig                # unchanged
signer_pubkey.pem        # unchanged
article12_report.md      # NEW: the human-readable mapping (or .html)
article12_summary.json   # NEW: machine-readable version of the report
verify_instructions.txt  # NEW: how the regulator checks the package
time_anchor.json         # NEW (v0.59.0): RFC 3161 anchor over the trail head,
                         #   present only when --anchor-tsa/--anchor-file is used
```

The report and summary are inside the signed set: the manifest's
`trail_sha256` covers `trail.jsonl`; the report is generated *from* that
trail, and `article12_summary.json` records the trail sha256 it was built
over, so a regulator can confirm the report describes the signed trail and
not a different one.

## Report contents (`article12_report.md`)

Generated, deterministic, from the trail + system-meta:

1. **Cover**: system identity from `--system-meta`, export timestamp,
   Vaara version, signer fingerprint, reporting period.
2. **Record-keeping summary**: total records, time span, chain-intact
   status, retention configuration if present.
3. **Article 12 obligation mapping**, a table: each Article 12 / 26(5)
   obligation against how the trail evidences it (event types present,
   counts, example record ids). Driven by `regulatory_articles` tags plus
   a static obligation checklist.
4. **Event inventory**: event-type histogram, per-type counts, first/last
   timestamps.
5. **Integrity statement**: the hash-chain + (if present) external time
   anchor status, with the trust-model one-liner: a valid signature proves
   integrity and provenance of these logs, not the truth of every recorded
   assertion. Links to `docs/signing-keys.md` and the trust model.
6. **How to verify**: the exact `vaara trail verify` /
   `scripts/verify_vaara_trail.py` invocation, mirrored in
   `verify_instructions.txt` so a regulator with no Vaara install can act.

The prose passes the strict anti-AI-tells pass before any of it is treated
as outbound (it is regulator-facing).

## External time anchor (Article 19), v0.59.0

The signature proves the logs are intact and who signed them. It does not,
on its own, prove *when* they existed: the timestamps and the signature both
come from the operator's own key, so a later key compromise could backdate an
alternate trail. Article 19 requires the automatically generated logs to be
kept for the appropriate period; "kept since when" needs an anchor outside
the operator's trust boundary.

`--anchor-tsa URL` takes the signed trail head (the last record's
`record_hash`) to an RFC 3161 Time-Stamp Authority and folds the returned
token in as `time_anchor.json`. Pinned to an eIDAS-qualified TSA the timestamp
is recognised EU-wide, so existence-in-time holds independently of the signing
key. `--anchor-file` folds a pre-fetched anchor instead (for air-gapped
issuance). The anchor binds to *this* trail: `export_article12` re-checks that
the anchor's `chain_position` is the head and that the token verifies over that
exact `record_hash` before writing it, so a package never claims an anchor it
cannot back.

The regulator verifies it offline with `vaara trail verify-anchor --zip
<package>.zip`: the anchored `chain_head_hash` must equal the last
`record_hash` in `trail.jsonl`, and the RFC 3161 token must verify under a TSA
the regulator trusts. This reuses `vaara/audit/timeanchor.py`
(`verify_anchor_over_records`); the report's "External time anchor (Article
19)" section and `article12_summary.json["time_anchor"]` surface the attested
time. Versus a blockchain-anchored digest, an eIDAS-qualified RFC 3161
timestamp is the form an EU regulator already recognises.

## Implementation sketch

- New module `vaara/audit/article12_export.py`: `build_article12_report()`
  (records + system_meta -> report dict) and `render_report()`
  (dict -> md/html). Pure, testable without crypto.
- New `export_article12()` in or beside `audit/export.py`: builds the
  report, then calls the existing signed-zip writer with the two extra
  files folded in before signing so they are covered.
- New CLI `_cmd_trail_export_article12` wiring the above, mirroring
  `_cmd_trail_export_incident`.

## Threshold interaction

If threshold signing (spec 1) is the active signer, the Article-12 package
is threshold-signed transparently: the manifest carries the k-of-n fields,
the report's integrity section states "signed by k of n named custodians."
The two features compose; build threshold first.

## Test plan

- Round-trip: export a known trail, report record-count / event histogram /
  chain status match the trail.
- Tamper: mutate a record after export, `vaara trail verify` fails and the
  report's stated `trail_sha256` no longer matches.
- system-meta absent: renders "not provided", does not crash.
- Period filter: only records in range are summarized; out-of-range
  excluded from counts but the signed trail is still whole-trail (document
  this clearly so the period is a report lens, not an evidence filter).
- Article mapping: a trail with `regulatory_articles` tags produces the
  obligation table; an untagged trail still renders with "no explicit tag".

## Out of scope for v0.5

- PDF rendering (md/html only; operators convert if a regulator needs PDF).
- Auto-submission to any regulator intake portal.
- Per-article legal interpretation beyond the static obligation checklist
  (semantic-correctness layer; stays human-owned per the trust model).

## Folding SEP-2828 evidence as sidecars (the fold)

The package can carry verified SEP-2828 evidence beside the trail: cross-org
handoff packages (Article 26(6) deployer custody) and confidential-VM
enforcement bindings ("where it ran"). One deployer command then produces one
package coherent across Article 12 (the signed trail), Article 19 (the eIDAS
anchor), and Article 26(6) (the handoff records), with an optional
confidential-VM attestation alongside.

### The architectural seam

Article 12 export operates on the **audit trail** (the hash-chained
`AuditRecord` trail, the governance plane). Handoff and enforcement operate on
**SEP-2828 execution records** (the attestation plane). These are two planes.
The fold attaches SEP-2828 evidence as verified sidecars to the trail package;
it does not claim the trail records are SEP-2828 records, and it does not bring
the sidecars under the trail signature. The sidecars are content-addressed
SEP-2828 records, verified at export.

### Layout

```
evidence/handoff/<name>.json
evidence/enforcement/<name>.{record.json,report.bin,vcek.pem}
evidence/attestations_summary.json
```

`attestations_summary.json` carries the roll-up `check_handoff_set` /
`check_enforcement_set` produce, plus the verifier-side inputs used at export
(per-package anchor times, the trusted DID document, the expected measurement),
so the folded evidence re-verifies offline from the package alone. The report
gains a "Cross-org handoff and enforcement evidence" section mapping the counts
to Article 26(6) custody and the confidential-VM "where enforcement ran"
evidence, each with the honesty note verbatim.

### Verified at export, fail closed

Each attachment is verified before any bytes are written, in default mode, by
the same `check_handoff_set` / `check_enforcement_set` the verify verbs use. A
single attachment that does not verify raises and writes no package: v0 never
ships evidence it cannot back, and there is no `--skip-invalid`.

### Honesty model (the brand applies)

- The eIDAS time anchor stays the **only** un-forgeable component of the package.
- A handoff is `verifiable` when the signature binds to a listed key inside its
  window and not revoked; it is `corroborated` only with a verified anchor that
  predates retirement, and `pinned` only against a `--trusted-did-document`. It
  is never silently upgraded.
- An enforcement binding is never "attested" in v0: `vcek_chain_basis` stays
  `caller_supplied_unverified` and `enforcement_logic_basis` stays
  `not_established`. `--expected-measurement` pins a launch image; the
  `attested` tier and a strict pass are reserved for the future KDS-chained tier.

### Conformance

`article12_fold_v0` vectors fold named subsets of the `cross_org_handoff_v0` and
`enforcement_attestation_v0` corpora into a real package and pin the roll-up and
`evidence/` membership. The Vaara-free checker reproduces every folded verdict
from the same bytes folded into the zip (not a re-snapshot), composing the two
single-verb suites' own evaluators.

### CLI

```
vaara trail export-article12 --trail t.jsonl --key signer.pem --out pack.zip \
    --anchor-tsa https://freetsa.org/tsr \
    --handoff one.json --handoffs ./handoffs \
    --enforcements ./enforced \
    --trusted-did-document issuer-did.json \
    --expected-measurement <96-hex>
```

`--handoff` is repeatable for single packages; `--handoffs` globs `*.json` in a
directory; `--enforcements` discovers `NAME.record.json` + `NAME.report.bin` +
`NAME.vcek.pem` triples by stem. Folding needs the attestation extra.
