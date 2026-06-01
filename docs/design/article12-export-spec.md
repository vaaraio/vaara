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
  [--format md|html]
```

`--system-meta` is a small operator-supplied JSON (system name, provider,
intended purpose, deployer, AI Act risk classification) that the report's
cover section needs and the trail does not carry. Optional; absent fields
render as "not provided" rather than failing.

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
```

The report and summary are inside the signed set: the manifest's
`trail_sha256` covers `trail.jsonl`; the report is generated *from* that
trail, and `article12_summary.json` records the trail sha256 it was built
over, so a regulator can confirm the report describes the signed trail and
not a different one.

## Report contents (`article12_report.md`)

Generated, deterministic, from the trail + system-meta:

1. **Cover** — system identity from `--system-meta`, export timestamp,
   Vaara version, signer fingerprint, reporting period.
2. **Record-keeping summary** — total records, time span, chain-intact
   status, retention configuration if present.
3. **Article 12 obligation mapping** — a table: each Article 12 / 26(5)
   obligation against how the trail evidences it (event types present,
   counts, example record ids). Driven by `regulatory_articles` tags plus
   a static obligation checklist.
4. **Event inventory** — event-type histogram, per-type counts, first/last
   timestamps.
5. **Integrity statement** — the hash-chain + (if present) external time
   anchor status, with the trust-model one-liner: a valid signature proves
   integrity and provenance of these logs, not the truth of every recorded
   assertion. Links to `docs/signing-keys.md` and the trust model.
6. **How to verify** — the exact `vaara trail verify` /
   `scripts/verify_vaara_trail.py` invocation, mirrored in
   `verify_instructions.txt` so a regulator with no Vaara install can act.

The prose passes the strict anti-AI-tells pass before any of it is treated
as outbound (it is regulator-facing).

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
