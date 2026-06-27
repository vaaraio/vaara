# Vaara eIDAS Qualified Electronic Ledger (QEL) Profile v0

This profile shapes a Vaara receipt chain into the evidence form an eIDAS 2.0
qualified electronic ledger (QEL) records, so that when a Qualified Trust Service
Provider (QTSP) operates the chain under its qualified status the records carry the
statutory presumptions of Regulation (EU) 2024/1183. It is a downstream profile: it
pins to `vaara.receipt/v1` and adds no new primitive. The envelope, capability,
credential, and governance-decision surfaces are unchanged.

## Status (read this first)

The QEL is a new trust service under eIDAS 2.0. The implementing act that sets the
reference standards for it (Article 45l(3)) was due 21 May 2025 and, as of June 2026,
is **not adopted** (the draft went through public consultation with feedback closing
5 March 2026; no Official Journal publication). Until it is adopted and a QTSP is
supervised and listed for the QEL service, no ledger can hold "qualified" status, and
no statutory presumption attaches.

This profile is therefore deliberately split:

- **Ready now.** The evidence schema and append-only ledger semantics over
  `vaara.receipt/v1`. A deployer can run this today as a non-qualified ledger: it is
  admissible (Art. 45k(1) bars denial of legal effect solely for electronic form) and
  technically immutable, but it carries no presumption.
- **Gated.** The "qualified" status itself. That is conferred on the QTSP by a
  supervisory body against the reference standards, not granted by this profile and
  not a property of any code. It attaches to the operating QTSP, not to Vaara.

The profile ships before the act adopts so that when the legal gate opens, the profile
is already defined and the substrate already conforms.

## Article 45k / 45l crosswalk against the shipped substrate

| eIDAS 2.0 requirement | What it demands | Where the substrate already meets it | Delta to "qualified" |
|---|---|---|---|
| **Art. 45k(2)** presumption of unique, accurate sequential chronological ordering | Records are ordered and the ordering is trustworthy | Append-only hash chain: each record SHA-256 chains to its predecessor (`record_hash`), so order is cryptographically fixed | The *presumption* is statutory, conferred by QTSP operation; the chain provides the technical ordering it presumes |
| **Art. 45k(2)** presumption of integrity | Records cannot be silently altered | Hash chain plus optional signed receipts; the qualified RFC 3161 timestamp over the chain head (`audit/timeanchor.py:314`, method `rfc3161-eidas-qualified` in `audit/receipt_anchor.py:186`) fixes existence-in-time | Statutory presumption attaches on qualified operation |
| **Art. 45l(d)** any subsequent change immediately detectable | Tamper-evidence | Recompute-from-bytes: any edit breaks the chain and the recomputed digest; verifiers import no producer code | Met technically; conformity assessment confirms it for the qualified listing |
| **Art. 45l(b)** establish the origin of data records | Records attributable to a source | Optional signed receipts bind origin; a qualified electronic seal over the export is the regulated origin leg | Add the qualified-seal leg (QTSP seal certificate) |
| **Art. 45l(a)** created and managed by one or more QTSPs | A QTSP operates the ledger | Operational, not cryptographic. The chain is built to be run by an operator who pins the qualified TSA and seal | QTSP key custody and supervised operation |

**Read of the delta.** It is not crypto. The substrate already delivers ordering,
integrity, change-detectability, and origin. What "qualified" adds is QTSP key custody,
a qualified seal over the export, and conformity assessment against the (pending)
reference standards. That is exactly the closed-enforcement half a QTSP partner
deploys, not a primitive Vaara is missing.

## What the profile adds over `vaara.receipt/v1`

1. A QEL evidence schema: the ledger record fields a QTSP needs to map a Vaara chain
   to the Art. 45l record model, expressed as a profile of the existing receipt, no new
   envelope.
2. Append-only ledger semantics already provided by the transparency-log machinery
   (`attestation/transparency_log.py`, RFC 6962 inclusion and consistency proofs); the
   profile names which proofs a QEL record carries.
3. The qualified-seal binding point: where the QTSP's qualified electronic seal signs
   the exported ledger segment to satisfy Art. 45l(b) origin.

## What a pass does and does not establish

A chain that conforms to this profile is an append-only, recompute-verifiable,
qualified-timestamp-anchored ledger. It is **not** a qualified electronic ledger until
a listed QTSP operates it under its qualified status against published reference
standards. The profile makes the substrate ready; the QTSP and the law make it
qualified.

## Sources

- Reg (EU) 2024/1183 (eIDAS 2.0), Art. 45k / 45l: https://eur-lex.europa.eu/eli/reg/2024/1183
- Status watch list and Stage 0 verdict: `research/eidas2_roadmap_20260627.md`
