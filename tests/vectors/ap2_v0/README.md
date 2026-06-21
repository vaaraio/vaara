# ap2_v0 conformance vectors

Binding profile: an AP2 checkout becomes the evidence a `vaara.receipt/v1`
authorization receipt names, and the post-checkout agent actions become a
gap-evident stream under the AP2 task scope. Verifiable by any third party with
only the AP2 Payment Evidence Frame, the held receipts, and the issuer's public
key. No AP2 access, no external witness, no Vaara import.

This is the artifact behind "AP2 can pin to vaara.receipt/v1 from the point the
Checkout Receipt ends, rather than define a new primitive." See the parent
`SPEC.md` section 5 and `tests/vectors/contiguity_v0/` for the completeness
half this reuses.

## The mapping

The AP2 checkout emits a Payment Evidence Frame (PEF, AP2 PR #274). The frame
wraps an AP2 Checkout Receipt as `receipt` and content-addresses it as
`receipt_hash` = `sha256(JCS(receipt))`. The whole frame is content-addressed as
`frame_id` = `sha256(JCS(frame))`, with `frame_id` and `signature` excluded from
the preimage. Canonicalisation is `urn:x402:canonicalisation:jcs-rfc8785-v1`
(JCS / RFC 8785), the same canonicalisation `vaara.receipt/v1` uses, so the
address joins cleanly with no re-canonicalisation.

Each post-checkout action is minted as a `vaara.authorization/v0` receipt at the
point it is authorized. The receipt:

- names the checkout it followed by content address:
  `decisionDerived.evidenceRef.ref` = `ap2:checkout/<frame_id>`, carried under
  the receipt signature;
- declares the AP2 task scope as `coverage.boundary`, inside the signed
  evidence;
- carries a signed `completeness` block (`boundaryId`, `seq`, `runningCount`)
  sequencing the actions under that boundary.

The inner Checkout Receipt here is representative. The binding rests on the #274
content-addressing discipline (`frame_id`, `receipt_hash`, JCS), not on the
inner receipt's exact field names.

## Cases

One checkout (`checkout/pef.json`) and one stream of three post-checkout actions
(seq 0..2 under boundary `ap2:task/checkout-7f3a`). The `dropped` case is the
`complete` set minus one file, so the gap is a genuine omission, not a re-mint.

- `complete/`: all three receipts held. Every signature verifies, every evidence
  binding resolves, every receipt names the checkout, contiguity `present` 3,
  `expected` 3, no missing seqs.
- `dropped/`: the seq-1 receipt is withheld. The two held receipts still carry
  the signed running count that says three exist, so seq 1 is a provable gap.
  Contiguity `present` 2, `expected` 3, `missingSeqs` `[1]`.

Frame verdicts hold over the one shared PEF: `frame_id_recomputes` and
`receipt_hash_recomputes`.

- `checkout/pef.json`: the AP2 Payment Evidence Frame for the checkout.
- `grant.json`: the capability grant the receipts authorize against, for
  context.
- `expected.json`: the verdict each case must produce.

## Honest limit

Contiguity proves nothing was dropped from the middle of the declared boundary;
a pure tail truncation (holding seq `0..k`) cannot be told from a complete
stream by contiguity alone, since the latest held count is then `k + 1`. The
frame binding proves the receipts name this exact checkout, not that the AP2
checkout itself was honest about what it settled; that is the payment side's
obligation, which the PEF content-addresses but does not adjudicate.

## Reproduce

Independent checker (standard library plus `cryptography` and `rfc8785`, no
Vaara import):

```
python tests/vectors/ap2_v0/_check_independent.py
```

Exit code 0 means every case matched its expected verdict. The post-checkout
streams are also consumable by the shipped CLI, since they carry standard
`completeness` blocks:

```
vaara verify-contiguity tests/vectors/ap2_v0/complete   # exit 0, contiguous
vaara verify-contiguity tests/vectors/ap2_v0/dropped    # exit 1, missing seq 1
```

Regenerate the vectors (ECDSA signatures are randomized, so signatures change
but verdicts do not) with:

```
python tests/vectors/ap2_v0/_generate.py
```
