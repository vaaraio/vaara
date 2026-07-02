# release-anchors/

Qualified RFC 3161 timestamp anchors, one per release tag.

Each `v<x.y.z>.json` binds a release to an **eIDAS-qualified** trusted timestamp:
a sha256 fingerprint of the release tree (`git ls-tree -r --full-tree <tag>`,
publicly recomputable) plus a qualified TSA token over that fingerprint. This is
the Article 41 legal-grade priority proof, on top of the Sigstore/SLSA
provenance and the PyPI/npm/GitHub publish records.

**Only qualified anchors belong here.** A file in this directory asserts the
token came from a QTSP listed on an EU Trusted List. Do not commit a
non-qualified token (e.g. a free public TSA used for a smoke test) — the
directory would then claim more than it proves.

Generate and verify with `scripts/anchor_release.py`:

```
python scripts/anchor_release.py --tag v<x.y.z> --tsa-url <qualified-QTSP-TSA>
python scripts/anchor_release.py --verify release-anchors/v<x.y.z>.json
```

`--verify` recomputes the tree fingerprint from the tag and checks the token
offline against the TSA's own key, so anyone can confirm the release tree
existed no later than the attested time without trusting us.
