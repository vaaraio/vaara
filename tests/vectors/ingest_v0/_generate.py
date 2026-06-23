#!/usr/bin/env python3
"""Generate the vaara.ingest/v0 conformance corpus from the registry.

This is the haymaker: the published vector set is not hand-authored. It is
a loop over the existing normalize input corpus. Each foreign source doc is
normalized, then sealed into a deterministic (fixed nonce + iat) ingest
envelope, and the {evidence, record} pair is written under ``cases/``. A
new SourceProfile that drops an input fixture lands in this corpus by
re-running this script; there is no per-format vector to write by hand.

The corpus signs with HS256 under a published test secret so the standalone
checker stays pure stdlib + rfc8785, mirroring the dependency-free ethos of
the peer did:web corpus. Conformance is recompute-determinism over the
vectors, not authenticity of this publisher; production identity is
asymmetric (ES256 / did:web), exactly as the normalize vectors note.

Run: ``python tests/vectors/ingest_v0/_generate.py`` then commit the output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785

from vaara.attestation.receipt import emit_ingest_receipt, normalize

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "normalize_v0" / "inputs"
CASES = HERE / "cases"

# Published conformance key. A test secret, never a production credential.
SECRET = b"ingest-conformance-shared-secret-0001"
FIXED = {
    "iss": "did:web:vaara.io",
    "sub": "sink",
    "secretVersion": "k1",
    "nonce": "ingest-fixed-nonce-000000",
    "iat": "2026-06-23T00:00:00Z",
}


def _emit(name: str):
    doc = json.loads((INPUTS / f"{name}.json").read_text())
    return emit_ingest_receipt(
        normalized=normalize(doc),
        iss=FIXED["iss"],
        sub=FIXED["sub"],
        secret_version=FIXED["secretVersion"],
        alg="HS256",
        signing_material=SECRET,
        nonce=FIXED["nonce"],
        iat=FIXED["iat"],
    )


def _corpus_digest(manifest: dict) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(manifest)).hexdigest()


def main() -> int:
    CASES.mkdir(exist_ok=True)
    names = sorted(p.stem for p in INPUTS.glob("*.json"))

    manifest: dict[str, dict[str, str]] = {}
    for name in names:
        r = _emit(name)
        (CASES / f"{name}.json").write_text(
            json.dumps({"record": r.record, "evidence": r.evidence}, indent=2,
                       sort_keys=True) + "\n"
        )
        manifest[name] = {
            "evidenceDigest": r.record["evidenceRef"]["digest"],
            "signature": r.record["signature"],
        }

    corpus = {
        "schema": "vaara.ingest-conformance/v0",
        "alg": "HS256",
        "sharedSecretHex": SECRET.hex(),
        "fixed": FIXED,
        "cases": names,
        "manifest": manifest,
        "corpusDigest": _corpus_digest(manifest),
    }
    (HERE / "corpus.json").write_text(
        json.dumps(corpus, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(names)} cases; corpusDigest={corpus['corpusDigest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
