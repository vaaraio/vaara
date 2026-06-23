#!/usr/bin/env python3
"""Independent checker for the vaara.ingest/v0 conformance corpus.

Imports no Vaara code. For each committed {record, evidence} pair it
reproduces, from the envelope rules alone:

  1. the content address  -> sha256 over the RFC 8785 (JCS) canonical
     bytes of the evidence object, compared to record.evidenceRef.digest;
  2. the HS256 signature   -> HMAC-SHA256 over the JCS bytes of the
     envelope (signature field removed), under the published test secret;
  3. the sourceFormat bind -> envelope sourceFormat == evidence sourceFormat;
  4. the corpus digest     -> sha256 over the JCS bytes of the manifest.

Steps 1, 2 and 4 are pure stdlib hashlib/hmac; only the JCS canonicalizer
(rfc8785) is a third-party dependency, and the whole run skips cleanly when
it is absent. Passing this is conformance to the published vectors, not to
any one producer's checker.

Run: ``python tests/vectors/ingest_v0/_check_independent.py``.
Exit 0 means every case and the corpus digest reproduced.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    import rfc8785
except ImportError:  # pragma: no cover - exercised only in a base install
    print("[SKIP] rfc8785 not installed; the ingest corpus needs JCS to recompute")
    sys.exit(0)


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _check_case(name: str, secret: bytes, want: dict) -> list[str]:
    pair = json.loads((HERE / "cases" / f"{name}.json").read_text())
    record, evidence = pair["record"], pair["evidence"]
    fails: list[str] = []

    recomputed = _digest(evidence)
    if recomputed != record["evidenceRef"]["digest"]:
        fails.append("content address does not match record.evidenceRef.digest")
    if recomputed != want["evidenceDigest"]:
        fails.append("content address does not match corpus manifest")

    body = {k: v for k, v in record.items() if k != "signature"}
    sig = hmac.new(secret, rfc8785.dumps(body), hashlib.sha256).hexdigest()
    if sig != record["signature"]:
        fails.append("HS256 signature does not match record.signature")
    if sig != want["signature"]:
        fails.append("HS256 signature does not match corpus manifest")

    if record.get("sourceFormat") != evidence.get("sourceFormat"):
        fails.append("envelope sourceFormat does not bind evidence sourceFormat")
    return fails


def main() -> int:
    corpus = json.loads((HERE / "corpus.json").read_text())
    secret = bytes.fromhex(corpus["sharedSecretHex"])
    manifest = corpus["manifest"]

    failures = 0
    for name in corpus["cases"]:
        fails = _check_case(name, secret, manifest[name])
        failures += len(fails)
        print(f"[{'OK' if not fails else 'FAIL'}] {name}")
        for f in fails:
            print(f"   {f}")

    if _digest(manifest) != corpus["corpusDigest"]:
        print("[FAIL] corpusDigest does not reproduce from the manifest")
        failures += 1
    else:
        print(f"[OK] corpusDigest {corpus['corpusDigest']}")

    total = len(corpus["cases"]) + 1
    print(f"\n{total - failures}/{total} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
