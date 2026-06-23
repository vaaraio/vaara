#!/usr/bin/env python3
"""Generate the acp_checkout_v0 conformance vector.

A recomputable ``{statement, expected-verdict}`` pair for an Agentic Commerce
Protocol (ACP) checkout session, the artifact offered for the governance
binding discussed on agentic-commerce-protocol#231.

The statement is a real-shaped ACP ``CheckoutSession`` in its terminal
``completed`` status, carrying the order it produced. ACP objects are **not
signed**: the protocol authenticates the API call at the transport, not the
session record. So unlike the in-toto ``agent-decision`` vector there is no
DSSE envelope and no signature to verify. The recomputable anchor is instead a
content commitment: the JCS (RFC 8785) canonical bytes of the statement,
digested. That digest, plus the SEP-2828 mapping, is what a reader reproduces.

The expected verdict is twofold:

1. **byte:** ``jcsSha256``, the sha256 of the JCS canonical statement bytes,
   reproducible by any reader from the document alone; and
2. **mapping:** the SEP-2828 normalization of the session (the outcome plane,
   the advisory context lifted, and what a complete signed receipt still
   ``missing``), reproduced from the shipped declarative profile spec.

The sibling ``_check_independent.py`` reproduces both importing **no Vaara
code**: only the standard library and ``rfc8785`` (for JCS). Conformance is
recompute-determinism over the bytes.

Run: ``python tests/vectors/acp_checkout_v0/_generate.py`` then commit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785

from vaara.attestation.receipt import normalize

HERE = Path(__file__).resolve().parent


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    statement = json.loads((HERE / "statement.json").read_text(encoding="utf-8"))

    jcs = rfc8785.dumps(statement)
    expected = {
        "status": statement["status"],
        "jcsSha256": "sha256:" + hashlib.sha256(jcs).hexdigest(),
        "normalized": normalize(statement).to_dict(),
    }
    _write(HERE / "expected.json", expected)
    print("wrote acp_checkout_v0 vector; jcsSha256=" + expected["jcsSha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
