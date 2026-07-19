#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mint and verify a self-hosted rfc3161 timestamp anchor over a Vaara receipt.

Fully offline: stands up a local Time-Stamping Authority, anchors the receipt's
signed-payload digest (SPEC.md Section 4), prints the timestampAnchors entry,
then verifies it back. No third party, no network.

    python scripts/anchor_receipt_demo.py [path/to/receipt.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from vaara.audit.receipt_anchor import SelfHostedTSA, verify_receipt_anchor

DEFAULT = (Path(__file__).resolve().parents[1]
           / "tests/vectors/x402_settlement_v0/generic/step1/receipt.json")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    receipt = json.loads(path.read_text())

    tsa = SelfHostedTSA.create()
    anchor = tsa.anchor_receipt(receipt)
    attested = verify_receipt_anchor(receipt, anchor)

    print(f"receipt        : {path}")
    print(f"authority      : {anchor['authority']} (self-hosted, not qualified)")
    print(f"method         : {anchor['method']}")
    print(f"anchoredDigest : {anchor['anchoredDigest']}")
    print(f"attested time  : {attested.isoformat()}")
    print(f"token (b64, {len(anchor['token'])} chars): {anchor['token'][:64]}...")
    print("\ntimestampAnchors entry:")
    print(json.dumps(anchor, indent=2))
    print(f"\nverified: token attests this receipt at {attested.isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
