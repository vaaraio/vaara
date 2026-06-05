#!/usr/bin/env python3
"""Independent conformance checker for the cross_stack_revocation_v0 vectors.

Imports only the standard library plus ``rfc8785``. It does not import
Vaara. For each committed case it reproduces three lenses over one receipt
and one registry, and checks they agree:

1. **Receipt-verifier lens.** Apply the revocation-in-time rule directly: a
   matching registry entry (identity-scope on ``iss``, or key-scope on the
   bound keyid) whose ``revoked_at`` is at or before the receipt's ``iat``
   revokes it. Unparseable instants fail closed.
2. **Transparency-log lens.** Recompute the RFC 6962 Merkle root from the
   receipt's canonical leaf bytes plus the committed inclusion proof, compare
   to the committed root, then combine with the revocation verdict: ``ok`` is
   inclusion and not revoked.
3. **Export lens.** Recompute the registry digest (SHA-256 over the RFC 8785
   canonical registry bytes) and match the committed digest, the value a
   signed Article-12 export pins.

A second implementation that runs this file demonstrates the cross-stack
revocation guarantee is consumable without depending on Vaara. Run:
``python tests/vectors/cross_stack_revocation_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent


def _parse_iso(value):
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _revoked_in_time(revoked_at, issued_at) -> bool:
    r = _parse_iso(revoked_at)
    i = _parse_iso(issued_at)
    if r is None or i is None:
        return True
    return r <= i


def _registry_revoked(registry: dict, iss: str, issued_at: str, keyid) -> bool:
    for entry in registry.get("entries", []):
        scope = entry.get("scope")
        subject = entry.get("subject")
        if scope == "key":
            if keyid is None or subject != keyid:
                continue
        elif scope == "identity":
            if subject != iss:
                continue
        else:
            continue
        if _revoked_in_time(entry.get("revoked_at"), issued_at):
            return True
    return False


def _registry_digest(registry: dict) -> str:
    entries = sorted(
        (
            {"scope": e["scope"], "subject": e["subject"], "revoked_at": e["revoked_at"]}
            for e in registry.get("entries", [])
        ),
        key=lambda d: (d["scope"], d["subject"], d["revoked_at"]),
    )
    canonical = rfc8785.dumps({"version": 1, "entries": entries})
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _hash_leaf(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _verify_inclusion(leaf: bytes, log_index, tree_size, siblings, root: bytes) -> bool:
    if not (0 <= log_index < tree_size):
        return False
    node = _hash_leaf(leaf)
    idx = log_index
    size = tree_size
    it = iter(siblings)
    while size > 1:
        last = size - 1
        if idx == last and idx % 2 == 0:
            pass  # unpaired right edge, promoted
        else:
            try:
                sib = next(it)
            except StopIteration:
                return False
            node = _hash_node(node, sib) if idx % 2 == 0 else _hash_node(sib, node)
        idx //= 2
        size = (size + 1) // 2
    if next(it, None) is not None:
        return False
    return node == root


_RECEIPT_KEYS = (
    "version", "alg", "backLink", "outcomeDerived", "receiptAsserted", "signature",
)


def _evaluate(case: dict) -> dict:
    receipt = case["receipt"]
    iss = receipt["receiptAsserted"]["iss"]
    iat = receipt["receiptAsserted"]["iat"]
    keyid = case.get("keyid")
    registry = case["registry"]

    revoked = _registry_revoked(registry, iss, iat, keyid)

    leaf = rfc8785.dumps({k: receipt[k] for k in _RECEIPT_KEYS})
    inc = case["inclusion"]
    siblings = [bytes.fromhex(h) for h in inc["siblings_hex"]]
    included = _verify_inclusion(
        leaf, inc["log_index"], inc["tree_size"], siblings,
        bytes.fromhex(inc["root_hex"]),
    )

    return {
        "receipt_lens_revoked": revoked,
        "log_lens_included": included,
        "log_lens_ok": included and not revoked,
        "registry_digest": _registry_digest(registry),
    }


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for case in cases:
        name = case["name"]
        got = _evaluate(case)
        want = expected[name]
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK {got}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall cross_stack_revocation_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
