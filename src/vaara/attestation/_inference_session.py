# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Inference-session manifest: collapse an ordered run of receipts to one record.

The bridge that makes the inference-receipt family bindable to a
hardware root at the same granularity a SEP-2828 tool-call record already is.

The TPM evidence chain (``vaara.tpm-evidence-chain/v0``) binds *one* JSON record
across an ordered window of TPM quotes. An inference proxy, by contrast, emits one
attestation+receipt pair per model call, many per window. This module closes that
mismatch: it folds a window of receipts into a single
``vaara.inference-session/v0`` manifest whose ``root`` commits to the ordered
``(attestationDigest, receiptDigest)`` list. Bind that one manifest to a continuous
TPM chain and every inference in the window traces to the hardware root,
transitively and byte-for-byte, with no change to the chain code: the manifest is
just another ``record``.

The manifest is a *binding index*, not a content copy. It carries digests, never
the receipts themselves, so it stays small and leaks nothing about the prompts. A
verifier is handed both the manifest (bound to the chain) and the receipts (signed
by the proxy) and confirms they agree; see :mod:`._inference_chain_verify`.

The ``attestationDigest`` and ``nonce`` of each link are read from the receipt's
own ``backLink`` rather than recomputed from a separate attestation file, so a
manifest is buildable from receipts alone. The composite verifier, when also given
the attestations, additionally confirms each back-link actually recomputes.

Schema ``vaara.inference-session/v0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from vaara.attestation._inference_emit import inference_receipt_digest
from vaara.attestation._inference_types import (
    InferenceReceipt,
    inference_receipt_from_dict,
)
from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation._sep2787_types import AttestationError

INFERENCE_SESSION_SCHEMA = "vaara.inference-session/v0"

_SESSION_KEYS = frozenset({"schema", "count", "links", "root"})
_LINK_KEYS = frozenset({"seq", "attestationDigest", "receiptDigest", "nonce"})


def session_links_root(links: list[dict[str, Any]]) -> str:
    """``sha256:<hex>`` over the JCS-canonical encoding of the ordered links.

    Computed over the ``links`` list alone (not the whole manifest), so a
    verifier recomputes it from the links independently of the ``root`` and
    ``count`` fields the manifest also carries.
    """
    return "sha256:" + hashlib.sha256(canonical_json(links)).hexdigest()


def _receipt_link(receipt: InferenceReceipt, seq: int) -> dict[str, Any]:
    """One manifest link pinning a single inference, in chain order."""
    return {
        "seq": seq,
        "attestationDigest": receipt.back_link.attestation_digest,
        "receiptDigest": inference_receipt_digest(receipt),
        "nonce": receipt.back_link.attestation_nonce,
    }


def build_session_manifest(receipts: list[InferenceReceipt]) -> dict[str, Any]:
    """Fold an ordered list of receipts into a ``vaara.inference-session/v0`` doc.

    ``receipts`` must already be in chain order (the proxy's monotonic counter
    order). Raises :class:`AttestationError` on an empty list, since a chain
    binds at least one record.
    """
    if not receipts:
        raise AttestationError("inference session needs at least one receipt")
    links = [_receipt_link(r, seq) for seq, r in enumerate(receipts, start=1)]
    return {
        "schema": INFERENCE_SESSION_SCHEMA,
        "count": len(links),
        "links": links,
        "root": session_links_root(links),
    }


def parse_session_manifest(doc: Any) -> dict[str, Any]:
    """Validate a session manifest against the closed schema and self-consistency.

    Checks: closed key set, schema tag, ``count == len(links)``, contiguous
    ``seq`` 1..N in order, ``sha256:`` digest shapes, non-empty nonces, and that
    the stored ``root`` recomputes from the links. Returns the doc unchanged on
    success; raises :class:`AttestationError` otherwise. A bound-but-tampered
    manifest (e.g. a link silently dropped) fails here rather than verifying.
    """
    if not isinstance(doc, dict):
        raise AttestationError("inference session manifest must be a JSON object")
    extra = set(doc) - _SESSION_KEYS
    if extra:
        raise AttestationError(
            f"inference session manifest carries unrecognized field(s) "
            f"{sorted(extra)!r}; the schema is closed"
        )
    if doc.get("schema") != INFERENCE_SESSION_SCHEMA:
        raise AttestationError(
            f"unexpected session schema {doc.get('schema')!r}; "
            f"expected {INFERENCE_SESSION_SCHEMA!r}"
        )
    links = doc.get("links")
    if not isinstance(links, list) or not links:
        raise AttestationError(
            "inference session manifest links must be a non-empty list"
        )
    count = doc.get("count")
    if count != len(links):
        raise AttestationError(
            f"inference session count {count!r} != number of links {len(links)}"
        )
    for idx, link in enumerate(links, start=1):
        if not isinstance(link, dict):
            raise AttestationError(f"session link {idx} must be a JSON object")
        link_extra = set(link) - _LINK_KEYS
        if link_extra:
            raise AttestationError(
                f"session link {idx} carries unrecognized field(s) {sorted(link_extra)!r}"
            )
        for required in _LINK_KEYS:
            if required not in link:
                raise AttestationError(
                    f"session link {idx} missing required field {required!r}"
                )
        if link["seq"] != idx:
            raise AttestationError(
                f"session links must be a contiguous 1..N sequence in order; "
                f"link at position {idx} has seq {link['seq']!r}"
            )
        for digest_field in ("attestationDigest", "receiptDigest"):
            if not str(link[digest_field]).startswith("sha256:"):
                raise AttestationError(
                    f"session link {idx} {digest_field} MUST be a 'sha256:' digest"
                )
        if not isinstance(link["nonce"], str) or not link["nonce"]:
            raise AttestationError(
                f"session link {idx} nonce MUST be a non-empty string"
            )
    if doc.get("root") != session_links_root(links):
        raise AttestationError(
            "inference session root does not recompute from its links "
            "(the manifest was altered after it was built)"
        )
    return doc


def session_manifest_matches_receipts(
    manifest: dict[str, Any], receipts: list[InferenceReceipt]
) -> bool:
    """True if ``manifest`` is exactly the session over ``receipts``.

    Rebuilds the manifest from the supplied receipts and compares roots, so a
    single mutated or reordered receipt yields a different root and ``False``.
    The manifest is assumed already structurally validated by
    :func:`parse_session_manifest`; this is the content-equality bridge between
    the hardware-bound record and the signed receipts.
    """
    try:
        rebuilt = build_session_manifest(receipts)
    except AttestationError:
        return False
    return (
        rebuilt["root"] == manifest.get("root")
        and rebuilt["count"] == manifest.get("count")
    )


def session_manifest_covers_prefix(
    manifest: dict[str, Any], receipts: list[InferenceReceipt]
) -> bool:
    """True if ``manifest`` is exactly the session over the FIRST ``count`` receipts.

    Where :func:`session_manifest_matches_receipts` requires the manifest to bind
    *every* supplied receipt, this requires only that the manifest binds an
    unbroken **prefix**: the first ``manifest['count']`` receipts in chain order
    rebuild to the manifest's root. Receipts beyond that prefix are signed turns
    the manifest does not claim, reported separately as an unbound tail by the
    composite verifier, not silently folded in.

    The security property over the bound prefix is unchanged from the exact match:
    any alteration, drop, or reorder *within* the first ``count`` receipts shifts
    the rebuilt root and yields ``False``. Returns ``False`` when fewer receipts
    are supplied than the manifest binds, since a receipt the manifest commits to
    is then missing.
    """
    count = manifest.get("count")
    if not isinstance(count, int) or count <= 0 or len(receipts) < count:
        return False
    return session_manifest_matches_receipts(manifest, receipts[:count])


# --- CLI: assemble a manifest from a receipts directory ---------------------


def _load_receipts_in_order(directory: Path) -> list[InferenceReceipt]:
    """Parse ``*-infer-receipt.json`` in the directory, in counter order.

    The proxy writes ``{counter:010d}-{nonce}-infer-receipt.json``; the
    zero-padded counter makes a plain filename sort the chain order.
    """
    receipts: list[InferenceReceipt] = []
    for path in sorted(directory.glob("*-infer-receipt.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        receipts.append(inference_receipt_from_dict(doc))
    return receipts


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inference-session",
        description=(
            "Assemble a vaara.inference-session/v0 manifest from a directory of "
            "signed inference receipts, ready to bind to a TPM evidence chain."
        ),
    )
    parser.add_argument(
        "--dir", required=True, help="Receipts directory (*-infer-receipt.json)."
    )
    parser.add_argument("--out", required=True, help="Path to write the manifest JSON.")
    args = parser.parse_args(argv)

    directory = Path(args.dir).expanduser()
    if not directory.is_dir():
        print(f"inference-session: not a directory: {directory}", file=sys.stderr)
        return 2
    try:
        manifest = build_session_manifest(_load_receipts_in_order(directory))
    except (AttestationError, ValueError, OSError) as exc:
        print(f"inference-session: {exc}", file=sys.stderr)
        return 1
    Path(args.out).expanduser().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.out}: {manifest['count']} inference(s), root {manifest['root']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
