# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""verify-inference: check a signed inference receipt (and its attestation).

A standalone CLI, the inference-side counterpart to ``vaara verify-record``:

- keyless structural mode: parse the receipt against the closed schema and, if
  the paired attestation is given, confirm the back-link pins it;
- keyed mode: with a verifying key, check the receipt signature (and the
  attestation signature + TTL when the attestation is given).

A ``--dir`` sweep pairs ``*-infer-attest.json`` with ``*-infer-receipt.json``
by the ``{counter}-{nonce}`` filename prefix and reports each chain, the exact
shape the proxy writes.

Run: ``python -m vaara.attestation._inference_verify RECEIPT.json \
[--attestation ATT.json] [--pubkey pubkey.pem | --secret KEY] [--json]``
or ``python -m vaara.attestation._inference_verify --dir RECEIPTS_DIR \
[--pubkey pubkey.pem] [--json]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


def _load_verifying_material(pubkey: Optional[str], secret: Optional[str]) -> Any:
    """Return verifying material, or ``None`` for keyless mode."""
    if pubkey and secret:
        raise ValueError("pass only one of --pubkey / --secret")
    if secret:
        return Path(secret).expanduser().read_bytes()
    if pubkey:
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_public_key(
            Path(pubkey).expanduser().read_bytes()
        )
    return None


def _verify_one(
    receipt_doc: Any, att_doc: Optional[Any], material: Any
) -> dict[str, Any]:
    """Verify a single receipt, optionally against its attestation + a key."""
    from vaara.attestation.inference import (
        parse_inference_attestation,
        parse_inference_receipt,
        verify_inference_attestation_detail,
        verify_inference_back_link,
        verify_inference_receipt_signature,
    )

    checks: dict[str, Any] = {}
    receipt = parse_inference_receipt(receipt_doc)  # closed-schema structural pass
    checks["receiptSchema"] = True
    checks["tier"] = receipt.outcome_derived.tier
    checks["status"] = receipt.outcome_derived.status

    attestation = None
    if att_doc is not None:
        attestation = parse_inference_attestation(att_doc)
        checks["attestationSchema"] = True
        checks["backLink"] = verify_inference_back_link(
            receipt, attestation=attestation
        )

    if material is not None:
        checks["receiptSignature"] = verify_inference_receipt_signature(
            receipt, verifying_material=material
        )
        if attestation is not None:
            detail = verify_inference_attestation_detail(
                attestation, verifying_material=material
            )
            # Signature gates; freshness is reported, not gated. A stored
            # receipt is archival and expected to outlive its live TTL window,
            # so an authentic-but-expired attestation stays VALID and is only
            # noted as expired -- never mislabeled as a signature failure.
            checks["attestationSignature"] = detail["signatureValid"]
            checks["attestationFresh"] = detail["fresh"]
            checks["attestationAgeSeconds"] = detail["ageSeconds"]
            checks["attestationExpSeconds"] = detail["expSeconds"]

    gating = [
        v for k, v in checks.items()
        if k in {"backLink", "receiptSignature", "attestationSignature"}
    ]
    checks["ok"] = all(gating) if gating else True
    return checks


def _fmt_minutes(seconds: Any) -> str:
    return f"{seconds / 60:.0f}m" if isinstance(seconds, (int, float)) else "?"


def _print_human(name: str, checks: dict[str, Any]) -> None:
    verdict = "VALID" if checks.get("ok") else "INVALID"
    print(f"{name}: {verdict}  (tier={checks.get('tier')}, status={checks.get('status')})")
    for key in ("backLink", "receiptSignature", "attestationSignature"):
        if key in checks:
            print(f"    [{'pass' if checks[key] else 'FAIL':4s}] {key}")
    # Freshness is informational, not a pass/FAIL gate: an expired credential
    # with a valid signature is the normal state of an archived receipt.
    if "attestationFresh" in checks:
        if checks["attestationFresh"]:
            print("    [info] attestationFresh: live")
        else:
            age = _fmt_minutes(checks.get("attestationAgeSeconds"))
            ttl = _fmt_minutes(checks.get("attestationExpSeconds"))
            print(
                f"    [info] attestationFresh: expired "
                f"(age {age} > ttl {ttl}; signature still valid)"
            )


def _pair_dir(directory: Path) -> "list[tuple[str, Optional[Path], Path]]":
    """Pair attest/receipt files by their ``{counter}-{nonce}`` prefix."""
    receipts = sorted(directory.glob("*-infer-receipt.json"))
    pairs: list[tuple[str, Optional[Path], Path]] = []
    for receipt_path in receipts:
        prefix = receipt_path.name[: -len("-infer-receipt.json")]
        att_path = directory / f"{prefix}-infer-attest.json"
        pairs.append((prefix, att_path if att_path.is_file() else None, receipt_path))
    return pairs


def _run_dir(args: argparse.Namespace, material: Any) -> int:
    directory = Path(args.dir).expanduser()
    if not directory.is_dir():
        print(f"verify-inference: not a directory: {directory}", file=sys.stderr)
        return 2
    pairs = _pair_dir(directory)
    if not pairs:
        print(f"verify-inference: no *-infer-receipt.json in {directory}", file=sys.stderr)
        return 2
    results: list[dict[str, Any]] = []
    all_ok = True
    for prefix, att_path, receipt_path in pairs:
        receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
        att_doc = (
            json.loads(att_path.read_text(encoding="utf-8")) if att_path else None
        )
        try:
            checks = _verify_one(receipt_doc, att_doc, material)
        except Exception as exc:  # parse/verify failure for this pair
            checks = {"ok": False, "error": str(exc)}
        all_ok = all_ok and bool(checks.get("ok"))
        if args.json:
            results.append({"pair": prefix, **checks})
        else:
            _print_human(prefix, checks)
    if args.json:
        print(json.dumps({"pairs": results, "allOk": all_ok}, indent=2))
    return 0 if all_ok else 1


def _run_single(args: argparse.Namespace, material: Any) -> int:
    receipt_path = Path(args.receipt).expanduser()
    if not receipt_path.is_file():
        print(f"verify-inference: not a file: {receipt_path}", file=sys.stderr)
        return 2
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    att_doc = None
    if args.attestation:
        att_doc = json.loads(Path(args.attestation).expanduser().read_text("utf-8"))
    try:
        checks = _verify_one(receipt_doc, att_doc, material)
    except Exception as exc:
        print(f"verify-inference: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        _print_human(receipt_path.name, checks)
    return 0 if checks.get("ok") else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify-inference",
        description="Verify a signed inference receipt and its attestation.",
    )
    parser.add_argument("receipt", nargs="?", help="Path to an inference receipt JSON.")
    parser.add_argument("--attestation", default=None,
        help="Paired InferenceAttestation JSON; enables the back-link check.")
    parser.add_argument("--dir", default=None,
        help="Sweep a receipts directory, pairing attest/receipt by filename.")
    parser.add_argument("--pubkey", default=None, help="PEM public key (ES256/RS256).")
    parser.add_argument("--secret", default=None, help="Raw shared-secret file (HS256).")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    if not args.dir and not args.receipt:
        parser.error("pass a receipt path or --dir")

    try:
        material = _load_verifying_material(args.pubkey, args.secret)
    except (ValueError, OSError) as exc:
        print(f"verify-inference: {exc}", file=sys.stderr)
        return 2

    if args.dir:
        return _run_dir(args, material)
    return _run_single(args, material)


if __name__ == "__main__":
    raise SystemExit(main())
