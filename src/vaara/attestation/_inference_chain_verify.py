# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""verify-inference-chain: the composite weld check, model turn to hardware root.

One verdict that ties three layers together, each independently
checkable offline by someone who trusts neither Vaara nor the operator:

1. the TPM evidence chain verifies and its bound ``record`` is a well-formed
   ``vaara.inference-session/v0`` manifest (:mod:`._inference_session`);
2. that manifest is *exactly* the session over the supplied signed receipts (its
   ``root`` recomputes from them), so the hardware-bound digest commits to these
   inferences and no others;
3. every inference receipt's signature and back-link verify on their own.

Compose those and the chain reads end to end: each model turn produced a signed
receipt, the ordered receipts fold to one manifest, that manifest is bound across
a continuous TPM quote loop on one uninterrupted boot, AK-signed by the platform's
attestation key. After this the model call and the tool call trace to the same
root the operator cannot forge.

The honesty model is inherited verbatim from the chain verdict: the AK is trusted
as supplied (no EK chain in v0), freshness rests on chain continuity not a live
challenge, and the weld proves platform *continuity*, never inference
*determinism* -- a ``tier=integrity`` receipt is not upgraded to ``replay`` by
being hardware-bound. Those two axes stay orthogonal in the verdict.

Run: ``python -m vaara.attestation._inference_chain_verify --chain CHAIN.json \
--dir RECEIPTS_DIR [--pubkey pubkey.pem | --secret KEY] [--strict] [--json]``.

Schema ``vaara.inference-chain/v0``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from vaara.attestation._inference_session import (
    parse_session_manifest,
    session_manifest_covers_prefix,
    session_manifest_matches_receipts,
)
from vaara.attestation._inference_verify import (
    _load_verifying_material,
    _pair_dir,
    _verify_one,
)
from vaara.attestation._attest_types import AttestationError

INFERENCE_CHAIN_SCHEMA = "vaara.inference-chain/v0"


def verify_inference_chain(
    chain_doc: Any,
    *,
    receipts: "list[tuple[Any, Optional[Any]]]",
    verifying_material: Any = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Composite weld verdict over a TPM chain bound to an inference session.

    ``chain_doc`` is a parsed ``vaara.tpm-evidence-chain/v0`` bundle whose bound
    record is the session manifest. ``receipts`` is the ordered list of
    ``(receipt_doc, attestation_doc | None)`` pairs the chain claims to bind, in
    chain order. ``verifying_material`` (a public key or HS256 secret) keys the
    per-receipt signature checks; ``None`` runs the structural/back-link checks
    only. Returns a verdict dict; raises :class:`ValueError` only on a
    structurally malformed chain bundle (the chain verifier's contract).
    """
    from vaara.attestation.receipt import verify_tpm_chain_bundle

    chain_verdict = verify_tpm_chain_bundle(chain_doc, strict=strict)
    chain_d = chain_verdict.to_dict()

    # The chain bound *some* record; require it to be a valid session manifest.
    try:
        manifest = parse_session_manifest(chain_verdict.record)
        record_is_session = True
        session_reason = ""
    except AttestationError as exc:
        manifest = None
        record_is_session = False
        session_reason = str(exc)

    # Per-receipt verification (signature gates only when keyed; back-link gates
    # whenever the attestation is supplied), reusing the inference verifier.
    per_receipt: list[dict[str, Any]] = []
    parsed_receipts: list[Any] = []
    from vaara.attestation.inference import parse_inference_receipt

    for receipt_doc, att_doc in receipts:
        try:
            checks = _verify_one(receipt_doc, att_doc, verifying_material)
            parsed_receipts.append(parse_inference_receipt(receipt_doc))
        except Exception as exc:  # parse/verify failure for this pair
            checks = {"ok": False, "error": str(exc)}
        per_receipt.append(checks)

    receipts_all_valid = bool(per_receipt) and all(
        c.get("ok") for c in per_receipt
    )
    tiers = sorted({c.get("tier") for c in per_receipt if c.get("tier")})

    all_parsed = len(parsed_receipts) == len(receipts)
    matches = (
        record_is_session
        and all_parsed
        and session_manifest_matches_receipts(manifest, parsed_receipts)
    )
    # Honest coverage: the manifest may bind an unbroken *prefix* of the receipts
    # (the operator chatted more after the last capture). The bound prefix is as
    # strongly rooted as an exact match; the tail is disclosed, never folded in.
    # ``ok`` stays strict (exact match) -- coverage is presentation, not verdict.
    covers_prefix = (
        record_is_session
        and all_parsed
        and session_manifest_covers_prefix(manifest, parsed_receipts)
    )
    bound_count = int(manifest["count"]) if record_is_session else 0
    unbound_tail = max(0, len(receipts) - bound_count) if covers_prefix else 0

    chain_bound = chain_verdict.tier != "unverified"
    ok = bool(
        chain_bound
        and record_is_session
        and matches
        and receipts_all_valid
        and (chain_verdict.ok if strict else True)
    )

    return {
        "schema": INFERENCE_CHAIN_SCHEMA,
        "ok": ok,
        "strict": strict,
        "chainTier": chain_verdict.tier,
        "chainBound": chain_bound,
        "chainContinuous": chain_verdict.tier == "continuous",
        "recordIsSession": record_is_session,
        "manifestMatchesReceipts": matches,
        "manifestCoversPrefix": covers_prefix,
        "boundCount": bound_count,
        "unboundTail": unbound_tail,
        "nReceipts": len(receipts),
        "receiptsAllValid": receipts_all_valid,
        "inferenceTiers": tiers,
        "basis": {
            "akChainBasis": chain_d.get("ak_chain_basis"),
            "freshnessBasis": chain_d.get("freshness_basis"),
            "weldProves": "platform_continuity_not_inference_determinism",
        },
        "reason": _compose_reason(
            ok=ok,
            chain_bound=chain_bound,
            chain_tier=chain_verdict.tier,
            record_is_session=record_is_session,
            session_reason=session_reason,
            matches=matches,
            covers_prefix=covers_prefix,
            bound_count=bound_count,
            n_receipts=len(receipts),
            receipts_all_valid=receipts_all_valid,
        ),
        "chain": chain_d,
        "receipts": per_receipt,
    }


def _compose_reason(
    *,
    ok: bool,
    chain_bound: bool,
    chain_tier: str,
    record_is_session: bool,
    session_reason: str,
    matches: bool,
    covers_prefix: bool,
    bound_count: int,
    n_receipts: int,
    receipts_all_valid: bool,
) -> str:
    if not chain_bound:
        return "the TPM chain did not verify: no hardware root is established"
    if not record_is_session:
        return f"the chain's bound record is not a valid inference session: {session_reason}"
    if not matches:
        if covers_prefix:
            tail = n_receipts - bound_count
            noun = "turn" if tail == 1 else "turns"
            verb = "is" if tail == 1 else "are"
            return (
                f"the manifest hardware-binds the first {bound_count} of "
                f"{n_receipts} receipts; the {tail} later {noun} {verb} signed "
                "but not yet bound (re-capture to extend coverage)"
            )
        return (
            "the bound session manifest does not match the supplied receipts "
            "(a receipt was altered, dropped, or reordered)"
        )
    if not receipts_all_valid:
        return "at least one inference receipt failed its signature or back-link check"
    cont = (
        "across a continuous, single-boot quote loop"
        if chain_tier == "continuous"
        else "across the quote chain (single tick: continuity not yet established)"
    )
    return (
        f"every inference receipt verifies and folds to the manifest bound {cont}; "
        "the AK is trusted as supplied (no EK chain in v0) and the weld proves "
        "platform continuity, not inference determinism"
    )


# --- CLI --------------------------------------------------------------------


def _load_pairs(directory: Path) -> "list[tuple[Any, Optional[Any]]]":
    """Load ``(receipt_doc, attestation_doc | None)`` pairs in chain order."""
    pairs: list[tuple[Any, Optional[Any]]] = []
    for _prefix, att_path, receipt_path in _pair_dir(directory):
        receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
        att_doc = (
            json.loads(att_path.read_text(encoding="utf-8")) if att_path else None
        )
        pairs.append((receipt_doc, att_doc))
    return pairs


def _print_human(verdict: dict[str, Any]) -> None:
    head = "VALID" if verdict["ok"] else "INVALID"
    print(
        f"inference-chain: {head}  "
        f"(chainTier={verdict['chainTier']}, receipts={verdict['nReceipts']}, "
        f"tiers={verdict['inferenceTiers']})"
    )
    for key in (
        "chainBound",
        "chainContinuous",
        "recordIsSession",
        "manifestMatchesReceipts",
        "manifestCoversPrefix",
        "receiptsAllValid",
    ):
        print(f"    [{'pass' if verdict[key] else 'FAIL':4s}] {key}")
    print(f"    basis: {verdict['basis']}")
    print(f"    {verdict['reason']}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify-inference-chain",
        description=(
            "Verify a TPM evidence chain bound to an inference-session manifest "
            "against the signed receipts it claims to bind."
        ),
    )
    parser.add_argument("--chain", required=True, help="TPM evidence-chain bundle JSON.")
    parser.add_argument(
        "--dir", required=True, help="Receipts directory (*-infer-{attest,receipt}.json)."
    )
    parser.add_argument("--pubkey", default=None, help="PEM public key (ES256/RS256).")
    parser.add_argument("--secret", default=None, help="Raw shared-secret file (HS256).")
    parser.add_argument(
        "--strict", action="store_true", help="Require the EK-rooted chain tier (v0: unreachable)."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    chain_path = Path(args.chain).expanduser()
    directory = Path(args.dir).expanduser()
    if not chain_path.is_file():
        print(f"verify-inference-chain: not a file: {chain_path}", file=sys.stderr)
        return 2
    if not directory.is_dir():
        print(f"verify-inference-chain: not a directory: {directory}", file=sys.stderr)
        return 2

    try:
        material = _load_verifying_material(args.pubkey, args.secret)
    except (ValueError, OSError) as exc:
        print(f"verify-inference-chain: {exc}", file=sys.stderr)
        return 2

    chain_doc = json.loads(chain_path.read_text(encoding="utf-8"))
    pairs = _load_pairs(directory)
    try:
        verdict = verify_inference_chain(
            chain_doc, receipts=pairs, verifying_material=material, strict=args.strict
        )
    except ValueError as exc:  # structurally malformed chain bundle
        print(f"verify-inference-chain: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(verdict, indent=2, sort_keys=True))
    else:
        _print_human(verdict)
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
