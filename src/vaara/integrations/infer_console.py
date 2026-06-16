# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Local web console for the sovereign-governed Vaara stack.

Serves a single page that drives the signing inference proxy and shows the live
proof of each turn: receipt + signature verified every turn, the TPM-chain weld
and the second-model cross-check on demand. The console is a keyless viewer; it
loads a *verifying* key only (the proxy keeps the receipt-signing key). A
cross-check signing key is optional and represents a distinct verifier identity.

Run::

    python -m vaara.integrations.infer_console \\
        --receipts-dir ./receipts/infer \\
        --pubkey ./receipts/infer/pubkey.pem \\
        [--listen 127.0.0.1:11456] [--proxy-url http://127.0.0.1:11435] \\
        [--chain CHAIN.json] [--judge-model M --crosscheck-key KEY.pem]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vaara.attestation._inference_verify import _load_verifying_material
from vaara.integrations._infer_console_app import build_app

logger = logging.getLogger("vaara.infer_console")

# The offload FTS+semantic index, when run from the repo tree
# (src/vaara/integrations/infer_console.py -> repo root is parents[3]).
_DEFAULT_INDEX_DB = (
    Path(__file__).resolve().parents[3] / "tools" / "offload" / "index.db"
)


def _parse_listen(value: str) -> "tuple[str, int]":
    host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("expected HOST:PORT, e.g. 127.0.0.1:11456")
    return host, int(port)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vaara-console",
        description="Local chat + live-proof console for the sovereign Vaara stack.",
    )
    parser.add_argument(
        "--listen", type=_parse_listen, default=("127.0.0.1", 11456),
        help="HOST:PORT bind (default 127.0.0.1:11456).",
    )
    parser.add_argument(
        "--proxy-url", default="http://127.0.0.1:11435",
        help="Signing inference-proxy base URL (default http://127.0.0.1:11435).",
    )
    parser.add_argument(
        "--receipts-dir", required=True,
        help="Directory the proxy writes the signed attestation/receipt pairs to.",
    )
    parser.add_argument(
        "--pubkey", default=None,
        help="PEM public key to verify signatures (ES256/RS256). "
             "Defaults to <receipts-dir>/pubkey.pem when present.",
    )
    parser.add_argument(
        "--secret", default=None, help="Raw shared-secret file for HS256 instead.",
    )
    parser.add_argument(
        "--chain", default=None,
        help="TPM evidence-chain bundle JSON; enables the hardware-chain button.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Default second model for the cross-check button. Optional: the page "
             "picks a judge per request, this is only the fallback.",
    )
    parser.add_argument(
        "--crosscheck-key", default=None,
        help="Signing key for the verifier-party cross-check record. Setting it "
             "enables the cross-check button (the judge is chosen per request).",
    )
    parser.add_argument(
        "--crosscheck-secret-version", default=None,
        help="Override the cross-check key's secret-version label.",
    )
    parser.add_argument(
        "--judge-num-gpu", type=int, default=0,
        help="GPU layers for the judge model (default 0 = CPU-only, so a light "
             "verifier never evicts the subject model from the GPU).",
    )
    parser.add_argument(
        "--upstream", default="http://127.0.0.1:11434",
        help="ollama base URL the judge model runs on (default :11434).",
    )
    parser.add_argument(
        "--recall-db", default=None,
        help="Offload FTS+semantic index to ground answers in "
             f"(default {_DEFAULT_INDEX_DB} when present).",
    )
    parser.add_argument(
        "--recall-k", type=int, default=6,
        help="Max memory slices to inject per turn (default 6).",
    )
    parser.add_argument(
        "--no-recall", action="store_true",
        help="Disable memory grounding even when an index is present.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    receipts_dir = Path(args.receipts_dir).expanduser()
    pubkey = args.pubkey
    if pubkey is None and args.secret is None:
        default_pub = receipts_dir / "pubkey.pem"
        if default_pub.is_file():
            pubkey = str(default_pub)
    try:
        verifying_material = _load_verifying_material(pubkey, args.secret)
    except Exception as exc:
        print(f"vaara-console: cannot load verifying key: {exc}")
        return 2

    crosscheck = None
    if args.crosscheck_key:
        from vaara.integrations._infer_proxy_sign import load_signing_key

        try:
            signing_material, alg, secret_version = load_signing_key(
                Path(args.crosscheck_key), args.crosscheck_secret_version
            )
        except Exception as exc:
            print(f"vaara-console: cannot load cross-check key: {exc}")
            return 2
        crosscheck = {
            "judge_model": args.judge_model,  # optional default; the page picks per request
            "upstream": args.upstream,
            "signing_material": signing_material,
            "alg": alg,
            "secret_version": secret_version,
            "num_gpu": args.judge_num_gpu,
        }
    elif args.judge_model:
        print("vaara-console: --judge-model requires --crosscheck-key (the verifier "
              "needs a signing identity)")
        return 2

    recall = None
    if not args.no_recall:
        db = Path(args.recall_db).expanduser() if args.recall_db else _DEFAULT_INDEX_DB
        if db.is_file():
            from vaara.integrations._infer_console_recall import MemoryRecall

            recall = MemoryRecall(db, k=args.recall_k)
            logger.info("memory grounding ON (%s, k=%d)", db, args.recall_k)
        else:
            logger.info("memory grounding OFF (no index at %s)", db)

    chain_path = Path(args.chain).expanduser() if args.chain else None
    host, port = args.listen
    logger.info(
        "vaara-console on %s:%d -> proxy %s (chain=%s, crosscheck=%s, recall=%s)",
        host, port, args.proxy_url, bool(chain_path), bool(crosscheck), bool(recall),
    )

    import uvicorn

    app = build_app(
        proxy_url=args.proxy_url,
        receipts_dir=receipts_dir,
        verifying_material=verifying_material,
        chain_path=chain_path,
        crosscheck=crosscheck,
        recall=recall,
    )
    uvicorn.run(app, host=host, port=port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
