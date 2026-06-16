# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""OpenAI/ollama-compatible inference proxy that signs the model call.

The MCP proxy
(``mcp_proxy.py``) governs ``tools/call`` and signs a SEP-2787 attestation +
execution-receipt pair per tool call. This proxy sits one layer down, in
front of the inference server: it fronts an OpenAI-compatible (or
ollama-native) chat endpoint and, per request, emits the sibling
``InferenceAttestation`` + ``InferenceReceipt`` pair binding which model, on
what silicon-resident weights, given what input, returned what output.

    Goose --(OpenAI /v1 or ollama /api/chat)--> vaara-infer-proxy --> ollama

Everything that is not a chat completion passes straight through, so
``/api/show``, ``/api/tags``, ``/v1/models``, embeddings, etc. all work and
the proxy is a drop-in replacement for the ollama base URL.

The app factory lives in ``_infer_proxy_app.build_app``; this module is the
CLI entry point.

Run: ``python -m vaara.integrations.infer_proxy --signing-key KEY \
--receipts-dir DIR [--listen 127.0.0.1:11435] [--upstream http://127.0.0.1:11434]``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from vaara.integrations._infer_proxy_app import build_app
from vaara.integrations._infer_proxy_emit import InferenceAttestEmitter
from vaara.integrations._infer_proxy_sign import (
    InferProxyConfigError,
    load_signing_key,
)

# Re-export the app factory at the public path.
__all__ = ["build_app", "main"]

logger = logging.getLogger("vaara.infer_proxy")


def _parse_listen(value: str) -> "tuple[str, int]":
    if ":" not in value:
        raise argparse.ArgumentTypeError("--listen must be HOST:PORT")
    host, _, port = value.rpartition(":")
    try:
        return host or "127.0.0.1", int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port in --listen: {port!r}") from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vaara-infer-proxy",
        description="OpenAI/ollama-compatible inference proxy that signs an "
        "InferenceAttestation + InferenceReceipt pair per chat call.",
    )
    parser.add_argument("--signing-key", required=True,
        help="Signing key path: PEM EC P-256 (ES256), PEM RSA (RS256), or raw "
             "bytes (HS256).")
    parser.add_argument("--receipts-dir", required=True,
        help="Directory where the signed attestation/receipt pair files land.")
    parser.add_argument("--listen", type=_parse_listen, default=("127.0.0.1", 11435),
        help="HOST:PORT to bind on (default 127.0.0.1:11435).")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434",
        help="Inference server base URL (default http://127.0.0.1:11434).")
    parser.add_argument("--secret-version", default=None,
        help="Override the secret-version label (defaults to a key fingerprint).")
    parser.add_argument("--exp-seconds", type=int, default=300,
        help="TTL for the pre-call attestation window (default 300).")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        signing_material, alg, secret_version = load_signing_key(
            Path(args.signing_key), args.secret_version
        )
        emitter = InferenceAttestEmitter(
            signing_key=signing_material, alg=alg,
            receipts_dir=Path(args.receipts_dir).expanduser(),
            secret_version=secret_version, exp_seconds=args.exp_seconds,
        )
    except InferProxyConfigError as exc:
        print(f"vaara-infer-proxy: {exc}", file=sys.stderr)
        return 2

    host, port = args.listen
    logger.info(
        "vaara-infer-proxy listening on %s:%d -> %s (alg=%s, receipts=%s)",
        host, port, args.upstream, alg, emitter.receipts_dir,
    )

    import uvicorn

    app = build_app(emitter=emitter, upstream=args.upstream)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
