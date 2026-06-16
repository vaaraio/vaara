# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""FastAPI app factory for the sovereign-governed Vaara console.

A thin, *keyless viewer* in front of the inference proxy. It drives the proxy
(which holds the signing key and emits the signed attestation/receipt pair) and
then runs the same verifiers a regulator would run: the per-turn receipt
signature check on every turn, and the TPM-chain weld + second-model cross-check
on demand. The console never signs an inference receipt. The one signature it can
produce is the cross-check verdict, and only as a distinct *verifier* identity
holding its own key, which is an honest second-party attestation.

    browser -> console (this app) -> infer proxy :11435 -> ollama :11434

Built by ``infer_console``; this module is the factory so it stays testable with
an injected ``httpx`` client and a fixture receipts dir.
"""

from __future__ import annotations

import base64
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from vaara.attestation._inference_verify import _verify_one
from vaara.integrations._infer_console_recall import ground_messages
from vaara.integrations._infer_console_view import CONSOLE_HTML
from vaara.integrations._infer_proxy_shape import StreamAccumulator, parse_ollama_response

logger = logging.getLogger("vaara.infer_console")

# Only the fields the proof panel renders; the runtime prompt/output and raw docs
# stay server-side (for a possible cross-check) and are never shipped to the page.
_VERDICT_KEYS = (
    "ok", "tier", "status", "backLink", "receiptSignature",
    "attestationSignature", "attestationFresh",
)
_SUFFIX = "-infer-receipt.json"

# Official Vaara wordmark (dark variant), inlined at serve time so the served
# page stays a single self-contained document with no extra asset route.
_WORDMARK_PNG = Path(__file__).resolve().parents[3] / "docs" / "vaara-wordmark-dark.png"


@lru_cache(maxsize=1)
def _wordmark_data_uri() -> str:
    """Base64 ``data:`` URI for the official wordmark, or ``""`` if unavailable."""
    try:
        raw = _WORDMARK_PNG.read_bytes()
    except OSError:
        logger.warning("wordmark asset not found at %s", _WORDMARK_PNG)
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _counter_of(receipt_path: Path) -> int:
    """The 10-digit ordinal the proxy prefixes onto each receipt filename."""
    try:
        return int(receipt_path.name.split("-", 1)[0])
    except (ValueError, IndexError):
        return -1


def _att_for(receipts_dir: Path, receipt_path: Path) -> Optional[Path]:
    prefix = receipt_path.name[: -len(_SUFFIX)]
    att_path = receipts_dir / f"{prefix}-infer-attest.json"
    return att_path if att_path.is_file() else None


def _sorted_receipts(receipts_dir: Path) -> "list[Path]":
    return sorted(receipts_dir.glob("*" + _SUFFIX), key=_counter_of)


def _load(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_pairs(receipts_dir: Path) -> "list[tuple[Any, Optional[Any]]]":
    """All ``(receipt_doc, attestation_doc|None)`` pairs in chain (counter) order."""
    pairs: list[tuple[Any, Optional[Any]]] = []
    for receipt_path in _sorted_receipts(receipts_dir):
        pairs.append((_load(receipt_path), _load(_att_for(receipts_dir, receipt_path))))
    return pairs


def _register_proof_routes(
    app: Any,
    receipts_dir: Path,
    verifying_material: Any,
    chain_path: Optional[Path],
    crosscheck: "Optional[dict[str, Any]]",
    client: Any,
    judge_factory: Any,
) -> None:
    """The on-demand "prove it harder" routes: TPM chain weld + cross-check."""

    @app.post("/api/verify-chain")
    async def verify_chain() -> Any:
        if chain_path is None:
            return JSONResponse(
                {"available": False, "reason": "no TPM evidence chain configured"}
            )
        from vaara.attestation._inference_chain_verify import verify_inference_chain

        chain_doc = _load(Path(chain_path))
        try:
            v = verify_inference_chain(
                chain_doc,
                receipts=_ordered_pairs(receipts_dir),
                verifying_material=verifying_material,
            )
        except ValueError as exc:  # structurally malformed chain bundle
            return JSONResponse({"available": True, "ok": False, "reason": str(exc)})
        return JSONResponse({
            "available": True,
            "ok": v["ok"],
            "chainTier": v["chainTier"],
            "chainContinuous": v["chainContinuous"],
            "manifestMatchesReceipts": v["manifestMatchesReceipts"],
            "manifestCoversPrefix": v["manifestCoversPrefix"],
            "boundCount": v["boundCount"],
            "unboundTail": v["unboundTail"],
            "receiptsAllValid": v["receiptsAllValid"],
            "nReceipts": v["nReceipts"],
            "reason": v["reason"],
        })

    @app.post("/api/crosscheck")
    async def crosscheck_turn(request: "Request") -> Any:
        if crosscheck is None:
            return JSONResponse(
                {"available": False, "reason": "no verifier identity configured"}
            )
        # The judge is chosen per request: the page sends the model the operator
        # picked in the judge dropdown. ``--judge-model`` is only a default for
        # requests that omit it. An empty selection with no default is the one
        # case we cannot proceed on.
        body = await request.body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        judge_model = (data.get("judge_model") or crosscheck.get("judge_model") or "").strip()
        if not judge_model:
            return JSONResponse(
                {"available": False, "reason": "no judge model selected"}
            )
        turn = app.state.last_turn
        if not turn or turn.get("output") is None:
            return JSONResponse(
                {"available": False, "reason": "no turn to cross-check yet"}
            )
        if turn.get("att_doc") is None:
            return JSONResponse({
                "available": False,
                "reason": "turn has no attestation to bind the cross-check to",
            })
        from vaara.attestation._inference_crosscheck import build_crosscheck
        from vaara.attestation.inference import (
            parse_inference_attestation,
            parse_inference_receipt,
        )

        try:
            record = build_crosscheck(
                attestation=parse_inference_attestation(turn["att_doc"]),
                receipt=parse_inference_receipt(turn["receipt_doc"]),
                messages=turn["messages"],
                response=turn["output"],
                judge=judge_factory(judge_model),
                secret_version=crosscheck["secret_version"],
                alg=crosscheck["alg"],
                signing_material=crosscheck["signing_material"],
            )
        except Exception as exc:
            return JSONResponse({"available": True, "error": str(exc)})
        return JSONResponse({
            "available": True,
            "agreement": record.agreement,
            "diverse": record.diverse,
            "subjectDigest": record.subject_receipt_digest,
            "verifierModel": record.verifier_model.model_ref,
        })

    @app.on_event("shutdown")
    async def _close() -> None:  # pragma: no cover - real wiring
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()


def build_app(
    *,
    proxy_url: str,
    receipts_dir: Path,
    verifying_material: Any = None,
    chain_path: Optional[Path] = None,
    crosscheck: "Optional[dict[str, Any]]" = None,
    recall: Any = None,
    client: Any = None,
    judge_factory: Any = None,
) -> Any:
    """Assemble the console.

    ``proxy_url`` is the signing inference proxy base URL. ``receipts_dir`` is
    where the proxy drops the signed pair files. ``verifying_material`` is the
    proxy's public key (or HS256 secret) used only to *verify*; the console holds
    no signing key for receipts. ``chain_path`` points at the TPM evidence-chain
    bundle (enables the hardware-chain button). ``crosscheck`` is an optional
    ``{judge_model, upstream, signing_material, alg, secret_version}`` enabling
    the second-model button as a verifier party. ``recall`` is an optional
    ``MemoryRecall`` that grounds each turn in the operator's local memory before
    the proxy signs it; grounding the prompt pre-signature keeps the receipt
    honest about what the model actually saw. ``judge_factory`` maps a per-request
    judge-model name to a :class:`Judge`; tests inject a stub, production defaults
    to a CPU-only :class:`OllamaJudge` so a light verifier never evicts the subject
    model from the GPU.
    """
    if client is None:  # pragma: no cover - real wiring, not exercised in tests
        import httpx

        client = httpx.AsyncClient(timeout=None)

    if judge_factory is None:  # pragma: no cover - real wiring, not exercised in tests
        cc = crosscheck or {}
        _upstream = cc.get("upstream", "http://127.0.0.1:11434")
        _num_gpu = cc.get("num_gpu")

        def judge_factory(model: str) -> Any:
            from vaara.attestation._inference_crosscheck import OllamaJudge

            return OllamaJudge(model=model, upstream=_upstream, num_gpu=_num_gpu)

    proxy_url = proxy_url.rstrip("/")
    receipts_dir = Path(receipts_dir).expanduser()
    app = FastAPI(title="vaara-console")
    app.state.last_turn = None

    def _verify_latest(before: int) -> "Optional[dict[str, Any]]":
        """Verify the receipt the just-finished turn emitted."""
        receipts = _sorted_receipts(receipts_dir)
        if not receipts:
            return None
        receipt_path = receipts[-1]
        counter = _counter_of(receipt_path)
        if counter <= before:  # the proxy emitted nothing new (e.g. upstream error)
            return None
        receipt_doc = _load(receipt_path)
        att_doc = _load(_att_for(receipts_dir, receipt_path))
        try:
            checks = _verify_one(receipt_doc, att_doc, verifying_material)
        except Exception as exc:  # malformed pair: report, do not crash the turn
            checks = {"ok": False, "error": str(exc)}
        return {
            "counter": counter,
            "receipt": receipt_path.name,
            "verdict": {k: checks[k] for k in _VERDICT_KEYS if k in checks},
            "receipt_doc": receipt_doc,
            "att_doc": att_doc,
        }

    def _public_turn(turn: "Optional[dict[str, Any]]") -> "dict[str, Any]":
        if not turn:
            return {"available": False}
        return {
            "available": True,
            "counter": turn["counter"],
            "receipt": turn["receipt"],
            "verdict": turn["verdict"],
            "grounded": turn.get("grounded", 0),
        }

    def _before_counter() -> int:
        receipts = _sorted_receipts(receipts_dir)
        return _counter_of(receipts[-1]) if receipts else -1

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Any:
        return HTMLResponse(CONSOLE_HTML.replace("__WORDMARK__", _wordmark_data_uri()))

    @app.get("/api/config")
    async def config() -> Any:
        """What the page needs to drive the stack: the local models available and
        whether memory grounding is wired. Model names come from the upstream
        ``/api/tags`` passed through the proxy; a down proxy yields an empty list
        rather than an error, so the page degrades to a free-text model field."""
        models: "list[str]" = []
        try:
            resp = await client.get(f"{proxy_url}/api/tags")
            for m in (resp.json().get("models") or []):
                name = m.get("name") or m.get("model")
                if name:
                    models.append(name)
        except Exception:
            models = []
        return JSONResponse({
            "models": models,
            "recall": recall is not None,
            "crosscheck": crosscheck is not None,
            "judgeDefault": (crosscheck or {}).get("judge_model") or "",
        })

    @app.post("/api/chat")
    async def chat(request: "Request") -> Any:
        body = await request.body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        messages = data.get("messages")
        stream = bool(data.get("stream"))

        # Ground in local memory before the proxy signs, so the receipt covers
        # the prompt the model actually saw. Off unless a recall engine is wired
        # and the client did not opt out for this turn.
        n_grounded = 0
        if recall is not None and data.get("ground", True):
            grounded, n_grounded = ground_messages(messages, recall)
            if n_grounded:
                messages = grounded
                data = {**data, "messages": grounded}
                body = json.dumps(data).encode()

        before = _before_counter()
        url = f"{proxy_url}/api/chat"
        headers = {"content-type": "application/json"}

        def _stash(turn: "Optional[dict[str, Any]]", output: Any) -> None:
            if turn is not None:
                turn["messages"] = messages
                turn["output"] = output or {"content": ""}
                turn["grounded"] = n_grounded
                app.state.last_turn = turn

        if not stream:
            resp = await client.post(url, content=body, headers=headers)
            try:
                output = parse_ollama_response(resp.json())[0]
            except Exception:  # non-JSON / upstream error body
                output = {"content": ""}
            turn = _verify_latest(before)
            _stash(turn, output)
            return JSONResponse({"message": output, "turn": _public_turn(turn)})

        acc = StreamAccumulator(True)

        async def _tee() -> Any:
            async with client.stream(
                "POST", url, content=body, headers=headers
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    acc.feed(chunk)
                    yield chunk
            output, _ = acc.finalize()
            _stash(_verify_latest(before), output)

        return StreamingResponse(_tee(), media_type="application/x-ndjson")

    @app.get("/api/turn/latest")
    async def turn_latest() -> Any:
        return JSONResponse(_public_turn(app.state.last_turn))

    _register_proof_routes(
        app, receipts_dir, verifying_material, chain_path, crosscheck, client,
        judge_factory,
    )
    return app
