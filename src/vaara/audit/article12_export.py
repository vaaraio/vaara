"""EU AI Act Article 12 one-command regulator export.

Composes the existing signed-export path with a generated Article 12
record-keeping report, packaged in one zip. See
``docs/design/article12-export-spec.md`` and ``article12_report.py``.

The signed core (trail.jsonl, manifest.json, signature, pubkey) is written
first by ``export_signed`` / ``export_signed_threshold``. The report is then
built from the *signed* trail bytes read back out of the zip, so it always
describes exactly the trail that was signed, and folded into the zip:

    article12_report.md      (or .html)
    article12_summary.json
    verify_instructions.txt
    time_anchor.json         (when an external time anchor is supplied)

The report is bound to the signed trail by the manifest ``trail_sha256`` it
records. It is not itself signed; the verify instructions say so plainly. An
optional RFC 3161 time anchor over the signed trail head adds Article 19
existence-in-time evidence that holds independently of the signing key.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from vaara.audit.timeanchor import TimeAnchor

from vaara.audit.article12_report import (
    build_article12_report,
    render_report_html,
    render_report_md,
    verify_instructions_text,
)
from vaara.audit.export import (
    ExportResult,
    export_signed,
    export_signed_threshold,
)
from vaara.audit.trail import AuditRecord


def _records_from_zip(zip_path: Path) -> tuple[list[AuditRecord], dict]:
    """Read ``trail.jsonl`` + ``manifest.json`` back out of the signed zip.

    Reading the trail from the just-written zip (rather than re-snapshotting
    the live trail) guarantees the report describes exactly the bytes that
    were signed: same records, same ``trail_sha256``.
    """
    records: list[AuditRecord] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        trail_bytes = zf.read("trail.jsonl")
        manifest = json.loads(zf.read("manifest.json"))
    for raw in trail_bytes.splitlines():
        line = raw.strip()
        if not line:
            continue
        records.append(AuditRecord.from_dict(json.loads(line)))
    return records, manifest


def export_article12(
    trail,
    out_path: Union[str, Path],
    *,
    signer_key=None,
    signer=None,
    signers: Optional[list] = None,
    threshold_k: Optional[int] = None,
    system_meta: Optional[dict] = None,
    period: Optional[tuple] = None,
    time_anchor: "Optional[TimeAnchor]" = None,
    fmt: str = "md",
    agent_id: str = "",
) -> ExportResult:
    """Write a signed Article 12 regulator package.

    The signer arguments mirror the underlying export functions: pass
    ``signer_key`` or ``signer`` for the single-signer path, or ``signers``
    plus ``threshold_k`` for the k-of-n threshold path. ``system_meta`` and
    ``period`` are passed through to the report builder; ``fmt`` selects
    ``"md"`` or ``"html"`` for the human-readable report.

    ``time_anchor`` is an optional
    :class:`~vaara.audit.timeanchor.TimeAnchor` over the signed trail head. It
    must anchor the final record (the chain head); the binding and the RFC 3161
    token are verified here, and a mismatch raises. When present it is folded
    in as ``time_anchor.json`` and surfaced in the report as Article 19
    existence-in-time evidence, independent of the signing key.

    Returns the underlying :class:`ExportResult` (its ``manifest`` covers the
    signed core; the Article 12 files are folded in afterwards).
    """
    if fmt not in ("md", "html"):
        raise ValueError(f"fmt must be 'md' or 'html', got {fmt!r}")
    out_path = Path(out_path)

    threshold_mode = signers is not None or threshold_k is not None
    if threshold_mode:
        if not signers or threshold_k is None:
            raise ValueError(
                "threshold export needs both `signers` and `threshold_k`"
            )
        result = export_signed_threshold(
            trail, out_path, signers=signers,
            threshold_k=threshold_k, agent_id=agent_id,
        )
    else:
        result = export_signed(
            trail, out_path, signer_key=signer_key,
            signer=signer, agent_id=agent_id,
        )

    # Build the report from the trail bytes that were actually signed.
    records, manifest = _records_from_zip(out_path)

    # An optional external time anchor must bind to the head of the trail we
    # just signed. Verify both the chain-head binding and the RFC 3161 token
    # before folding it in, so a package never claims an anchor it cannot back.
    anchor_dict: Optional[dict] = None
    if time_anchor is not None:
        from vaara.audit.timeanchor import verify_anchor_over_records

        head_position = len(records) - 1
        if time_anchor.chain_position != head_position:
            raise ValueError(
                "time_anchor must anchor the trail head (position "
                f"{head_position}), got {time_anchor.chain_position}"
            )
        # Raises TimeAnchorError if the anchor refers to a different chain or
        # the token does not verify over its attested digest.
        verify_anchor_over_records(time_anchor, [r.record_hash for r in records])
        anchor_dict = time_anchor.to_dict()

    report = build_article12_report(
        records, manifest, system_meta=system_meta, period=period,
        time_anchor=anchor_dict,
    )

    report_name = f"article12_report.{fmt}"
    report_body = (
        render_report_html(report) if fmt == "html" else render_report_md(report)
    )
    summary_bytes = json.dumps(report, indent=2, sort_keys=False).encode("utf-8")
    instructions = verify_instructions_text(report)

    with zipfile.ZipFile(out_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(report_name, report_body)
        zf.writestr("article12_summary.json", summary_bytes)
        zf.writestr("verify_instructions.txt", instructions + "\n")
        if anchor_dict is not None:
            zf.writestr(
                "time_anchor.json",
                json.dumps(anchor_dict, indent=2, sort_keys=True).encode("utf-8"),
            )

    return result
