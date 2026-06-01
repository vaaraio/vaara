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

The report is bound to the signed trail by the manifest ``trail_sha256`` it
records. It is not itself signed; the verify instructions say so plainly.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional, Union

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
    fmt: str = "md",
    agent_id: str = "",
) -> ExportResult:
    """Write a signed Article 12 regulator package.

    The signer arguments mirror the underlying export functions: pass
    ``signer_key`` or ``signer`` for the single-signer path, or ``signers``
    plus ``threshold_k`` for the k-of-n threshold path. ``system_meta`` and
    ``period`` are passed through to the report builder; ``fmt`` selects
    ``"md"`` or ``"html"`` for the human-readable report.

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
    report = build_article12_report(
        records, manifest, system_meta=system_meta, period=period,
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

    return result
