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
from typing import TYPE_CHECKING, Optional, Sequence, Union

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


def _member_base(name: str) -> str:
    """A safe zip-member base from a caller-supplied attachment name.

    Strips any directory components so a hostile name cannot fold a file
    outside ``evidence/``. The result is used to build ``evidence/.../<base>``
    paths; an empty result is rejected by the caller.
    """
    return Path(str(name)).name


def _prepare_folded_evidence(
    handoffs: "Optional[Sequence[tuple[str, dict, Optional[str]]]]",
    enforcements: "Optional[Sequence[tuple[str, dict, bytes, bytes]]]",
    *,
    trusted_did_document: Optional[dict],
    expected_measurement: Optional[str],
) -> "tuple[Optional[dict], Optional[dict], list[tuple[str, bytes]]]":
    """Verify the optional SEP-2828 attachments and stage them for folding.

    Returns ``(report_attestations, summary_doc, files)``:

    * ``report_attestations`` is the counts-only roll-up handed to the report
      builder (``None`` when nothing is folded).
    * ``summary_doc`` is the self-describing ``attestations_summary.json`` body,
      carrying the full set verdicts plus the verifier-side inputs (per-package
      anchor times, the trusted DID document, the expected measurement) so the
      folded evidence re-verifies offline from the package alone.
    * ``files`` are ``(zip_member, bytes)`` pairs to fold under ``evidence/``.

    Fails closed: every attachment is verified here in default mode, and a
    single attachment that does not verify raises :class:`ValueError` naming the
    offender. The package never ships evidence it cannot back. Needs the
    attestation extra; a missing extra raises :class:`ImportError`.
    """
    if not handoffs and not enforcements:
        return None, None, []

    from vaara.attestation.receipt import (
        check_enforcement_set,
        check_handoff_set,
    )

    report_attestations: dict = {}
    summary: dict = {
        "schema": "vaara-article12-attestations",
        "schemaVersion": 1,
        "handoff": {"present": False},
        "enforcement": {"present": False},
    }
    files: list[tuple[str, bytes]] = []
    seen: set[str] = set()

    def _fold(member: str, body: bytes) -> None:
        if member in seen:
            raise ValueError(f"duplicate folded evidence path: {member!r}")
        seen.add(member)
        files.append((member, body))

    if handoffs:
        h_report = check_handoff_set(
            handoffs, trusted_did_document=trusted_did_document, strict=False,
        )
        if not h_report.ok:
            bad = [
                f"{e.name} ({e.error or 'did not verify'})"
                for e in h_report.entries if not e.ok
            ]
            raise ValueError(
                "handoff attachment(s) failed verification, refusing to fold: "
                + "; ".join(bad)
            )
        anchored_times: dict[str, Optional[str]] = {}
        for name, doc, anchor_time in handoffs:
            base = _member_base(name)
            if base.endswith(".json"):
                base = base[: -len(".json")]
            if not base:
                raise ValueError(f"empty handoff attachment name: {name!r}")
            _fold(
                f"evidence/handoff/{base}.json",
                json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"),
            )
            # Key by the folded base, not the raw name, so the anchor time maps
            # to the file an offline reader sees in the zip.
            anchored_times[base] = anchor_time
        summary["handoff"] = {
            "present": True,
            "strict": h_report.strict,
            "trustedDidDocument": trusted_did_document,
            "anchoredTimes": anchored_times,
            "report": h_report.to_dict(),
        }
        report_attestations["handoff"] = {
            "total": h_report.total,
            "passed": h_report.passed,
            "verifiable": h_report.verifiable,
            "corroborated": h_report.corroborated,
            "pinned": h_report.pinned,
            "pinning_gap": h_report.pinning_gap,
            "strict": h_report.strict,
        }

    if enforcements:
        e_report = check_enforcement_set(
            enforcements, expected_measurement=expected_measurement, strict=False,
        )
        if not e_report.ok:
            bad = [
                f"{e.name} ({e.error or 'did not verify'})"
                for e in e_report.entries if not e.ok
            ]
            raise ValueError(
                "enforcement attachment(s) failed verification, refusing to "
                "fold: " + "; ".join(bad)
            )
        for name, record, report_bytes, vcek_pem in enforcements:
            base = _member_base(name)
            if not base:
                raise ValueError(f"empty enforcement attachment name: {name!r}")
            _fold(
                f"evidence/enforcement/{base}.record.json",
                json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
            )
            _fold(f"evidence/enforcement/{base}.report.bin", report_bytes)
            _fold(f"evidence/enforcement/{base}.vcek.pem", vcek_pem)
        summary["enforcement"] = {
            "present": True,
            "strict": e_report.strict,
            "expectedMeasurement": expected_measurement,
            "report": e_report.to_dict(),
        }
        report_attestations["enforcement"] = {
            "total": e_report.total,
            "passed": e_report.passed,
            "bound": e_report.bound,
            "measurement_pinned": e_report.measurement_pinned,
            "tier_counts": dict(e_report.tier_counts),
            "pinning_gap": e_report.pinning_gap,
            "strict": e_report.strict,
        }

    return report_attestations, summary, files


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
    handoffs: "Optional[Sequence[tuple[str, dict, Optional[str]]]]" = None,
    enforcements: "Optional[Sequence[tuple[str, dict, bytes, bytes]]]" = None,
    trusted_did_document: Optional[dict] = None,
    expected_measurement: Optional[str] = None,
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

    ``handoffs`` and ``enforcements`` fold verified SEP-2828 evidence in as
    sidecars under ``evidence/``: cross-org handoff packages (Article 26(6)
    deployer custody) as ``(name, document, anchor_attested_time)`` tuples, and
    confidential-VM enforcement bindings as ``(name, record, report_bytes,
    vcek_pem)`` tuples, mirroring ``check_handoff_set`` / ``check_enforcement_set``.
    ``trusted_did_document`` pins handoff producer identity out of band;
    ``expected_measurement`` pins enforcement launch images. Each attachment is
    verified at export in default mode and a single one that does not verify
    raises :class:`ValueError`: the package never ships evidence it cannot back.
    The roll-up is written as ``evidence/attestations_summary.json`` and
    surfaced in the report; folding these needs the attestation extra. The
    eIDAS anchor stays the only un-forgeable component, and the report says so.

    Returns the underlying :class:`ExportResult` (its ``manifest`` covers the
    signed core; the Article 12 files are folded in afterwards).
    """
    if fmt not in ("md", "html"):
        raise ValueError(f"fmt must be 'md' or 'html', got {fmt!r}")
    out_path = Path(out_path)

    # Verify the optional SEP-2828 attachments up front, before any bytes are
    # written, so a bad attachment fails closed without leaving a partial zip.
    report_attestations, attestations_summary, folded_files = (
        _prepare_folded_evidence(
            handoffs, enforcements,
            trusted_did_document=trusted_did_document,
            expected_measurement=expected_measurement,
        )
    )

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
        time_anchor=anchor_dict, attestations=report_attestations,
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
        if attestations_summary is not None:
            zf.writestr(
                "evidence/attestations_summary.json",
                json.dumps(
                    attestations_summary, indent=2, sort_keys=True
                ).encode("utf-8"),
            )
            for member, body in folded_files:
                zf.writestr(member, body)

    return result
