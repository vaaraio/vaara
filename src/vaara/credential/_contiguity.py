"""Gap-evident completeness over a stream of authorization receipts.

The per-boundary sequence assigned at issuance is contiguous by construction
(see ``vaara.integrations._mcp_attest.AttestPairEmitter._next_completeness``).
This module checks that property after the fact, over the evidence records a
third party holds, with no access to the issuer and no external witness: a
missing ``seq`` is a provable gap, and the highest ``runningCount`` makes a
short set self-evidently incomplete.

The input is a list of ``vaara.authorization/v0`` evidence records (the
``evidence`` half of each persisted authorization receipt). Records without a
``completeness`` block are ignored: completeness is opt-in, and a stream that
never asserted a sequence has nothing to be incomplete against.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ContiguityReport:
    """The result of checking one coverage boundary for gaps.

    ``ok`` is true only when the held records form a contiguous ``0..expected-1``
    run with consistent running counts and no duplicates. ``missing_seqs`` names
    the gaps; ``expected`` is the count the latest receipt asserts exists, so a
    short set reports the tail it is missing.

    ``worst_case_class`` is the ``maxClass`` a sealing record carried, when one
    did: the highest action class the boundary authorized. A dropped record took
    its own contents with it, so the most a gap could have hidden is an action of
    this class. It lets a holder bound a gap's worst case from the held set
    alone, with no issuer. ``None`` when no seal named a class.
    """

    boundary_id: str
    present: int
    expected: int
    missing_seqs: list[int]
    duplicate_seqs: list[int]
    count_mismatches: list[dict[str, int]]
    ok: bool
    worst_case_class: Optional[str] = None

    def gap_report(self) -> str:
        """A short human-readable verdict naming any gaps."""
        if self.ok:
            if self.expected == 0:
                return f"boundary {self.boundary_id!r}: no completeness asserted"
            return (
                f"boundary {self.boundary_id!r}: complete, {self.present} receipt(s), "
                f"seq 0..{self.expected - 1} contiguous"
            )
        lines = [f"boundary {self.boundary_id!r}: INCOMPLETE"]
        lines.append(
            f"  asserted {self.expected}, hold {self.present}"
        )
        if self.missing_seqs:
            lines.append(f"  missing seq: {_compact(self.missing_seqs)}")
            if self.worst_case_class is not None:
                lines.append(
                    f"  gap worst-case: action class up to "
                    f"{self.worst_case_class!r} may be unrecorded"
                )
        if self.duplicate_seqs:
            lines.append(f"  duplicate seq: {_compact(self.duplicate_seqs)}")
        for mm in self.count_mismatches:
            lines.append(
                f"  count mismatch at seq {mm['seq']}: "
                f"runningCount={mm['runningCount']}, expected {mm['seq'] + 1}"
            )
        return "\n".join(lines)


def _compact(seqs: list[int]) -> str:
    """Render a sorted int list as comma-separated ranges, e.g. ``3, 5-7``."""
    if not seqs:
        return ""
    out: list[str] = []
    start = prev = seqs[0]
    for n in seqs[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(out)


def _completeness_records(
    evidence_records: list[dict[str, Any]], boundary_id: Optional[str]
) -> tuple[str, list[dict[str, Any]]]:
    """Select the completeness blocks for one boundary and resolve its id.

    Raises ``ValueError`` if no boundary is given and the records span more than
    one, since the caller must say which boundary they mean.
    """
    blocks = [
        ev["completeness"]
        for ev in evidence_records
        if isinstance(ev, dict) and isinstance(ev.get("completeness"), dict)
    ]
    if boundary_id is not None:
        return boundary_id, [b for b in blocks if b.get("boundaryId") == boundary_id]
    boundaries = {b.get("boundaryId") for b in blocks}
    boundaries.discard(None)
    if len(boundaries) > 1:
        raise ValueError(
            "records span multiple boundaries "
            f"({sorted(str(b) for b in boundaries)}); pass boundary_id"
        )
    resolved = next(iter(boundaries)) if boundaries else ""
    return str(resolved), blocks


def verify_contiguity(
    evidence_records: list[dict[str, Any]],
    boundary_id: Optional[str] = None,
) -> ContiguityReport:
    """Check a set of authorization-evidence records for gaps under one boundary.

    ``evidence_records`` is the ``evidence`` half of each authorization receipt.
    With ``boundary_id`` omitted, the boundary is inferred when the records name
    exactly one. The check is pure and offline: it needs only the held records.

    A receipt is missing when its ``seq`` does not appear and the latest
    ``runningCount`` says it should. ``runningCount`` must equal ``seq + 1`` for
    every held record, or the count itself is inconsistent. The report is
    self-contained evidence of completeness or of the specific receipts absent.

    An optional sealing block (``{"sealed": True, "total": N}``) finalizes a run
    and lets a dropped tail be caught; without it, dropping the last record(s)
    is invisible from the held set alone. See the inline note below. The seal may
    also carry ``maxClass`` (the highest action class the boundary authorized),
    surfaced as ``worst_case_class`` so a gap's worst case is read held-set-alone.
    """
    boundary, blocks = _completeness_records(evidence_records, boundary_id)
    # A sealing block (``{"sealed": True, "total": N}``) pins the run length
    # independently of the held decision records. It is optional and additive: a
    # stream carrying only ``{seq, runningCount}`` blocks behaves exactly as
    # before. When a seal is present, a dropped tail still shows as missing even
    # though the dropped records took their own ``seq`` with them. A suffix that
    # also suppresses the seal stays outside held-set-alone detection; an
    # external anchor (e.g. an rfc3161 timestamp over the run) closes that.
    seq_blocks = [b for b in blocks if not b.get("sealed")]
    sealed_total = max(
        (int(b["total"]) for b in blocks if b.get("sealed")), default=0
    )
    # The seal may name the boundary's highest action class. A gap could have
    # hidden an action of at most this class, so it bounds the worst case from
    # the held set alone. Class labels carry no generic ordering here, so we
    # surface the first one a seal asserts (one seal per boundary in practice).
    worst_case_class = next(
        (
            str(b["maxClass"])
            for b in blocks
            if b.get("sealed") and b.get("maxClass") is not None
        ),
        None,
    )

    if not seq_blocks:
        # Nothing but a possible seal. A seal asserting N over zero held records
        # is a fully-dropped run; no seal and no records is a no-op.
        return ContiguityReport(
            boundary_id=boundary,
            present=0,
            expected=sealed_total,
            missing_seqs=list(range(sealed_total)),
            duplicate_seqs=[],
            count_mismatches=[],
            ok=(sealed_total == 0),
            worst_case_class=worst_case_class,
        )

    seqs = [int(b["seq"]) for b in seq_blocks]
    counts = [int(b["runningCount"]) for b in seq_blocks]
    max_seq = max(seqs)
    max_running = max(counts)
    # What the stream claims exists: the furthest seq, a runningCount that
    # outruns it (a later receipt was dropped, taking its own seq with it), or
    # the sealed total when the run was finalized.
    expected = max(max_seq + 1, max_running, sealed_total)

    seq_counts = Counter(seqs)
    duplicate_seqs = sorted(s for s, n in seq_counts.items() if n > 1)
    missing_seqs = sorted(set(range(expected)) - set(seqs))
    count_mismatches = [
        {"seq": int(b["seq"]), "runningCount": int(b["runningCount"])}
        for b in seq_blocks
        if int(b["runningCount"]) != int(b["seq"]) + 1
    ]

    present = len(seq_blocks)
    ok = (
        not missing_seqs
        and not duplicate_seqs
        and not count_mismatches
        and present == expected
    )
    return ContiguityReport(
        boundary_id=boundary,
        present=present,
        expected=expected,
        missing_seqs=missing_seqs,
        duplicate_seqs=duplicate_seqs,
        count_mismatches=count_mismatches,
        ok=ok,
        worst_case_class=worst_case_class,
    )
