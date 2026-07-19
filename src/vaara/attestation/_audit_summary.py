# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one-page audit summary: the record-set verdict a regulator reads.

``check_record_set`` answers the machine question and ``verify-records`` prints
a terse roll-up. A regulator wants the same facts as a page of plain prose:
what was checked, how many records conform, where the gaps are, and why the
answer can be trusted without a key. ``render_record_set_summary`` turns the
report into that page.

The page is a derived view of the set check, not new evidence. It states only
what the report already found and adds no judgement of its own. Every count and
finding traces back to a record any party can re-run with ``verify-records``,
so the summary is reproducible from the records alone. It carries no timestamp
and no key: the same set always renders the same page.

Markdown, pure standard library, importable without the attestation extra.
"""

from __future__ import annotations

from vaara.attestation._record_set_conformance import RecordSetReport

SUMMARY_SCHEMA = "vaara-record-set-summary/1"

_DEFAULT_TITLE = "Execution-record conformance summary"

_WHAT = (
    "Each record was checked against the SEP-2828 execution-record schema for "
    "its type, decision or outcome, with no signing key: the wire shape and the "
    "record's own self-proving digest. Across the set, calls were checked for "
    "duplicates and decisions were paired with their outcomes."
)

_KEYLESS = (
    "No signing key was used to produce this summary. Any party can reproduce "
    "every count and finding above from the records alone with "
    "`vaara verify-records`."
)


def render_record_set_summary(
    report: RecordSetReport, *, title: str = _DEFAULT_TITLE
) -> str:
    """Render a record-set report as a one-page Markdown audit summary.

    Deterministic: the page depends only on ``report`` (no clock, no version),
    so the same set renders byte-identical every time. The verdict line, the
    counts, and the findings restate the report; nothing is inferred beyond it.
    """
    lines: list[str] = [f"# {title}", ""]

    verdict = "CONFORMS" if report.conforms else "NON-CONFORMING"
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    if report.total == 0:
        lines.append("No records were supplied.")
    else:
        lines.append(
            f"{report.total} record{_s(report.total)} checked, "
            f"{report.conforming} conform{'s' if report.conforming == 1 else ''}."
        )
    lines.append("")

    lines.append("## What was checked")
    lines.append("")
    lines.append(_WHAT)
    lines.append("")

    if report.verdict_counts or report.status_counts:
        lines.append("## Records")
        lines.append("")
        if report.verdict_counts:
            lines.append(f"- Decisions: {_tally(report.verdict_counts)}")
        if report.status_counts:
            lines.append(f"- Outcomes: {_tally(report.status_counts)}")
        lines.append("")

    required = [f for f in report.findings if f.severity == "required"]
    advisory = [f for f in report.findings if f.severity == "advisory"]
    lines.append("## Findings")
    lines.append("")
    if not report.findings:
        lines.append("No cross-record findings.")
        lines.append("")
    else:
        if required:
            lines.append("Required (these gate conformance):")
            lines.append("")
            for f in required:
                lines.append(f"- **{f.id}**: {f.detail} ({_names(f.records)})")
            lines.append("")
        if advisory:
            lines.append("Advisory (gaps that do not gate conformance):")
            lines.append("")
            for f in advisory:
                lines.append(f"- **{f.id}**: {f.detail} ({_names(f.records)})")
            lines.append("")

    nonconforming = [e for e in report.entries if not e.conforms]
    if nonconforming:
        lines.append("## Non-conforming records")
        lines.append("")
        for e in nonconforming:
            why = ", ".join(e.required_failed) if e.required_failed else "did not conform"
            lines.append(f"- `{e.name}` ({e.kind}): {why}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(_KEYLESS)
    lines.append("")
    return "\n".join(lines)


def _s(n: int) -> str:
    return "" if n == 1 else "s"


def _tally(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _names(records: tuple[str, ...]) -> str:
    return ", ".join(records)
