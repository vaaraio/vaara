# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Four public lines say what the record does not do, as well as what it does.

Each line once ran past its own limit:

* the capability index called the five witness roots "independent", but the
  self-hosted timestamp is the operator's own and TPM and confidential-VM
  attestation root in the platform;
* two trail pages said deleting a record, or removing it from the run, breaks
  the chain. A record cut off the end breaks no hash: the shorter trail still
  verifies, and the cut shows only against a seal or an anchor that covers it;
* the capability index said "a read is an event", but a read is recorded only
  when the deployer's code calls ``record_access``;
* the access-record docstring said the anchored time is attested by a party
  outside the operator, which holds only when the authority is not operator-run.
"""
from __future__ import annotations

import re
from pathlib import Path

import vaara.audit.access as access

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _flat(text: str) -> str:
    # The pages wrap lines; compare on one line of single spaces.
    return re.sub(r"\s+", " ", text)


def _page(name: str) -> str:
    return _flat((DOCS / name).read_text(encoding="utf-8"))


def _paragraph(name: str, starts_with: str) -> str:
    for block in (DOCS / name).read_text(encoding="utf-8").split("\n\n"):
        if block.lstrip().startswith(starts_with):
            return _flat(block)
    raise AssertionError(f"docs/{name} has no paragraph starting {starts_with!r}")


def _capabilities_row(first_cell: str) -> str:
    for line in (DOCS / "capabilities.md").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| {first_cell} |"):
            return line
    raise AssertionError(f"docs/capabilities.md has no row for {first_cell}")


# -- the witness roots ------------------------------------------------------


def test_the_roots_are_not_called_independent():
    assert "independent roots of trust" not in _page("capabilities.md")


def test_the_summary_names_which_roots_sit_outside_the_operator():
    summary = _paragraph("capabilities.md", "Vaara governs every agent tool call")
    assert "up to five roots of trust" in summary
    assert "outside witnesses" in summary
    assert "qualified timestamp authority" in summary
    assert "public transparency log" in summary


def test_the_witness_section_says_who_runs_each_kind_of_root():
    intro = _paragraph("capabilities.md", "The same record can be anchored")
    assert "run by someone else" in intro
    assert "self-hosted" in intro
    assert "operator" in intro
    assert "TPM and confidential-VM attestation" in intro


# -- the hash chain and a removed tail --------------------------------------


def test_the_chain_paragraph_says_a_removed_tail_breaks_no_hash():
    para = _paragraph("tamper-evident-audit-trail.md", "**Hash chain.**")
    assert "from the middle" in para
    assert "from the end breaks no hash" in para
    assert "seal" in para and "anchor" in para
    assert '"the log is complete"' not in para


def test_deleting_a_record_is_not_said_to_always_break_the_chain():
    text = _page("tamper-evident-audit-trail.md")
    assert "Deleting a record breaks the chain the same way" not in text


def test_the_proof_page_limits_the_delete_claim_to_records_with_successors():
    para = _paragraph("prove-what-an-ai-agent-did.md", "**Hash-chained.**")
    assert "deleting any record breaks every hash after it" not in para
    assert "deleting one that has records after it" in para
    assert "shorter chain that still verifies" in para
    assert "A missing entry becomes a provable gap" not in para


def test_the_four_questions_say_inside_the_run():
    text = _page("prove-what-an-ai-agent-did.md")
    assert "nothing was removed from the run" not in text
    assert "nothing inside the run was removed" in text


# -- the access record ------------------------------------------------------


def test_a_read_is_not_said_to_be_an_event_by_itself():
    row = _capabilities_row("Access records")
    assert "A read is an event" not in row
    assert "calls `record_access`" in row
    assert "encrypted retention archive" in row


def test_the_access_docstring_says_the_time_is_outside_only_when_the_authority_is():
    doc = _flat(access.__doc__ or "")
    assert "attested by a party outside the operator rather than" not in doc
    assert "not run by the operator" in doc
    assert "self-hosted" in doc
