#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate the conformance-statement golden vectors from the live corpus.

The ``conformance_statement_v0`` vectors pin what ``vaara conformance-statement``
produces when run against the published corpus under ``conformance/sep2828`` and
the emitter records committed beside them. Because the statement names the exact
corpus byte set (version plus corpusDigest), the goldens move whenever the
corpus does; this script regenerates them so the test ``test_conformance_statement``
stays a drift guard rather than a chore.

It writes, for three scenarios, the structured statement (``expected.json``) and
the rendered Markdown page (``pages/<scenario>.md``):

* ``selftest_only`` - no emitter records, just the corpus self-test.
* ``clean`` - emitter records that conform.
* ``flawed`` - emitter records with one non-conforming record.
* ``duplicate`` - records that each conform but fail a required set property
  (two outcomes pin one call), so the set does not conform.
* ``unproved`` - the clean records plus one file that will not parse. Nothing
  read disagrees with the spec, and the set check never saw the third file, so
  the statement grades ``unproved`` rather than ``false``. This is the case a
  boolean cannot express.

Run: ``python scripts/build_conformance_statement_vectors.py``. Commit the
result; the test fails if the committed goldens drift from this output or from
the independent re-derivation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vaara.attestation.receipt import (
    build_conformance_statement,
    render_conformance_statement,
)

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "conformance" / "sep2828"
VECTORS = REPO / "tests" / "vectors" / "conformance_statement_v0"
EMITTER = VECTORS / "emitter_records"
PAGES = VECTORS / "pages"

SCENARIOS = ["selftest_only", "clean", "flawed", "duplicate", "unproved"]


def _load_records(
    scenario: str,
) -> tuple[list[tuple[str, object]] | None, list[tuple[str, str]]]:
    """Read a scenario's records the way the CLI reads a real directory.

    A file that will not parse is reported as unreadable rather than dropped, so
    the statement can say the check never saw it instead of quietly narrowing
    the set it claims to have checked.
    """
    if scenario == "selftest_only":
        return None, []
    records: list[tuple[str, object]] = []
    unreadable: list[tuple[str, str]] = []
    for path in sorted((EMITTER / scenario).glob("*.json")):
        try:
            records.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path.name, type(exc).__name__))
    return records, unreadable


def build() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    expected: dict[str, object] = {}
    for scenario in SCENARIOS:
        records, unreadable = _load_records(scenario)
        statement = build_conformance_statement(
            CORPUS, records=records, unreadable=unreadable
        )
        expected[scenario] = statement.to_dict()
        (PAGES / f"{scenario}.md").write_text(
            render_conformance_statement(statement), encoding="utf-8"
        )
    text = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    (VECTORS / "expected.json").write_text(text, encoding="utf-8")
    print(f"wrote {len(SCENARIOS)} scenarios to {VECTORS.relative_to(REPO)}")


if __name__ == "__main__":
    build()
    sys.exit(0)
