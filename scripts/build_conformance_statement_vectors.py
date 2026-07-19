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

SCENARIOS = ["selftest_only", "clean", "flawed", "duplicate"]


def _load_records(scenario: str) -> list[tuple[str, object]] | None:
    if scenario == "selftest_only":
        return None
    paths = sorted((EMITTER / scenario).glob("*.json"))
    return [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def build() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    expected: dict[str, object] = {}
    for scenario in SCENARIOS:
        statement = build_conformance_statement(CORPUS, records=_load_records(scenario))
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
