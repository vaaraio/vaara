# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The doors the reachability graph reads out of a document.

examples/langchain_agent.py imports ``InterceptionPipeline`` on one line
and ``TraceGenerator`` on the next. The import pattern's name class
included whitespace, so the first match ran on through the newline and
ate the ``from vaara`` of the second statement: ``TraceGenerator`` was
reported as reachable from nothing, and the whole cold-start calibration
module was classified for deletion. Every document that lists its vaara
imports one under the other lost every second one the same way.
"""

from __future__ import annotations

from tests.reachability import _DOC_IMPORT, _imported_names


def _doors(text: str) -> list[tuple[str, str]]:
    out = []
    for match in _DOC_IMPORT.finditer(text):
        for name in _imported_names(match.group(2) or match.group(3) or ""):
            out.append((match.group(1), name))
    return out


def test_consecutive_import_lines_each_open_a_door():
    text = (
        "from vaara.pipeline import InterceptionPipeline\n"
        "from vaara.sandbox.trace_gen import TraceGenerator\n"
        "from vaara.taxonomy.actions import (\n"
        "    ActionCategory,\n"
        "    ActionType,\n"
        ")\n"
    )
    assert _doors(text) == [
        ("vaara.pipeline", "InterceptionPipeline"),
        ("vaara.sandbox.trace_gen", "TraceGenerator"),
        ("vaara.taxonomy.actions", "ActionCategory"),
        ("vaara.taxonomy.actions", "ActionType"),
    ]


def test_aliases_and_trailing_comments_bind_the_imported_name():
    text = "from vaara.audit.access import record_access as read, find_accesses  # the door\n"
    assert _doors(text) == [
        ("vaara.audit.access", "record_access"),
        ("vaara.audit.access", "find_accesses"),
    ]


def test_a_conditional_export_is_still_an_export():
    """vaara.scorer adds the trained-gate names to ``__all__`` with
    ``__all__.extend([...])`` under ``if _TRAINED_GATE_AVAILABLE``. The graph
    read only the assignment, so every name an optional extra exports was a
    definition with no door."""
    from tests.reachability import Graph

    exports = Graph().modules["vaara.scorer"].exports
    assert "AdaptiveScorer" in exports
    assert "create_stacked_scorer" in exports
