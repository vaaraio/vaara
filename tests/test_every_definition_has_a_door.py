# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every function and class in src/vaara has a door a user can open.

``ExternalScorer`` and ``CompositeScorer`` shipped in v0.14.0 and were tested
for five months, each on its own. Neither was ever reached from a command,
a hook, a server or an import any document shows, so nobody saw that the
pipeline read ``action`` from a scorer that answered ``decision``: every call
through either was denied. ``AuditTrail.enable_auto_publish`` was documented
as a configurable cadence and nothing could switch it on. A symbol only a
test calls is the shape both hid in.

So reachability is computed from the doors (see tests/reachability.py): the
console scripts, ``python -m`` modules, ``scripts/``, the plugin hooks, the
public names of packages and of modules a document names, and the imports
README.md, docs/ and examples/ show a reader. A definition no door reaches is either dead, and goes, or a library
capability with no documented way in, and gets one in docs/library.md.

The allow-list is for definitions a framework or the type system uses in a
way static resolution cannot follow. Each entry says which.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

pytest.importorskip("tomllib", reason="tomllib is Python 3.11+")

from tests.reachability import Graph  # noqa: E402

#: id -> why nothing in the tree has to name it.
ALLOWED: dict[str, str] = {
    f"vaara.integrations.langchain:VaaraCallbackHandler.{method}": (
        "LangChain calls it on the handler; the handler is duck-typed so the "
        "framework stays optional, and the graph only follows callbacks on a "
        "subclass of a framework base"
    )
    for method in (
        "on_agent_action", "on_agent_finish", "on_chain_end", "on_chain_error",
        "on_chain_start", "on_chat_model_start", "on_custom_event", "on_llm_end",
        "on_llm_error", "on_llm_new_token", "on_llm_start", "on_retriever_end",
        "on_retriever_error", "on_retriever_start", "on_retry", "on_stream_event",
        "on_text", "on_tool_end", "on_tool_error", "on_tool_start",
    )
}


def _unreachable() -> set[str]:
    return {d.id for d in Graph().unreachable()}


def test_every_definition_is_reachable_from_a_door():
    missing = sorted(_unreachable() - set(ALLOWED))
    by_module = defaultdict(list)
    for def_id in missing:
        module, qual = def_id.split(":", 1)
        by_module[module].append(qual)
    report = "\n".join(f"  {module}: {', '.join(quals)}" for module, quals in sorted(by_module.items()))
    assert not missing, (
        f"{len(missing)} definitions no command, hook, server, export or documented "
        f"import reaches. Delete what is dead; give a library capability a section "
        f"in docs/library.md:\n{report}"
    )


def test_the_allow_list_names_only_what_is_still_unreachable():
    stale = sorted(set(ALLOWED) - _unreachable())
    assert not stale, f"reachable now, drop from ALLOWED: {stale}"
