# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Defects a full-tree mypy run found, each pinned by behaviour.

mypy ran over all of src with the project's own config. Most findings were
annotations; the ones here fail at runtime too.
"""
from __future__ import annotations


def test_every_registered_action_type_is_hashable_with_set_domains():
    # shell.exec was built with a list, so it was the one action type that
    # could not be hashed or have its domains combined with another's.
    from vaara.integrations.shell_proxy import SHELL_EXEC
    from vaara.taxonomy.actions import create_default_registry

    registry = create_default_registry()
    registry.register(SHELL_EXEC)
    for action in [SHELL_EXEC, *registry.all_types.values()]:
        hash(action)
        assert isinstance(action.regulatory_domains, frozenset), action.name
        assert action.regulatory_domains | frozenset() == action.regulatory_domains
