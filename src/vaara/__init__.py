# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara. Runtime action governance for AI agents.

Sits between AI agents and their execution environment.
Classifies and scores each proposed action, returns allow/escalate/deny,
and writes a hash-chained audit trail suitable for EU AI Act Article 14
oversight.
"""

__version__ = "1.38.0"

from vaara.pipeline import InterceptionPipeline, InterceptionResult

# The one-liner. ``from vaara.govern import govern`` binds the decorator onto
# the package namespace, so ``@vaara.govern`` is the decorator (not the
# submodule). See vaara/govern.py.
from vaara.govern import govern, Blocked

Pipeline = InterceptionPipeline

__all__ = [
    "InterceptionPipeline",
    "Pipeline",
    "InterceptionResult",
    "govern",
    "Blocked",
    "__version__",
]
