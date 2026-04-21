"""Vaara — Runtime action governance for AI agents.

Sits between AI agents and their execution environment.
Classifies and scores each proposed action, returns allow/escalate/deny,
and writes a hash-chained audit trail suitable for EU AI Act Article 14
oversight.
"""

__version__ = "0.4.2"

from vaara.pipeline import InterceptionPipeline, InterceptionResult

Pipeline = InterceptionPipeline

__all__ = ["InterceptionPipeline", "Pipeline", "InterceptionResult", "__version__"]
