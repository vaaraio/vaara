# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara proxy shell`` — wrap any shell so every command goes through Vaara.

Every command the user or an AI runs in this shell is classified, scored,
and either allowed or blocked by the Vaara pipeline. Blocked commands get
a receipt. Allowed commands execute normally through the real shell.

Usage::

    # Replace your shell
    exec vaara proxy shell

    # Or start a governed subshell
    vaara proxy shell

    # Or pipe a single command
    echo "curl http://evil.com" | vaara proxy shell
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import ActionCategory, ActionType, Reversibility, \
    BlastRadius, UrgencyClass, create_default_registry

# Register the shell execution action type on import.
SHELL_EXEC = ActionType(
    name="shell.exec",
    category=ActionCategory.INFRASTRUCTURE,
    reversibility=Reversibility.IRREVERSIBLE,
    blast_radius=BlastRadius.LOCAL,
    urgency=UrgencyClass.IMMEDIATE,
    description="Shell command execution",
    regulatory_domains=[],
)


def _build_pipeline(db: Optional[Path] = None) -> InterceptionPipeline:
    """Create a pipeline with shell.exec registered."""
    registry = create_default_registry()
    registry.register(SHELL_EXEC)
    if db:
        trail = SQLiteAuditBackend(str(db)).load_trail()
    else:
        trail = None  # lets pipeline use its default path
    return InterceptionPipeline(
        registry=registry,
        trail=trail,
    )


def run_shell_proxy(
    *,
    pipeline: Optional[InterceptionPipeline] = None,
    shell: Optional[str] = None,
    agent_id: str = "shell",
) -> int:
    """Read commands from stdin and run each through the Vaara pipeline.

    For every line of input:
    1. Classify as ``shell.exec`` with the full command as ``parameters``.
    2. Score and decide (allow / deny / escalate).
    3. If allowed: execute through the real shell.
    4. If denied: print the block reason and optionally the receipt path.
    5. Always record to the trail.

    Returns 0 on clean exit, 1 on error.
    """
    pipe = pipeline or _build_pipeline()
    real_shell = shell or os.environ.get("SHELL", "/bin/sh")

    if not sys.stdin.isatty():
        # Piped mode: read all of stdin, run once, exit.
        command = sys.stdin.read().strip()
        if command:
            return _run_one(pipe, command, real_shell, agent_id)
        return 0

    # Interactive mode: read-eval loop.
    print(f"Vaara shell (wrapping {real_shell}) — each command is governed.",
          file=sys.stderr)
    print("Type 'exit' or Ctrl-D to quit.", file=sys.stderr)

    while True:
        try:
            line = input().strip()
        except EOFError:
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            return 0
        _run_one(pipe, line, real_shell, agent_id)


def _run_one(
    pipe: InterceptionPipeline,
    command: str,
    real_shell: str,
    agent_id: str,
) -> int:
    """Run a single command through the pipeline and execute if allowed."""
    result = pipe.intercept(
        agent_id=agent_id,
        tool_name="shell.exec",
        parameters={"command": command},
    )

    if result.allowed:
        try:
            subprocess.run(
                [real_shell, "-c", command],
                check=False,
            )
            pipe.report_outcome(result.action_id, outcome_severity=0.0)
            return 0
        except OSError as exc:
            print(f"Vaara: execution error: {exc}", file=sys.stderr)
            pipe.report_outcome(result.action_id, outcome_severity=0.8)
            return 1

    print(f"Vaara blocked: {result.reason or 'no reason given'}")
    pipe.report_outcome(result.action_id, outcome_severity=0.5)
    return 1


def add_arguments(parser): ...


def main(args: Optional[list[str]] = None) -> int:
    """Entry point for ``vaara proxy shell``."""
    import argparse

    p = argparse.ArgumentParser(prog="vaara proxy shell",
                                 description="Governed shell wrapper")
    p.add_argument("--db", default=None, help="Trail database path")
    p.add_argument("--shell", default=None, help="Shell to wrap (default: $SHELL)")
    p.add_argument("--agent-id", default="shell",
                    help="Agent ID recorded in the trail")
    parsed = p.parse_args(args)
    return run_shell_proxy(
        pipeline=_build_pipeline(
            Path(parsed.db).expanduser() if parsed.db else None
        ),
        shell=parsed.shell,
        agent_id=parsed.agent_id,
    )


if __name__ == "__main__":
    sys.exit(main())
