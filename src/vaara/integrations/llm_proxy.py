# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara llm-proxy`` — govern LLM API calls from coding agents.

Intercepts prompts, strips secrets, enforces model/rate policies, and forwards
to the configured upstream provider.  Everything recorded in the Vaara audit
trail.

Usage::

    # Start the proxy, forwarding to Melious
    vaara llm-proxy \\
        --upstream https://api.melious.ai/v1 \\
        --api-key-file /path/to/key

    # Enforce model allow-list
    vaara llm-proxy --upstream ... --model-allow "claude-sonnet-4-*,deepseek-*"

    # Agent points at 127.0.0.1:8790/v1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import create_default_registry
from .llm_actions import LLM_ACTIONS


def _build_pipeline(db: Optional[Path] = None) -> InterceptionPipeline:
    """Create a pipeline with LLM action types registered."""
    registry = create_default_registry()
    for at in LLM_ACTIONS:
        registry.register(at)
        registry.map_tool(at.name, at.name)
    if db:
        db.parent.mkdir(parents=True, exist_ok=True)
        trail = SQLiteAuditBackend(str(db)).load_trail()
    else:
        trail = None
    return InterceptionPipeline(registry=registry, trail=trail)


def add_arguments(parser): ...


def main(args: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="vaara llm-proxy",
        description="Govern LLM API calls: intercept prompts, "
                    "strip secrets, enforce model/rate policies, "
                    "record everything in the Vaara audit trail.",
    )
    p.add_argument(
        "--upstream", required=True,
        help="Provider base URL, e.g. https://api.melious.ai/v1",
    )
    key_group = p.add_mutually_exclusive_group(required=True)
    key_group.add_argument(
        "--api-key", default=None,
        help="Upstream API key (visible in process list; prefer --api-key-file)",
    )
    key_group.add_argument(
        "--api-key-file", default=None,
        help="Path to file containing the upstream API key (mode 0400)",
    )
    p.add_argument(
        "--api-key-header", default="x-api-key",
        help="Header name for the API key (default: x-api-key; "
             "use 'authorization' for OpenAI-compatible)",
    )
    p.add_argument(
        "--mode", default="relay", choices=["relay", "govern"],
        help="relay: blind route without inspecting prompts (default). "
             "govern: inspect, scan, redact, full audit.",
    )
    p.add_argument(
        "--audit", default="meta", choices=["meta", "hash", "full"],
        help="Audit detail level: meta (model/tokens/timestamp only), "
             "hash (meta + sha256 of prompt), full (meta + redacted prompt). "
             "Default: meta.",
    )
    p.add_argument(
        "--listen", default="127.0.0.1:8790",
        help="Bind address (default: 127.0.0.1:8790)",
    )
    p.add_argument(
        "--trail", default=None,
        help="Trail database path (default: ~/.vaara/llm-proxy/audit.db)",
    )
    p.add_argument(
        "--enforce", action="store_true",
        help="Gate instead of observe-only (only in govern mode)",
    )
    p.add_argument(
        "--model-allow", action="append", default=None, metavar="GLOB",
        help="Allowed model name glob (repeatable, e.g. deepseek-*)",
    )
    p.add_argument(
        "--model-deny", action="append", default=None, metavar="GLOB",
        help="Denied model name glob (repeatable)",
    )
    p.add_argument(
        "--rate-limit", type=int, default=0,
        help="Max requests per minute per agent (0 = unlimited)",
    )
    p.add_argument(
        "--redact", action="append", default=None, metavar="REGEX",
        help="Additional regex pattern for prompt redaction (repeatable)",
    )
    p.add_argument(
        "--agent-id-header", default="x-agent-id",
        help="Request header carrying the agent identity (default: x-agent-id)",
    )
    p.add_argument(
        "--version", action="version",
        version="vaara llm-proxy 1.55.0",
    )

    parsed = p.parse_args(args)

    api_key = parsed.api_key
    if parsed.api_key_file:
        key_path = Path(parsed.api_key_file).expanduser()
        if not key_path.exists():
            print(f"Error: API key file not found: {key_path}", file=sys.stderr)
            return 1
        api_key = key_path.read_text().strip()

    if not api_key:
        print("Error: no API key provided", file=sys.stderr)
        return 1

    trail_path = parsed.trail
    if not trail_path:
        trail_path = str(Path.home() / ".vaara" / "llm-proxy" / "audit.db")

    pipeline = _build_pipeline(Path(trail_path).expanduser())

    try:
        from uvicorn import Config, Server
    except ImportError as exc:
        print(
            f"vaara llm-proxy: missing dependency ({exc.name}). "
            "Install with: pip install 'vaara[llm-proxy]'",
            file=sys.stderr,
        )
        return 1

    from ._llm_proxy_app import build_app

    app = build_app(
        upstream=parsed.upstream,
        api_key=api_key,
        api_key_header=parsed.api_key_header,
        pipeline=pipeline,
        mode=parsed.mode,
        audit_level=parsed.audit,
        enforce=parsed.enforce,
        model_allow=parsed.model_allow,
        model_deny=parsed.model_deny,
        rate_limit_rpm=parsed.rate_limit,
        redact_patterns=parsed.redact,
        agent_id_header=parsed.agent_id_header,
    )

    host, _, port_str = parsed.listen.rpartition(":")
    port = int(port_str) if port_str else 8790
    host = host or "127.0.0.1"

    config = Config(app=app, host=host, port=port, log_level="info")
    server = Server(config=config)

    print(f"vaara llm-proxy: {parsed.mode} mode, audit={parsed.audit}, "
          f"listening on {host}:{port} -> {parsed.upstream}", file=sys.stderr)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
