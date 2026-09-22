# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara llm-proxy`` — govern LLM API calls from coding agents.

Governs ``POST /v1/chat/completions`` and ``POST /v1/messages``: each call is
checked against the model and rate policy and recorded in the Vaara audit trail
with its prompt. Every other call is recorded by method, path, size and
sha256. The secrets named in ``--seal-file`` are replaced on every path before
the request leaves. Nothing beyond those values is removed from a request;
``--redact`` masks only the trail's copy of the prompt.

Usage::

    # Start the proxy, forwarding to Melious
    vaara llm-proxy \\
        --upstream https://api.melious.ai/v1 \\
        --api-key-file /path/to/key

    # Enforce model allow-list
    vaara llm-proxy --upstream ... --model-allow "claude-sonnet-4-*,deepseek-*"

    # Agent points at 127.0.0.1:8790/v1

    # Govern an agent that carries its own subscription credential, where
    # there is no operator API key to inject
    vaara llm-proxy \\
        --upstream https://api.anthropic.com \\
        --auth-passthrough --seal-file ~/.vaara/seal.json

    # Then: ANTHROPIC_BASE_URL=http://127.0.0.1:8790 claude
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from vaara import __version__ as _VAARA_VERSION
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import create_default_registry
from .llm_actions import LLM_ACTIONS

#: What the proxy does, stated to the edge of what it does. Shared by
#: ``vaara llm-proxy --help`` and the subcommand list so the two cannot differ.
DESCRIPTION = (
    "Govern LLM API calls. POST /v1/chat/completions and POST /v1/messages "
    "are checked against the model and rate policy and recorded in the Vaara "
    "audit trail with their prompt. Every other call is recorded by method, "
    "path, size and sha256, never by content. The secrets named in "
    "--seal-file are replaced on every path before the request leaves. "
    "Nothing beyond the --seal-file values is removed from a request; "
    "--redact masks only the trail's copy of the prompt. A call the trail "
    "cannot record is refused unless --fail-open is set."
)


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


def _timestamped_log_config():
    """uvicorn's own logging config, with a dated prefix on every line.

    uvicorn's defaults carry no time at all, so an operator correlating a
    502 in this log against a trail entry has nothing to join on. The
    sibling proxies all set %(asctime)s through basicConfig; that does not
    work here because uvicorn installs its own handlers, so the timestamp
    has to go into uvicorn's formatters instead.
    """
    from copy import deepcopy

    from uvicorn.config import LOGGING_CONFIG

    config = deepcopy(LOGGING_CONFIG)
    for formatter in config["formatters"].values():
        formatter["fmt"] = "%(asctime)s " + formatter["fmt"]
        # Offset included, so a log shipped off this box stays unambiguous.
        formatter["datefmt"] = "%Y-%m-%dT%H:%M:%S%z"
    return config


def main(args: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="vaara llm-proxy",
        description=DESCRIPTION,
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
    key_group.add_argument(
        "--auth-passthrough", action="store_true",
        help="Hold no key. Forward the caller's own credential to the "
             "provider untouched. Use this to govern an agent that "
             "authenticates with its own subscription (Claude Code, Cursor), "
             "which has no API key to hand over. The proxy still intercepts, "
             "seals, audits and enforces; it just does not supply identity.",
    )
    p.add_argument(
        "--api-key-header", default="x-api-key",
        help="Header name for the API key (default: x-api-key; "
             "use 'authorization' for OpenAI-compatible)",
    )
    p.add_argument(
        "--mode", default="relay", choices=["relay", "govern"],
        help="relay: blind route without inspecting prompts (default). "
             "govern: inspect and scan the prompt, redact the trail copy.",
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
        help="Regex masked in the trail's copy of the prompt (govern mode, "
             "--audit full; repeatable). The request sent to the provider "
             "is unchanged.",
    )
    p.add_argument(
        "--agent-id-header", default="x-agent-id",
        help="Request header carrying the agent identity (default: x-agent-id)",
    )
    p.add_argument(
        "--agent-id", default="llm-agent", metavar="ID",
        help="Identity recorded when the caller sends no agent-id header "
             "(default: llm-agent). A caller like Claude Code sends none, so "
             "without this every record names the same anonymous agent.",
    )
    p.add_argument(
        "--seal-file", default=None, metavar="PATH",
        help="JSON object of {\"name\": \"secret\"} whose values are replaced "
             "with stable placeholders before the request reaches the "
             "provider, and restored in the response. Covers only what you "
             "name in advance. Missing or unreadable file means sealing is "
             "off and the proxy runs unchanged.",
    )
    p.add_argument(
        "--allow-unsealed", action="store_true",
        help="Start even when --seal-file loads no secrets. Without this the "
             "proxy refuses, because an operator who passed --seal-file "
             "believes secrets are being held back, and a proxy that runs "
             "unsealed while the flag is set claims what it does not do.",
    )
    p.add_argument(
        "--compact-history", type=int, default=0, metavar="N",
        help="Keep the last N assistant turns verbatim and replace older tool "
             "results and tool inputs with a size-and-digest stub before the "
             "request leaves. User and assistant text is never touched. A "
             "payload then leaves the machine once, when it is fresh, "
             "instead of on every later call. The record carries bytes before "
             "and after. 0 (default) is off. This is a privacy control, not a "
             "cost control: the stub boundary moves every turn, so against a "
             "provider with prompt caching (Anthropic, OpenAI) every earlier "
             "message changes on every call and the cached prefix is lost. "
             "Measured at 6 to 10 times the input cost in front of a coding "
             "agent that re-sends its whole context. Leave it at 0 for clients that rely on the cache.",
    )
    p.add_argument(
        "--markers-file", default=None, metavar="PATH",
        help="JSON object of {\"id\": \"marker\"}. Each prompt record lists "
             "the ids whose marker string was inside the bytes that left, so "
             "a marker later seen elsewhere can be traced to the call that "
             "carried it. Ids only reach the trail; the strings never do. "
             "Re-read on every request, so markers can be added while the "
             "proxy runs.",
    )
    p.add_argument(
        "--seal-listen-unix", default=None, metavar="PATH",
        help="Bind a unix socket instead of TCP. No listening port, so "
             "filesystem permissions decide who can reach the proxy.",
    )
    p.add_argument(
        "--allow-origin", action="append", default=None, metavar="ORIGIN",
        help="Browser origin permitted to call the proxy, e.g. "
             "https://console.example (repeatable, matched exactly). By "
             "default any request carrying an Origin header from another "
             "site is refused, which is what stops a page you visit from "
             "spending your upstream key. Native clients send no Origin.",
    )
    p.add_argument(
        "--fail-open", action="store_true",
        help="Forward a request even when its audit record could not be "
             "written. Off by default: the proxy first repairs the trail and "
             "retries, and if the record still cannot be written it answers "
             "503 and forwards nothing. With this flag the request goes out "
             "unrecorded and the log says so on every one.",
    )
    p.add_argument(
        # Read from the package rather than restated here. The literal that
        # used to sit in this line said 1.56.0 long after the package moved
        # on, so --version reported a release this code is not.
        "--version", action="version",
        version=f"vaara llm-proxy {_VAARA_VERSION}",
    )

    parsed = p.parse_args(args)

    api_key = parsed.api_key
    if parsed.api_key_file:
        key_path = Path(parsed.api_key_file).expanduser()
        if not key_path.exists():
            print(f"Error: API key file not found: {key_path}", file=sys.stderr)
            return 1
        api_key = key_path.read_text().strip()

    if parsed.auth_passthrough:
        # None is the signal the app layer reads as "hold nothing, forward
        # what the caller sent". Kept distinct from the empty string so a key
        # file that happens to be empty still fails loudly below.
        api_key = None
    elif not api_key:
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
    from .llm_envelope import MarkerWatch
    from .llm_seal import SealRegistry

    markers = MarkerWatch.from_file(str(parsed.markers_file)) \
        if parsed.markers_file else None

    # Neither the path nor the contents of the seal file reach a log line.
    seal_path = str(parsed.seal_file) if parsed.seal_file else None
    seal = SealRegistry.from_file(seal_path) if seal_path \
        else SealRegistry()
    if seal_path and not seal.active:
        # An operator who passed --seal-file believes secrets are being held
        # back. Running anyway would be the proxy claiming what it does not
        # do, so it refuses unless told the unsealed run is intended.
        if not parsed.allow_unsealed:
            print(
                "vaara llm-proxy: the --seal-file loaded nothing to seal; "
                "refusing to start unsealed. Fix the file, or pass "
                "--allow-unsealed to run with sealing OFF.",
                file=sys.stderr,
            )
            return 2
        print(
            "vaara llm-proxy: the --seal-file loaded nothing to seal; "
            "sealing is OFF for this run (--allow-unsealed).",
            file=sys.stderr,
        )

    # Requests a dead process left pending would look in flight forever.
    # Close them as orphaned before taking new ones.
    from ._llm_proxy_app import sweep_orphaned_outcomes
    orphaned = sweep_orphaned_outcomes(pipeline, process_started=time.time())
    if orphaned:
        print(f"vaara llm-proxy: closed {orphaned} orphaned outcome(s) from "
              "an earlier process.", file=sys.stderr)

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
        seal_registry=seal,
        redact_patterns=parsed.redact,
        agent_id_header=parsed.agent_id_header,
        agent_id_default=parsed.agent_id,
        allowed_origins=parsed.allow_origin,
        marker_watch=markers,
        compact_keep_turns=parsed.compact_history,
        fail_open=parsed.fail_open,
    )

    seal_note = f", sealing {len(seal)} secret(s)" if seal.active \
        else ""
    if markers is not None and markers.active:
        seal_note += f", watching {len(markers)} marker(s)"
    if parsed.compact_history > 0:
        seal_note += f", compacting history beyond {parsed.compact_history} turn(s)"
    if parsed.fail_open:
        seal_note += ", FAIL-OPEN (forwards requests it cannot record)"

    if parsed.seal_listen_unix:
        sock_path = str(Path(parsed.seal_listen_unix).expanduser())
        config = Config(app=app, uds=sock_path, log_level="info",
                        log_config=_timestamped_log_config())
        where = f"unix:{sock_path}"
    else:
        host, _, port_str = parsed.listen.rpartition(":")
        port = int(port_str) if port_str else 8790
        host = host or "127.0.0.1"
        config = Config(app=app, host=host, port=port, log_level="info",
                        log_config=_timestamped_log_config())
        where = f"{host}:{port}"
    server = Server(config=config)

    # Say which identity goes upstream. An operator who cannot tell whether
    # the proxy is supplying a key or forwarding the caller's own cannot tell
    # whose quota is being spent.
    auth_note = ", auth=passthrough" if api_key is None else ""
    print(f"vaara llm-proxy: {parsed.mode} mode, audit={parsed.audit}"
          f"{seal_note}{auth_note}, listening on {where} -> {parsed.upstream}",
          file=sys.stderr)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
