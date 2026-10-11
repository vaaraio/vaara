# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara net-gate``: run the gateway and manage the agents behind it.

Usage::

    # Give an agent a credential and a mandate. The token is printed once.
    vaara net-gate add-agent builder --allow api.github.com --allow "*.pypi.org"

    # Run the gateway
    vaara net-gate serve --listen 127.0.0.1:8791

    # Point the agent at it
    HTTPS_PROXY=http://builder:<token>@127.0.0.1:8791 \\
    HTTP_PROXY=http://builder:<token>@127.0.0.1:8791 the-agent

    # The brake: every open connection of the agent ends now, on the record
    vaara net-gate revoke builder

    # A WireGuard client config that forces everything else through the gateway
    vaara net-gate wg-config --address 10.70.0.2/32 --gateway-key <pub> \\
        --endpoint gw.example:51820 --direct 192.168.0.0/16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from ._mandate import MandateStore, default_path

DESCRIPTION = (
    "Govern agents' network access where it leaves: every connection is "
    "attributed to one agent by its own credential, checked against that "
    "agent's mandate and the policy, recorded on the trail, and ended the "
    "moment the agent is revoked."
)


def _store(args) -> MandateStore:
    return MandateStore(Path(args.agents).expanduser() if args.agents else None)


def _cmd_add(args) -> int:
    store = _store(args)
    try:
        token = store.add(args.agent, args.allow or [], allow_private=args.allow_private)
    except ValueError as exc:
        print(f"vaara net-gate: {exc}", file=sys.stderr)
        return 2
    print(f"agent:  {args.agent}")
    print(f"token:  {token}")
    print(f"allow:  {', '.join(args.allow or []) or '(nothing: every connection is refused)'}")
    print("The token is shown once; only its sha256 is kept. Use it as the proxy password:")
    print(f"  HTTPS_PROXY=http://{args.agent}:{token}@<gateway>")
    return 0


def _cmd_revoke(args) -> int:
    store = _store(args)
    try:
        store.revoke(args.agent)
    except KeyError:
        print(f"vaara net-gate: no agent {args.agent!r}", file=sys.stderr)
        return 2
    print(f"revoked {args.agent} at {store.revoked[args.agent]}; "
          "a running gateway ends its connections within a second")
    return 0


def _cmd_list(args) -> int:
    store = _store(args)
    rows = []
    for a in store.agents.values():
        rows.append({"agent": a.agent_id, "allow": [str(r) for r in a.allow],
                     "allow_private": a.allow_private, "created": a.created,
                     "revoked": store.revoked.get(a.agent_id, "")})
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_wg(args) -> int:
    from ._wg import client_config

    try:
        sys.stdout.write(client_config(address=args.address,
                                       gateway_public_key=args.gateway_key,
                                       endpoint=args.endpoint,
                                       direct=args.direct or [], dns=args.dns or ""))
    except ValueError as exc:
        print(f"vaara net-gate: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_serve(args) -> int:
    from . import NetGate, build_pipeline

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    host, _, port = args.listen.rpartition(":")
    if not port.isdigit():
        print(f"vaara net-gate: --listen wants HOST:PORT, got {args.listen!r}",
              file=sys.stderr)
        return 2
    policy = None
    if args.policy:
        from vaara.policy.validate import validate_source

        policy_path = Path(args.policy).expanduser()
        policy, report = validate_source(policy_path)
        if policy is None:
            print(f"vaara net-gate: policy {policy_path} failed validation:",
                  file=sys.stderr)
            for issue in report.issues:
                print(f"  {issue.level.value}: {issue.message}", file=sys.stderr)
            return 2
    trail = Path(args.trail).expanduser() if args.trail else \
        Path.home() / ".vaara" / "netgate" / "audit.db"
    store = _store(args)
    gate = NetGate(store, build_pipeline(trail, policy=policy), host=host or "127.0.0.1",
                   port=int(port), challenge=args.challenge)
    print(f"vaara net-gate on {gate.host}:{gate.port}, agents {store.path}, trail {trail}",
          file=sys.stderr)
    try:
        asyncio.run(gate.serve_forever())
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vaara net-gate", description=DESCRIPTION)
    p.add_argument("--agents", default=None, metavar="FILE",
                   help=f"Agents file (default: {default_path()}, or $VAARA_NETGATE_AGENTS)")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    s = sub.add_parser("serve", help="Run the gateway")
    s.add_argument("--listen", default="127.0.0.1:8791", metavar="HOST:PORT")
    s.add_argument("--trail", default=None, metavar="DB",
                   help="Trail database (default: ~/.vaara/netgate/audit.db)")
    s.add_argument("--policy", default=None, metavar="FILE",
                   help="Policy file scored on every net.connect, the format "
                        "`vaara serve --policy` takes")
    s.add_argument("--challenge", action="store_true",
                   help="Answer an unauthenticated client with 407 instead of "
                        "silence, for clients that only send credentials when asked")
    s.set_defaults(func=_cmd_serve)

    a = sub.add_parser("add-agent", help="Create or re-key an agent; prints its token once")
    a.add_argument("agent")
    a.add_argument("--allow", action="append", metavar="PATTERN",
                   help="host, *.suffix or *, optional :port (repeatable)")
    a.add_argument("--allow-private", action="store_true",
                   help="Let this agent reach loopback, link-local and private "
                        "addresses (never the cloud-metadata address)")
    a.set_defaults(func=_cmd_add)

    r = sub.add_parser("revoke", help="Revoke an agent and end its open connections")
    r.add_argument("agent")
    r.set_defaults(func=_cmd_revoke)

    ls = sub.add_parser("list", help="Agents, their mandates and revocations")
    ls.set_defaults(func=_cmd_list)

    w = sub.add_parser("wg-config", help="WireGuard client config through the gateway")
    w.add_argument("--address", required=True, help="The agent's tunnel address, e.g. 10.70.0.2/32")
    w.add_argument("--gateway-key", required=True, help="The gateway's WireGuard public key")
    w.add_argument("--endpoint", required=True, help="The gateway's HOST:PORT")
    w.add_argument("--direct", action="append", metavar="CIDR",
                   help="A range that bypasses the gateway (split tunnel; repeatable)")
    w.add_argument("--dns", default=None)
    w.set_defaults(func=_cmd_wg)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
