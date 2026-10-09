# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara cage``: start, stop and read an agent's cage through a driver.

    vaara cage drivers
    vaara cage drivers --json
    vaara cage run --driver openshell --policy policy.yaml -- claude
    vaara cage run --driver vaara-cage -- codex
    vaara cage run --driver gvisor --image python:3.12 -- python agent.py
    vaara cage run --driver apple-container --image python:3.12 --read-only -- python agent.py
    vaara cage status --driver openshell demo
    vaara cage events --driver openshell demo --since 10m
    vaara cage stop --driver openshell demo
    vaara cage observe

``drivers`` says which cages are ready on this machine: Vaara ships the
drivers, the operator installs the cages (Vaara's own is built in).
``observe`` prints the block a decision made by this process would carry,
which is how to check from inside a sandbox what the records will say.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

from vaara.cage import DRIVERS, load_driver, observe
from vaara.cage.driver import CageError

_DURATION = re.compile(r"^(\d+)([smhd]?)$")
_UNIT = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def _since(value: str) -> float:
    m = _DURATION.match(value.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"{value!r}: use a number with s, m, h or d")
    return time.time() - int(m.group(1)) * _UNIT[m.group(2)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vaara cage", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("drivers", help="The cage drivers Vaara ships, and which are "
                        "ready on this machine.")
    pd.add_argument("--json", action="store_true", help="Print the readiness as JSON.")

    pr = sub.add_parser("run", help="Start an agent inside a cage.")
    pr.add_argument("--driver", required=True, choices=DRIVERS)
    pr.add_argument("--policy", type=Path, default=None,
                    help="The cage's policy file (OpenShell: sandbox policy YAML).")
    pr.add_argument("--name", default=None, help="The launch's name in the cage.")
    pr.add_argument("--image", default=None,
                    help="The sandbox image: openshell (--from), gvisor, kata, microsandbox, "
                         "apple-container.")
    pr.add_argument("--provider", action="append", default=[],
                    help="openshell: attach a credential provider. Repeatable.")
    pr.add_argument("--template", default=None, help="e2b: the template id.")
    pr.add_argument("--permission-profile", default=None,
                    help="codex: a named permission profile instead of --policy.")
    pr.add_argument("--security-opt", action="append", default=[],
                    help="gvisor, kata: an engine security option. Repeatable.")
    pr.add_argument("--read-only", action="store_true",
                    help="apple-container: mount the root filesystem read-only.")
    pr.add_argument("--cap-drop", action="append", default=[],
                    help="apple-container: drop a Linux capability. Repeatable.")
    pr.add_argument("--allow", action="append", default=[],
                    help="nono: a directory to allow read and write. Repeatable.")
    pr.add_argument("agent", nargs=argparse.REMAINDER, help="The agent and its arguments, after --")

    for name, text in (("status", "What the cage reports about a launch."),
                       ("stop", "End a launch."),
                       ("events", "The cage's own events for a launch.")):
        ps = sub.add_parser(name, help=text)
        ps.add_argument("--driver", required=True, choices=DRIVERS)
        ps.add_argument("name", nargs="?" if name == "status" else None,
                        help="The launch's name in the cage.")
        if name == "events":
            ps.add_argument("--since", type=_since, default=0.0,
                            help="How far back, e.g. 10m, 2h (default: all the cage keeps).")
        if name == "status":
            ps.add_argument("--json", action="store_true", help="Print the raw status.")

    sub.add_parser("observe", help="The cage block a decision made here would carry.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return _dispatch(args)
    except CageError as exc:
        print(f"vaara cage: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "drivers":
        from vaara.cage.readiness import check_all, platform_names

        found = check_all()
        if args.json:
            print(json.dumps([r.to_dict() for r in found], indent=2))
            return 0
        width = max(len(name) for name in DRIVERS)
        for r in found:
            state = f"ready  {r.version}".rstrip() if r.ready else "missing"
            print(f"{r.driver:<{width}}  {state}")
            print(f"{'':<{width}}  runs on {platform_names(r.driver)}")
            for problem in r.missing:
                print(f"{'':<{width}}  - {problem}")
        return 0
    if args.cmd == "observe":
        print(json.dumps(observe().to_record(), indent=2))
        return 0

    d = load_driver(args.driver)
    if args.cmd == "run":
        agent = [a for a in args.agent if a != "--"] if args.agent else []
        if not agent:
            raise CageError("give the agent after --")
        import inspect

        offered = {"image": args.image, "providers": args.provider or None,
                   "template": args.template, "permission_profile": args.permission_profile,
                   "security_opt": args.security_opt or None, "allow": args.allow or None,
                   "read_only": args.read_only or None, "cap_drop": args.cap_drop or None}
        accepted = inspect.signature(d.start).parameters
        kwargs = {k: v for k, v in offered.items() if k in accepted and v is not None}
        launch = d.start(agent, args.policy, name=args.name, **kwargs)
        print(json.dumps({"driver": launch.driver, "name": launch.name, "pid": launch.pid,
                          "cage": launch.state.to_record()}, indent=2))
        return 0
    if args.cmd == "status":
        state = d.enforcement_state(args.name)
        out = state.to_record()
        if getattr(args, "json", False):
            out = {"cage": out, "detail": state.detail}
        print(json.dumps(out, indent=2, default=str))
        return 0 if state.confirmed else 1
    if args.cmd == "stop":
        d.stop(args.name)
        return 0
    if args.cmd == "events":
        for event in d.events(args.name, args.since):
            print(json.dumps(event, default=str))
        return 0
    raise CageError(f"unknown command {args.cmd}")
