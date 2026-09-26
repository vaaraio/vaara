# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara os-layer``: the operator's side of the OS layer, without the app.

    vaara os-layer status
    vaara os-layer folder ~/clients ask        # record | ask | block | off
    vaara os-layer app /usr/local/bin/copilot  # --remove to detach
    vaara os-layer pending
    vaara os-layer approve <action_id>         # or: deny <action_id>

Changes go to ``~/.vaara/os-layer.json``; a running guard picks them up
within a couple of seconds, and at once when it can be told.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from vaara.oslayer import floor, selection
from vaara.oslayer.guard import SOCKET_PATH

MODES_OR_OFF = selection.MODES + ("off",)


def _home() -> str:
    return str(Path.home())


def _guard(op: str) -> Optional[dict]:
    from vaara.oslayer.client import GuardError, request

    try:
        return request({"op": op}, socket_path=SOCKET_PATH)
    except (GuardError, OSError):
        return None


def guard_status() -> Optional[dict]:
    """The running guard's status, or None when no guard answers."""
    return _guard("status")


def save(sel: selection.Selection, home: Optional[str] = None) -> tuple[Path, bool]:
    """Write ``sel`` and tell a running guard. (path, whether the guard was told)."""
    path = selection.save(home or _home(), sel)
    return path, _guard("reload") is not None


def _cmd_status(args: argparse.Namespace) -> int:
    status = guard_status()
    if args.json:
        print(json.dumps(status or {"ok": False, "guard": "not running"}, indent=2))
        return 0 if status else 1
    if status is None:
        sel = selection.load(_home())
        print("guard: not running (start it with: sudo vaara os-guard)")
        print(f"floor: {_floor_state()}")
        folders, apps = sel.folders, sel.apps
        launches: list = []
    else:
        print(f"guard: running as pid {status['pid']} for {status['user']}")
        print(f"floor: {'loaded' if status.get('profile_loaded') else 'NOT loaded'}")
        print(f"trail: {status['trail']}")
        folders = [selection.Folder(f["path"], f["mode"]) for f in status["folders"]]
        apps, launches = status["apps"], status["launches"]
    print("folders:" if folders else "folders: none picked")
    for f in folders:
        print(f"  {f.mode:<7} {f.path}")
    print("apps:" if apps else "apps: none attached")
    for a in apps:
        print(f"  {a}")
    for launch in launches:
        age = int(time.time() - launch["started"])
        print(f"launch {launch['launch']}: {launch['agent']} pid {launch['pid']} ({age}s)")
    return 0


def _floor_state() -> str:
    if not floor.apparmor_enabled():
        return "AppArmor is not enabled"
    try:
        Path("/sys/kernel/security/apparmor/profiles").read_text()
    except OSError:
        return "unknown (reading the loaded profiles needs root)"
    return "loaded" if floor.profile_loaded() else "not loaded"


def _save(sel: selection.Selection) -> int:
    path, told = save(sel)
    print(f"saved {path}" + ("; the guard has it" if told else
                             "; the guard picks it up when it runs"))
    return 0


def _cmd_folder(args: argparse.Namespace) -> int:
    mode = None if args.mode == "off" else args.mode
    sel = selection.set_folder(selection.load(_home()), args.path, mode)
    return _save(sel)


def _cmd_app(args: argparse.Namespace) -> int:
    sel = selection.set_app(selection.load(_home()), args.path, not args.remove)
    return _save(sel)


def pending(approvals_dir: Path) -> list[dict]:
    """Requests from the guard still waiting on the operator, oldest first."""
    out = []
    for path in sorted(approvals_dir.glob("*.request.json")):
        try:
            request = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(request, dict) and str(request.get("tool_name", "")).startswith("os."):
            out.append(request)
    return sorted(out, key=lambda r: r.get("requested_at", 0))


def _cmd_pending(args: argparse.Namespace) -> int:
    from vaara.approvals import APPROVALS_DIR

    waiting = pending(APPROVALS_DIR)
    if args.json:
        print(json.dumps(waiting, indent=2))
        return 0
    if not waiting:
        print("nothing waiting")
    for request in waiting:
        age = int(time.time() - float(request.get("requested_at", time.time())))
        print(f"{request['action_id']}  {age:>3}s  {request.get('reason', '')}")
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    from vaara.approvals import write_decision

    decision = "approve" if args.cmd == "approve" else "deny"
    if not write_decision(args.action_id, decision):
        print(f"vaara os-layer: no request {args.action_id} is waiting, or there is no key "
              "to sign with", file=sys.stderr)
        return 1
    print(f"{'approved' if decision == 'approve' else 'denied'} {args.action_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vaara os-layer",
                                description="Pick folders and apps for the Linux OS layer, "
                                            "see the guard, answer its questions.")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND", required=True)

    ps = sub.add_parser("status", help="The guard, the floor, your folders, apps and launches")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=_cmd_status)

    pf = sub.add_parser("folder", help="Set a folder's mode: record, ask, block, or off")
    pf.add_argument("path")
    pf.add_argument("mode", choices=MODES_OR_OFF)
    pf.set_defaults(func=_cmd_folder)

    pa = sub.add_parser("app", help="Attach the profile to an app by path, so it is governed "
                                    "however it is started")
    pa.add_argument("path")
    pa.add_argument("--remove", action="store_true", help="Detach instead")
    pa.set_defaults(func=_cmd_app)

    pp = sub.add_parser("pending", help="Opens and execs waiting on your answer")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=_cmd_pending)

    for verb in ("approve", "deny"):
        pd = sub.add_parser(verb, help=f"{verb.capitalize()} one waiting request, signed")
        pd.add_argument("action_id")
        pd.set_defaults(func=_cmd_decide)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except selection.SelectionError as exc:
        print(f"vaara os-layer: {exc}", file=sys.stderr)
        return 2
