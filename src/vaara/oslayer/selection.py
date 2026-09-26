# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The operator's picks: ``~/.vaara/os-layer.json``.

    {"version": 1,
     "folders": [{"path": "/home/me/clients", "mode": "ask"},
                 {"path": "/home/me/notes", "mode": "record"},
                 {"path": "/home/me/keys-offline", "mode": "block"}],
     "apps": ["/usr/local/bin/copilot"],
     "ask_timeout": 60}

A folder's mode applies to every agent process:

- ``record``: the open or exec goes ahead, and the first one of each file per
  launch is written to the trail.
- ``ask``: the open or exec waits for a signed approval from the operator, and
  is refused when none arrives before ``ask_timeout``.
- ``block``: nothing in the folder is read, written, moved or deleted. Block
  folders join the floor, so this holds in the kernel with or without the
  guard.

An app is a harness binary the profile attaches to by path, so it is governed
when started from a menu or a dock, without ``vaara run``.

The file lives under ``~/.vaara``, which no process in an agent's tree can
open, so an agent cannot change its own rules.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

FILENAME = "os-layer.json"
MODES = ("record", "ask", "block")
DEFAULT_ASK_TIMEOUT = 60.0

# Folders a pick may not be, or sit inside: the guard reads these itself, or
# the kernel and the floor already own them.
_REFUSED_ROOTS = ("/proc", "/sys", "/dev", "/run", "/var/lib/vaara", "/etc/vaara",
                  "/etc/apparmor.d", "/boot")

# Characters AppArmor treats as pattern syntax. A path holding one cannot be
# written as a literal rule, so it cannot be picked.
_PATTERN_CHARS = set('*?[]{}^,"\x00')


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Folder:
    path: str
    mode: str


@dataclass
class Selection:
    folders: list[Folder] = field(default_factory=list)
    apps: list[str] = field(default_factory=list)
    ask_timeout: float = DEFAULT_ASK_TIMEOUT

    def folders_in(self, mode: str) -> list[str]:
        return [f.path for f in self.folders if f.mode == mode]

    def match(self, path: str) -> Optional[Folder]:
        """The picked folder ``path`` is in, the deepest one when they nest."""
        best: Optional[Folder] = None
        for folder in self.folders:
            if path == folder.path or path.startswith(folder.path + "/"):
                if best is None or len(folder.path) > len(best.path):
                    best = folder
        return best

    def to_json(self) -> dict:
        return {
            "version": 1,
            "folders": [{"path": f.path, "mode": f.mode} for f in self.folders],
            "apps": list(self.apps),
            "ask_timeout": self.ask_timeout,
        }


def path_for(home: str) -> Path:
    return Path(home) / ".vaara" / FILENAME


def check_folder(path: str) -> str:
    """``path`` as a picked folder: absolute, real, a directory. Raises otherwise."""
    if not path or not os.path.isabs(path):
        raise SelectionError(f"not an absolute path: {path!r}")
    if _PATTERN_CHARS & set(path):
        raise SelectionError(f"path holds a character AppArmor reads as a pattern: {path!r}")
    real = os.path.realpath(path)
    if real == "/":
        raise SelectionError("the whole filesystem cannot be one folder")
    for root in _REFUSED_ROOTS:
        if real == root or real.startswith(root + "/"):
            raise SelectionError(f"{real} is under {root}, which the OS layer owns")
    if not os.path.isdir(real):
        raise SelectionError(f"not a directory: {real}")
    return real


def check_app(path: str) -> str:
    if not path or not os.path.isabs(path):
        raise SelectionError(f"not an absolute path: {path!r}")
    real = os.path.realpath(path)
    if _PATTERN_CHARS & set(real) or any(c.isspace() for c in real):
        raise SelectionError(f"app path cannot be attached by AppArmor: {real!r}")
    if not os.path.isfile(real):
        raise SelectionError(f"not a file: {real}")
    return real


def parse(data: object) -> Selection:
    """A Selection from decoded JSON. Entries that do not check out are dropped."""
    if not isinstance(data, dict):
        return Selection()
    folders: dict[str, Folder] = {}
    for entry in data.get("folders") or []:
        if not isinstance(entry, dict) or entry.get("mode") not in MODES:
            continue
        try:
            real = check_folder(str(entry.get("path", "")))
        except SelectionError:
            continue
        folders[real] = Folder(real, entry["mode"])
    apps: dict[str, None] = {}
    for entry in data.get("apps") or []:
        try:
            apps[check_app(str(entry))] = None
        except SelectionError:
            continue
    timeout = data.get("ask_timeout", DEFAULT_ASK_TIMEOUT)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 1 <= timeout <= 3600:
        timeout = DEFAULT_ASK_TIMEOUT
    return Selection(folders=list(folders.values()), apps=list(apps), ask_timeout=float(timeout))


def load(home: str) -> Selection:
    try:
        return parse(json.loads(path_for(home).read_text()))
    except (OSError, ValueError):
        return Selection()


def save(home: str, selection: Selection) -> Path:
    """Write ``selection`` in one step, readable by the owner only."""
    path = path_for(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".os-layer.", suffix=".tmp")
    with os.fdopen(fd, "w") as out:
        json.dump(selection.to_json(), out, indent=2)
        out.write("\n")
    os.replace(tmp, path)
    return path


def set_folder(selection: Selection, path: str, mode: Optional[str]) -> Selection:
    """``selection`` with ``path`` set to ``mode``, or removed when ``mode`` is None."""
    if mode is not None and mode not in MODES:
        raise SelectionError(f"mode must be one of {', '.join(MODES)}")
    real = check_folder(path) if mode is not None else os.path.realpath(path)
    folders = [f for f in selection.folders if f.path != real]
    if mode is not None:
        folders.append(Folder(real, mode))
    return Selection(folders=folders, apps=list(selection.apps),
                     ask_timeout=selection.ask_timeout)


def set_app(selection: Selection, path: str, attached: bool) -> Selection:
    real = check_app(path) if attached else os.path.realpath(path)
    apps = [a for a in selection.apps if a != real]
    if attached:
        apps.append(real)
    return Selection(folders=list(selection.folders), apps=apps,
                     ask_timeout=selection.ask_timeout)
