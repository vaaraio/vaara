# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which cages can run on this machine, and what is missing for the rest.

Vaara ships the drivers; the cages themselves are installed by the
operator (except Vaara's own, which is built in). ``check(name)`` answers
for one driver: does it run on this operating system, is every tool it
shells out to on ``PATH`` (or at its variable), and what does the cage
report as its version. Nothing is installed and nothing is started; the
only command run is the tool's own version query, and only when the tool
was found.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

from vaara.cage import DRIVERS, load_driver
from vaara.cage._cli import Tool

#: Where each cage runs, as ``sys.platform`` prefixes. ``any`` means the
#: driver is a client of something that runs elsewhere (a cluster, a
#: service) and works from any operating system.
PLATFORMS: dict[str, tuple[str, ...]] = {
    "vaara-cage": ("linux",),
    "openshell": ("linux", "darwin"),
    "codex": ("linux", "darwin"),
    "sandbox-runtime": ("linux", "darwin"),
    "nono": ("linux", "darwin"),
    "gvisor": ("linux",),
    "kata": ("linux",),
    "firecracker": ("linux",),
    "microsandbox": ("linux", "darwin", "win32"),
    "apple-container": ("darwin",),
    "agent-sandbox": ("any",),
    "e2b": ("any",),
}

_OS_NAMES = {"linux": "Linux", "darwin": "macOS", "win32": "Windows", "any": "any OS"}


@dataclass
class Readiness:
    driver: str
    platforms: list[str]
    on_this_os: bool
    ready: bool
    version: str = ""
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def platform_names(driver: str) -> str:
    return ", ".join(_OS_NAMES.get(p, p) for p in PLATFORMS.get(driver, ()))


def _runs_here(driver: str, platform: str) -> bool:
    return any(p == "any" or platform.startswith(p) for p in PLATFORMS.get(driver, ()))


def _tool_missing(tool: Tool) -> Optional[str]:
    try:
        tool.path()
    except Exception as exc:  # noqa: BLE001 - CageError carries the install hint
        return str(exc)
    return None


def _missing(name: str, d: object) -> list[str]:
    if name == "vaara-cage":
        from vaara.oslayer import floor

        return [] if floor.apparmor_enabled() else [
            "AppArmor is not enabled; the Vaara cage needs it (Ubuntu, Debian, SUSE)"]
    if name == "e2b":
        return [f"{var} is not set" for var in ("E2B_API_URL", "E2B_API_KEY")
                if not os.environ.get(var)]
    out: list[str] = []
    if name == "openshell":
        binary = getattr(d, "_binary", "openshell")
        if not (shutil.which(binary) if "/" not in binary else os.path.exists(binary)):
            out.append(f"{binary}: not found; install OpenShell "
                       "(github.com/NVIDIA/OpenShell), or set OPENSHELL_BIN")
    for value in vars(d).values():
        if isinstance(value, Tool):
            problem = _tool_missing(value)
            if problem:
                out.append(problem)
    if name == "firecracker" and not os.path.exists("/dev/kvm"):
        out.append("/dev/kvm is not there; Firecracker needs KVM")
    return out


def check(name: str, platform: Optional[str] = None) -> Readiness:
    """Readiness of one driver on this machine (or as if on ``platform``)."""
    platform = platform or sys.platform
    here = _runs_here(name, platform)
    result = Readiness(driver=name, platforms=list(PLATFORMS.get(name, ())),
                       on_this_os=here, ready=False)
    if not here:
        result.missing = [f"runs on {platform_names(name)}, not here"]
        return result
    d = load_driver(name)
    result.missing = _missing(name, d)
    if result.missing:
        return result
    result.ready = True
    try:
        result.version = d.upstream_version()
    except Exception:  # noqa: BLE001 - a version query never fails the check
        result.version = ""
    return result


def check_all(platform: Optional[str] = None) -> list[Readiness]:
    return [check(name, platform) for name in DRIVERS]
