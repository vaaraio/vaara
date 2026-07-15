# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Run the model proxy as a managed service (P2 of one-install governance).

``vaara proxy`` in a terminal dies with the terminal; the governance net has
to survive logout and come back on boot. This installs the proxy as a user
service — a launchd LaunchAgent on macOS, a systemd user unit on Linux — and
reverses it cleanly. Folded into ``vaara init --proxy-service`` /
``vaara ungovern``.

Activation shells out to ``launchctl`` / ``systemctl --user`` through an
injectable runner. When the service manager is unavailable (containers,
minimal hosts), the unit file is still written so a later boot picks it up,
and the caller gets a warning instead of a failure — installing must never
break ``vaara init``.
"""
from __future__ import annotations

import plistlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_LISTEN = "127.0.0.1:8788"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"

LAUNCHD_LABEL = "io.vaara.proxy"
SYSTEMD_UNIT = "vaara-proxy.service"

__all__ = [
    "LAUNCHD_LABEL",
    "SYSTEMD_UNIT",
    "ServiceReport",
    "install_proxy_service",
    "render_launchd_plist",
    "render_systemd_unit",
    "uninstall_proxy_service",
    "unit_path",
]


@dataclass
class ServiceReport:
    """Outcome of ``install_proxy_service`` for CLI rendering."""

    installed: bool = False
    path: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)


def _proxy_argv(vaara_bin: str, listen: str, upstream: str,
                trail_db: str) -> list[str]:
    return [
        vaara_bin, "proxy",
        "--listen", listen,
        "--upstream", upstream,
        "--trail", trail_db,
    ]


def render_launchd_plist(
    *,
    vaara_bin: str,
    listen: str,
    upstream: str,
    trail_db: str,
    log_dir: str,
) -> str:
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": _proxy_argv(vaara_bin, listen, upstream, trail_db),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": f"{log_dir}/proxy.out.log",
        "StandardErrorPath": f"{log_dir}/proxy.err.log",
    }
    return plistlib.dumps(plist).decode()


def render_systemd_unit(
    *,
    vaara_bin: str,
    listen: str,
    upstream: str,
    trail_db: str,
) -> str:
    exec_start = " ".join(_proxy_argv(vaara_bin, listen, upstream, trail_db))
    return (
        "[Unit]\n"
        "Description=Vaara model-endpoint governance proxy\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        f"ExecStart={exec_start}\n"
        "Restart=on-failure\n"
        "RestartSec=2\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def unit_path(system: str, home: Path) -> Optional[Path]:
    """Where the service unit lives for ``system``; None when unsupported."""
    if system == "darwin":
        return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if system == "linux":
        return home / ".config" / "systemd" / "user" / SYSTEMD_UNIT
    return None


def _run_quiet(runner: Callable[..., Any], cmd: list[str]) -> bool:
    """Run an activation command; False (never raise) when it can't run."""
    try:
        runner(cmd, capture_output=True, check=False)
        return True
    except OSError:
        return False


def install_proxy_service(
    *,
    vaara_bin: str,
    listen: str = DEFAULT_LISTEN,
    upstream: str = DEFAULT_UPSTREAM,
    trail_db: Optional[str] = None,
    home: Optional[Path] = None,
    system: Optional[str] = None,
    runner: Callable[..., Any] = subprocess.run,
) -> ServiceReport:
    """Write the user service unit and activate it. Idempotent."""
    home = home or Path.home()
    system = system or sys.platform
    trail_db = trail_db or str(home / ".vaara" / "trail" / "audit.db")
    report = ServiceReport()

    path = unit_path(system, home)
    if path is None:
        report.warnings.append(
            f"proxy service is not supported on {system!r}; run "
            "'vaara proxy' manually or under your own supervisor."
        )
        return report

    if system == "darwin":
        log_dir = home / ".vaara" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        text = render_launchd_plist(
            vaara_bin=vaara_bin, listen=listen, upstream=upstream,
            trail_db=trail_db, log_dir=str(log_dir),
        )
        activate = [
            ["launchctl", "unload", str(path)],  # replace any prior version
            ["launchctl", "load", "-w", str(path)],
        ]
        manager = "launchctl"
    else:
        text = render_systemd_unit(
            vaara_bin=vaara_bin, listen=listen, upstream=upstream,
            trail_db=trail_db,
        )
        activate = [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT],
        ]
        manager = "systemctl"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    report.installed = True
    report.path = path

    if not all(_run_quiet(runner, cmd) for cmd in activate):
        report.warnings.append(
            f"{manager} unavailable; the service unit was written to "
            f"{path} but not started. It will activate on next login/boot."
        )
    return report


def uninstall_proxy_service(
    *,
    home: Optional[Path] = None,
    system: Optional[str] = None,
    runner: Callable[..., Any] = subprocess.run,
) -> bool:
    """Deactivate and remove the service unit. True when one was removed."""
    home = home or Path.home()
    system = system or sys.platform
    path = unit_path(system, home)
    if path is None or not path.exists():
        return False
    if system == "darwin":
        _run_quiet(runner, ["launchctl", "unload", str(path)])
    else:
        _run_quiet(runner, ["systemctl", "--user", "disable", "--now",
                            SYSTEMD_UNIT])
    path.unlink()
    return True
