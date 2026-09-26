# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The floor, rendered as the ``vaara-agent`` AppArmor profile.

Two profiles, one family. ``vaara-agent`` is the harness program itself
(the process ``vaara run`` starts, and any known harness binary it execs).
Everything a harness execs that is not a harness binary runs in
``vaara-agent//tool``, and everything under that stays there. The harness
keeps write access to its own settings, because it writes them in normal use;
its tools do not. Both keep no access at all to Vaara's own state.

The operator's block folders join the floor: no process in the agent's tree
reads, writes, moves or deletes anything in them. In the ask and record
folders the guard decides each open and exec; the profile only keeps a hard
link from pointing into them, since a second name outside the folder would be
opened without the guard seeing the folder's path.

Apps the operator picked are attached by path, so the profile applies to them
however they are started.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Optional

PROFILE = "vaara-agent"
TOOL = "tool"

# No access at all, relative to each governed home.
HOME_SEALED = (".vaara",)

# Never written by any process in the tree, relative to each governed home:
# pure hook files no harness writes itself, and what a login shell, the
# session, systemd, SSH and git run from.
HOME_FIXED = (
    ".codex/hooks.json",
    ".copilot/hooks",
    ".cursor/hooks.json",
    ".cursor/hooks",
    ".claude/hooks",
    ".config/opencode/plugin",
    ".config/opencode/plugins",
    ".bashrc", ".bash_profile", ".bash_login", ".bash_logout", ".profile",
    ".zshrc", ".zprofile", ".zshenv", ".zlogin", ".zlogout",
    ".config/fish/config.fish", ".config/fish/conf.d",
    ".pam_environment", ".xprofile", ".xsessionrc", ".xinitrc",
    ".config/environment.d",
    ".config/systemd/user",
    ".local/share/systemd/user",
    ".config/autostart",
    ".ssh/authorized_keys", ".ssh/authorized_keys2", ".ssh/rc", ".ssh/environment",
    ".gitconfig", ".config/git/config",
)

# Written by the harness program, never by its tools, relative to each home.
HOME_HARNESS = (
    ".claude.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/skills",
    ".claude/agents",
    ".claude/plugins",
    ".claude/scheduled_tasks.json",
    ".claude/CLAUDE.md",
    ".codex/config.toml",
    ".gemini/settings.json",
    ".copilot/config.json",
    ".copilot/settings.json",
    ".copilot/settings.local.json",
    ".copilot/mcp-config.json",
    ".cursor/mcp.json",
    ".config/opencode/opencode.json",
    ".config/opencode/opencode.jsonc",
)

# The same, in any project directory.
PROJECT_FIXED = (
    "/**/.github/hooks/",
    "/**/.opencode/plugin/",
    "/**/.opencode/plugins/",
    "/**/.git/hooks/{applypatch-msg,commit-msg,fsmonitor-watchman,post-applypatch,"
    "post-checkout,post-commit,post-merge,post-receive,post-rewrite,post-update,"
    "pre-applypatch,pre-auto-gc,pre-commit,pre-merge-commit,pre-push,pre-rebase,"
    "pre-receive,prepare-commit-msg,push-to-checkout,reference-transaction,"
    "sendemail-validate,update}",
)
PROJECT_HARNESS = (
    "/**/.claude/settings.json",
    "/**/.claude/settings.local.json",
    "/**/.claude/hooks/",
    "/**/.claude/agents/",
    "/**/.claude/skills/",
    "/**/.mcp.json",
    "/**/.gemini/settings.json",
    "/**/.codex/config.toml",
    "/**/.codex/hooks.json",
    "/**/.cursor/hooks.json",
    "/**/.cursor/mcp.json",
    "/**/.github/copilot/",
    "/**/.claude-plugin/",
)

# System paths, for a process in the tree that runs as root through sudo.
SYSTEM_SEALED = ("/var/lib/vaara/", "/etc/vaara/", "/run/vaara/")
SYSTEM_FIXED = (
    "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/gshadow",
    "/etc/sudoers", "/etc/sudoers.d/",
    "/etc/crontab", "/etc/cron.d/", "/etc/cron.hourly/", "/etc/cron.daily/",
    "/etc/cron.weekly/", "/etc/cron.monthly/", "/var/spool/cron/",
    "/etc/systemd/", "/lib/systemd/", "/usr/lib/systemd/",
    "/etc/init.d/", "/etc/rc.local",
    "/etc/ld.so.preload", "/etc/ld.so.conf", "/etc/ld.so.conf.d/",
    "/etc/profile", "/etc/profile.d/", "/etc/bash.bashrc", "/etc/zsh/",
    "/etc/environment", "/etc/pam.d/", "/etc/security/", "/etc/ssh/",
    "/etc/apparmor/", "/etc/apparmor.d/", "/etc/udev/",
    "/etc/modprobe.d/", "/etc/modules-load.d/", "/boot/",
    "/sys/kernel/security/", "/sys/fs/cgroup/", "/proc/sys/",
    "/proc/sysrq-trigger",
    "/dev/mem", "/dev/kmem", "/dev/port",
    "/dev/sd*", "/dev/nvme*", "/dev/mmcblk*", "/dev/vd*", "/dev/xvd*",
    "/dev/dm-*", "/dev/mapper/", "/dev/loop*",
)

# Sockets of services that start programs on a caller's behalf outside the
# tree. Connecting to a path socket needs write access to its path.
LAUNCHER_SOCKETS = (
    "/run/docker.sock", "/var/run/docker.sock", "/run/containerd/",
    "/run/podman/", "/run/user/*/podman/", "/run/user/*/docker.sock",
    "/run/systemd/private", "/run/user/*/systemd/",
    # AppArmor rule paths for launcher sockets, not temp files this code opens.
    "/tmp/tmux-*/", "/run/screen/", "/var/run/screen/",  # nosec B108
    "/tmp/.X11-unix/", "/run/user/*/wayland-*", "/run/user/*/vscode-*.sock",  # nosec B108
)

DENIED_CAPABILITIES = (
    "mac_admin", "mac_override", "sys_admin", "sys_module", "sys_rawio",
    "sys_boot", "sys_ptrace", "dac_read_search", "bpf", "perfmon",
    "audit_control", "linux_immutable", "syslog",
)

# npm loaders that start the real harness binary as a child, and where npm
# puts that binary: hoisted beside the loader's package, or nested in it.
_LOADER_COMPANIONS = {
    ("@github", "copilot", "npm-loader.js"): (
        "@github/copilot-*/copilot",
        "@github/copilot/node_modules/@github/copilot-*/copilot",
    ),
}


def harness_paths(paths: Iterable[str]) -> list[str]:
    """Concrete harness binary paths for the top profile.

    Exact paths, never globs: AppArmor refuses two exec rules that overlap
    with different transitions, and an exact path is the one kind that
    overrides the catch-all ``/**`` rule. A path an npm loader starts as a
    child comes along with the loader.
    """
    import glob

    out: dict[str, None] = {}
    for raw in paths:
        real = os.path.realpath(raw)
        if not os.path.isfile(real):
            continue
        out[real] = None
        parts = Path(real).parts
        for (scope, pkg, name), companions in _LOADER_COMPANIONS.items():
            if parts[-3:] == (scope, pkg, name):
                node_modules = Path(*parts[:-3])
                for companion in companions:
                    for hit in sorted(glob.glob(str(node_modules / companion))):
                        if os.path.isfile(hit):
                            out[os.path.realpath(hit)] = None
    return list(out)


def _q(path: str) -> str:
    """A path for an AppArmor rule: quoted when it holds a space."""
    return f'"{path}"' if any(c.isspace() for c in path) else path


def _tree(path: str) -> list[str]:
    """The entry and everything under it, for a file or a directory path."""
    base = path.rstrip("/")
    return [base, f"{base}/", f"{base}/**"] if path.endswith("/") else [base, f"{base}/**"]


def _ancestors(home: str, rel: str) -> list[str]:
    """Each directory between ``home`` and ``rel``, so none can be renamed away."""
    parts = Path(rel).parts[:-1]
    return [f"{home}/{'/'.join(parts[: i + 1])}/" for i in range(len(parts))]


def vaara_install_paths() -> list[str]:
    """What the unconfined side runs from: the vaara package and its Python."""
    import vaara

    paths = {str(Path(vaara.__file__).resolve().parent) + "/"}
    for key in ("stdlib", "platstdlib", "purelib", "platlib", "scripts"):
        p = sysconfig.get_paths().get(key)
        if p:
            paths.add(str(Path(p).resolve()) + "/")
    exe = Path(sys.executable)
    paths.add(str(exe))
    paths.add(str(exe.resolve()))
    return sorted(paths)


def abi_line(abi_dir: Path = Path("/etc/apparmor.d/abi")) -> str:
    """The newest policy ABI this machine's parser ships."""
    for version in ("4.0", "3.0"):
        if (abi_dir / version).exists():
            return f"abi <abi/{version}>,"
    return ""


def attachment(apps: Iterable[str]) -> str:
    """The profile's attachment for exact app paths, or ``""`` for none."""
    paths = sorted({a for a in apps if a.startswith("/") and len(a) > 1})
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    return "/{" + ",".join(p[1:] for p in paths) + "}"


def render(homes: Iterable[str], *,
           block_folders: Iterable[str] = (),
           watched_folders: Iterable[str] = (),
           harness_binaries: Iterable[str] = (),
           apps: Iterable[str] = (),
           install_paths: Optional[Iterable[str]] = None,
           abi: Optional[str] = None) -> str:
    """The profile text for ``homes``, the operator's folders and apps."""
    homes = [str(Path(h)).rstrip("/") for h in homes]
    watched = [str(Path(f)).rstrip("/") + "/**" for f in watched_folders]
    installs = list(vaara_install_paths() if install_paths is None else install_paths)
    harness = harness_paths(harness_binaries)

    sealed: list[str] = []
    fixed: list[str] = []
    harness_files: list[str] = []
    for home in homes:
        for rel in HOME_SEALED:
            sealed += _tree(f"{home}/{rel}/")
        for rel in HOME_FIXED:
            fixed += _ancestors(home, rel) + _tree(f"{home}/{rel}")
        for rel in HOME_HARNESS:
            fixed += _ancestors(home, rel)
            harness_files += _tree(f"{home}/{rel}")
    for folder in block_folders:
        sealed += _tree(str(Path(folder)).rstrip("/") + "/")
    for path in SYSTEM_SEALED:
        sealed += _tree(path)
    for path in list(PROJECT_FIXED) + list(SYSTEM_FIXED) + list(LAUNCHER_SOCKETS) + installs:
        fixed += _tree(path) if path.endswith("/") else [path]
    for path in PROJECT_HARNESS:
        harness_files += _tree(path) if path.endswith("/") else [path]

    def rules(tool: bool) -> list[str]:
        out = ["  /** rwlkm,"]
        if tool:
            out.append("  /** ix,")
        else:
            out.append(f"  /** Cx -> {TOOL},")
        for b in harness:
            out.append(f"  {_q(b)} Px -> {PROFILE},")
        out += [
            "  capability,",
            *[f"  audit deny capability {c}," for c in DENIED_CAPABILITIES],
            "  network,",
            "  unix,",
            "  signal (receive),",
            # A peer glob does not cross the //, so the child profile is named.
            f"  signal (send) peer={PROFILE},",
            f"  signal (send) peer={PROFILE}//*,",
            "  ptrace (read, readby, tracedby),",
            f"  ptrace (trace) peer={PROFILE},",
            f"  ptrace (trace) peer={PROFILE}//*,",
            "  audit deny dbus send bus=session peer=(name=org.freedesktop.systemd1),",
            "  audit deny dbus send bus=system peer=(name=org.freedesktop.systemd1),",
            "  dbus,",
            "  audit deny mount,",
            "  audit deny umount,",
            "  audit deny pivot_root,",
        ]
        # `audit deny`, not `deny`: AppArmor keeps an explicit deny out of the
        # kernel log, and the guard records the floor's refusals from there.
        out += [f"  audit deny link /** -> {_q(p)}," for p in dict.fromkeys(watched)]
        out += [f"  audit deny {_q(p)} mrwlkx," for p in dict.fromkeys(sealed)]
        out += [f"  audit deny {_q(p)} wl," for p in dict.fromkeys(fixed)]
        if tool:
            out += [f"  audit deny {_q(p)} wl," for p in dict.fromkeys(harness_files)]
        return out

    abi = abi_line() if abi is None else abi
    lines = [
        "# Written by vaara os-guard. Regenerated on every start; edits are lost.",
        *( [abi] if abi else [] ),
        "include <tunables/global>",
        f"profile {' '.join(filter(None, (PROFILE, attachment(apps))))} "
        f"flags=(attach_disconnected) {{",
        *rules(tool=False),
        "",
        f"  profile {TOOL} flags=(attach_disconnected) {{",
        *["  " + r for r in rules(tool=True)],
        "  }",
        "}",
        "",
    ]
    return "\n".join(lines)


def label_of(pid: int) -> str:
    """The AppArmor label of ``pid``, without the mode suffix, or ``""``."""
    for attr in (f"/proc/{pid}/attr/apparmor/current", f"/proc/{pid}/attr/current"):
        try:
            raw = Path(attr).read_text().strip()
        except OSError:
            continue
        return raw.rsplit(" (", 1)[0] if raw.endswith(")") else raw
    return ""


def is_agent_label(label: str) -> bool:
    return label == PROFILE or label.startswith(PROFILE + "//")


def apparmor_enabled() -> bool:
    try:
        return Path("/sys/module/apparmor/parameters/enabled").read_text().strip() == "Y"
    except OSError:
        return False


def profile_loaded(name: str = PROFILE) -> bool:
    """Whether the kernel has ``name`` loaded (root, or a readable policy dir)."""
    try:
        text = Path("/sys/kernel/security/apparmor/profiles").read_text()
    except OSError:
        return False
    return any(line.split(" (", 1)[0] == name for line in text.splitlines())


def home_of(uid: int) -> str:
    import pwd

    return pwd.getpwuid(uid).pw_dir


def exec_attr_path() -> str:
    new = "/proc/self/attr/apparmor/exec"
    return new if os.path.exists(new) else "/proc/self/attr/exec"
