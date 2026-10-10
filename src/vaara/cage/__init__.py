# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The cage layer: the containment an agent runs in, as a choosable part.

Vaara decides and records; the cage confines. Several cages exist and more
are coming, so Vaara does not insist on its own. A cage joins through a
driver (:mod:`vaara.cage.driver`), and every decision record carries a
``cage`` block saying which cage the deciding process ran in, what its
effective configuration was, and whether the confinement was confirmed
live at decision time. A record from an unconfined run says so.

Two sides, one contract:

- The launcher side. A driver starts the agent in its cage and hands the
  governed tree four environment variables: ``VAARA_CAGE`` (the driver
  name), ``VAARA_CAGE_DIGEST`` (``sha256:`` of the effective cage
  configuration), ``VAARA_CAGE_UPSTREAM`` (the cage's own name and
  version) and ``VAARA_CAGE_NAME`` (this launch's name in the cage).
- The deciding side. :func:`observe` runs inside the tree, at decision
  time. It reads the declaration and then asks the kernel whether the
  confinement that cage imposes is on this process: the ``vaara-agent``
  AppArmor label for the Vaara cage, a seccomp filter with
  ``no_new_privs`` for OpenShell. The declaration says which cage; the
  kernel says whether it holds. ``confirmed`` is true only when both agree.

Without a declaration the block says ``driver: none``. A process that is
confined but was not started through a driver is not guessed at: a
container's default seccomp profile looks the same from inside as a cage,
and a record must not claim a cage on that evidence.

Drivers shipped: ``vaara-cage`` (``vaara run``: AppArmor, cgroup, the
fanotify guard) and ``openshell`` (NVIDIA OpenShell through its CLI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CAGE_ENV = "VAARA_CAGE"
DIGEST_ENV = "VAARA_CAGE_DIGEST"
UPSTREAM_ENV = "VAARA_CAGE_UPSTREAM"
NAME_ENV = "VAARA_CAGE_NAME"
# Set by ``vaara run`` for the hook it runs on behalf of an agent inside the
# Vaara cage: the agent's pid, from the relay socket's peer credentials.
PEER_ENV = "VAARA_CAGE_PEER_PID"

#: The driver names Vaara ships, in the order of .shared's driver list.
DRIVERS = (
    "vaara-cage", "openshell", "codex", "sandbox-runtime", "gvisor",
    "firecracker", "kata", "agent-sandbox", "microsandbox", "nono", "e2b",
    "apple-container",
)

NONE = "none"

# How a confirmation was reached. The deciding side names the kernel fact it
# read; the operator side names the cage's own control plane.
BASIS_NONE = "none"                   # nothing declared, nothing claimed
BASIS_DECLARED = "declared"           # declared by the launcher, not confirmed
BASIS_APPARMOR = "apparmor_label"     # the process carries the vaara-agent label
BASIS_SECCOMP = "seccomp_filter"      # a seccomp filter and no_new_privs are on
BASIS_NO_NEW_PRIVS = "no_new_privs"   # no_new_privs is on (what Landlock requires)
BASIS_BWRAP = "bwrap_init"            # pid 1 of this pid namespace is bubblewrap
BASIS_GVISOR = "gvisor_kernel_log"    # the kernel log is gVisor's own
BASIS_VM = "hypervisor_present"       # the CPU reports a hypervisor underneath
BASIS_GUARD = "guard_status"          # the Vaara OS guard reports it (operator side)
BASIS_GATEWAY = "gateway_status"      # the OpenShell gateway reports it (operator side)
BASIS_PROCESS = "process_alive"       # the launcher's child is still running (operator side)
BASIS_ENGINE = "engine_status"        # the container engine reports it (operator side)
BASIS_API = "api_status"              # the cage's HTTP API reports it (operator side)
BASIS_CONTROL = "control_plane"       # a cluster or session store reports it (operator side)


@dataclass(frozen=True)
class CageState:
    """What is known about the cage around one process or one launch."""

    driver: str = NONE
    upstream: str = ""
    config_digest: str = ""
    confirmed: bool = False
    basis: str = BASIS_NONE
    name: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """The ``cage`` block written into a decision record.

        Stable keys, strings and one boolean, no floats, nothing machine
        specific beyond what the launcher declared. An unconfined run is
        two keys; a declared cage adds the declaration and the basis.
        """
        block: dict[str, Any] = {"driver": self.driver, "confirmed": bool(self.confirmed)}
        if self.driver == NONE:
            return block
        block["upstream"] = self.upstream
        block["config_digest"] = self.config_digest
        block["basis"] = self.basis
        if self.name:
            block["name"] = self.name
        return block


# The same declaration on a guest kernel's command line, for the microVM
# cages: a launcher cannot set environment variables inside a guest it only
# boots, but it writes the boot arguments.
CMDLINE_KEYS = {
    "vaara.cage": CAGE_ENV, "vaara.cage.digest": DIGEST_ENV,
    "vaara.cage.upstream": UPSTREAM_ENV, "vaara.cage.name": NAME_ENV,
}


def cmdline_declaration(path: str = "/proc/cmdline") -> dict[str, str]:
    """The ``vaara.cage*`` tokens of the kernel command line, as env names."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for token in text.split():
        key, sep, value = token.partition("=")
        if sep and key in CMDLINE_KEYS:
            out[CMDLINE_KEYS[key]] = value
    return out


def declared(environ: Optional[dict[str, str]] = None) -> Optional[CageState]:
    """The cage the launcher declared to this tree, unconfirmed, or None.

    The environment is read first; a guest with no declaration there is
    checked for one on its kernel command line.
    """
    env = os.environ if environ is None else environ
    if not (env.get(CAGE_ENV) or "").strip():
        env = cmdline_declaration() if environ is None else env
    driver = (env.get(CAGE_ENV) or "").strip()
    if not driver:
        return None
    return CageState(
        driver=driver,
        upstream=(env.get(UPSTREAM_ENV) or "").strip(),
        config_digest=(env.get(DIGEST_ENV) or "").strip(),
        confirmed=False,
        basis=BASIS_DECLARED,
        name=(env.get(NAME_ENV) or "").strip(),
    )


def _proc_status(pid: str = "self") -> dict[str, str]:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def seccomp_filter_on(pid: str = "self") -> bool:
    """True when the kernel says this process runs under a seccomp filter
    and cannot gain privileges. Both are what OpenShell's sandbox sets on
    the main process and everything it starts."""
    status = _proc_status(pid)
    return status.get("Seccomp") == "2" and status.get("NoNewPrivs") == "1"


def apparmor_agent_label(pid: int | str = "self") -> bool:
    """True when this process carries the ``vaara-agent`` AppArmor label."""
    from vaara.oslayer import floor

    try:
        label = floor.label_of(pid)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - a probe never raises into a decision
        return False
    return floor.is_agent_label(label)


def no_new_privs_on(pid: str = "self") -> bool:
    """True when the process cannot gain privileges. Landlock requires it,
    so a Landlock-only cage (nono with ``--sandbox-policy landlock``) shows
    this and nothing else from inside."""
    return _proc_status(pid).get("NoNewPrivs") == "1"


def bwrap_is_init() -> bool:
    """True when pid 1 of this pid namespace is bubblewrap, which is what
    sandbox-runtime's ``--unshare-pid`` leaves in place."""
    try:
        return Path("/proc/1/comm").read_text(encoding="utf-8").strip() == "bwrap"
    except OSError:
        return False


def gvisor_kernel_log() -> bool:
    """True when the kernel log is gVisor's: its sentry answers
    ``syslog(SYSLOG_ACTION_READ_ALL)`` with its own fixed opening line."""
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        buf = ctypes.create_string_buffer(8192)
        n = libc.klogctl(3, buf, len(buf))  # SYSLOG_ACTION_READ_ALL
    except (OSError, AttributeError):
        return False
    if n <= 0:
        return False
    return b"Starting gVisor" in buf.raw[:n]


def hypervisor_present() -> bool:
    """True when the CPU reports a hypervisor underneath this kernel: the
    ``hypervisor`` flag on x86, the hypervisor node of the device tree on
    arm64, or a ``/sys/hypervisor/type``. What a microVM or a VM-backed
    container shows from inside; a bare container does not."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("flags") and " hypervisor" in line:
                return True
    except OSError:
        pass
    for probe in ("/sys/hypervisor/type", "/proc/device-tree/hypervisor/compatible",
                  "/sys/firmware/devicetree/base/hypervisor/compatible"):
        if Path(probe).exists():
            return True
    return False


# Per driver: the kernel check that confirms the declaration from inside.
# A driver maps to a tuple of (probe, basis) pairs tried in order; the first
# that holds names the basis.
_CONFIRM: dict[str, tuple[tuple[Any, str], ...]] = {
    "vaara-cage": ((apparmor_agent_label, BASIS_APPARMOR),),
    "openshell": ((seccomp_filter_on, BASIS_SECCOMP),),
    "codex": ((seccomp_filter_on, BASIS_SECCOMP),),
    "sandbox-runtime": ((bwrap_is_init, BASIS_BWRAP),),
    "nono": ((seccomp_filter_on, BASIS_SECCOMP), (no_new_privs_on, BASIS_NO_NEW_PRIVS)),
    "gvisor": ((gvisor_kernel_log, BASIS_GVISOR),),
    "agent-sandbox": ((gvisor_kernel_log, BASIS_GVISOR), (hypervisor_present, BASIS_VM)),
    "kata": ((hypervisor_present, BASIS_VM),),
    "firecracker": ((hypervisor_present, BASIS_VM),),
    "microsandbox": ((hypervisor_present, BASIS_VM),),
    "e2b": ((hypervisor_present, BASIS_VM),),
    "apple-container": ((hypervisor_present, BASIS_VM),),
}


def observe(environ: Optional[dict[str, str]] = None) -> CageState:
    """The cage around the calling process, confirmed where the kernel can.

    Called by the trail for every decision it records. Cheap: one
    environment read and one ``/proc/self`` read.
    """
    state = declared(environ)
    if state is None:
        return CageState()
    for probe, basis in _CONFIRM.get(state.driver, ()):
        try:
            held = bool(probe())
        except Exception:  # noqa: BLE001 - a probe never raises into a decision
            held = False
        if held:
            return CageState(
                driver=state.driver, upstream=state.upstream,
                config_digest=state.config_digest, confirmed=True, basis=basis,
                name=state.name,
            )
    # Inside the Vaara cage the hook runs outside the floor, in vaara run, so
    # the process to check is the agent that asked: vaara run names it.
    env = os.environ if environ is None else environ
    peer = (env.get(PEER_ENV) or "").strip()
    if state.driver == "vaara-cage" and peer.isdigit() and apparmor_agent_label(int(peer)):
        return CageState(
            driver=state.driver, upstream=state.upstream,
            config_digest=state.config_digest, confirmed=True, basis=BASIS_APPARMOR,
            name=state.name, detail={"pid": int(peer)},
        )
    return state


def environ_for(state: CageState) -> dict[str, str]:
    """The four variables a launcher hands the governed tree."""
    env = {CAGE_ENV: state.driver, DIGEST_ENV: state.config_digest,
           UPSTREAM_ENV: state.upstream}
    if state.name:
        env[NAME_ENV] = state.name
    return env


_MODULES = {
    "vaara-cage": ("vaara.cage.vaara_cage", "VaaraCageDriver"),
    "openshell": ("vaara.cage.openshell", "OpenShellDriver"),
    "codex": ("vaara.cage.codex", "CodexSandboxDriver"),
    "sandbox-runtime": ("vaara.cage.sandbox_runtime", "SandboxRuntimeDriver"),
    "gvisor": ("vaara.cage.gvisor", "GVisorDriver"),
    "firecracker": ("vaara.cage.firecracker", "FirecrackerDriver"),
    "kata": ("vaara.cage.kata", "KataDriver"),
    "agent-sandbox": ("vaara.cage.agent_sandbox", "AgentSandboxDriver"),
    "microsandbox": ("vaara.cage.microsandbox", "MicrosandboxDriver"),
    "nono": ("vaara.cage.nono", "NonoDriver"),
    "e2b": ("vaara.cage.e2b", "E2BDriver"),
    "apple-container": ("vaara.cage.apple_container", "AppleContainerDriver"),
}


def load_driver(name: str, **kwargs: Any):
    """The driver called ``name``. Raises ``ValueError`` for an unknown one."""
    try:
        module_name, class_name = _MODULES[name]
    except KeyError:
        raise ValueError(
            f"unknown cage driver {name!r}; Vaara ships {', '.join(DRIVERS)}"
        ) from None
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


__all__ = [
    "CAGE_ENV", "DIGEST_ENV", "UPSTREAM_ENV", "NAME_ENV", "DRIVERS", "NONE",
    "CageState", "declared", "observe", "environ_for", "load_driver",
    "seccomp_filter_on", "apparmor_agent_label", "no_new_privs_on", "bwrap_is_init",
    "gvisor_kernel_log", "hypervisor_present", "cmdline_declaration",
]
