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

#: The driver names Vaara ships. The registry maps each to its module.
DRIVERS = ("vaara-cage", "openshell")

NONE = "none"

# How a confirmation was reached.
BASIS_NONE = "none"                   # nothing declared, nothing claimed
BASIS_DECLARED = "declared"           # declared by the launcher, not confirmed
BASIS_APPARMOR = "apparmor_label"     # the process carries the vaara-agent label
BASIS_SECCOMP = "seccomp_filter"      # a seccomp filter and no_new_privs are on
BASIS_GUARD = "guard_status"          # the Vaara OS guard reports it (operator side)
BASIS_GATEWAY = "gateway_status"      # the OpenShell gateway reports it (operator side)


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


def declared(environ: Optional[dict[str, str]] = None) -> Optional[CageState]:
    """The cage the launcher declared to this tree, unconfirmed, or None."""
    env = os.environ if environ is None else environ
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
        text = Path(f"/proc/{pid}/status").read_text()
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


# Per driver: the kernel check that confirms the declaration from inside.
_CONFIRM = {
    "vaara-cage": (apparmor_agent_label, BASIS_APPARMOR),
    "openshell": (seccomp_filter_on, BASIS_SECCOMP),
}


def observe(environ: Optional[dict[str, str]] = None) -> CageState:
    """The cage around the calling process, confirmed where the kernel can.

    Called by the trail for every decision it records. Cheap: one
    environment read and one ``/proc/self`` read.
    """
    state = declared(environ)
    if state is None:
        return CageState()
    check = _CONFIRM.get(state.driver)
    if check is None:
        return state
    probe, basis = check
    try:
        held = bool(probe())
    except Exception:  # noqa: BLE001 - a probe never raises into a decision
        held = False
    if not held:
        return state
    return CageState(
        driver=state.driver, upstream=state.upstream, config_digest=state.config_digest,
        confirmed=True, basis=basis, name=state.name,
    )


def environ_for(state: CageState) -> dict[str, str]:
    """The four variables a launcher hands the governed tree."""
    env = {CAGE_ENV: state.driver, DIGEST_ENV: state.config_digest,
           UPSTREAM_ENV: state.upstream}
    if state.name:
        env[NAME_ENV] = state.name
    return env


def load_driver(name: str, **kwargs: Any):
    """The driver called ``name``. Raises ``ValueError`` for an unknown one."""
    if name == "vaara-cage":
        from vaara.cage.vaara_cage import VaaraCageDriver

        return VaaraCageDriver(**kwargs)
    if name == "openshell":
        from vaara.cage.openshell import OpenShellDriver

        return OpenShellDriver(**kwargs)
    raise ValueError(f"unknown cage driver {name!r}; Vaara ships {', '.join(DRIVERS)}")


__all__ = [
    "CAGE_ENV", "DIGEST_ENV", "UPSTREAM_ENV", "NAME_ENV", "DRIVERS", "NONE",
    "CageState", "declared", "observe", "environ_for", "load_driver",
    "seccomp_filter_on", "apparmor_agent_label",
]
