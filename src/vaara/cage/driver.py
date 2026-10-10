# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The driver interface every cage implements.

A driver is the operator-side half of a cage: it starts an agent inside the
cage with a policy, stops it, reports its state, and hands back the cage's
own events. The deciding-side half is :func:`vaara.cage.observe`, which
runs inside the tree and needs no driver.

Five calls, the same for every cage:

- ``start(agent, policy, name=...)``: start ``agent`` (an argv) under the
  cage with ``policy`` (a path the cage understands, or None for the cage's
  own default). Returns a :class:`CageLaunch`.
- ``stop(name)``: end the launch.
- ``status(name)``: what the cage reports about the launch, as a dict.
- ``enforcement_state(name)``: the same, reduced to a :class:`CageState`:
  the cage, its version, the digest of the effective configuration, and
  whether the cage reports the confinement as on. This is the operator's
  view; the record's view comes from inside, at decision time.
- ``events(name, since)``: the cage's own events for the launch since a
  unix time, each as a dict with at least ``ts``, ``source`` and ``message``.

Drivers depend on unmodified upstream releases and shell out to the cage's
own tools. No forks, and no driver code outside Vaara's repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from vaara.cage import CageState


class CageError(RuntimeError):
    """The cage refused, is missing, or did not answer."""


@dataclass
class CageLaunch:
    """A launch a driver started."""

    driver: str
    name: str
    state: CageState
    pid: Optional[int] = None
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CageDriver(Protocol):
    name: str

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None) -> CageLaunch: ...

    def stop(self, name: str) -> None: ...

    def status(self, name: str) -> dict[str, Any]: ...

    def enforcement_state(self, name: Optional[str] = None) -> CageState: ...

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]: ...

    def upstream_version(self) -> str: ...
