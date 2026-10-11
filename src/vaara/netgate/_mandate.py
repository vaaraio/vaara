# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agents, their credentials and their mandates, in one file the gateway reloads.

An agent's mandate is the set of destinations it may reach. Everything outside
it is unreachable through the gateway, whatever the policy would have scored.
The credential is a random token the agent presents as its proxy password; the
file holds only its sha256, so a copy of the file does not let anyone act as
the agent.

Revocation is a field in the same file. The gateway watches the file and, the
moment an agent is revoked, closes every tunnel that agent has open. That is
the emergency brake: deterministic, outside the model, and on the record.

File shape::

    {"agents": {"builder": {"token_sha256": "...",
                            "allow": ["api.github.com", "*.pypi.org:443"],
                            "allow_private": false,
                            "created": "2026-10-11T12:00:00Z"}},
     "revoked": {"builder": "2026-10-11T12:30:00Z"}}

Allow patterns: ``host``, ``*.suffix`` (any subdomain, not the bare suffix),
``*`` (any public host), each with an optional ``:port``. With no port a
pattern admits 443 and 80 only, the two ports an agent's web traffic uses.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ports an entry without one admits, and the entry parser: both are
# ``vaara run``'s, so a mandate entry means the same thing on both surfaces.
from vaara.oslayer.egress import DEFAULT_PORTS, parse_rule

_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# A host name or IPv4 literal, with an optional leading "*.". Checked before
# the shared parser, which accepts any token without whitespace or "/".
_HOST = re.compile(r"^(\*\.)?[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)*$")


def default_path() -> Path:
    return Path(os.environ.get("VAARA_NETGATE_AGENTS",
                               Path.home() / ".vaara" / "netgate" / "agents.json"))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class AllowRule:
    """One mandate entry, in the same language as ``vaara run``'s egress list.

    The parsing and matching are :mod:`vaara.oslayer.egress`'s, so an entry
    means the same thing on both surfaces. The one addition is ``*``: any
    public host, for an agent whose mandate is "the web" and whose limits come
    from the policy and the egress floor instead of a list.
    """
    text: str
    _rule: Any = None

    @classmethod
    def parse(cls, text: str) -> "AllowRule":
        text = text.strip().lower()
        head, _, port = text.partition(":")
        if head == "*":
            if port and (not port.isdigit() or not 0 < int(port) < 65536):
                raise ValueError(f"bad port in allow rule {text!r}")
            return cls(text, None)
        if text.startswith("["):  # an IPv6 literal, [::1]:443
            literal = text[1:].partition("]")[0]
            try:
                ipaddress.ip_address(literal)
            except ValueError:
                raise ValueError(f"bad address in allow rule {text!r}") from None
        elif not _HOST.match(head):
            raise ValueError(f"bad host in allow rule {text!r}")
        try:
            return cls(text, parse_rule(text))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"bad allow rule {text!r}: {exc}") from exc

    def admits(self, host: str, port: int) -> bool:
        if self._rule is not None:
            return bool(self._rule.matches(host, port))
        _, _, wanted = self.text.partition(":")
        return port == int(wanted) if wanted else port in DEFAULT_PORTS

    def __str__(self) -> str:
        return self.text


@dataclass
class Agent:
    agent_id: str
    token_sha256: str
    allow: list[AllowRule] = field(default_factory=list)
    allow_private: bool = False
    created: str = ""

    def admits(self, host: str, port: int) -> bool:
        return any(rule.admits(host, port) for rule in self.allow)

    def to_json(self) -> dict:
        return {"token_sha256": self.token_sha256,
                "allow": [str(r) for r in self.allow],
                "allow_private": self.allow_private,
                "created": self.created}


class MandateStore:
    """The agents file, re-read whenever it changes on disk."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_path()
        self.agents: dict[str, Agent] = {}
        self.revoked: dict[str, str] = {}
        self._stamp: Optional[tuple[int, int]] = None
        self.reload(force=True)

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def reload(self, force: bool = False) -> bool:
        """Re-read the file if it changed. True when something was loaded."""
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            stamp = None
        if not force and stamp == self._stamp:
            return False
        data = self._read()
        agents = {}
        for agent_id, row in (data.get("agents") or {}).items():
            agents[agent_id] = Agent(
                agent_id=agent_id,
                token_sha256=str(row.get("token_sha256", "")),
                allow=[AllowRule.parse(r) for r in row.get("allow", [])],
                allow_private=bool(row.get("allow_private", False)),
                created=str(row.get("created", "")),
            )
        self.agents = agents
        self.revoked = dict(data.get("revoked") or {})
        self._stamp = stamp
        return True

    def authenticate(self, agent_id: str, token: str) -> Optional[Agent]:
        """The agent whose token this is, or None. Constant-time on the hash."""
        agent = self.agents.get(agent_id)
        if agent is None or not agent.token_sha256:
            return None
        if not hmac.compare_digest(agent.token_sha256, _sha256(token)):
            return None
        return agent

    def is_revoked(self, agent_id: str) -> bool:
        return agent_id in self.revoked

    # Writes. Each one rewrites the whole file atomically, so the gateway never
    # reads a half-written mandate.

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"agents": {a.agent_id: a.to_json() for a in self.agents.values()},
                "revoked": self.revoked}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".agents.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.reload(force=True)

    def add(self, agent_id: str, allow: list[str],
            allow_private: bool = False) -> str:
        """Create or re-key an agent. Returns the new token, shown once."""
        if not _AGENT_ID.match(agent_id):
            raise ValueError(f"bad agent id {agent_id!r}")
        rules = [AllowRule.parse(r) for r in allow]
        token = secrets.token_urlsafe(32)
        self.reload(force=True)
        self.agents[agent_id] = Agent(agent_id, _sha256(token), rules,
                                      allow_private, _now())
        self.revoked.pop(agent_id, None)
        self._write()
        return token

    def revoke(self, agent_id: str) -> None:
        self.reload(force=True)
        if agent_id not in self.agents:
            raise KeyError(agent_id)
        self.revoked[agent_id] = _now()
        self._write()
