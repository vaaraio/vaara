# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The gateway: every connection an agent opens, decided, recorded, revocable.

An agent reaches the network only through this proxy: by ``HTTPS_PROXY`` /
``HTTP_PROXY`` inside a cage whose firewall allows nothing else, or by a
WireGuard full tunnel whose far end allows nothing else (see ``_wg``). Each
connection goes through five checks, in order, and the first that fails ends
it:

1. **Who.** The agent's id and token, as the proxy username and password. An
   unknown or wrong credential gets no reply at all: the socket is closed
   without a byte, so a stranger cannot learn what is behind the gateway.
2. **Still trusted.** A revoked agent is refused.
3. **Mandate.** The destination must be in the agent's allow list. Outside it,
   refused, whatever the policy would have said.
4. **Policy.** The Vaara pipeline scores ``net.connect`` like any other action.
   A deny is refused. A hold for a human is refused too, and closed on the
   record as refused, because a proxy connection cannot wait for a person.
5. **Egress floor.** The name is resolved once, private and cloud-metadata
   addresses are refused unless the agent is allowed private hosts, and the
   socket dials the exact address that passed, so a DNS answer that changes
   between the check and the connect cannot redirect it.

Every decision, allowed or refused, is one ``net.connect`` entry on the trail,
and every connection that opened gets a ``net.close`` entry with its byte
counts, its duration and why it ended. Revoking an agent closes its open
connections at once; each of those ends with reason ``revoked``.

What this does not see: traffic that never reaches the gateway. That is the
cage's or the tunnel's job, and the reason the gateway is meant to be the only
way out.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Optional
from urllib.parse import urlsplit

from vaara.integrations._egress_guard import EgressBlocked, pick_egress_ip

from ._mandate import Agent, MandateStore

logger = logging.getLogger("vaara.netgate")

_MAX_HEAD = 16 * 1024
_HEAD_TIMEOUT = 10.0
_CONNECT_TIMEOUT = 10.0
_CHUNK = 64 * 1024
#: How often the gateway re-reads the agents file for revocations.
_WATCH_INTERVAL = 0.25
#: One trail entry per stranger address per this many seconds, so a flood of
#: unauthenticated connections cannot be used to fill the trail.
_STRANGER_RECORD_EVERY = 10.0


@dataclass
class _Live:
    """One open connection, so a revocation can find and end it."""
    conn_id: str
    agent_id: str
    host: str
    port: int
    action_id: str
    started: float
    tasks: list = field(default_factory=list)
    writers: list = field(default_factory=list)
    up: int = 0
    down: int = 0
    reason: str = "closed"


class _Refused(Exception):
    def __init__(self, status: int, message: str, silent: bool = False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.silent = silent


def _parse_basic(value: str) -> Optional[tuple[str, str]]:
    scheme, _, blob = value.strip().partition(" ")
    if scheme.lower() != "basic" or not blob:
        return None
    try:
        user, sep, password = base64.b64decode(blob.strip(), validate=True).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        return None
    return (user, password) if sep else None


class NetGate:
    """An asyncio forward proxy that governs agents' outbound connections."""

    def __init__(self, store: MandateStore, pipeline: Any, *,
                 host: str = "127.0.0.1", port: int = 8791,
                 challenge: bool = False):
        self.store = store
        self.pipeline = pipeline
        self.host = host
        self.port = port
        # Off by default: an unauthenticated client gets silence. A client
        # that only sends credentials after a 407 needs this on.
        self.challenge = challenge
        self.live: dict[str, _Live] = {}
        self._stranger_seen: dict[str, float] = {}
        self._server: Optional[asyncio.base_events.Server] = None
        self._watch: Optional[asyncio.Task] = None

    # Recording. A record that fails is logged and never decides a connection
    # on its own: the decision was already made by the checks above.

    def _record(self, agent_id: str, tool: str, params: dict, **kwargs) -> Any:
        try:
            return self.pipeline.intercept(agent_id=agent_id, tool_name=tool,
                                           parameters=params, **kwargs)
        except Exception as exc:
            logger.error("could not record %s for %s: %r", tool, agent_id, exc)
            return None

    def _refuse_record(self, agent_id: str, params: dict, policy_id: str,
                       reason: str) -> None:
        self._record(agent_id, "net.connect", dict(params, refused=policy_id),
                     policy_decision="deny", policy_reason=reason,
                     policy_id=policy_id)

    # Lifecycle

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        sock = self._server.sockets[0].getsockname()
        self.port = sock[1]
        self._watch = asyncio.ensure_future(self._watch_revocations())
        logger.info("net-gate listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._watch:
            self._watch.cancel()
        for live in list(self.live.values()):
            self._end(live, "gateway stopped")
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    # The brake

    async def _watch_revocations(self) -> None:
        while True:
            await asyncio.sleep(_WATCH_INTERVAL)
            try:
                if self.store.reload():
                    self.enforce_revocations()
            except Exception as exc:  # a bad edit must not stop the watcher
                logger.error("agents file could not be reloaded: %r", exc)

    def enforce_revocations(self) -> int:
        """Close every live connection of every revoked agent. Returns how many."""
        n = 0
        for live in list(self.live.values()):
            if self.store.is_revoked(live.agent_id):
                self._end(live, "revoked")
                n += 1
        return n

    def _end(self, live: _Live, reason: str) -> None:
        live.reason = reason
        for w in live.writers:
            try:
                w.transport.abort()
            except Exception:
                pass
        for t in live.tasks:
            t.cancel()

    # One client connection

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        peer_s = f"{peer[0]}:{peer[1]}" if peer else "?"
        try:
            await self._serve(reader, writer, peer_s)
        except _Refused as r:
            if not r.silent:
                body = (f"Vaara net-gate: {r.message}\n").encode()
                head = (f"HTTP/1.1 {r.status} {HTTPStatus(r.status).phrase}\r\n"
                        f"Content-Type: text/plain\r\nContent-Length: {len(body)}\r\n"
                        f"Connection: close\r\n")
                if r.status == 407:
                    head += 'Proxy-Authenticate: Basic realm="vaara-net-gate"\r\n'
                try:
                    writer.write(head.encode() + b"\r\n" + body)
                    await writer.drain()
                except Exception:
                    pass
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
            pass
        except Exception as exc:
            logger.exception("net-gate connection from %s failed: %r", peer_s, exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _serve(self, reader, writer, peer: str) -> None:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), _HEAD_TIMEOUT)
        if len(head) > _MAX_HEAD:
            raise _Refused(400, "request head too large", silent=True)
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, version = lines[0].split(" ")
        except ValueError:
            raise _Refused(400, "malformed request line", silent=True)
        headers: list[tuple[str, str]] = []
        for line in lines[1:]:
            if not line:
                continue
            name, _, value = line.partition(":")
            headers.append((name.strip(), value.strip()))
        lower = {n.lower(): v for n, v in headers}

        # The agents file as it is now, not as the watcher last saw it: an agent
        # added or revoked a moment ago is judged on that change. A stat call;
        # the file is only re-read when it changed.
        try:
            if self.store.reload():
                self.enforce_revocations()
        except Exception as exc:
            logger.error("agents file could not be reloaded: %r", exc)

        # 1. Who
        creds = _parse_basic(lower.get("proxy-authorization", ""))
        agent: Optional[Agent] = None
        if creds:
            agent = self.store.authenticate(*creds)
        if agent is None:
            self._record_stranger(peer, creds[0] if creds else "")
            raise _Refused(407, "proxy credentials required", silent=not self.challenge)

        # Where to
        if method.upper() == "CONNECT":
            host, _, raw_port = target.rpartition(":")
            scheme = "tunnel"
        else:
            parts = urlsplit(target)
            if parts.scheme != "http" or not parts.hostname:
                raise _Refused(400, "only CONNECT and absolute http:// requests are proxied")
            host, raw_port, scheme = parts.hostname, str(parts.port or 80), "http"
        host = host.strip("[]").lower()
        if not host or not raw_port.isdigit():
            raise _Refused(400, f"bad destination {target!r}")
        port = int(raw_port)
        params = {"host": host, "port": port, "scheme": scheme,
                  "method": method.upper(), "peer": peer}

        # 2. Still trusted
        if self.store.is_revoked(agent.agent_id):
            reason = f"agent {agent.agent_id} was revoked at {self.store.revoked[agent.agent_id]}"
            self._refuse_record(agent.agent_id, params, "net_gate:revoked", reason)
            raise _Refused(403, reason)

        # 3. Mandate
        if not agent.admits(host, port):
            reason = f"{host}:{port} is outside agent {agent.agent_id}'s mandate"
            self._refuse_record(agent.agent_id, params, f"net_mandate:{agent.agent_id}", reason)
            raise _Refused(403, reason)

        # 4. Policy, and the record of the decision
        result = self._record(agent.agent_id, "net.connect", params)
        if result is None:
            raise _Refused(503, "the trail could not record this connection; refused")
        if result.decision != "allow":
            if result.decision == "escalate":
                try:
                    self.pipeline.resolve_escalation(
                        result.action_id, "deny", reviewer="net-gate",
                        justification=("held for a human; a proxied connection "
                                       "cannot wait, so the hold is refused"),
                        approver="policy")
                except Exception as exc:
                    logger.error("could not record resolution of %s: %r",
                                 result.action_id, exc)
            raise _Refused(403, f"policy {result.decision}: {result.reason}")

        # 5. Egress floor, resolved once and pinned
        loop = asyncio.get_running_loop()
        try:
            ip = await loop.run_in_executor(
                None, lambda: pick_egress_ip(host, port, allow_private=agent.allow_private))
        except EgressBlocked as exc:
            self._close_record(agent.agent_id, result.action_id, host, port, 0, 0, 0.0,
                               f"egress floor: {exc}")
            raise _Refused(403, str(exc))

        try:
            up_r, up_w = await asyncio.wait_for(asyncio.open_connection(ip, port),
                                                _CONNECT_TIMEOUT)
        except (OSError, asyncio.TimeoutError) as exc:
            self._close_record(agent.agent_id, result.action_id, host, port, 0, 0, 0.0,
                               f"upstream unreachable: {exc!r}")
            raise _Refused(502, f"could not reach {host}:{port}")

        live = _Live(conn_id=uuid.uuid4().hex, agent_id=agent.agent_id, host=host,
                     port=port, action_id=result.action_id, started=time.monotonic(),
                     writers=[writer, up_w])
        self.live[live.conn_id] = live
        try:
            if scheme == "tunnel":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                up_w.write(self._origin_request(method, target, version, headers))
                await up_w.drain()
            # A revocation that landed while this connection was being set up.
            if self.store.is_revoked(agent.agent_id):
                live.reason = "revoked"
                return
            up_t = asyncio.ensure_future(self._pipe(reader, up_w, live, "up"))
            down_t = asyncio.ensure_future(self._pipe(up_r, writer, live, "down"))
            live.tasks = [up_t, down_t]
            done, _ = await asyncio.wait(live.tasks, return_when=asyncio.FIRST_COMPLETED)
            # The agent finished sending: the answer may still be on its way,
            # so the downstream half runs to its end. The far side finished:
            # nothing more can come back, so the connection is over.
            if up_t in done and not down_t.done():
                await asyncio.wait([down_t])
            for t in live.tasks:
                t.cancel()
        finally:
            self.live.pop(live.conn_id, None)
            try:
                up_w.close()
            except Exception:
                pass
            self._close_record(agent.agent_id, live.action_id, host, port, live.up,
                               live.down, time.monotonic() - live.started, live.reason)

    @staticmethod
    def _origin_request(method: str, target: str, version: str,
                        headers: list[tuple[str, str]]) -> bytes:
        """The request as the origin expects it: path only, no proxy headers,
        and one request per connection, so a second request on a kept-alive
        connection cannot reach a host the first one was not checked for."""
        parts = urlsplit(target)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        drop = {"proxy-authorization", "proxy-connection", "connection", "keep-alive"}
        out = [f"{method} {path} {version}"]
        out += [f"{n}: {v}" for n, v in headers if n.lower() not in drop]
        out.append("Connection: close")
        return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1")

    async def _pipe(self, src, dst, live: _Live, direction: str) -> None:
        try:
            while True:
                data = await src.read(_CHUNK)
                if not data:
                    break
                if direction == "up":
                    live.up += len(data)
                else:
                    live.down += len(data)
                dst.write(data)
                await dst.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            try:
                if dst.can_write_eof():
                    dst.write_eof()
            except Exception:
                pass

    def _close_record(self, agent_id: str, action_id: str, host: str, port: int,
                      up: int, down: int, seconds: float, reason: str) -> None:
        self._record(agent_id, "net.close",
                     {"host": host, "port": port, "connect_action_id": action_id,
                      "bytes_up": up, "bytes_down": down,
                      "seconds": round(seconds, 3), "reason": reason})

    def _record_stranger(self, peer: str, claimed: str) -> None:
        ip = peer.rsplit(":", 1)[0]
        now = time.monotonic()
        if now - self._stranger_seen.get(ip, -_STRANGER_RECORD_EVERY) < _STRANGER_RECORD_EVERY:
            return
        self._stranger_seen[ip] = now
        self._refuse_record("unauthenticated", {"peer": peer, "claimed_agent": claimed},
                            "net_gate:unauthenticated",
                            "no valid agent credential; closed without a reply")
