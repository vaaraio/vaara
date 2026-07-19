# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transport-aware notification routing for the Vaara MCP proxy.

Upstream MCP servers can push notifications back to the client mid-call
(progress events, log messages, sampling requests). The stdio transport
writes those notifications to its single stdout. The Streamable HTTP
transport has zero-to-many SSE subscribers keyed by ``Mcp-Session-Id``,
and a notification must be delivered to the session that owns the
originating tools/call (matched via progressToken at call time) or
broadcast across sessions when no session scope is known.

``NotificationRouter`` is the small surface the proxy uses regardless of
transport. ``HttpRouter`` holds per-session queues plus a bounded replay
buffer indexed by monotonically increasing event IDs so a reconnecting
client can resume with ``Last-Event-ID``. Events evicted from the bounded
buffer before reconnect are not recoverable: the caller sees a gap.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections import deque
from typing import Deque, Optional, Protocol

from vaara.integrations._mcp_upstream import strict_json_dumps

logger = logging.getLogger(__name__)

_DEFAULT_REPLAY_BUFFER = 100


class NotificationRouter(Protocol):
    def deliver(
        self,
        message: dict,
        *,
        session_id: Optional[str] = None,
        upstream: str = "default",
        tenant: Optional[str] = None,
    ) -> None:
        ...


class StdioRouter:
    """Stdio: serialise to stdout under a shared lock; session_id ignored."""

    def __init__(self, stdout_lock: threading.Lock) -> None:
        self._lock = stdout_lock

    def deliver(
        self,
        message: dict,
        *,
        session_id: Optional[str] = None,
        upstream: str = "default",
        tenant: Optional[str] = None,
    ) -> None:
        with self._lock:
            sys.stdout.write(strict_json_dumps(message) + "\n")
            sys.stdout.flush()


class _SessionState:
    """One subscribed HTTP client connected via GET /mcp.

    Mutation crosses threads: the upstream reader thread enqueues from a
    sync context while the SSE handler drains from inside FastAPI's event
    loop, so ``call_soon_threadsafe`` lands queue puts on the right loop.
    """

    def __init__(
        self,
        session_id: str,
        upstream: str,
        tenant: str,
        loop: asyncio.AbstractEventLoop,
        replay_buffer_size: int,
    ) -> None:
        self.session_id = session_id
        self.upstream = upstream
        self.tenant = tenant
        self._loop = loop
        self._queue: asyncio.Queue[Optional[tuple[int, dict]]] = asyncio.Queue()
        self._buffer: Deque[tuple[int, dict]] = deque(maxlen=replay_buffer_size)
        self._next_id = 1
        self._lock = threading.Lock()
        self._closed = False

    def enqueue(self, message: dict) -> None:
        with self._lock:
            if self._closed:
                return
            event_id = self._next_id
            self._next_id += 1
            entry = (event_id, message)
            self._buffer.append(entry)
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, entry)
        except RuntimeError:
            logger.debug("session %s loop closed during enqueue", self.session_id)

    async def drain(self) -> Optional[tuple[int, dict]]:
        """Yield the next event, or None when the session has been closed."""
        return await self._queue.get()

    def replay_since(self, last_event_id: int) -> list[tuple[int, dict]]:
        with self._lock:
            return [entry for entry in self._buffer if entry[0] > last_event_id]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        except RuntimeError:
            pass


class HttpRouter:
    """Per-session SSE notification delivery for Streamable HTTP transport.

    ``deliver`` with a session_id targets exactly that session. Without one,
    an ``tenant``-attributed broadcast (a progress notification correlated to
    a tools/call) reaches only that tenant's sessions on the upstream; an
    unattributed broadcast (a server-level log with no progressToken) reaches
    every session on the upstream only when they all share one tenant scope,
    and is otherwise suppressed so one tenant's upstream push cannot leak to
    another tenant subscribed to the same upstream.
    """

    def __init__(self, replay_buffer_size: int = _DEFAULT_REPLAY_BUFFER) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()
        self._replay_buffer_size = replay_buffer_size

    def register_session(
        self,
        session_id: str,
        upstream: str,
        tenant: str,
        loop: asyncio.AbstractEventLoop,
    ) -> _SessionState:
        state = _SessionState(
            session_id=session_id,
            upstream=upstream,
            tenant=tenant,
            loop=loop,
            replay_buffer_size=self._replay_buffer_size,
        )
        with self._lock:
            prior = self._sessions.get(session_id)
            self._sessions[session_id] = state
        # Close the prior state outside the lock so its sentinel put doesn't
        # land while another thread holds the router lock. The prior SSE
        # handler unblocks on its next drain() and runs its finally cleanup.
        if prior is not None:
            prior.close()
        return state

    def unregister_session(
        self, session_id: str, expected: Optional["_SessionState"] = None,
    ) -> None:
        """Remove a session's state.

        When ``expected`` is given, the entry is popped only if it is still the
        same ``_SessionState`` object. On reconnect, ``register_session``
        replaces the map entry with a fresh state and closes the prior one; the
        old stream's ``finally`` then runs ``unregister_session``. Without the
        identity check that finally would pop the NEW state, silently dropping
        notifications for the live reconnected session. The identity check makes
        the stale teardown a no-op.
        """
        with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                return
            if expected is not None and current is not expected:
                # A newer session took this id; leave it registered.
                return
            state = self._sessions.pop(session_id)
        state.close()

    def deliver(
        self,
        message: dict,
        *,
        session_id: Optional[str] = None,
        upstream: str = "default",
        tenant: Optional[str] = None,
    ) -> None:
        with self._lock:
            if session_id is not None:
                state = self._sessions.get(session_id)
                targets = [state] if state is not None else []
            else:
                candidates = [
                    s for s in self._sessions.values() if s.upstream == upstream
                ]
                if tenant is not None:
                    # Attributable broadcast (progress correlated to a
                    # tools/call): only the originating tenant's sessions.
                    targets = [s for s in candidates if s.tenant == tenant]
                else:
                    # Unattributable broadcast (a server-level log with no
                    # progressToken). Safe only when every subscriber on this
                    # upstream shares one tenant scope; across distinct tenants
                    # it would leak one tenant's upstream push to another, so
                    # suppress rather than fan out.
                    scopes = {s.tenant for s in candidates}
                    if len(scopes) <= 1:
                        targets = candidates
                    else:
                        logger.debug(
                            "suppressing unattributable broadcast on upstream "
                            "%s spanning %d tenant scopes",
                            upstream,
                            len(scopes),
                        )
                        targets = []
        for state in targets:
            state.enqueue(message)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
