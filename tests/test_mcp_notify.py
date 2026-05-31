"""Tests for vaara.integrations._mcp_notify.

Covers the transport-aware notification routing surfaces introduced for
v0.41 GET-SSE: per-session enqueue, broadcast on missing session id,
bounded replay buffer eviction, and the close-sentinel drain contract.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading

from vaara.integrations._mcp_notify import (
    HttpRouter,
    StdioRouter,
    _SessionState,
)


def _new_loop() -> asyncio.AbstractEventLoop:
    """Create a fresh loop for tests that only need it as a target for
    ``call_soon_threadsafe`` from the enqueue path. No running required."""
    return asyncio.new_event_loop()


def test_stdio_router_writes_json_line_to_stdout(monkeypatch):
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    router = StdioRouter(threading.Lock())
    router.deliver({"jsonrpc": "2.0", "method": "notifications/x"})
    line = captured.getvalue().strip()
    assert json.loads(line) == {"jsonrpc": "2.0", "method": "notifications/x"}


def test_http_router_register_and_unregister(monkeypatch):
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        router.register_session("sess-a", upstream="alpha", tenant="t1", loop=loop)
        assert router.session_count() == 1
        router.unregister_session("sess-a")
        assert router.session_count() == 0
    finally:
        loop.close()


def test_http_router_deliver_targets_named_session():
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        state_a = router.register_session("sess-a", "alpha", "", loop)
        state_b = router.register_session("sess-b", "alpha", "", loop)
        msg = {"jsonrpc": "2.0", "method": "notifications/progress"}
        router.deliver(msg, session_id="sess-a", upstream="alpha")
        # Buffer reflects the enqueue side-effect even without draining.
        assert state_a.replay_since(0) == [(1, msg)]
        assert state_b.replay_since(0) == []
    finally:
        loop.close()


def test_http_router_deliver_without_session_broadcasts_on_upstream():
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        state_alpha = router.register_session("sess-a", "alpha", "", loop)
        state_beta = router.register_session("sess-b", "beta", "", loop)
        msg = {"jsonrpc": "2.0", "method": "notifications/message"}
        router.deliver(msg, session_id=None, upstream="alpha")
        assert state_alpha.replay_since(0) == [(1, msg)]
        # beta upstream subscriber must not receive the alpha-scoped message.
        assert state_beta.replay_since(0) == []
    finally:
        loop.close()


def test_http_router_attributed_broadcast_is_tenant_scoped():
    # Two tenants subscribed to the SAME upstream slot. A progress
    # notification correlated to tenant t1's call (tenant="t1", no session_id)
    # must reach only t1's sessions, never t2's.
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        t1 = router.register_session("sess-1", "alpha", "t1", loop)
        t2 = router.register_session("sess-2", "alpha", "t2", loop)
        msg = {"jsonrpc": "2.0", "method": "notifications/progress"}
        router.deliver(msg, session_id=None, upstream="alpha", tenant="t1")
        assert t1.replay_since(0) == [(1, msg)]
        assert t2.replay_since(0) == []
    finally:
        loop.close()


def test_http_router_unattributable_broadcast_suppressed_across_tenants():
    # A log notification (no progressToken, tenant=None) on an upstream shared
    # by two distinct tenants is suppressed rather than fanned out, so one
    # tenant's upstream push cannot leak to another.
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        t1 = router.register_session("sess-1", "alpha", "t1", loop)
        t2 = router.register_session("sess-2", "alpha", "t2", loop)
        msg = {"jsonrpc": "2.0", "method": "notifications/message"}
        router.deliver(msg, session_id=None, upstream="alpha", tenant=None)
        assert t1.replay_since(0) == []
        assert t2.replay_since(0) == []
    finally:
        loop.close()


def test_http_router_unattributable_broadcast_ok_within_one_tenant():
    # When every subscriber on the upstream shares one tenant scope, an
    # unattributable broadcast is safe and still fans out (single-tenant
    # deployments and the empty-tenant default keep their behavior).
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        a = router.register_session("sess-1", "alpha", "t1", loop)
        b = router.register_session("sess-2", "alpha", "t1", loop)
        msg = {"jsonrpc": "2.0", "method": "notifications/message"}
        router.deliver(msg, session_id=None, upstream="alpha", tenant=None)
        assert a.replay_since(0) == [(1, msg)]
        assert b.replay_since(0) == [(1, msg)]
    finally:
        loop.close()


def test_http_router_deliver_to_unknown_session_is_noop():
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        state = router.register_session("sess-a", "alpha", "", loop)
        router.deliver({"foo": 1}, session_id="ghost", upstream="alpha")
        assert state.replay_since(0) == []
    finally:
        loop.close()


def test_session_state_replay_buffer_is_bounded():
    loop = _new_loop()
    try:
        state = _SessionState("s", "alpha", "", loop, replay_buffer_size=3)
        for i in range(5):
            state.enqueue({"n": i})
        replayed = state.replay_since(0)
        # Only the last 3 events remain. Event IDs reflect monotonic
        # assignment; the first two are gone but their IDs were 1, 2.
        assert [eid for eid, _ in replayed] == [3, 4, 5]
        assert [payload["n"] for _, payload in replayed] == [2, 3, 4]
    finally:
        loop.close()


def test_session_state_replay_since_filters_by_cursor():
    loop = _new_loop()
    try:
        state = _SessionState("s", "alpha", "", loop, replay_buffer_size=10)
        for i in range(5):
            state.enqueue({"n": i})
        # Resume after seeing event id 3. Replay only events 4 and 5.
        replayed = state.replay_since(3)
        assert [eid for eid, _ in replayed] == [4, 5]
    finally:
        loop.close()


def test_session_state_close_is_idempotent():
    loop = _new_loop()
    try:
        state = _SessionState("s", "alpha", "", loop, replay_buffer_size=10)
        state.close()
        # Second close must not raise and must not double-post a sentinel.
        state.close()
    finally:
        loop.close()


def test_session_state_enqueue_after_close_is_ignored():
    loop = _new_loop()
    try:
        state = _SessionState("s", "alpha", "", loop, replay_buffer_size=10)
        state.close()
        state.enqueue({"after_close": True})
        assert state.replay_since(0) == []
    finally:
        loop.close()


def test_drain_yields_events_then_sentinel_on_close():
    """drain() returns enqueued events, then None once close() runs.

    Exercised inside an actual running loop: enqueue schedules puts via
    ``call_soon_threadsafe``, so the queue mutation must happen inside the
    loop's execution.
    """
    async def run() -> list:
        loop = asyncio.get_running_loop()
        state = _SessionState("s", "alpha", "", loop, replay_buffer_size=10)
        state.enqueue({"n": 1})
        state.enqueue({"n": 2})
        # Schedule close after the queued events are pulled.
        out: list = []
        for _ in range(2):
            entry = await state.drain()
            out.append(entry)
        state.close()
        sentinel = await state.drain()
        out.append(sentinel)
        return out

    result = asyncio.run(run())
    assert result == [(1, {"n": 1}), (2, {"n": 2}), None]


def test_http_router_re_registration_replaces_session():
    """A second register_session under the same id swaps the state.

    The new state has a fresh buffer; the old state is closed via the
    sentinel path so any drain awaiting on it would unblock.
    """
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        old = router.register_session("sess-a", "alpha", "", loop)
        old.enqueue({"old": True})
        new = router.register_session("sess-a", "alpha", "", loop)
        assert new is not old
        new.enqueue({"new": True})
        assert old.replay_since(0) == [(1, {"old": True})]
        assert new.replay_since(0) == [(1, {"new": True})]
    finally:
        loop.close()


def test_stale_unregister_does_not_drop_reconnected_session():
    """The reconnect race: the old stream's teardown must not unregister the
    NEW session that took its id.

    register_session("sess-a") returns old; a reconnect registers new under the
    same id and closes old. When old's SSE finally block runs unregister with
    its own state as the identity guard, the map entry (now new) is left intact
    so delivery to sess-a still reaches the live reconnected session.
    """
    router = HttpRouter(replay_buffer_size=10)
    loop = _new_loop()
    try:
        old = router.register_session("sess-a", "alpha", "", loop)
        new = router.register_session("sess-a", "alpha", "", loop)
        # Old stream tears down and runs its identity-checked unregister.
        router.unregister_session("sess-a", expected=old)
        assert router.session_count() == 1, "stale teardown dropped the live session"
        # The live session still receives a targeted notification.
        router.deliver({"jsonrpc": "2.0", "method": "notifications/x"},
                       session_id="sess-a", upstream="alpha")
        assert new.replay_since(0) == [
            (1, {"jsonrpc": "2.0", "method": "notifications/x"}),
        ]
        # An unguarded unregister still works for the normal teardown path.
        router.unregister_session("sess-a", expected=new)
        assert router.session_count() == 0
    finally:
        loop.close()
