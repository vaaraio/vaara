"""Upstream MCP subprocess client for the proxy.

Owns the subprocess lifecycle of an upstream MCP server, demuxes responses
by JSON-RPC id, and routes notifications to a callback for the proxy to
forward downstream.

Internal module. Public surface is :class:`vaara.integrations.mcp_proxy.VaaraMCPProxy`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ProxyError(Exception):
    """The proxy itself cannot serve a request.

    Distinct from upstream-emitted JSON-RPC errors, which are forwarded
    verbatim. ProxyError is raised when the proxy-side machinery fails
    (upstream subprocess crashed, stdin write failed, response timeout)
    and the caller should surface JSON-RPC -32603 Internal error downstream.
    """


def strict_json_dumps(obj: Any, **kwargs: Any) -> str:
    """JSON dump that fails on NaN/Infinity (RFC 8259 strict).

    Python's default ``json.dumps`` emits ``NaN``/``Infinity``/``-Infinity``
    literals that strict JSON parsers (Go, Rust, browsers, many MCP clients)
    reject. Forcing strict output surfaces escaped non-finite values loudly
    in tests rather than silently emitting invalid wire format.
    """
    return json.dumps(obj, allow_nan=False, **kwargs)


@dataclass
class _UpstreamRequest:
    id: Any
    event: threading.Event
    response: Optional[dict] = None


class UpstreamMCPClient:
    """Spawn an upstream MCP server and communicate over its stdio.

    The reader runs on a background thread that parks responses keyed by
    JSON-RPC id and routes notifications to ``on_notification``. The main
    thread synchronously calls :meth:`request` and waits.
    """

    def __init__(
        self,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        on_notification: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._on_notification = on_notification
        self._pending: dict[Any, _UpstreamRequest] = {}
        self._lock = threading.Lock()
        self._closed = False

        # stderr passes through so upstream logs surface in the proxy's
        # stderr without contaminating the JSON-RPC channel on stdout.
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env or os.environ.copy(),
            bufsize=1,
            text=True,
        )

        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="upstream-reader",
        )
        self._reader_thread.start()

    def request(self, payload: dict, timeout: float = 30.0) -> dict:
        """Send a request, wait for the matching response by id.

        Raises :class:`ProxyError` if the upstream has died or the response
        does not arrive within ``timeout``.
        """
        if self._closed:
            raise ProxyError("Upstream MCP server is closed")
        if "id" not in payload:
            raise ValueError("request() requires a JSON-RPC id; use notify() for notifications")

        pending = _UpstreamRequest(id=payload["id"], event=threading.Event())
        with self._lock:
            self._pending[payload["id"]] = pending
        try:
            self._write(payload)
            if not pending.event.wait(timeout=timeout):
                raise ProxyError(f"Upstream MCP server did not respond within {timeout}s")
            # event was set but response stays None when the reader thread
            # exited (upstream closed stdout) and woke us as a shutdown signal.
            # An assert would either raise AssertionError (escapes the caller's
            # ProxyError handler) or be optimized out under -O (return None,
            # silently breaking the contract). Raise ProxyError explicitly.
            if pending.response is None:
                raise ProxyError("Upstream MCP server closed before responding")
            if not isinstance(pending.response, dict):
                raise ProxyError("Upstream MCP server returned non-object JSON-RPC")
            return pending.response
        finally:
            with self._lock:
                self._pending.pop(payload["id"], None)

    def notify(self, payload: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self._closed:
            return
        self._write(payload)

    def _write(self, payload: dict) -> None:
        if self._proc.stdin is None:
            raise ProxyError("Upstream MCP server stdin is closed")
        try:
            self._proc.stdin.write(strict_json_dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ProxyError(f"Upstream MCP server stdin write failed: {e}") from e

    def _read_loop(self) -> None:
        if self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Upstream emitted non-JSON line: %r", line[:200])
                continue

            # Notifications (no id) route to the callback for downstream forward.
            if isinstance(message, dict) and "id" not in message:
                if self._on_notification is not None:
                    try:
                        self._on_notification(message)
                    except Exception:
                        logger.exception("Notification handler raised")
                continue

            # Responses demux by id.
            response_id = message.get("id") if isinstance(message, dict) else None
            with self._lock:
                pending = self._pending.get(response_id)
            if pending is None:
                logger.warning("Upstream response for unknown id %r", response_id)
                continue
            pending.response = message
            pending.event.set()

        # Reader exited: upstream closed stdout. Wake all waiters so they
        # fail with ProxyError rather than hanging forever.
        self._closed = True
        with self._lock:
            for pending in self._pending.values():
                pending.event.set()

    def close(self) -> None:
        self._closed = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        except Exception:
            pass
