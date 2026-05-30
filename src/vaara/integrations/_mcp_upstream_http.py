"""Remote HTTP/SSE upstream MCP client for the proxy.

Speaks the MCP Streamable HTTP transport (spec revisions 2025-03-26 and
2025-06-18) so the proxy can sit in front of MCP servers that are not local
stdio subprocesses. Built on the standard library only (``urllib``) to keep
vaara's zero-dependency core intact.

What this implements:
  * POST a JSON-RPC request to the single MCP endpoint and read the reply,
    whether the server answers with ``application/json`` (one object) or a
    ``text/event-stream`` (SSE) it closes once the reply has been sent.
  * Capture the ``Mcp-Session-Id`` the server assigns on ``initialize`` and
    echo it on every later request; send ``MCP-Protocol-Version`` once the
    negotiated revision is known.
  * Open a standing GET ``text/event-stream`` channel after the handshake to
    receive server-initiated notifications, reconnecting with ``Last-Event-ID``.
  * Inject caller-supplied static headers (e.g. ``Authorization: Bearer ...``)
    on every call so authenticated remote servers are reachable.

Deliberately NOT implemented: the deprecated 2024-11-05 two-endpoint HTTP+SSE
transport, and interactive OAuth / dynamic client registration. Auth is
static-header passthrough only.

Internal module. Public surface is :class:`vaara.integrations.mcp_proxy.VaaraMCPProxy`.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Optional

from vaara.integrations._mcp_upstream import ProxyError, strict_json_dumps

logger = logging.getLogger(__name__)

# Bounded wait between reconnect attempts on the standing server-to-client SSE
# stream so a flapping upstream cannot spin the reconnect thread hot.
_SSE_RECONNECT_BACKOFF_SECONDS = 1.0

# Socket read timeout on the standing GET stream. Bounds how long a blocked
# read parks, so close() (which only flips the closed flag) is observed within
# this window. We deliberately do NOT close the in-flight response from another
# thread to unblock it: that path goes through the BufferedReader lock the
# blocked read already holds and would deadlock. Comfortably longer than a
# compliant MCP server's keepalive cadence so a live stream never trips it.
_SSE_READ_TIMEOUT_SECONDS = 30.0

# Cap on an error-response body quoted back in a ProxyError, so a chatty
# upstream cannot blow up the proxy's logs or the downstream error message.
_ERROR_BODY_SNIPPET = 200


class _ServerPushUnsupported(Exception):
    """The upstream offers no GET server-to-client SSE channel.

    Raised internally when the standing-stream GET is met with 404/405/501 so
    the listener loop stops cleanly instead of reconnecting forever.
    """


class HttpUpstreamClient:
    """Talk to a remote MCP server over the Streamable HTTP transport.

    Matches the :class:`~vaara.integrations._mcp_upstream.UpstreamClient`
    surface (:meth:`request`, :meth:`notify`, :meth:`close`) so the proxy treats
    a remote upstream exactly like a local stdio one. ``on_notification`` is
    called with every server-initiated JSON-RPC notification, both those that
    arrive inline on a POST response stream and those on the standing GET stream.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        on_notification: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._url = url
        # Caller-supplied static headers (auth). Applied first so the transport
        # control headers always win on the keys the protocol owns.
        self._extra_headers = dict(headers or {})
        self._on_notification = on_notification
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._session_id: Optional[str] = None
        self._protocol_version: Optional[str] = None
        # The standing GET-SSE listener starts lazily after the first request
        # (the MCP handshake), the earliest point a session id can exist.
        self._listener_started = False
        self._listener_last_event_id: Optional[str] = None

    # -- public UpstreamClient surface ------------------------------------

    def request(self, payload: dict, timeout: float = 30.0) -> dict:
        if self._closed.is_set():
            raise ProxyError("Upstream MCP server is closed")
        if "id" not in payload:
            raise ValueError(
                "request() requires a JSON-RPC id; use notify() for notifications",
            )
        resp = self._post(payload, timeout=timeout)
        try:
            self._capture_session(resp)
            if self._is_event_stream(resp):
                response = self._reply_from_sse(resp, payload["id"])
            else:
                response = self._reply_from_json(resp)
        finally:
            resp.close()
        # The matched reply must echo our id; a reply for another id is a
        # protocol violation, not a result to return.
        if response.get("id") != payload["id"]:
            raise ProxyError(
                f"Upstream MCP server replied to id {response.get('id')!r}, "
                f"expected {payload['id']!r}",
            )
        self._capture_protocol_version(payload, response)
        self._ensure_listener()
        return response

    def notify(self, payload: dict) -> None:
        if self._closed.is_set():
            return
        try:
            resp = self._post(payload, timeout=30.0)
        except ProxyError as e:
            # A notification has no reply to wait on; a delivery failure is
            # logged not raised, matching the stdio client's fire-and-forget.
            logger.warning("Upstream MCP server rejected notification: %s", e)
            return
        try:
            self._capture_session(resp)
            if self._is_event_stream(resp):
                # Some servers answer a notification POST with a short SSE
                # stream of server-initiated messages; drain and route them.
                for message in self._iter_messages(resp):
                    self._dispatch_unsolicited(message)
        finally:
            resp.close()

    def close(self) -> None:
        # Flip the flag only. The standing-stream listener is a daemon thread
        # that observes it within _SSE_READ_TIMEOUT_SECONDS, or at once when the
        # server closes the connection. We must NOT close the in-flight response
        # from here: that acquires the BufferedReader lock the blocked read
        # already holds and would deadlock.
        self._closed.set()

    # -- HTTP plumbing ----------------------------------------------------

    def _post(self, payload: dict, timeout: float) -> Any:
        body = strict_json_dumps(payload).encode("utf-8")
        headers = self._headers(accept="application/json, text/event-stream")
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
        except urllib.error.HTTPError as e:
            raise ProxyError(
                f"Upstream MCP server returned HTTP {e.code}: {self._error_snippet(e)}",
            ) from e
        except urllib.error.URLError as e:
            raise ProxyError(f"Upstream MCP server unreachable: {e.reason}") from e
        except (TimeoutError, OSError) as e:
            raise ProxyError(f"Upstream MCP server request failed: {e}") from e

    def _headers(self, *, accept: str) -> dict[str, str]:
        headers = dict(self._extra_headers)
        headers["Accept"] = accept
        headers["Content-Type"] = "application/json"
        with self._lock:
            session_id = self._session_id
            protocol_version = self._protocol_version
        if session_id is not None:
            headers["Mcp-Session-Id"] = session_id
        if protocol_version is not None:
            headers["MCP-Protocol-Version"] = protocol_version
        return headers

    def _capture_session(self, resp: Any) -> None:
        session_id = resp.headers.get("Mcp-Session-Id")
        if session_id:
            with self._lock:
                self._session_id = session_id

    def _capture_protocol_version(self, payload: dict, response: dict) -> None:
        if payload.get("method") != "initialize":
            return
        result = response.get("result")
        version = result.get("protocolVersion") if isinstance(result, dict) else None
        if isinstance(version, str) and version:
            with self._lock:
                self._protocol_version = version

    @staticmethod
    def _is_event_stream(resp: Any) -> bool:
        return "text/event-stream" in (resp.headers.get("Content-Type") or "").lower()

    @staticmethod
    def _error_snippet(e: urllib.error.HTTPError) -> str:
        try:
            return e.read().decode("utf-8", "replace")[:_ERROR_BODY_SNIPPET]
        except Exception:
            return ""

    # -- reply extraction -------------------------------------------------

    def _reply_from_json(self, resp: Any) -> dict:
        raw = resp.read()
        if not raw:
            raise ProxyError("Upstream MCP server returned an empty response body")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ProxyError(f"Upstream MCP server returned non-JSON: {e}") from e
        if not isinstance(message, dict):
            raise ProxyError("Upstream MCP server returned non-object JSON-RPC")
        return message

    def _reply_from_sse(self, resp: Any, want_id: Any) -> dict:
        # Stream events until the one bearing our id arrives; route anything
        # else (server notifications) onward. The server closes the stream
        # after delivering the reply.
        for message in self._iter_messages(resp):
            if message.get("id") == want_id:
                return message
            self._dispatch_unsolicited(message)
        raise ProxyError(
            f"Upstream MCP server closed the SSE stream before replying to id {want_id!r}",
        )

    def _iter_messages(self, resp: Any) -> Iterator[dict]:
        """Yield each JSON-RPC object carried by the SSE stream's data events."""
        for event in self._iter_sse(resp):
            data = event.get("data", "")
            if not data:
                continue
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Upstream emitted non-JSON SSE data: %r", data[:200])
                continue
            if isinstance(message, dict):
                yield message
            else:
                logger.warning("Upstream emitted non-object JSON-RPC over SSE")

    @staticmethod
    def _iter_sse(resp: Any) -> Iterator[dict]:
        """Parse a server-sent-events byte stream into ``{data, id}`` events.

        Implements the subset of the SSE grammar MCP uses: ``data:`` (joined by
        newline when multi-line), ``id:`` (for ``Last-Event-ID`` resume),
        comment lines (``:`` prefix, heartbeats) ignored, ``event:``/``retry:``
        ignored. An empty line dispatches the accumulated event.
        """
        data_lines: list[str] = []
        event_id: Optional[str] = None
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    yield {"data": "\n".join(data_lines), "id": event_id}
                data_lines = []
                event_id = None
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            elif field == "id":
                event_id = value
        if data_lines:
            yield {"data": "\n".join(data_lines), "id": event_id}

    def _dispatch_unsolicited(self, message: dict) -> None:
        # Server-initiated requests (id + method) cannot be answered by the
        # proxy on either transport, so they are dropped with a warning the
        # same way the stdio reader treats an unknown id. Only true
        # notifications (no id) reach the proxy's handler.
        if "id" in message:
            logger.warning(
                "Upstream sent an unsolicited message with id %r; dropping",
                message.get("id"),
            )
            return
        if self._on_notification is None:
            return
        try:
            self._on_notification(message)
        except Exception:
            logger.exception("Notification handler raised")

    # -- standing server-to-client SSE channel ----------------------------

    def _ensure_listener(self) -> None:
        with self._lock:
            if self._listener_started or self._closed.is_set():
                return
            self._listener_started = True
        thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="upstream-http-sse",
        )
        thread.start()

    def _listen_loop(self) -> None:
        while not self._closed.is_set():
            try:
                self._read_standing_stream()
            except _ServerPushUnsupported:
                logger.info(
                    "Upstream %s offers no server-to-client SSE channel", self._url,
                )
                return
            except Exception as e:  # network drop, parse error, closed socket
                if self._closed.is_set():
                    return
                logger.debug("Upstream SSE stream ended (%s); reconnecting", e)
            # Interruptible backoff: close() sets the event and we return early.
            if self._closed.wait(_SSE_RECONNECT_BACKOFF_SECONDS):
                return

    def _read_standing_stream(self) -> None:
        headers = self._headers(accept="text/event-stream")
        headers.pop("Content-Type", None)  # no body on the GET
        if self._listener_last_event_id is not None:
            headers["Last-Event-ID"] = self._listener_last_event_id
        req = urllib.request.Request(self._url, headers=headers, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=_SSE_READ_TIMEOUT_SECONDS)  # noqa: S310
        except urllib.error.HTTPError as e:
            if e.code in (404, 405, 501):
                raise _ServerPushUnsupported from e
            raise
        # resp is read and closed only on this listener thread, so the close in
        # the finally never races a read on another thread.
        try:
            for event in self._iter_sse(resp):
                if self._closed.is_set():
                    return
                if event.get("id") is not None:
                    self._listener_last_event_id = event["id"]
                data = event.get("data", "")
                if not data:
                    continue
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Upstream emitted non-JSON SSE data: %r", data[:200])
                    continue
                if isinstance(message, dict):
                    self._dispatch_unsolicited(message)
        finally:
            try:
                resp.close()
            except Exception:
                pass
