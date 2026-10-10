# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The egress proxy: the one way out of a ``vaara run`` launch with egress locked.

``vaara run`` starts it outside the agent's tree, on a loopback port, and
hands the tree ``HTTPS_PROXY`` and ``HTTP_PROXY``. Landlock lets the tree
connect to that port and nothing else, and seccomp refuses UDP and raw IP
sockets (:mod:`vaara.oslayer.harden`), so every connection the agent makes
comes here and is decided against the operator's allow list:

- ``CONNECT host:port`` (HTTPS and anything tunnelled) and absolute-form
  plain HTTP requests are understood. The proxy resolves the name itself;
  the tree needs no DNS.
- A host is allowed when it matches an entry: ``example.com`` (ports 443
  and 80), ``*.example.com`` (any subdomain, same ports) or
  ``example.com:8443`` (that port only).
- Whatever the list says, a name that resolves to loopback, link-local
  (cloud metadata at 169.254.169.254), multicast or the unspecified
  address is refused, unless the entry names that address literally.
- Every connection, allowed or refused, is handed to ``record`` with the
  host, port, method and verdict, and for an allowed one the bytes each
  way when it closes.

The proxy does not look inside TLS. It decides by name and records by name;
a credential for a host the model must not see goes through ``vaara
llm-proxy``, which holds the key itself.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

MAX_HEAD = 64 * 1024
DEFAULT_PORTS = (443, 80)
_REFUSED_NETS = ("loopback", "link_local", "multicast", "unspecified")
# Instance metadata on IPv6 sits in unique-local space, not link-local: AWS.
_METADATA_V6 = (ipaddress.ip_network("fd00:ec2::254/128"),)


@dataclass(frozen=True)
class Rule:
    host: str          # lower case, without a leading "*."
    wildcard: bool
    port: Optional[int]

    def matches(self, host: str, port: int) -> bool:
        host = host.lower().rstrip(".")
        if self.port is None and port not in DEFAULT_PORTS:
            return False
        if self.port is not None and port != self.port:
            return False
        if self.wildcard:
            return host.endswith("." + self.host)
        return host == self.host


def parse_rule(entry: str) -> Rule:
    entry = entry.strip().lower()
    if not entry or any(c.isspace() for c in entry) or "/" in entry:
        raise ValueError(f"not a host entry: {entry!r}")
    port: Optional[int] = None
    if entry.startswith("["):  # an IPv6 literal, [::1]:443
        host, _, rest = entry[1:].partition("]")
        if rest:
            port = int(rest.lstrip(":"))
    elif entry.count(":") == 1:
        host, _, p = entry.partition(":")
        port = int(p)
    else:
        host = entry
    if port is not None and not 0 < port < 65536:
        raise ValueError(f"port out of range in {entry!r}")
    wildcard = host.startswith("*.")
    if wildcard:
        host = host[2:]
    if not host or "*" in host:
        raise ValueError(f"only a leading '*.' is allowed: {entry!r}")
    return Rule(host=host, wildcard=wildcard, port=port)


def _literal(host: str) -> Optional[ipaddress._BaseAddress]:
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def _refused_address(addr: ipaddress._BaseAddress) -> Optional[str]:
    # An IPv4 address carried in IPv6 is judged as the IPv4 address: Python
    # releases before the CVE-2024-4032 fix do not look through the mapping.
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    if any(addr in net for net in _METADATA_V6):
        return "metadata"
    for kind in _REFUSED_NETS:
        if getattr(addr, f"is_{kind}"):
            return kind.replace("_", "-")
    return None


class EgressProxy:
    """A threaded HTTP proxy on 127.0.0.1 that decides each connection."""

    def __init__(self, allow: list[str],
                 record: Optional[Callable[[dict], None]] = None,
                 resolve: Callable[..., list] = socket.getaddrinfo,
                 connect_timeout: float = 15.0) -> None:
        self.rules = [parse_rule(e) for e in allow]
        self._record = record or (lambda event: None)
        self._resolve = resolve
        self._timeout = connect_timeout
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="vaara-egress")
        self._thread.start()
        return self.port

    @property
    def port(self) -> int:
        assert self._sock is not None, "not started"
        return self._sock.getsockname()[1]

    def environ(self) -> dict[str, str]:
        url = f"http://127.0.0.1:{self.port}"
        return {"HTTPS_PROXY": url, "HTTP_PROXY": url, "https_proxy": url,
                "http_proxy": url, "ALL_PROXY": url, "all_proxy": url,
                "NO_PROXY": "", "no_proxy": "",
                # Node's built-in fetch reads the proxy variables only with this.
                "NODE_USE_ENV_PROXY": "1"}

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ── decision ──────────────────────────────────────────────────

    def decide(self, host: str, port: int) -> tuple[bool, str, Optional[tuple]]:
        """(allowed, reason, address to connect to)."""
        host = host.lower().rstrip(".")
        rule = next((r for r in self.rules if r.matches(host, port)), None)
        if rule is None:
            return False, "not on the egress allow list", None
        literal = _literal(host)
        try:
            infos = self._resolve(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            return False, f"name did not resolve: {exc}", None
        for family, _t, _p, _c, sockaddr in infos:
            addr = ipaddress.ip_address(sockaddr[0])
            refused = _refused_address(addr)
            if refused and literal != addr:
                return False, f"{host} resolves to a {refused} address ({addr})", None
        family, _t, _p, _c, sockaddr = infos[0]
        return True, f"allowed by {'*.' if rule.wildcard else ''}{rule.host}" + (
            f":{rule.port}" if rule.port else ""), (family, sockaddr)

    # ── serving ───────────────────────────────────────────────────

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True,
                             name="vaara-egress-conn").start()

    def _client(self, conn: socket.socket) -> None:
        conn.settimeout(30)
        try:
            head, rest = self._read_head(conn)
            if head is None:
                return
            line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = line.split(" ")
            if len(parts) != 3:
                conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                return
            method, target, _version = parts
            if method.upper() == "CONNECT":
                host, port = _split_hostport(target, 443)
                self._tunnel(conn, method, host, port, b"", connect_reply=True)
            elif target.lower().startswith("http://"):
                host, port, path = _split_url(target)
                lines = head.split(b"\r\n")
                # One request per client connection: a keep-alive request for
                # another host on the same connection would reach this
                # upstream without a decision, so the upstream is told to close.
                kept = [f"{method} {path} {_version}".encode("latin-1")]
                kept += [h for h in lines[1:] if h and not h.lower().startswith(
                    (b"proxy-", b"connection:", b"keep-alive:"))]
                kept.append(b"Connection: close")
                lower = [h.lower() for h in lines[1:]]
                chunked = any(h.startswith(b"transfer-encoding:") and b"chunked" in h
                              for h in lower)
                body_left: Optional[int] = None
                if not chunked:
                    # Without a length the body is empty. Bytes past the body
                    # are the next request, and they do not go to this host.
                    length = next((int(h.split(b":", 1)[1]) for h in lower
                                   if h.startswith(b"content-length:")), 0)
                    rest, body_left = rest[:length], max(length - len(rest), 0)
                request = b"\r\n".join(kept) + b"\r\n\r\n" + rest
                self._tunnel(conn, method, host, port, request, connect_reply=False,
                             body_left=body_left)
            else:
                conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    def _read_head(self, conn: socket.socket) -> tuple[Optional[bytes], bytes]:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                return None, b""
            data += chunk
            if len(data) > MAX_HEAD:
                conn.sendall(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                return None, b""
        head, _, rest = data.partition(b"\r\n\r\n")
        return head, rest

    def _tunnel(self, conn: socket.socket, method: str, host: str, port: int,
                first: bytes, connect_reply: bool,
                body_left: Optional[int] = None) -> None:
        allowed, reason, target = self.decide(host, port)
        event = {"ts": time.time(), "host": host, "port": port, "method": method.upper(),
                 "allowed": allowed, "reason": reason}
        if not allowed or target is None:
            # Refused by policy: the one event kind that is a deny.
            self._record(dict(event, kind="refused"))
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n"
                         b"Connection: close\r\n\r\nvaara egress: " + reason.encode() + b"\n")
            return
        # Allowed: recorded now, before the connect, so a long stream is on
        # the record while it runs and stays there if this process dies.
        # What follows for the same connection id is an outcome, not a
        # decision: closed with the bytes, or failed when the upstream did
        # not answer. A 502 is transport, not policy.
        event["connection"] = str(uuid.uuid4())
        self._record(dict(event, kind="opened"))
        family, sockaddr = target
        upstream = socket.socket(family, socket.SOCK_STREAM)
        upstream.settimeout(self._timeout)
        try:
            upstream.connect(sockaddr)
        except OSError as exc:
            upstream.close()
            self._record(dict(event, kind="failed", error=f"upstream did not answer: {exc}"))
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            return
        if connect_reply:
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        if first:
            upstream.sendall(first)
        if body_left is not None:
            # A plain request with a known body: send the rest of the body,
            # then nothing more from the client reaches this upstream.
            sent = _copy_exact(conn, upstream, body_left)
            up, down = sent, _drain(upstream, conn)
            upstream.close()
        else:
            up, down = _pipe(conn, upstream)
        self._record(dict(event, kind="closed", bytes_up=up + len(first), bytes_down=down))


def _split_hostport(target: str, default: int) -> tuple[str, int]:
    if target.startswith("["):
        host, _, rest = target[1:].partition("]")
        return host, int(rest.lstrip(":") or default)
    host, sep, port = target.rpartition(":")
    if not sep:
        return target, default
    return host, int(port)


def _split_url(url: str) -> tuple[str, int, str]:
    rest = url[len("http://"):]
    authority, slash, path = rest.partition("/")
    host, port = _split_hostport(authority, 80)
    return host, port, "/" + path if slash else "/"


def _pipe(a: socket.socket, b: socket.socket) -> tuple[int, int]:
    """Copy both ways until either side closes. Returns (a->b, b->a) bytes."""
    counts = [0, 0]

    def copy(src: socket.socket, dst: socket.socket, i: int) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
                counts[i] += len(data)
        except OSError:
            pass
        finally:
            # Pass the end of this direction on and leave the other running:
            # a client that half-closes after its request still gets the reply.
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    a.settimeout(None)
    b.settimeout(None)
    t = threading.Thread(target=copy, args=(b, a, 1), daemon=True)
    t.start()
    copy(a, b, 0)
    t.join()
    b.close()
    return counts[0], counts[1]


def _copy_exact(src: socket.socket, dst: socket.socket, n: int) -> int:
    """Copy exactly ``n`` bytes from ``src`` to ``dst``, fewer if it closes."""
    src.settimeout(None)
    done = 0
    while done < n:
        data = src.recv(min(65536, n - done))
        if not data:
            break
        dst.sendall(data)
        done += len(data)
    return done


def _drain(src: socket.socket, dst: socket.socket) -> int:
    """Copy ``src`` to ``dst`` until ``src`` closes."""
    src.settimeout(None)
    done = 0
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
            done += len(data)
    except OSError:
        pass
    return done
