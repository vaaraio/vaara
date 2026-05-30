"""SSRF egress guard for the remote MCP HTTP connector.

The ``--upstream-url`` connector hands a user-supplied URL to ``urllib`` and
follows redirects. Without a guard a hostile or compromised upstream (or an
attacker who controls a redirect target) can point the proxy at the cloud
instance-metadata service, a loopback admin port, or an internal RFC1918 host,
and have the proxy fetch it with the operator's static auth headers attached.
This module is the host-resolution floor that refuses those targets before any
socket is opened, plus the custom ``urllib`` opener that re-applies the floor
on every redirect hop and drops the ``Authorization`` header on a cross-origin
redirect.

Default posture is SAFE: loopback, link-local (IPv4 169.254/16 and IPv6
fe80::/10), RFC1918, IPv6 ULA (fc00::/7), the cloud-metadata addresses, and the
dotless decimal/hex encodings of 169.254.169.254 are all refused. An operator
who needs a trusted internal host opts in explicitly, per client
(``allow_private_hosts=True``) or process-wide
(``VAARA_MCP_ALLOW_PRIVATE_UPSTREAM=1``). The opt-in never disables the
cross-origin ``Authorization`` drop, never raises the redirect cap, and never
reopens the metadata address.

Internal module. Public surface is :mod:`vaara.integrations._mcp_upstream_http`.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlsplit

# urllib's default redirect cap is 10; a remote MCP endpoint that needs more
# than a couple of redirects to answer a JSON-RPC POST is broken or hostile.
_MAX_REDIRECTS = 3

# Process-wide opt-in to permit private/loopback targets. Read at call time so
# tests and embedders can set it per process.
_ALLOW_ENV = "VAARA_MCP_ALLOW_PRIVATE_UPSTREAM"

_METADATA_V4 = ipaddress.IPv4Address("169.254.169.254")
_METADATA_V6 = ipaddress.IPv6Address("fe80::a9fe:a9fe")


class EgressBlocked(Exception):
    """An upstream URL resolves to an address the egress floor refuses."""


def _env_allows_private() -> bool:
    return os.environ.get(_ALLOW_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _is_metadata(ip: ipaddress._BaseAddress) -> bool:
    """True iff the address is a cloud instance-metadata endpoint.

    Refused unconditionally (even under the private-host opt-in): there is no
    legitimate reason to dial instance-metadata through the proxy.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:a.b.c.d judged on the embedded v4 address
        ip = mapped
    return ip == _METADATA_V4 or ip == _METADATA_V6


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    """True iff this resolved address must never be reached by default.

    Covers the metadata addresses plus loopback, link-local (IPv4 169.254/16
    and IPv6 fe80::/10), private (RFC1918 and ULA fc00::/7), unspecified,
    reserved, and multicast.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:a.b.c.d judged on the embedded v4 address
        ip = mapped
    return (
        _is_metadata(ip)
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def _coerce_dotless_host(host: str) -> Optional[ipaddress._BaseAddress]:
    """Parse a bare decimal or hex integer host (``2852039166``, ``0xa9fea9fe``).

    Browsers and ``inet_aton`` accept these as IPv4; ``ipaddress`` does not, so
    we decode them ourselves. Returns the address when the host is such an
    integer, else None.
    """
    base = 16 if host.lower().startswith("0x") else 10
    try:
        value = int(host, base)
    except ValueError:
        return None
    if 0 <= value <= 0xFFFFFFFF:
        return ipaddress.IPv4Address(value)
    return None


def assert_url_egress_allowed(url: str, *, allow_private: bool = False) -> None:
    """Refuse ``url`` if its host resolves to a blocked address.

    Resolves the hostname (every returned A/AAAA record is checked, so a
    DNS-rebinding answer mixing a public and a private address is still refused)
    and applies the floor. A literal IP host is checked directly. The dotless
    encodings of the metadata address are refused even when ``allow_private`` is
    set: there is no legitimate reason to dial instance-metadata through the
    proxy. Raises :class:`EgressBlocked` on a refused or unresolvable target.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise EgressBlocked(f"upstream URL scheme must be http or https: {url!r}")
    host = parts.hostname
    if not host:
        raise EgressBlocked(f"upstream URL has no host: {url!r}")

    dotless = _coerce_dotless_host(host)
    if dotless is not None:
        if dotless == _METADATA_V4:
            raise EgressBlocked(
                f"upstream URL targets the cloud-metadata address: {url!r}",
            )
        if not allow_private and _ip_is_blocked(dotless):
            raise EgressBlocked(f"upstream URL resolves to a blocked address: {url!r}")
        return

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_metadata(literal):
            raise EgressBlocked(
                f"upstream URL targets the cloud-metadata address: {url!r}",
            )
        if not allow_private and _ip_is_blocked(literal):
            raise EgressBlocked(f"upstream URL resolves to a blocked address: {url!r}")
        return

    try:
        infos = socket.getaddrinfo(host, parts.port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise EgressBlocked(f"upstream host does not resolve: {host!r} ({exc})") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _is_metadata(ip):
            raise EgressBlocked(
                f"upstream host {host!r} resolves to the cloud-metadata address",
            )
        if not allow_private and _ip_is_blocked(ip):
            raise EgressBlocked(
                f"upstream host {host!r} resolves to a blocked address {ip}",
            )


def pick_egress_ip(host: str, port: Optional[int], *, allow_private: bool = False) -> str:
    """Resolve ``host`` and return the single IP the caller must dial.

    Mirrors :func:`assert_url_egress_allowed`'s floor but *returns the
    address to connect to* instead of only validating. The caller dials
    that IP literal, so the kernel never performs a second DNS lookup at
    socket-connect time. That closes the rebind TOCTOU: the address that
    passed the floor is the exact address the socket reaches, even if the
    name re-resolves to a blocked target a millisecond later.

    Every resolved address is checked (a rebind answer mixing a public and
    a blocked address is still refused); the first one is pinned. Literal
    and dotless-integer hosts are returned directly after the same checks.
    Raises :class:`EgressBlocked` on a refused or unresolvable target.
    """
    dotless = _coerce_dotless_host(host)
    if dotless is not None:
        if dotless == _METADATA_V4:
            raise EgressBlocked(f"upstream host targets the cloud-metadata address: {host!r}")
        if not allow_private and _ip_is_blocked(dotless):
            raise EgressBlocked(f"upstream host resolves to a blocked address: {host!r}")
        return str(dotless)

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_metadata(literal):
            raise EgressBlocked(f"upstream host targets the cloud-metadata address: {host!r}")
        if not allow_private and _ip_is_blocked(literal):
            raise EgressBlocked(f"upstream host resolves to a blocked address: {host!r}")
        return str(literal)

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise EgressBlocked(f"upstream host does not resolve: {host!r} ({exc})") from exc
    chosen: Optional[str] = None
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if _is_metadata(ip):
            raise EgressBlocked(f"upstream host {host!r} resolves to the cloud-metadata address")
        if not allow_private and _ip_is_blocked(ip):
            raise EgressBlocked(f"upstream host {host!r} resolves to a blocked address {ip}")
        if chosen is None:
            chosen = addr
    if chosen is None:
        raise EgressBlocked(f"upstream host does not resolve: {host!r}")
    return chosen


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that validates+pins the host's IP at connect time.

    ``self.host`` (the original name) stays the Host header; the socket is
    opened to the validated IP literal so no re-resolution can occur
    between the egress check and the connect.
    """

    def __init__(self, host: str, *, _allow_private: bool = False, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._allow_private = _allow_private

    def connect(self) -> None:  # noqa: D102
        ip = pick_egress_ip(self.host, self.port, allow_private=self._allow_private)
        self.sock = socket.create_connection((ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS counterpart to :class:`_PinnedHTTPConnection`.

    The TCP connect targets the pinned IP; the TLS handshake still uses the
    original hostname for SNI and certificate verification, so a rebind to
    an unvalidated address cannot also present a valid certificate.
    """

    def __init__(self, host: str, *, _allow_private: bool = False, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._allow_private = _allow_private

    def connect(self) -> None:  # noqa: D102
        ip = pick_egress_ip(self.host, self.port, allow_private=self._allow_private)
        sock = socket.create_connection((ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            server_hostname = self._tunnel_host
        else:
            server_hostname = self.host
        self.sock = self._context.wrap_socket(sock, server_hostname=server_hostname)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    """urllib handler that dials plain HTTP through a validated, pinned IP."""

    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self._allow_private = allow_private

    def http_open(self, req: urllib.request.Request) -> Any:  # noqa: D102
        allow_private = self._allow_private

        def factory(host: str, **kwargs: Any) -> _PinnedHTTPConnection:
            return _PinnedHTTPConnection(host, _allow_private=allow_private, **kwargs)

        return self.do_open(factory, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """urllib handler that dials HTTPS through a validated, pinned IP."""

    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self._allow_private = allow_private

    def https_open(self, req: urllib.request.Request) -> Any:  # noqa: D102
        allow_private = self._allow_private

        def factory(host: str, **kwargs: Any) -> _PinnedHTTPSConnection:
            return _PinnedHTTPSConnection(host, _allow_private=allow_private, **kwargs)

        return self.do_open(
            factory, req, context=self._context, check_hostname=self._check_hostname
        )


def _same_origin(a: str, b: str) -> bool:
    """True iff two URLs share scheme, host, and effective port."""
    pa, pb = urlsplit(a), urlsplit(b)
    if pa.scheme != pb.scheme:
        return False
    if (pa.hostname or "").lower() != (pb.hostname or "").lower():
        return False
    default = {"http": 80, "https": 443}
    port_a = pa.port if pa.port is not None else default.get(pa.scheme)
    port_b = pb.port if pb.port is not None else default.get(pb.scheme)
    return port_a == port_b


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that re-applies the egress floor and strips auth.

    Each redirect (1) must not exceed :data:`_MAX_REDIRECTS`, (2) runs the new
    target through :func:`assert_url_egress_allowed`, and (3) drops the auth
    headers when the redirect crosses origin so the upstream bearer token never
    leaks to a different host.
    """

    max_redirections = _MAX_REDIRECTS

    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self._allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            assert_url_egress_allowed(newurl, allow_private=self._allow_private)
        except EgressBlocked as exc:
            raise urllib.error.HTTPError(
                newurl, code, f"blocked redirect target: {exc}", headers, fp,
            ) from exc
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _same_origin(req.full_url, newurl):
            for key in ("Authorization", "Proxy-Authorization", "Cookie"):
                new.remove_header(key)
                new.remove_header(key.lower())
        return new


def build_guarded_opener(allow_private: bool = False) -> urllib.request.OpenerDirector:
    """An ``OpenerDirector`` whose redirects are guarded and auth-stripped.

    Use this opener's ``open`` instead of ``urllib.request.urlopen`` so every
    redirect hop is re-checked and cross-origin hops drop the auth header. The
    pinned HTTP/HTTPS handlers re-resolve, re-validate, and pin the target IP
    at connect time on every hop, so a DNS name that re-resolves to a blocked
    address between the floor check and the socket connect (DNS rebinding) is
    still refused. The initial URL is also checked by the caller with
    :func:`assert_url_egress_allowed` for a fail-fast error before the request
    is built.
    """
    return urllib.request.build_opener(
        _GuardedRedirectHandler(allow_private=allow_private),
        _PinnedHTTPHandler(allow_private),
        _PinnedHTTPSHandler(allow_private),
    )
