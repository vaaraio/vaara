# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic parameter-content risk signals for the base rule scorer.

Stdlib only, so it runs in a zero-config install with no ML extra and with no
dependency on the integrations layer. The one signal here is the cloud-metadata
endpoint: a call parameter pointing at 169.254.169.254 (or its IPv6 / dotless /
hex encodings) is a known SSRF attack, not a probabilistic risk, so it is
treated as a hard decision floor rather than one averaged expert among many.

The boundary matches the MCP proxy egress guard
(vaara.integrations._egress_guard, the canonical network-enforcement point):
the metadata address has no legitimate reason to be dialed, while private and
loopback hosts stay reachable and are not floored here. The detection is
duplicated rather than imported to keep the scorer free of any dependency on
the integrations package.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Optional
from urllib.parse import urlsplit

# The cloud instance-metadata endpoints, refused unconditionally.
_METADATA_V4 = ipaddress.IPv4Address("169.254.169.254")
_METADATA_V6 = ipaddress.IPv6Address("fe80::a9fe:a9fe")

# Risk assigned when a metadata endpoint is found. High enough that the
# scorer's decision floor lands the call in deny/escalate regardless of the
# benign taxonomy base for a network read.
_METADATA_RISK = 0.95

# Bare host tokens that look like an IP or a dotless/hex integer, pulled from
# any string value in the parameters (not only well-formed URLs).
_HOSTISH = re.compile(r"(?:\[[0-9A-Fa-f:]+\]|[0-9A-Fa-f.x]+)")


def _coerce_dotless_host(host: str) -> Optional[ipaddress.IPv4Address]:
    """Parse a bare decimal or hex integer host (``2852039166``, ``0xa9fea9fe``).

    Browsers and ``inet_aton`` accept these as IPv4; ``ipaddress`` does not, so
    decode them explicitly. Returns the address when the host is such an
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


def _is_metadata(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:a.b.c.d judged on the embedded v4 address
        ip = mapped
    return ip == _METADATA_V4 or ip == _METADATA_V6


def _host_is_metadata(host: str) -> bool:
    host = host.strip().strip("[]")
    if not host:
        return False
    dotless = _coerce_dotless_host(host)
    if dotless is not None and _is_metadata(dotless):
        return True
    try:
        return _is_metadata(ipaddress.ip_address(host))
    except ValueError:
        return False


def _string_hits_metadata(value: str) -> bool:
    # Try the value as a URL first (covers scheme://host:port/path), then
    # scan any host-shaped tokens so an unschemed or embedded address is
    # still caught.
    host = urlsplit(value).hostname
    if host and _host_is_metadata(host):
        return True
    return any(_host_is_metadata(tok) for tok in _HOSTISH.findall(value))


def _walk(value: Any) -> bool:
    if isinstance(value, str):
        return _string_hits_metadata(value)
    if isinstance(value, dict):
        return any(_walk(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_walk(v) for v in value)
    return False


def metadata_endpoint_risk(parameters: Any) -> float:
    """Return ``_METADATA_RISK`` if any parameter targets a cloud-metadata
    endpoint, else 0.0. Recurses through nested dicts and lists."""
    return _METADATA_RISK if _walk(parameters) else 0.0
