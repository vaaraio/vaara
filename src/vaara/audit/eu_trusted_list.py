# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parse the EU List of Trusted Lists (LOTL) and national trusted lists.

The LOTL (ETSI TS 119 612, EU Commission Implementing Decision 2015/1505) is a
top-level XML that points to each member state's national trusted list. Each
national list enumerates trust service providers and their services, of which
the qualified timestamping services (``ServiceTypeIdentifier`` ending in
``TSA/QTST``) are the ones a Vaara operator can pick as a qualified RFC 3161
anchor provider.

This module only parses bytes into plain records. Fetching them over the
network, caching, and presenting a picker live in their own layers, so this
part stays pure and testable from committed fixtures. It endorses no provider:
it surfaces the official public list for the operator to choose from.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

# The official EU List of Trusted Lists, published by the European Commission.
LOTL_URL = "https://ec.europa.eu/tools/lotl/eu-lotl.xml"

# ETSI service type and status URIs (TS 119 612).
_QTST = "http://uri.etsi.org/TrstSvc/Svctype/TSA/QTST"
_GRANTED = "http://uri.etsi.org/TrstSvc/TrustedList/Svcstatus/granted"
_XML_TL_MIME = "application/vnd.etsi.tsl+xml"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


@dataclass(frozen=True)
class TSLPointer:
    """A pointer from the LOTL to one member state's national trusted list."""

    territory: str
    location: str


@dataclass(frozen=True)
class QualifiedTSA:
    """A qualified timestamping service an operator can anchor against."""

    territory: str
    provider: str
    service_name: str
    endpoint: str


def _ln(elem: ET.Element) -> str:
    """Local name of a tag, dropping any ``{namespace}`` prefix."""
    return elem.tag.rsplit("}", 1)[-1]


def _iter(root: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in root.iter() if _ln(e) == name]


def _first(root: ET.Element, name: str) -> ET.Element | None:
    for e in root.iter():
        if _ln(e) == name:
            return e
    return None


def _text(elem: ET.Element | None) -> str:
    return (elem.text or "").strip() if elem is not None else ""


def _name_text(container: ET.Element | None) -> str:
    """First ``Name`` under a container, preferring the English variant."""
    if container is None:
        return ""
    names = _iter(container, "Name")
    for n in names:
        if n.get(_XML_LANG) == "en":
            return _text(n)
    return _text(names[0]) if names else ""


def parse_lotl(data: bytes) -> list[TSLPointer]:
    """Return the pointers to member-state XML trusted lists in a LOTL.

    Pointers to human-readable renderings (PDF) or other MIME types are
    skipped, so only machine-readable national lists remain.
    """
    root = ET.fromstring(data)
    pointers: list[TSLPointer] = []
    for ptr in _iter(root, "OtherTSLPointer"):
        location = _text(_first(ptr, "TSLLocation"))
        territory = _text(_first(ptr, "SchemeTerritory"))
        mime = ""
        for m in _iter(ptr, "MimeType"):
            mime = _text(m)
            if mime:
                break
        if not location or mime != _XML_TL_MIME:
            continue
        pointers.append(TSLPointer(territory=territory, location=location))
    return pointers


def parse_trusted_list(data: bytes, territory: str = "") -> list[QualifiedTSA]:
    """Return the granted qualified timestamping services in a national list."""
    root = ET.fromstring(data)
    out: list[QualifiedTSA] = []
    for tsp in _iter(root, "TrustServiceProvider"):
        provider = _name_text(_first(tsp, "TSPName"))
        for svc in _iter(tsp, "TSPService"):
            if _text(_first(svc, "ServiceTypeIdentifier")) != _QTST:
                continue
            status = _text(_first(svc, "ServiceStatus"))
            if status and status != _GRANTED:
                continue
            supply = _first(svc, "ServiceSupplyPoint")
            out.append(
                QualifiedTSA(
                    territory=territory,
                    provider=provider,
                    service_name=_name_text(_first(svc, "ServiceName")),
                    endpoint=_text(supply),
                )
            )
    return out


def providers_for_country(
    country: str, fetch: Callable[[str], bytes]
) -> list[QualifiedTSA]:
    """Fetch the LOTL, find the national list for ``country``, and return its
    granted qualified timestamping services.

    ``fetch`` maps a URL to bytes; it is injected so this stays testable and
    the network transport (and any caching) lives in the caller.
    """
    for ptr in parse_lotl(fetch(LOTL_URL)):
        if ptr.territory == country:
            return parse_trusted_list(fetch(ptr.location), territory=country)
    return []
