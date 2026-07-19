# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regex-based PII extractor.

Six categories cover the buyer-visible "PII detection" feature without
adding a heavy ML dependency:

- email
- phone (E.164 with country code + common US shapes)
- US Social Security Number (with a 000-area and trailing-0000 reject)
- IPv4 address
- credit-card number (13–19 digits, Luhn-checked)
- IBAN (rough length + checksum)

Findings carry the literal match, the category, and (offset, length) so
audit trails can highlight or redact in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIFinding:
    """One PII match in the scanned text."""

    category: str
    value: str
    offset: int
    length: int

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "value": self.value,
            "offset": self.offset,
            "length": self.length,
        }


@dataclass(frozen=True)
class PIIResult:
    """Aggregate result of a PII scan."""

    detected: bool
    findings: tuple[PIIFinding, ...]
    categories: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "categories": list(self.categories),
            "findings": [f.to_dict() for f in self.findings],
        }


_RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_RE_PHONE = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[(]))"
    r"\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}"
    r"(?:(?=\s)|(?=$)|(?=[).,;:!?]))"
)
_RE_SSN = re.compile(r"\b(?!000|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)
_RE_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


def _luhn_ok(digits: str) -> bool:
    s = 0
    parity = (len(digits) - 2) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0


def _iban_ok(value: str) -> bool:
    rotated = value[4:] + value[:4]
    expanded = "".join(
        ch if ch.isdigit() else str(ord(ch) - 55) for ch in rotated
    )
    try:
        return int(expanded) % 97 == 1
    except ValueError:
        return False


def detect_pii(text: str) -> PIIResult:
    """Scan free text for the six supported PII categories."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    findings: list[PIIFinding] = []
    for m in _RE_EMAIL.finditer(text):
        findings.append(PIIFinding("email", m.group(0), m.start(), m.end() - m.start()))
    for m in _RE_PHONE.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if 7 <= len(digits) <= 15:
            findings.append(PIIFinding(
                "phone", raw, m.start(), m.end() - m.start(),
            ))
    for m in _RE_SSN.finditer(text):
        findings.append(PIIFinding("ssn", m.group(0), m.start(), m.end() - m.start()))
    for m in _RE_IPV4.finditer(text):
        findings.append(PIIFinding("ipv4", m.group(0), m.start(), m.end() - m.start()))
    for m in _RE_CARD.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"[ -]", "", raw)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            findings.append(PIIFinding(
                "credit_card", raw, m.start(), m.end() - m.start(),
            ))
    for m in _RE_IBAN.finditer(text):
        raw = m.group(0)
        if _iban_ok(raw):
            findings.append(PIIFinding(
                "iban", raw, m.start(), m.end() - m.start(),
            ))

    categories = tuple(sorted({f.category for f in findings}))
    return PIIResult(
        detected=bool(findings),
        findings=tuple(findings),
        categories=categories,
    )
