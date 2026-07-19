# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Named detector aliases over Vaara's existing scoring surface.

Three buyer-visible categories that the EU AI Act and the agentic-AI
peer landscape brand as headline features:

- **Prompt injection** — wraps the existing adversarial scorer for free
  text. Routes through the same model that produces vaara-bench-v1's
  published numbers.
- **PII** — small regex-based extractor for emails, phone numbers, SSNs,
  IPv4 addresses, credit-card numbers (with Luhn check), and IBANs.
  Zero extra dependencies.

The detectors return structured ``DetectionResult`` objects with a
boolean ``detected`` flag, a normalized score in [0, 1] when meaningful,
and per-finding details suitable for audit-trail attachment.
"""

from vaara.detect.injection import InjectionResult, detect_injection
from vaara.detect.pii import (
    PIIFinding,
    PIIResult,
    detect_pii,
)

__all__ = [
    "InjectionResult",
    "PIIFinding",
    "PIIResult",
    "detect_injection",
    "detect_pii",
]
