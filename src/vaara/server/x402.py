# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inbound x402 payment gate for paid Vaara endpoints (HTTP 402).

The gate is *pricing-agnostic plumbing*: it turns a route into a paid resource
by emitting an HTTP 402 challenge with x402 payment requirements, and admits the
request only once a valid ``X-PAYMENT`` header is presented and settled through
an x402 facilitator.

Configuration is entirely by environment, so the same server binary runs free in
development and paid in production:

    VAARA_X402_ENABLED      "1"/"true" to enforce payment. Default off (free).
    VAARA_X402_PAY_TO       receiving wallet address (e.g. 0x... on Base).
    VAARA_X402_NETWORK      settlement network id. Default "base".
    VAARA_X402_ASSET        asset contract address (e.g. USDC on Base).
    VAARA_X402_PRICE        amount required, atomic units, as a string.
    VAARA_X402_FACILITATOR  facilitator base URL for /verify + /settle.

When ``VAARA_X402_ENABLED`` is unset the gate is a no-op: every request is
admitted, so tests and self-hosters get the full pipeline for free. The paid
path is wired and inert until an operator turns it on.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

_X402_VERSION = 1


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class X402Config:
    """x402 gate configuration, sourced from the environment."""

    enabled: bool
    pay_to: str
    network: str
    asset: str
    price: str
    facilitator: Optional[str]

    @classmethod
    def from_env(cls) -> "X402Config":
        return cls(
            enabled=_truthy(os.environ.get("VAARA_X402_ENABLED")),
            pay_to=os.environ.get("VAARA_X402_PAY_TO", "").strip(),
            network=os.environ.get("VAARA_X402_NETWORK", "base").strip(),
            asset=os.environ.get("VAARA_X402_ASSET", "").strip(),
            price=os.environ.get("VAARA_X402_PRICE", "0").strip(),
            facilitator=(
                os.environ.get("VAARA_X402_FACILITATOR", "").strip() or None
            ),
        )


class X402Gate:
    """A pricing-agnostic x402 payment gate.

    ``check`` returns ``None`` when a request is admitted (free mode, or a
    settled payment) and a ready-to-return ``JSONResponse`` carrying the 402
    challenge otherwise. Route handlers call it first and short-circuit on a
    non-None result.
    """

    def __init__(self, config: Optional[X402Config] = None) -> None:
        self.config = config or X402Config.from_env()

    def price_for(self, declared_amount: Optional[str]) -> str:
        """Amount required for this call.

        Flat by default (``VAARA_X402_PRICE``). ``declared_amount`` is the hook
        for value-at-risk pricing (a share of a settled amount the caller
        declares); the plumbing stays flat until a pricing model is chosen, so
        no model is baked in here.
        """
        return self.config.price

    def requirements(self, resource: str, description: str, amount: str) -> dict:
        return {
            "x402Version": _X402_VERSION,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": self.config.network,
                    "maxAmountRequired": amount,
                    "resource": resource,
                    "description": description,
                    "mimeType": "application/json",
                    "payTo": self.config.pay_to,
                    "asset": self.config.asset,
                    "maxTimeoutSeconds": 60,
                }
            ],
            "error": "payment required",
        }

    def check(
        self,
        request: Request,
        *,
        resource: str,
        description: str,
        amount: Optional[str] = None,
    ) -> Optional[JSONResponse]:
        """Admit (return None) or challenge (return a 402 JSONResponse)."""
        if not self.config.enabled:
            return None
        need = amount or self.config.price
        header = request.headers.get("X-PAYMENT")
        if not header:
            return JSONResponse(
                status_code=402,
                content=self.requirements(resource, description, need),
            )
        if not self._settle(header, resource, need):
            body = self.requirements(resource, description, need)
            body["error"] = "payment invalid or unsettled"
            return JSONResponse(status_code=402, content=body)
        return None

    def _settle(self, payment_header: str, resource: str, amount: str) -> bool:
        """Verify and settle a presented payment through the facilitator.

        Returns True only when the facilitator confirms settlement. With no
        facilitator configured a presented payment cannot be settled, so the
        call is refused rather than trusted: an enabled gate never admits an
        unverifiable payment.
        """
        if not self.config.facilitator:
            return False
        payload = json.dumps(
            {
                "x402Version": _X402_VERSION,
                "paymentHeader": payment_header,
                "resource": resource,
                "amount": amount,
                "payTo": self.config.pay_to,
                "asset": self.config.asset,
                "network": self.config.network,
            }
        ).encode("ascii")
        base = self.config.facilitator.rstrip("/")
        for path in ("/verify", "/settle"):
            try:
                req = urllib.request.Request(
                    base + path,
                    data=payload,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                    report = json.load(resp)
            except Exception:
                return False
            if not (
                report.get("isValid")
                or report.get("success")
                or report.get("settled")
            ):
                return False
        return True
