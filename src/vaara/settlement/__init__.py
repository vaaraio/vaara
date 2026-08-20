# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settlement-side surfaces: what a receipt is worth when money is involved.

Everything else in Vaara runs one direction. A payment gates access, and the
settlement evidence lands inside a receipt: that is the x402 gate in
``vaara.server.x402`` and the ``x402.settlement.*/v0`` profile in SPEC.md
Section 5.2.

This package is the inversion. A release condition holds money against a named,
signed statement of what must be proved, and a Vaara receipt proving the
authorised action happened is what releases it. The receipt gates the payment.

Public surface is :mod:`vaara.settlement.release`.
"""

from __future__ import annotations
