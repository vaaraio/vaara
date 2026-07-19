# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-knowledge decisionProof engine (behind the attestation extra).

Transparent commit-and-prove over P-256 (secp256r1). Proves that a SEP-2828
decision verdict is the correct threshold output over committed policy, intent,
and inputs, without revealing them. Keyless and additive; the ML risk score is a
committed input, not part of the circuit.
"""
