# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public parameters for the decisionProof engine and their digest.

The setup is transparent: the only public parameters are the curve, the second
Pedersen generator H (itself derived by hash-to-curve), the range width, and the
fixed-point scale. `params_digest()` is the `verifierParamsDigest` carried in the
proof envelope, so a verifier can confirm it used the same parameters as the
prover.
"""

from __future__ import annotations

import hashlib
import json

from ._commit import H

PROOF_SYSTEM = "vaara-p256-cap-v0"

# Risk scores and thresholds are floats in [0, 1]. They are scaled to fixed-point
# integers in [0, 2**RANGE_BITS) before entering the circuit.
RANGE_BITS = 32
SCALE = 10**6


def params_digest() -> str:
    obj = {
        "system": PROOF_SYSTEM,
        "curve": "P-256",
        "H": H.to_bytes().hex(),
        "rangeBits": RANGE_BITS,
        "scale": SCALE,
    }
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()
