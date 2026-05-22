"""Atheris fuzz target: OVERT 1.0 envelope CBOR decode + verify.

Mirrors the attack surface of `vaara overt verify`: attacker-controlled CBOR
bytes that get loaded by `cbor2.loads`, structurally validated against the
closed 9-field schema, mapped into a `BaseEnvelope`, then signature-checked.

A finding is anything that escapes the documented error contract: any
exception other than the ones the verifier path is expected to swallow
(`EnvelopeError`, `CBORDecodeError`, `TypeError`, `ValueError`, `KeyError`,
`OverflowError`, `MemoryError`). Hangs and uncaught crashes count too.
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import cbor2
    from cbor2 import CBORDecodeError

    from vaara.attestation.overt import (
        BaseEnvelope,
        EnvelopeError,
        verify_base_envelope,
    )

_REQUIRED_KEYS = (
    "blinded_identifier",
    "request_commitment",
    "encoder_binary_identity",
    "non_content_metadata",
    "monotonic_counter",
    "nanosecond_timestamp",
    "key_identifier",
    "arbiter_instance_identifier",
    "signature",
)

# Fixed 32-byte all-zero pubkey: cheap, deterministic, never collides with
# a real key_identifier so verify_base_envelope must always reach the
# signature path or reject earlier.
_DUMMY_PUBKEY = b"\x00" * 32


def TestOneInput(data: bytes) -> None:
    try:
        decoded = cbor2.loads(data)
    except (CBORDecodeError, MemoryError, OverflowError, ValueError):
        return
    except Exception:
        # Anything else from the CBOR decoder is interesting.
        raise

    if not isinstance(decoded, dict):
        return
    if any(k not in decoded for k in _REQUIRED_KEYS):
        return

    try:
        envelope = BaseEnvelope(
            blinded_identifier=decoded["blinded_identifier"],
            request_commitment=decoded["request_commitment"],
            encoder_binary_identity=decoded["encoder_binary_identity"],
            non_content_metadata=decoded["non_content_metadata"],
            monotonic_counter=int(decoded["monotonic_counter"]),
            nanosecond_timestamp=int(decoded["nanosecond_timestamp"]),
            key_identifier=decoded["key_identifier"],
            arbiter_instance_identifier=decoded["arbiter_instance_identifier"],
            signature=decoded["signature"],
        )
    except (TypeError, ValueError, OverflowError, KeyError):
        return

    try:
        verify_base_envelope(envelope, _DUMMY_PUBKEY)
    except (EnvelopeError, TypeError, ValueError):
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
