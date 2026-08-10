# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OVERT example in standards.md is executed end to end.

The page emitted an envelope in Python and then said `vaara overt verify
RECEIPT.cbor --pubkey-file PUB.bin` validates it. The step between the
two was missing, and the two obvious guesses both fail:
``canonical_cbor(envelope)`` raises ``CBOREncodeError`` because the
envelope is a dataclass, and canonical CBOR over ``envelope.to_dict()``
produces a file the verifier rejects. The one function that works,
``envelope_to_canonical_cbor``, was never named on the page.

This test runs the documented Python, writes the two files the way the
page now says to, and shells out to the documented command. It fails if
either half drifts.
"""

from __future__ import annotations

import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytest.importorskip("cbor2")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)

import vaara  # noqa: E402
from vaara.attestation import envelope_to_canonical_cbor  # noqa: E402
from vaara.attestation.overt import (  # noqa: E402
    emit_base_envelope, encoder_binary_identity, make_request_commitment,
)

DOC = Path(__file__).resolve().parents[1] / "docs" / "standards.md"


def _emit():
    key = Ed25519PrivateKey.generate()
    envelope = emit_base_envelope(
        signing_key=key,
        request_commitment=make_request_commitment(b"payload", operator_key=b"k" * 32),
        encoder_binary_identity=encoder_binary_identity(
            arbiter_version=f"vaara/{vaara.__version__}", policy_hash=b"h" * 32,
        ),
        non_content_metadata={"action_class": "tx.transfer", "decision": "escalate"},
        monotonic_counter=42,
        arbiter_instance_identifier=uuid.uuid4().bytes,
    )
    return key, envelope


def test_the_documented_emit_and_verify_round_trips(tmp_path):
    key, envelope = _emit()
    receipt = tmp_path / "receipt.cbor"
    pubkey = tmp_path / "pub.bin"
    receipt.write_bytes(envelope_to_canonical_cbor(envelope))
    pubkey.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw,
        )
    )

    done = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "overt", "verify",
         str(receipt), "--pubkey-file", str(pubkey)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert '"valid": true' in done.stdout


def test_the_page_names_the_serializer_that_works(tmp_path):
    """Regression: the page skipped from an envelope object to a .cbor file."""
    text = DOC.read_text(encoding="utf-8")
    assert "envelope_to_canonical_cbor" in text, (
        "standards.md shows `vaara overt verify RECEIPT.cbor` without saying "
        "how to produce RECEIPT.cbor, and the obvious guesses do not work"
    )
    command = re.search(r"`vaara overt verify ([^`]+)`", text)
    assert command, "standards.md no longer shows the verify command"
    assert "--pubkey-file" in command.group(1)


def test_the_envelope_carries_the_nine_documented_fields():
    """"closed 9-field schema" is a claim about the wire form."""
    import dataclasses

    _, envelope = _emit()
    assert len(dataclasses.fields(envelope)) == 9
