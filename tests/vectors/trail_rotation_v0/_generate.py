#!/usr/bin/env python3
"""Regenerate cases.json for the trail_rotation_v0 vectors.

Stable: the keys, the clock and the record ids are fixed, so a second run
writes the same bytes. The expected verdicts are written here by hand, from
the rule the vectors state, never read back from a verifier.

    python tests/vectors/trail_rotation_v0/_generate.py
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.trail import AuditTrail
from vaara.audit.verify import ROTATION_RECORD_TYPE, rotation_payload

HERE = Path(__file__).resolve().parent
# TEST keys. A deployment's key is its own and is never published.
SEEDS = {"signer": bytes(range(32)), "other_signer": bytes(range(32, 64))}
KEYS = {name: Ed25519PrivateKey.from_private_bytes(seed) for name, seed in SEEDS.items()}
ARCHIVE = bytes(range(256)) * 4  # stands in for a sealed (encrypted) archive file
ARCHIVE_SHA = hashlib.sha256(ARCHIVE).hexdigest()
T0 = 1_792_000_000.0


def raw_public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


FINGERPRINT = hashlib.sha256(raw_public(KEYS["signer"])).hexdigest()[:32]


class _Clock:
    """A clock that moves one second per call, and ids that count."""

    def __init__(self) -> None:
        self.n = 0

    def reset(self) -> None:
        self.n = 0
        time.time = lambda: T0 + self._tick()
        uuid.uuid4 = lambda: uuid.UUID(int=self._tick_id())

    def _tick(self) -> float:
        self.n += 1
        return float(self.n)

    def _tick_id(self) -> int:
        self.n += 1
        return 0xA11CE000000000000000000000000000 + self.n


CLOCK = _Clock()


def signed_by(name: str, *, tamper: dict | None = None):
    key = KEYS[name]

    def sign(data: dict) -> dict:
        covered = dict(data, **(tamper or {}))
        return {"type": ROTATION_RECORD_TYPE, "alg": "Ed25519",
                "signature": key.sign(rotation_payload(covered)).hex()}
    return sign


def chain(*, sign=None, last_of: int = 1, extra_after_seam: int = 0, keep: slice, **overrides):
    """Three decisions (a, b, c), then a seam naming decision ``last_of`` as the
    last one archived, then ``extra_after_seam`` more decisions. Returns the
    records in ``keep`` as dicts."""
    CLOCK.reset()
    trail = AuditTrail()
    for name in ("a", "b", "c"):
        trail.record_decision(name, "agent-x", "read_file", "allow", "low risk", 0.1)
    fields = dict(
        last_hash=trail._records[last_of].record_hash,
        purged_records=last_of + 1,
        archive_sha256=ARCHIVE_SHA,
        archive_records=3,
        archive_kind="sealed",
        boundary_utc="2026-10-20T00:00:00Z",
        signer_fingerprint=FINGERPRINT,
    )
    fields.update(overrides)
    trail.record_rotation(sign=sign or signed_by("signer"), **fields)
    for name in range(extra_after_seam):
        trail.record_decision(f"d{name}", "agent-x", "read_file", "allow", "low risk", 0.1)
    return [r.to_dict() for r in trail._records][keep]


def second_rotation():
    """a, b, c, rotation 1 (names b), d, rotation 2 (names d, the head). Only
    rotation 2 survives, and it links to the hash it names."""
    CLOCK.reset()
    trail = AuditTrail()
    for name in ("a", "b", "c"):
        trail.record_decision(name, "agent-x", "read_file", "allow", "low risk", 0.1)
    base = dict(archive_sha256=ARCHIVE_SHA, archive_records=3, archive_kind="sealed",
                boundary_utc="2026-10-20T00:00:00Z", signer_fingerprint=FINGERPRINT,
                sign=signed_by("signer"))
    trail.record_rotation(last_hash=trail._records[1].record_hash, purged_records=2, **base)
    trail.record_decision("d", "agent-x", "read_file", "allow", "low risk", 0.1)
    trail.record_rotation(last_hash=trail._records[-1].record_hash, purged_records=5, **base)
    return [trail._records[-1].to_dict()]


def seam(index, state, begins):
    return {"index": index, "state": state, "begins_here": begins}


def expect(verdict, chain_, begins, seams, archive=None):
    return {"verdict": verdict, "chain": chain_, "begins_at_seam": begins,
            "seams": seams, "archive_check": archive}


def cases() -> list[dict]:
    last = slice(2, None)  # the survivors of a purge that took a and b
    out = [
        ("a_signed_seam_begins_the_trail",
         "The first record links to b, and a signed rotation record in the trail names b.",
         chain(keep=last), None, None,
         expect("pass", "ok", True, [seam(1, "signed", True)])),
        ("a_trail_that_begins_at_its_first_record",
         "No purge happened: the first record links to nothing, and there is no seam to find.",
         chain(keep=slice(0, 3)), None, None,
         expect("pass", "ok", False, [])),
        ("a_first_record_that_links_to_nothing_held",
         "The trail begins at c and holds no rotation record: a chain break.",
         chain(keep=slice(2, 3)), None, None,
         expect("fail", "break", False, [])),
        ("a_seam_that_names_another_hash",
         "The rotation record is signed but names a, not b: c links to nothing held.",
         chain(last_of=0, purged_records=2, keep=last), None, None,
         expect("fail", "break", False, [seam(1, "signed", False)])),
        ("an_unsigned_seam",
         "The rotation record names b and carries no signature: it accounts for nothing.",
         chain(sign=lambda data: None, keep=last), None, None,
         expect("fail", "break", False, [seam(1, "unsigned", False)])),
        ("a_seam_signed_over_other_fields",
         "The signature is over a purged count one higher than the record states.",
         chain(sign=signed_by("signer", tamper={"purged_records": 3}), keep=last), None, None,
         expect("fail", "break", False, [seam(1, "bad_signature", False)])),
        ("a_seam_signed_by_another_key",
         "The rotation record is signed by a key that did not sign the export.",
         chain(sign=signed_by("other_signer"), keep=last), None, None,
         expect("fail", "break", False, [seam(1, "bad_signature", False)])),
        ("a_seam_signed_by_another_key_that_is_named",
         "The same record, with the key that signed it given as the rotation key.",
         chain(sign=signed_by("other_signer"), keep=last), "other_signer", None,
         expect("pass", "ok", True, [seam(1, "signed", True)])),
        ("a_seam_that_does_not_begin_the_trail",
         "The whole chain, with a rotation record at the tip whose purge never ran.",
         chain(keep=slice(0, None)), None, None,
         expect("pass", "ok", False, [seam(3, "signed", False)])),
        ("a_second_rotation_begins_at_its_own_record",
         "A later rotation purged the first one's record too: the one record left names the head it links to.",
         second_rotation(), None, None,
         expect("pass", "ok", True, [seam(0, "signed", True)])),
        ("a_malformed_seam",
         "The rotation record states a purged count of 0.",
         chain(purged_records=0, keep=last), None, None,
         expect("fail", "break", False, [seam(1, "malformed", False)])),
        ("records_written_after_the_seam_still_chain",
         "A record written after the rotation links to it: the seam is not the last record.",
         chain(extra_after_seam=2, keep=last), None, None,
         expect("pass", "ok", True, [seam(1, "signed", True)])),
        ("the_archive_is_the_one_the_record_names",
         "The sealed archive file hashes to the digest in the record. Contents are not read.",
         chain(keep=last), None, "match",
         expect("pass", "ok", True, [seam(1, "signed", True)], "digest_only")),
        ("the_archive_is_a_different_file",
         "One byte of the archive differs from the digest the record states.",
         chain(keep=last), None, "altered",
         expect("fail", "ok", True, [seam(1, "signed", True)], "digest_mismatch")),
    ]
    return [
        {"name": name, "description": desc, "records": records, "rotation_key": rotation_key,
         "archive": archive, "expected": expected}
        for name, desc, records, rotation_key, archive, expected in out
    ]


def main() -> None:
    altered = bytearray(ARCHIVE)
    altered[0] ^= 1
    doc = {
        "format": "vaara.trail-rotation/v0",
        "rule": (
            "A trail whose first record links to a hash outside the trail is accepted "
            "only if a rotation record in the trail names that hash as last_hash and its "
            "Ed25519 signature verifies over the JCS payload; otherwise it is a chain break."
        ),
        "signers": {name: {"seed_hex": SEEDS[name].hex(), "public_hex": raw_public(key).hex()}
                    for name, key in KEYS.items()},
        "export_signer": "signer",
        "archive_hex": ARCHIVE.hex(),
        "altered_archive_hex": bytes(altered).hex(),
        "cases": cases(),
    }
    (HERE / "cases.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(doc['cases'])} cases")


if __name__ == "__main__":
    main()
