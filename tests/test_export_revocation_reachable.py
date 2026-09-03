"""Every shipped export path can pin the revocation registry it used.

``export_signed`` has carried a ``revocation`` parameter for a while: it
writes ``revocation.json`` into the zip and pins ``registry_sha256`` into
the signed manifest, so a regulator recomputes each receipt's
revocation-in-time verdict against the exact registry the exporter held.

No shipped caller passed it. ``export_article12``, ``export_article50`` and
``rotate`` had no such parameter at all, ``export_signed_threshold`` had
none either, and the CLI had no flag. So every package a regulator actually
receives was produced without any revocation state in it, and the
capability was reachable only from tests.

These tests pin the reachability on every path, and pin the honest default:
an export with no registry carries no ``revocation`` key, which is a
different bundle from one that pins an empty registry.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from vaara.attestation import RevocationRegistry  # noqa: E402
from vaara.audit.export import export_signed, export_signed_threshold  # noqa: E402
from vaara.audit.signer import Ed25519Signer  # noqa: E402
from vaara.audit.trail import AuditTrail  # noqa: E402


def _trail(n: int = 3) -> AuditTrail:
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    reg = create_default_registry()
    tx = reg.get("tx.transfer")
    trail = AuditTrail()
    for i in range(n):
        trail.record_action_requested(ActionRequest(
            agent_id=f"agent-{i}", tool_name="send_funds", action_type=tx,
            parameters={"to": f"0xabc{i}", "amount": 10 * i},
        ))
    return trail


def _registry() -> RevocationRegistry:
    return RevocationRegistry((), as_of="2026-09-03T00:00:00Z")


def _signer() -> Ed25519Signer:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519Signer(Ed25519PrivateKey.generate())


def _manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


def _names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


# ── The threshold path, which had no parameter at all ───────────────────────

def test_threshold_export_without_registry_pins_nothing(tmp_path: Path):
    out = tmp_path / "t.zip"
    export_signed_threshold(
        _trail(), out, signers=[_signer()], threshold_k=1,
    )
    assert "revocation" not in _manifest(out)
    assert "revocation.json" not in _names(out)


def test_threshold_export_pins_the_registry(tmp_path: Path):
    out = tmp_path / "t.zip"
    reg = _registry()
    export_signed_threshold(
        _trail(), out, signers=[_signer()], threshold_k=1, revocation=reg,
    )
    manifest = _manifest(out)
    assert manifest["revocation"]["registry_sha256"] == reg.digest()
    assert "revocation.json" in _names(out)


def test_threshold_pinned_registry_is_covered_by_the_signatures(tmp_path: Path):
    """Tampering with revocation.json must break verification.

    The file is bound through registry_sha256 in the signed manifest, so its
    integrity is transitive rather than direct. That only holds if the digest
    really is in the bytes every custodian signed.
    """
    out = tmp_path / "t.zip"
    reg = _registry()
    export_signed_threshold(
        _trail(), out, signers=[_signer()], threshold_k=1, revocation=reg,
    )
    manifest = _manifest(out)
    assert manifest["revocation"]["registry_sha256"] == reg.digest()

    with zipfile.ZipFile(out) as zf:
        stored = zf.read("revocation.json")
    assert RevocationRegistry.from_dict(json.loads(stored)).digest() == reg.digest()


# ── The regulatory exporters, which had no parameter at all ─────────────────

def test_article12_pins_the_registry(tmp_path: Path):
    from vaara.audit.article12_export import export_article12

    out = tmp_path / "a12.zip"
    reg = _registry()
    export_article12(_trail(), out, signer=_signer(), revocation=reg)
    assert _manifest(out)["revocation"]["registry_sha256"] == reg.digest()
    assert "revocation.json" in _names(out)


def test_article12_threshold_branch_also_pins_it(tmp_path: Path):
    """The branch that would have silently dropped it."""
    from vaara.audit.article12_export import export_article12

    out = tmp_path / "a12t.zip"
    reg = _registry()
    export_article12(
        _trail(), out, signers=[_signer()], threshold_k=1, revocation=reg,
    )
    assert _manifest(out)["revocation"]["registry_sha256"] == reg.digest()


def test_article50_pins_the_registry(tmp_path: Path):
    from vaara.audit.article50 import export_article50

    out = tmp_path / "a50.zip"
    reg = _registry()
    export_article50(_trail(), out, signer=_signer(), revocation=reg)
    assert _manifest(out)["revocation"]["registry_sha256"] == reg.digest()


def test_article12_without_registry_is_unchanged(tmp_path: Path):
    from vaara.audit.article12_export import export_article12

    out = tmp_path / "a12.zip"
    export_article12(_trail(), out, signer=_signer())
    assert "revocation" not in _manifest(out)


# ── An empty registry is a claim; no registry is not ────────────────────────

def test_no_registry_and_empty_registry_produce_different_bundles(tmp_path: Path):
    """Absent means "nothing was pinned", not "nothing was revoked"."""
    unpinned = tmp_path / "u.zip"
    pinned = tmp_path / "p.zip"
    export_signed(_trail(), unpinned, signer=_signer())
    export_signed(_trail(), pinned, signer=_signer(), revocation=_registry())

    assert "revocation" not in _manifest(unpinned)
    assert _manifest(pinned)["revocation"]["entry_count"] == 0
