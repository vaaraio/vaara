"""Integrity pinning for classifier bundles.

joblib.load is pickle: loading a tampered bundle is arbitrary code execution.
The shipped default bundle is verified against a pinned SHA-256 before it is
unpickled, and a caller loading their own bundle can pass expected_sha256 to
get the same fail-closed check.
"""

from __future__ import annotations

import hashlib
import shutil

import pytest

pytest.importorskip("joblib")
pytest.importorskip("xgboost")

import vaara.adversarial_classifier as ac


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tampered_copy(tmp_path):
    dst = tmp_path / "tampered.joblib"
    shutil.copy(ac._DEFAULT_BUNDLE, dst)
    raw = bytearray(dst.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    dst.write_bytes(bytes(raw))
    return dst


def test_pinned_hash_matches_shipped_bundle():
    assert ac._DEFAULT_BUNDLE_SHA256 == _sha256(ac._DEFAULT_BUNDLE)


def test_default_bundle_load_verifies_pin(monkeypatch, tmp_path):
    tampered = _tampered_copy(tmp_path)
    monkeypatch.setattr(ac, "_DEFAULT_BUNDLE", tampered)
    with pytest.raises(ValueError, match="integrity"):
        ac.AdversarialClassifier()


def test_explicit_path_with_expected_sha256_rejects_tampered(tmp_path):
    tampered = _tampered_copy(tmp_path)
    with pytest.raises(ValueError, match="integrity"):
        ac.AdversarialClassifier(
            bundle_path=str(tampered),
            expected_sha256=ac._DEFAULT_BUNDLE_SHA256,
        )


def test_explicit_path_with_matching_sha256_loads(tmp_path):
    copy = tmp_path / "copy.joblib"
    shutil.copy(ac._DEFAULT_BUNDLE, copy)
    clf = ac.AdversarialClassifier(
        bundle_path=str(copy),
        expected_sha256=_sha256(copy),
    )
    assert clf.bundle_version


def test_explicit_path_without_pin_still_loads(tmp_path):
    # A caller's own bundle with no declared pin keeps working; the trust
    # decision is theirs.
    copy = tmp_path / "copy.joblib"
    shutil.copy(ac._DEFAULT_BUNDLE, copy)
    clf = ac.AdversarialClassifier(bundle_path=str(copy))
    assert clf.bundle_version
