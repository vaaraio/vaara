"""Offline tests for scripts/anchor_release.py.

Covers the recomputable tree fingerprint and the tamper-detection branch of
--verify. The token-verification success path needs a live qualified TSA and is
intentionally out of the unit suite.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "anchor_release.py"


def _load():
    spec = importlib.util.spec_from_file_location("anchor_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tree_fingerprint_deterministic_and_recomputable():
    mod = _load()
    fp1 = mod.tree_fingerprint("HEAD")
    fp2 = mod.tree_fingerprint("HEAD")
    assert fp1 == fp2  # stable

    # Matches the documented public recompute: sha256 over the ls-tree bytes.
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--full-tree", "HEAD"])
    assert fp1 == hashlib.sha256(out).hexdigest()


def test_verify_detects_changed_tree(tmp_path: Path):
    mod = _load()
    wrong = "de" * 32  # not the real HEAD fingerprint
    doc = {
        "release": "HEAD",
        "treeFingerprint": {"sha256": wrong, "recompute": "git ls-tree ..."},
        "anchor": {
            "chain_position": 0,
            "chain_head_hash": wrong,
            "backend": "rfc3161",
            "tsa_url": "https://tsa.example/qualified",
            "hash_algorithm": "sha256",
            "token_b64": "AA==",
            "anchored_time": "2026-07-02T00:00:00+00:00",
        },
    }
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    # HEAD's real fingerprint won't equal `wrong`, so verify fails closed
    # before it ever touches the token.
    assert mod.verify(path) == 1
