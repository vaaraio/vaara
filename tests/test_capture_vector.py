# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate the vaara.capture/v0 vectors: the library and an independent checker agree.

The checker imports no Vaara and recomputes every digest from the policy, the
test key and the input arguments. The library test feeds the same policy and
key to ``vaara._capture.mask`` and compares with the committed cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")

from vaara import _capture  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "capture_digest_only_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, "-I", str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_library_reproduces_every_committed_case(tmp_path, monkeypatch):
    doc = json.loads((VECTORS / "cases.json").read_text(encoding="utf-8"))
    (tmp_path / "capture.json").write_text(json.dumps(doc["policy"]), encoding="utf-8")
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "capture-hmac.key").write_text(doc["key_hex"], encoding="utf-8")
    monkeypatch.setenv("VAARA_CAPTURE_POLICY", str(tmp_path / "capture.json"))
    for case in doc["cases"]:
        assert _capture.mask(case["tool"], case["parameters"]) == case["expected"], case["name"]
