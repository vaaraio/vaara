# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plain ``pip install vaara`` delivers a signed receipt per decision.

The README promises a verifiable receipt for every autonomous action and leads
with ``pip install vaara``. Receipts need ``cryptography`` and ``rfc8785``, so
those have to be base dependencies, not an extra. The end-to-end half of this
(build the wheel, install it into an empty venv, run decisions through the
hook, verify the receipts) is the ``default-install`` job in ci.yml.
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _base_dependencies() -> list[str]:
    # tomllib is 3.11+, and the matrix runs 3.10, so read the one line.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^dependencies = \[(.*?)\]", text, re.M | re.S)
    assert m, "no [project] dependencies line in pyproject.toml"
    return [re.split(r"[<>=!~\[ ]", d.strip().strip('"'))[0]
            for d in m.group(1).split(",") if d.strip()]


def test_signing_libraries_are_base_dependencies():
    deps = _base_dependencies()
    for name in ("cryptography", "rfc8785", "cbor2"):
        assert name in deps, f"{name} missing from the base install: {deps}"


def test_missing_signing_libraries_warn_once(tmp_path, monkeypatch, caplog):
    dr = importlib.import_module("vaara.audit.decision_receipts")
    monkeypatch.delenv("VAARA_RECEIPTS", raising=False)
    monkeypatch.setattr(dr, "signing_available", lambda: False)
    monkeypatch.setattr(dr, "_warned_unsigned", False)
    db = tmp_path / "audit.db"
    with caplog.at_level(logging.WARNING, logger=dr.__name__):
        assert dr.default_sink(db) is None
        assert dr.default_sink(db) is None
    warnings = [r for r in caplog.records if "decision receipts are OFF" in r.getMessage()]
    assert len(warnings) == 1


def test_switch_off_stays_quiet(tmp_path, monkeypatch, caplog):
    dr = importlib.import_module("vaara.audit.decision_receipts")
    monkeypatch.setenv("VAARA_RECEIPTS", "0")
    monkeypatch.setattr(dr, "_warned_unsigned", False)
    with caplog.at_level(logging.WARNING, logger=dr.__name__):
        assert dr.default_sink(tmp_path / "audit.db") is None
    assert not caplog.records
