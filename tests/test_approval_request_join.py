# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The macOS app reads every field the engine puts in an approval request.

approvals.request_approval writes the request; the menu-bar app is the
surface that asks the human. When the request gained the call's arguments
and their digest, the app kept reading only the tool name and the reason, so
a held delete reached the human as "Bash" with no command. This pins the
join: a key the writer adds and the app never reads fails here.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from vaara import approvals

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "clients" / "macos" / "Sources" / "VaaraMenuBar" / "Model.swift"


def _request_keys(tmp_path: Path) -> set[str]:
    seen: dict = {}

    def grab():
        for _ in range(200):
            files = list(tmp_path.glob("*.request.json"))
            if files:
                seen.update(json.loads(files[0].read_text()))
                return
            threading.Event().wait(0.01)

    t = threading.Thread(target=grab)
    t.start()
    approvals.request_approval("a-1", "Bash", "destructive", approvals_dir=tmp_path,
                               timeout=0.5, poll_interval=0.05,
                               parameters={"command": "rm -rf build/"})
    t.join()
    assert seen, "no request file was written"
    return set(seen)


def test_the_app_reads_every_request_field(tmp_path):
    written = _request_keys(tmp_path)
    read = set(re.findall(r'req\["([a-z_0-9]+)"\]', MODEL.read_text(encoding="utf-8")))
    assert {"parameters", "parameters_sha256"} <= written
    assert written <= read, f"written but never read by the app: {sorted(written - read)}"
