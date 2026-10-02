# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The keys the Rancher guide tells you to make are the keys the code accepts.

docs/kubernetes-rancher.md used to create one key with ``vaara keygen --dev``
and hand it to both the proxy (``--signing-key``) and ``vaara trail export``
(``--key``). keygen --dev writes Ed25519. The proxy only takes EC P-256 or
RSA, and the export only takes Ed25519, so whichever key an operator made,
one of the two documented commands failed on first use.

These tests run the guide's own keygen lines and feed each key to the loader
that will read it in the cluster, so the guide and the code cannot drift
apart again without a red build.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from vaara.audit.export import _load_private_key
from vaara.cli import main as vaara_main
from vaara.integrations._infer_proxy_sign import load_signing_key

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "kubernetes-rancher.md"
CHART = ROOT / "deploy" / "helm" / "vaara"


def _keygen_lines() -> dict[str, list[str]]:
    """Map output filename to the keygen argv the guide prints for it."""
    lines: dict[str, list[str]] = {}
    for raw in GUIDE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("vaara keygen "):
            continue
        argv = shlex.split(line)[1:]
        out = argv[argv.index("--out") + 1]
        lines[out] = argv
    return lines


def _generate(tmp_path: Path, name: str) -> Path:
    argv = list(_keygen_lines()[name])
    target = tmp_path / name
    argv[argv.index("--out") + 1] = str(target)
    assert vaara_main(argv) in (0, None)
    return target


def test_the_guide_makes_both_keys():
    assert set(_keygen_lines()) == {"signing_key.pem", "trail_key.pem"}


def test_the_proxy_key_from_the_guide_loads_in_the_proxy(tmp_path):
    _, alg, _ = load_signing_key(_generate(tmp_path, "signing_key.pem"), None)
    assert alg == "ES256"


def test_the_trail_key_from_the_guide_loads_in_the_export(tmp_path):
    _load_private_key(_generate(tmp_path, "trail_key.pem"))


def test_every_export_command_uses_the_trail_key():
    texts = {
        GUIDE: GUIDE.read_text(encoding="utf-8"),
        CHART / "templates" / "NOTES.txt": (CHART / "templates" / "NOTES.txt").read_text(encoding="utf-8"),
    }
    for path, text in texts.items():
        keys = re.findall(r"vaara trail export\b.*?--key (\S+)", text, flags=re.S)
        assert keys, f"{path.name} prints no trail export command"
        for key in keys:
            assert key.endswith("trail_key.pem"), f"{path.name}: export signs with {key}"


def test_the_chart_hands_the_proxy_the_es256_file():
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert re.search(r"^\s*secretKey:\s*signing_key\.pem\s*$", values, flags=re.M)
