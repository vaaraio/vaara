# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every flag the docs tell a reader to type has to exist.

docs/COMPLIANCE.md is a compliance document, and its copy-paste block for
producing a signed regulator handoff called ``vaara keygen --out-dir`` (the
flag is ``--out``, and keygen refuses to run at all without ``--dev``) and
``vaara trail verify --public-key`` (the flag is ``--pubkey``). Three of the
four commands in that block failed as printed.

The check walks the real argparse tree of every installed console script, so
it tracks whatever the CLI actually accepts rather than a second list that
can drift the same way the docs did.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Console scripts declared in pyproject's [project.scripts].
CONSOLE_SCRIPTS = ("vaara", "vaara-audit", "vaara-mcp-proxy", "vaara-mcp-server")

#: Modules with a __main__ CLI that is not a console script.
MODULE_CLIS = (
    "vaara.integrations.infer_proxy",
    "vaara.integrations.infer_console",
    "vaara.integrations.llm_proxy",
)

#: Flags the docs mention that belong to something other than a Vaara CLI,
#: or that appear inside a sentence saying they do NOT exist.
NOT_OURS = {
    "--vectors-dir",  # scripts/conformance_runner.py
    "--list",         # scripts/conformance_runner.py
    "--skip-invalid",  # article12-export-spec.md says there is no such flag
    # helm and kubectl, in docs/kubernetes-rancher.md. The deployment guide
    # has to show the commands that install the chart, and those belong to
    # other tools. Keep this list to flags a reader types into helm/kubectl,
    # never a Vaara flag that "should" exist.
    "--create-namespace",  # helm install
    "--namespace",         # helm, kubectl
    "--from-file",         # kubectl create secret generic
}


def _help(argv: list[str]) -> str:
    try:
        done = subprocess.run(
            argv + ["--help"], capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout + done.stderr


def _walk(argv: list[str], depth: int = 0, seen: set | None = None):
    seen = set() if seen is None else seen
    text = _help(argv)
    yield text
    if depth >= 2 or not text:
        return
    for sub in set(re.findall(r"^\s{2,6}([a-z][a-z0-9-]*)\s{2,}\S", text, re.M)):
        key = tuple(argv + [sub])
        if key in seen:
            continue
        seen.add(key)
        yield from _walk(argv + [sub], depth + 1, seen)


def _script(name: str) -> str | None:
    """The console script belonging to THIS interpreter, not whatever is on PATH.

    A second Vaara on PATH answers --help from a different install, which is
    how a check like this quietly starts testing the wrong build.
    """
    import sys

    beside = Path(sys.executable).parent / name
    if beside.exists():
        return str(beside)
    return shutil.which(name)


def _known_flags() -> set[str]:
    import sys

    flags: set[str] = set()
    for script in CONSOLE_SCRIPTS:
        path = _script(script)
        if path:
            for text in _walk([path]):
                flags |= set(re.findall(r"(--[a-z][\w-]+)", text))
    for module in MODULE_CLIS:
        for text in _walk([sys.executable, "-m", module]):
            flags |= set(re.findall(r"(--[a-z][\w-]+)", text))
    return flags


def _cited_flags() -> dict[str, str]:
    cited: dict[str, str] = {}
    for doc in sorted(ROOT.glob("docs/**/*.md")) + [ROOT / "README.md"]:
        for flag in re.findall(r"(--[a-z][\w-]{2,})", doc.read_text()):
            cited.setdefault(flag, str(doc.relative_to(ROOT)))
    return cited


@pytest.fixture(scope="module")
def known() -> set[str]:
    flags = _known_flags()
    if len(flags) < 50:
        pytest.skip("vaara console scripts are not installed in this environment")
    return flags


def test_docs_cite_flags_that_exist(known):
    unknown = {
        flag: doc
        for flag, doc in _cited_flags().items()
        if flag not in known and flag not in NOT_OURS
    }
    assert not unknown, (
        "docs mention flags no Vaara CLI accepts: "
        + ", ".join(f"{flag} ({doc})" for flag, doc in sorted(unknown.items()))
    )


def test_the_compliance_handoff_block_uses_real_flags(known):
    """The block a regulator-facing reader copies verbatim."""
    block = (ROOT / "docs" / "COMPLIANCE.md").read_text()
    assert "vaara keygen --dev --out " in block
    assert "--out-dir" not in block
    assert "--public-key" not in block
    assert "--pubkey" in block


def test_keygen_without_dev_produces_no_key(known, tmp_path):
    """Documenting keygen without --dev would print a refusal, not a key."""
    path = _script("vaara")
    target = tmp_path / "signing_key.pem"
    done = subprocess.run(
        [path, "keygen", "--out", str(target)],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode != 0
    assert not target.exists()
