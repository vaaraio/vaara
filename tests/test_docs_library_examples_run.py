# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every Python example in the library documents runs.

docs/library.md is the door to the capabilities that have no command of
their own: a reader gets there from capabilities.md, copies a block, and
it has to work. docs/adapters.md, docs/northern-lights.md and
docs/COMPLIANCE.md show the framework, guardrail, routing and article
override doors the same way. A block that cannot
run is a door painted on a wall, which is what ``ExternalScorer`` and
``CompositeScorer`` were for five months.

Each block runs on its own, in a throwaway home directory, with no
network: the examples that reach out (a remote scorer, a public log) have
to show what the reader sees when the peer is not there, and an example
that silently needed a live service would pass here and fail for the
reader. A block that imports an optional package the environment lacks is
skipped by name, after it has been compiled, so a syntax error never hides
behind a missing extra; a missing ``vaara`` module is a failure.
"""

from __future__ import annotations

import os
import re
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("docs/library.md", "docs/adapters.md", "docs/northern-lights.md", "docs/COMPLIANCE.md")

_FENCE = re.compile(r"^(#{1,6})[ \t]+([^\n]*)$|```python\n(.*?)```", re.M | re.S)


def _blocks(doc: Path) -> list[tuple[str, str]]:
    """(heading the block sits under, code) for every ```python fence."""
    out, heading = [], "(top)"
    for match in _FENCE.finditer(doc.read_text(encoding="utf-8")):
        if match.group(1):
            heading = match.group(2).strip()
        else:
            out.append((heading, match.group(3)))
    return out


def _cases():
    for rel in DOCS:
        doc = ROOT / rel
        seen: dict[str, int] = {}
        for heading, code in _blocks(doc):
            seen[heading] = seen.get(heading, 0) + 1
            suffix = f" #{seen[heading]}" if seen[heading] > 1 else ""
            yield pytest.param(rel, heading, code, id=f"{rel}:{heading}{suffix}")


def _loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else str(address)
    return host in ("127.0.0.1", "::1", "localhost")


@pytest.fixture()
def no_network(monkeypatch):
    """Refuse every connection and every name lookup that leaves the machine."""
    real_connect, real_lookup = socket.socket.connect, socket.getaddrinfo

    def connect(self, address):
        if not _loopback(address):
            raise OSError(f"documented examples run without a network: {address!r}")
        return real_connect(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host in ("127.0.0.1", "::1", "localhost", None, ""):
            return real_lookup(host, *args, **kwargs)
        raise socket.gaierror(f"documented examples run without a network: {host!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize("rel,heading,code", list(_cases()))
def test_documented_block_runs(rel, heading, code, tmp_path, monkeypatch, no_network):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    for var in [v for v in os.environ if v.startswith("VAARA_")]:
        monkeypatch.delenv(var)

    compiled = compile(code, f"{rel}#{heading}", "exec")  # syntax holds even if skipped
    namespace = {"__name__": "__doc_example__"}
    try:
        exec(compiled, namespace)  # noqa: S102
    except ModuleNotFoundError as exc:
        top = (exc.name or "").split(".")[0]
        if top == "vaara" or not top:
            raise
        pytest.skip(f"{rel} under {heading!r} needs {exc.name}, not installed here")


def test_every_library_document_has_python_blocks():
    for rel in DOCS:
        assert _blocks(ROOT / rel), f"{rel} shows no Python; the doors it documents have no example"
