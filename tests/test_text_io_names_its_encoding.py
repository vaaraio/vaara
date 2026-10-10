# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every text read and write in the package names its encoding.

Without one, Python uses the platform's: UTF-8 on Linux and macOS, the ANSI
code page on Windows. An agent config with a non-ASCII path or user name
(``C:\\Users\\Mäki``) then decoded wrong or not at all on Windows, and a
config Vaara wrote back was no longer UTF-8 for the agent that reads it.

Writes also pin ``\\n``: text mode on Windows writes ``\\r\\n``, so the trail
export and every receipt file came out as different bytes there.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vaara"


def test_text_io_names_its_encoding_and_line_ending():
    missing = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("read_text", "write_text")
                    and "encoding" not in {k.arg for k in node.keywords}):
                missing.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text"
                    and "newline" not in {k.arg for k in node.keywords}):
                missing.append(f"{path.relative_to(SRC.parent)}:{node.lineno} (newline)")
    assert not missing, "text I/O without an encoding:\n" + "\n".join(missing)
