"""Build the .mcpb bundle for one-click desktop install.

An MCP Bundle is a zip holding a manifest.json and whatever the server needs.
A desktop host opens it and shows an install dialog, so a user gets a governed
MCP server without a terminal, a virtualenv, or hand-edited JSON config.

WHY THE MANIFEST IS GENERATED RATHER THAN COMMITTED. It carries the version
number, and a version number written by hand is a version number that goes
stale. Every field below is derived: the version from the installed package
metadata, the description and the entry point from the same values already
published in the MCP registry server.json. One source, one release, no drift.

WHAT IS DELIBERATELY NOT IN THE BUNDLE. No vendored dependencies. The server
runs through `uvx --from vaara==<version> vaara-mcp-server`, so uv resolves the
pinned wheel from PyPI at install time and the archive stays a few kilobytes.
Vendoring would mean shipping a second copy of the package that can disagree
with the one on PyPI, and disagreeing copies of an audit tool is the exact
failure the product exists to prevent.

Build:  python packaging/mcpb/build.py [output-dir]
Output: <output-dir>/vaara-<version>.mcpb, defaulting to dist/

The output directory is overridable because the release workflow must NOT put
this next to the wheel. `dist/` is uploaded wholesale to PyPI, which rejects an
unknown archive type and would fail the publish for every release.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "dist"

#: Kept identical to the `description` published for io.github.vaaraio/vaara-server
#: in the MCP registry. Two descriptions for one server is how a listing starts
#: telling a different story from the package.
DESCRIPTION = (
    "Accountable autonomy MCP server: gating, tamper-evident audit for every action"
)

LONG_DESCRIPTION = (
    "Gates every AI agent tool call against your policy before it runs, then "
    "writes a hash-chained, tamper-evident record of what was decided and what "
    "happened. Anyone can verify that record offline, without your software, "
    "your systems, or any key you hold. Your environment, no SaaS, no telemetry."
)


#: `__version__ = "1.83.0"` in the package's `__init__.py`.
_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)


def version() -> str:
    """The version, read from the package rather than typed here.

    Read from `src/vaara/__init__.py` and NOT from pyproject, deliberately.
    `tomllib` arrived in Python 3.11 and this project supports 3.10, so parsing
    pyproject here would make the build script fail on the oldest interpreter
    the package itself claims to support. CI caught exactly that on the first
    run of this file.

    The version is declared in two places by release convention, pyproject and
    here. A test asserts the two agree, on the interpreters that can read TOML.
    """
    src = (ROOT / "src" / "vaara" / "__init__.py").read_text(encoding="utf-8")
    match = _VERSION_RE.search(src)
    if not match:
        raise RuntimeError("no __version__ in src/vaara/__init__.py")
    return match.group(1)


def manifest(ver: str) -> dict:
    return {
        "manifest_version": "0.4",
        "name": "vaara",
        "display_name": "Vaara",
        "version": ver,
        "description": DESCRIPTION,
        "long_description": LONG_DESCRIPTION,
        "author": {
            "name": "Henri Sirkkavaara",
            "email": "hello@vaara.io",
            "url": "https://vaara.io",
        },
        "homepage": "https://vaara.io",
        "documentation": "https://github.com/vaaraio/vaara#readme",
        "support": "https://github.com/vaaraio/vaara/issues",
        "repository": {
            "type": "git",
            "url": "https://github.com/vaaraio/vaara",
        },
        "license": "AGPL-3.0-or-later",
        "icon": "icon.png",
        "keywords": [
            "audit-trail",
            "ai-governance",
            "eu-ai-act",
            "attestation",
            "compliance",
            "agent-security",
        ],
        "server": {
            "type": "uv",
            "entry_point": "vaara-mcp-server",
            "mcp_config": {
                "command": "uvx",
                # Pinned to the bundle's own version. An unpinned bundle would
                # install whatever is newest, which makes the version printed on
                # the install dialog a guess rather than a fact.
                "args": ["--from", f"vaara=={ver}", "vaara-mcp-server"],
                "env": {
                    "VAARA_DB": "${user_config.db_path}",
                    "VAARA_API_KEY": "${user_config.api_key}",
                },
            },
        },
        # Both optional, matching the environmentVariables already declared in
        # the registry entry. Neither is required for a working single-user
        # install, which is the whole point of a one-click bundle.
        "user_config": {
            "db_path": {
                "type": "string",
                "title": "Audit database path",
                "description": (
                    "Where the hash-chained audit trail is written. Leave blank "
                    "for vaara_audit.db in the working directory."
                ),
                "required": False,
            },
            "api_key": {
                "type": "string",
                "title": "Shared secret (optional)",
                "description": (
                    "When set, every tool call must carry _api_key in its "
                    "arguments. Leave blank for a single-user install, where "
                    "process isolation is already the boundary."
                ),
                "sensitive": True,
                "required": False,
            },
        },
        "compatibility": {
            "runtimes": {"python": ">=3.10"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]).resolve() if argv else OUT
    out.mkdir(parents=True, exist_ok=True)

    ver = version()
    stage = out / "mcpb-stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    (stage / "manifest.json").write_text(
        json.dumps(manifest(ver), indent=2) + "\n", encoding="utf-8"
    )

    # The dialog shows this next to the name, at a recommended 512 square.
    # `webpage/favicon-180.png` is 180 and upscaling it looked like upscaling.
    # `icon.png` here is the same geometry as `webpage/favicon.svg`, rendered
    # once at 512: rounded rect #1A2226 with a #7B9C8A edge, #78A08A triangle.
    # Redraw it from the SVG if the mark ever changes.
    icon_src = HERE / "icon.png"
    if icon_src.is_file():
        shutil.copy2(icon_src, stage / "icon.png")
    else:
        print(f"[mcpb] WARNING: no icon at {icon_src}, bundling without one")

    for name in ("LICENSE", "README.md"):
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, stage / name)

    print(f"[mcpb] staged v{ver} in {stage}")

    packed = out / f"vaara-{ver}.mcpb"
    if packed.exists():
        packed.unlink()

    try:
        subprocess.run(
            ["npx", "--yes", "@anthropic-ai/mcpb", "pack", str(stage), str(packed)],
            check=True,
            cwd=ROOT,
        )
    except FileNotFoundError:
        print("[mcpb] npx not found. Install Node, or zip the stage directory.")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[mcpb] pack failed with exit {exc.returncode}")
        return exc.returncode

    print(f"[mcpb] wrote {packed} ({packed.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
