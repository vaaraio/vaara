"""The .mcpb manifest must stay in step with the package it installs.

An MCP Bundle is a zip a desktop host opens to install a local server in one
click. Its manifest carries a version and a pinned package spec, and a manifest
that drifts from pyproject installs a different Vaara from the one the dialog
names. These tests exist so that drift fails in CI rather than on a user's
machine.

Nothing here calls npx or writes an archive. The build script's packing step
needs Node, and a test that needs Node to check a JSON document would be a test
that gets skipped.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "packaging" / "mcpb" / "build.py"


def _pyproject() -> dict:
    """Parsed pyproject, or skip.

    `tomllib` is 3.11+ and this project supports 3.10, so the tests that need
    to read TOML skip on the oldest interpreter rather than failing collection
    for the whole module. Importing it at module scope broke the 3.10 CI leg.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is Python 3.11+")
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _build_module():
    spec = importlib.util.spec_from_file_location("mcpb_build", BUILD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mcpb():
    return _build_module()


@pytest.fixture(scope="module")
def manifest(mcpb):
    return mcpb.manifest(mcpb.version())


def test_version_comes_from_the_package(mcpb):
    """One source of truth. A hand-typed version is a version that goes stale."""
    import vaara

    assert mcpb.version() == vaara.__version__


def test_the_two_declared_versions_agree(mcpb):
    """Release convention puts the version in pyproject AND in __init__.py.

    The build script reads the second, because tomllib is 3.11+ and this
    package supports 3.10. That only stays safe while the two agree, so this
    checks it on every interpreter that can read TOML.
    """
    assert mcpb.version() == _pyproject()["project"]["version"]


def test_required_manifest_fields_present(manifest):
    for field in (
        "manifest_version",
        "name",
        "version",
        "description",
        "author",
        "server",
    ):
        assert manifest.get(field), f"required manifest field missing: {field}"
    assert manifest["author"].get("name")


def test_pinned_spec_matches_the_manifest_version(manifest):
    """The dialog shows a version. It has to be the version that installs.

    An unpinned `--from vaara` would resolve to whatever is newest at install
    time, which makes the number on the dialog a guess.
    """
    args = manifest["server"]["mcp_config"]["args"]
    assert f"vaara=={manifest['version']}" in args
    assert "vaara-mcp-server" in args


def test_entry_point_exists_in_pyproject(manifest):
    """The bundle must launch a console script this package actually installs."""
    scripts = _pyproject()["project"]["scripts"]
    assert manifest["server"]["entry_point"] in scripts


def test_licence_is_stated_and_matches_the_project(manifest):
    """Six of ten third-party directories show no licence for Vaara, because
    the MCP registry schema has no licence field. The bundle has one, so it
    gets used, and it has to agree with pyproject."""
    assert "AGPL-3.0" in manifest["license"]
    declared = _pyproject()["project"]["license"]
    text = declared if isinstance(declared, str) else declared.get("text", "")
    assert "AGPL-3.0" in text


def test_no_user_config_is_required(manifest):
    """A one-click install that demands configuration is not one click.

    Both options map to environment variables the server already treats as
    optional, so the default install must work with every field blank.
    """
    for name, opt in manifest["user_config"].items():
        assert opt.get("required") is False, f"{name} must not be required"


def test_secret_field_is_marked_sensitive(manifest):
    assert manifest["user_config"]["api_key"]["sensitive"] is True


def test_description_matches_the_published_registry_entry(mcpb):
    """Two descriptions for one server is how a listing starts telling a
    different story from the package."""
    assert mcpb.DESCRIPTION == (
        "Accountable autonomy MCP server: gating, tamper-evident audit "
        "for every action"
    )


def test_icon_is_present_and_square_at_512(mcpb):
    pillow = pytest.importorskip("PIL.Image")
    icon = Path(mcpb.HERE) / "icon.png"
    assert icon.is_file(), "bundle icon missing"
    with pillow.open(icon) as im:
        assert im.size == (512, 512), f"icon is {im.size}, host wants 512 square"


def test_manifest_is_json_serialisable(manifest):
    json.dumps(manifest)
