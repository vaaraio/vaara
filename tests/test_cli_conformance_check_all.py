# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""`vaara conformance check --all` runs every published suite.

The site's test-it-yourself page tells a reader: `pip install vaara`, then
`vaara conformance check --all`, and "one command runs every published
test suite and prints a pass or fail per suite". The parser had no `--all`,
so the command printed an argparse error. Now it finds the published tree
(a directory given with `--vectors`, the checkout the package runs from,
or the release archive for the installed version, fetched once and kept)
and runs the neutral runner, `scripts/conformance_runner.py`, over it.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from vaara import __version__
from vaara import _conformance_all as all_suites
from vaara.cli import main

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests" / "vectors"



def test_all_runs_every_published_suite_and_prints_a_verdict_per_suite(capsys):
    rc = main(["conformance", "check", "--all", "--vectors", str(VECTORS)])
    out = capsys.readouterr().out
    assert rc == 0, out
    suites = sorted(p.parent.name for p in VECTORS.glob("*/_check_independent.py"))
    assert len(suites) >= 50
    for suite in suites:
        assert suite in out, suite
    assert "across %d suites" % len(suites) in out
    assert "PASS" in out


def test_all_with_json_prints_the_runners_record(capsys):
    rc = main(["conformance", "check", "--all", "--vectors", str(VECTORS), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    report = json.loads(out)
    assert report["tool"] == "vaara-conformance-runner"
    assert report["all_passed"] is True
    assert report["totals"]["suites"] >= 50
    assert report["record_sha256"]


def test_a_failing_suite_fails_the_run(tmp_path, capsys):
    good = tmp_path / "vectors" / "good_v0"
    bad = tmp_path / "vectors" / "bad_v0"
    for suite, code in ((good, 0), (bad, 1)):
        suite.mkdir(parents=True)
        (suite / "_check_independent.py").write_text(
            f"import sys\nprint('{suite.name} checked')\nsys.exit({code})\n"
        )
    rc = main(["conformance", "check", "--all", "--vectors", str(tmp_path / "vectors")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "PASS  good_v0" in out
    assert "FAIL  bad_v0" in out


def test_all_and_a_path_do_not_mix(capsys):
    with pytest.raises(SystemExit):
        main(["conformance", "check", "--all", str(VECTORS)])
    assert "--all" in capsys.readouterr().err


def test_without_all_a_path_is_required(capsys):
    with pytest.raises(SystemExit):
        main(["conformance", "check"])
    assert "path" in capsys.readouterr().err


# ── Finding the published tree ──────────────────────────────────────────

def test_the_checkout_the_package_runs_from_is_found_first(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(all_suites.TREE_ENV, raising=False)
    assert all_suites.locate_vectors(None) == (VECTORS, "checkout")
    assert all_suites.locate_runner(VECTORS) == ROOT / "scripts" / "conformance_runner.py"


def test_an_explicit_directory_may_be_the_vectors_or_the_tree(tmp_path):
    assert all_suites.locate_vectors(VECTORS) == (VECTORS, "given")
    assert all_suites.locate_vectors(ROOT) == (VECTORS, "given")
    with pytest.raises(all_suites.NoSuitesError):
        all_suites.locate_vectors(tmp_path)


def test_the_environment_variable_names_the_tree(monkeypatch, tmp_path):
    monkeypatch.setenv(all_suites.TREE_ENV, str(ROOT))
    assert all_suites.locate_vectors(None) == (VECTORS, "environment")


def _release_archive(version: str) -> bytes:
    """A tarball shaped like GitHub's archive of the release tag: one top
    directory, the suites and the runner inside it, and a decoy outside."""
    buf = io.BytesIO()
    top = f"vaara-{version}"
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(rel: Path, data: bytes):
            info = tarfile.TarInfo(f"{top}/{rel.as_posix()}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add(Path("scripts/conformance_runner.py"),
            (ROOT / "scripts" / "conformance_runner.py").read_bytes())
        for suite in ("tap_v0", "x402_settlement_v0"):
            for path in sorted((VECTORS / suite).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    add(path.relative_to(ROOT), path.read_bytes())
        add(Path("README.md"), b"not extracted\n")
        info = tarfile.TarInfo("../escape.py")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    return buf.getvalue()


def test_without_a_checkout_the_release_archive_is_fetched_once(monkeypatch, tmp_path, capsys):
    """A reader who ran `pip install vaara` has no checkout. The suites for the
    installed version are fetched from the release archive, only the suites
    and the runner are extracted, the download is kept for the next run, and
    the command says where the bytes came from and what they hashed to."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(all_suites.TREE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(all_suites, "_checkout_above", lambda *_: None)
    archive = _release_archive(__version__)
    fetched: list[str] = []

    def fake_urlopen(request, timeout=0):
        fetched.append(request.full_url)
        return io.BytesIO(archive)

    monkeypatch.setattr(all_suites.urllib.request, "urlopen", fake_urlopen)

    rc = main(["conformance", "check", "--all"])
    out = capsys.readouterr()
    assert rc == 0, out.out + out.err
    assert fetched == [all_suites.release_archive_url(__version__)]
    assert f"v{__version__}" in out.err and "sha256" in out.err
    assert "PASS  tap_v0" in out.out and "PASS  x402_settlement_v0" in out.out
    extracted = tmp_path / "cache" / "vaara" / "conformance" / f"v{__version__}"
    assert (extracted / "tests" / "vectors" / "tap_v0" / "_check_independent.py").is_file()
    assert not (extracted / "README.md").exists()
    assert not (tmp_path / "escape.py").exists() and not (tmp_path / "cache" / "escape.py").exists()

    rc = main(["conformance", "check", "--all"])
    assert rc == 0
    assert len(fetched) == 1, "the kept archive was not reused"


def test_a_failed_download_is_said_plainly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(all_suites.TREE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(all_suites, "_checkout_above", lambda *_: None)

    def refuse(request, timeout=0):
        raise OSError("no route to host")

    monkeypatch.setattr(all_suites.urllib.request, "urlopen", refuse)
    rc = main(["conformance", "check", "--all"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no route to host" in err
    assert "--vectors" in err
