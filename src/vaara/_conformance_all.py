# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run every published conformance suite from one command.

``vaara conformance check --all`` is the command the site hands a reader
after ``pip install vaara``. The suites live in the repository under
``tests/vectors/<suite>/``, each with an independent checker that imports
no Vaara code, and ``scripts/conformance_runner.py`` runs them all and
prints a verdict per suite. Neither is in the wheel, so this module finds
them: a directory named on the command line, the directory
``VAARA_CONFORMANCE_TREE`` names, the checkout this package runs from, or
the release archive for the installed version, fetched once from GitHub
and kept under the user's cache directory. The command says which it used,
and for a download, the URL and the SHA-256 of the bytes.

The runner stays the neutral party: this module locates it and starts it,
and the verdicts are the checkers' own.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

from vaara import __version__

#: A directory of suites, or a tree holding ``tests/vectors`` and the runner.
TREE_ENV = "VAARA_CONFORMANCE_TREE"
RUNNER = Path("scripts") / "conformance_runner.py"
VECTORS = Path("tests") / "vectors"
CHECKER = "_check_independent.py"
_ARCHIVE = "https://github.com/vaaraio/vaara/archive/refs/tags/v{version}.tar.gz"
_MAX_ARCHIVE_BYTES = 64 << 20


class NoSuitesError(Exception):
    """No published suites could be found or fetched; the message says why
    and what the reader can pass instead."""


def release_archive_url(version: str) -> str:
    return _ARCHIVE.format(version=version)


def _has_suites(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() for p in path.glob(f"*/{CHECKER}"))


def _is_tree(path: Path) -> bool:
    return (path / RUNNER).is_file() and _has_suites(path / VECTORS)


def _checkout_above(start: Path) -> Optional[Path]:
    """The repository tree containing ``start``, or None."""
    for candidate in (start, *start.parents):
        if _is_tree(candidate):
            return candidate
    return None


def _vectors_in(path: Path) -> Optional[Path]:
    """``path`` as a directory of suites, or the suites of the tree it is."""
    if _has_suites(path):
        return path
    if _has_suites(path / VECTORS):
        return path / VECTORS
    return None


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "vaara" / "conformance"


def fetch_release_tree(version: str = __version__, *, log=None) -> Path:
    """The suites and the runner of release ``version``, downloaded once.

    Only ``tests/vectors/`` and ``scripts/conformance_runner.py`` leave the
    archive, as regular files under the cache directory; nothing else in
    the tarball is written anywhere.
    """
    target = cache_dir() / f"v{version}"
    if _is_tree(target):
        return target
    url = release_archive_url(version)
    say = log or (lambda *_: None)
    say(f"no checkout found; fetching the published suites for v{version} from {url}")
    request = urllib.request.Request(url, headers={"user-agent": f"vaara/{version}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            data = response.read(_MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise NoSuitesError(
            f"could not fetch {url}: {exc}. Pass --vectors DIR with a clone's "
            f"tests/vectors, or set {TREE_ENV}."
        ) from exc
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise NoSuitesError(f"{url} is larger than {_MAX_ARCHIVE_BYTES} bytes; refusing")
    say(f"fetched {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    staging = target.with_name(target.name + ".partial")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar:
                parts = Path(member.name).parts
                # GitHub's archive has one top directory; what follows is the
                # repository path.
                if len(parts) < 2 or not member.isfile():
                    continue
                rel = Path(*parts[1:])
                if ".." in rel.parts or rel.is_absolute():
                    continue
                if rel != RUNNER and not rel.is_relative_to(VECTORS):
                    continue
                payload = tar.extractfile(member)
                if payload is None:
                    continue
                out = staging / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(payload.read())
    except (tarfile.TarError, OSError) as exc:
        raise NoSuitesError(f"the archive at {url} could not be read: {exc}") from exc
    if not _is_tree(staging):
        raise NoSuitesError(f"the archive at {url} carries no conformance suites")
    if target.exists():
        import shutil

        shutil.rmtree(target)
    staging.rename(target)
    return target


def locate_vectors(given: Optional[Path], *, log=None) -> tuple[Path, str]:
    """The directory of suites to run and where it came from: ``given``,
    ``environment``, ``checkout`` or ``release``."""
    if given is not None:
        found = _vectors_in(Path(given).expanduser().resolve())
        if found is None:
            raise NoSuitesError(
                f"{given} holds no conformance suites (no */{CHECKER}) and is "
                f"not a tree with {VECTORS.as_posix()}"
            )
        return found, "given"
    named = os.environ.get(TREE_ENV)
    if named:
        found = _vectors_in(Path(named).expanduser().resolve())
        if found is None:
            raise NoSuitesError(f"{TREE_ENV}={named} holds no conformance suites")
        return found, "environment"
    checkout = _checkout_above(Path(__file__).resolve().parent) or _checkout_above(Path.cwd())
    if checkout is not None:
        return checkout / VECTORS, "checkout"
    return fetch_release_tree(log=log) / VECTORS, "release"


def locate_runner(vectors: Path, *, log=None) -> Path:
    """The runner beside the suites, else the checkout's, else the release's."""
    beside = vectors.parent.parent / RUNNER
    if beside.is_file():
        return beside
    checkout = _checkout_above(Path(__file__).resolve().parent)
    if checkout is not None:
        return checkout / RUNNER
    return fetch_release_tree(log=log) / RUNNER


def run_all(vectors: Path, runner: Path, *, as_json: bool, suites: Optional[list[str]] = None,
            stdout=None) -> int:
    """Start the runner over ``vectors`` and return its exit code: 0 when
    every suite that ran passed, 1 on a failure, 2 when nothing could run."""
    out = stdout or sys.stdout
    argv = [sys.executable, str(runner), "--vectors-dir", str(vectors)]
    for suite in suites or []:
        argv += ["--corpus", suite]
    if as_json:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            done = subprocess.run(argv + ["--json", str(report), "--no-submit-link"],
                                  capture_output=True, text=True)
            if report.is_file():
                out.write(report.read_text(encoding="utf-8"))
                out.flush()
            else:
                sys.stderr.write(done.stdout + done.stderr)
            return done.returncode
    done = subprocess.run(argv, capture_output=True, text=True)
    out.write(done.stdout)
    out.flush()
    if done.stderr:
        sys.stderr.write(done.stderr)
    return done.returncode
