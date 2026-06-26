"""The aggregate conformance runner discovers every suite and grades it.

``scripts/conformance_runner.py`` is the neutral runner over the whole vector
corpus: it invokes each suite's ``_check_independent.py`` and aggregates one
verdict. These tests pin its contract — discovery covers every suite that ships
a checker, an argument-only suite is reported SKIP rather than failed, and the
runner reaches a clean exit on a known-good suite — without re-running all 36
subprocesses on every test invocation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "conformance_runner.py"
VECTORS = REPO / "tests" / "vectors"


def _load_runner():
    spec = importlib.util.spec_from_file_location("conformance_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_discovers_every_suite_with_a_checker() -> None:
    runner = _load_runner()
    found = set(runner.discover(VECTORS))
    on_disk = {p.parent.name for p in VECTORS.glob("*/_check_independent.py")}
    assert found == on_disk
    assert len(found) >= 36


def test_argument_only_suite_is_skipped_not_failed() -> None:
    runner = _load_runner()
    row = runner.run_suite(VECTORS, "article12_fold_v0")
    assert row["status"] == "SKIP"
    assert row["reason"]
    assert row["returncode"] is None


def test_known_good_suite_passes_through_main() -> None:
    runner = _load_runner()
    rc = runner.main(["--corpus", "capability_scope_v0"])
    assert rc == 0


def test_missing_vectors_dir_exits_two() -> None:
    runner = _load_runner()
    rc = runner.main(["--vectors-dir", str(REPO / "does_not_exist")])
    assert rc == 2
