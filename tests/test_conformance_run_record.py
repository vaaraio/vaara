"""A run is provable without being listed.

The Vaara Conformance Results page takes a permanent, public, non-removable
row, and consent to that permanence is enforced by three required tickboxes.
That is right for a page that cannot take rows down. It also means anyone
unwilling to be permanently listed had, until now, exactly one alternative:
an unsigned JSON file that proved nothing.

So the only way to be believed was to be seen, and the people most careful
about permanent public records are the ones most worth hearing from. A run
that fails is even less likely to be submitted than one that passes, which
biases the page toward passes while it claims a failed row is as welcome.

The report now pins what was actually graded and carries its own digest, so a
runner can hand the file to whoever they like and be checked without a page,
without consent, and without the maintainer in the trust path.

Everything here uses the standard library only, because the runner does. It
grades outside implementations and must not require a canonicalisation
library to be installed in order to state its own result.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from conformance_runner import (  # noqa: E402
    CANONICALIZATION,
    RECORD_VERSION,
    build_report,
    canonical_bytes,
    pin_corpus,
    seal,
)


@pytest.fixture(scope="module")
def report():
    """One real suite through the same code path the CLI uses."""
    from conformance_runner import run_suite
    vectors = REPO / "tests" / "vectors"
    rows = [run_suite(vectors, "tap_v0")]
    return build_report(rows, vectors, "2026-01-01T00:00:00Z")


def _recompute(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "record_sha256"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def test_record_digest_recomputes_from_its_own_bytes(report):
    assert report["record_sha256"] == _recompute(report)


def test_record_names_its_own_canonicalisation(report):
    """A verifier must not have to guess how the digest was taken."""
    assert report["canonicalization"] == CANONICALIZATION
    assert report["record_version"] == RECORD_VERSION


def test_editing_the_totals_breaks_the_digest(report):
    tampered = json.loads(json.dumps(report))
    tampered["totals"]["passed"] = 999
    assert _recompute(tampered) != tampered["record_sha256"]


def test_editing_a_suite_verdict_breaks_the_digest(report):
    tampered = json.loads(json.dumps(report))
    tampered["suites"][0]["status"] = "PASS"
    tampered["suites"][0]["suite"] = "not_the_one_that_ran"
    assert _recompute(tampered) != tampered["record_sha256"]


def test_corpus_digest_recomputes_from_the_pinned_suites(report):
    suites = report["corpus"]["suites"]
    assert (
        hashlib.sha256(canonical_bytes(suites)).hexdigest()
        == report["corpus"]["corpus_sha256"]
    )


def test_swapping_a_case_file_digest_breaks_the_corpus_digest(report):
    tampered = json.loads(json.dumps(report))
    suite = next(iter(tampered["corpus"]["suites"]))
    files = tampered["corpus"]["suites"][suite]["files_sha256"]
    files[next(iter(files))] = "0" * 64
    assert (
        hashlib.sha256(canonical_bytes(tampered["corpus"]["suites"])).hexdigest()
        != tampered["corpus"]["corpus_sha256"]
    )


def test_the_checker_itself_is_pinned(report):
    """The checker decides the verdicts, so it has to be in the pin."""
    suite = report["corpus"]["suites"]["tap_v0"]
    assert len(suite["checker_sha256"]) == 64
    real = REPO / "tests" / "vectors" / "tap_v0" / "_check_independent.py"
    assert suite["checker_sha256"] == hashlib.sha256(real.read_bytes()).hexdigest()


def test_the_commit_is_pinned(report):
    """"49 passed" against which bytes is the whole question."""
    assert report["commit"] is None or len(report["commit"]) == 40
    assert isinstance(report["commit_dirty"], bool)


def test_no_absolute_local_paths_leak_into_a_shared_record(report):
    """The record gets handed to strangers. It should not carry a home dir."""
    blob = json.dumps(report)
    assert str(REPO) not in blob
    assert not report["vectors_dir"].startswith("/")


def test_captured_tracebacks_are_scrubbed_of_local_paths():
    """The leak this test found in CI, pinned so it cannot come back.

    On a machine without rfc8785 a checker skips with a full traceback, and a
    traceback names the file it was raised in. That put the runner's
    filesystem into a record meant to be handed to strangers.
    """
    from conformance_runner import _scrub

    vectors = REPO / "tests" / "vectors"
    line = f'File "{vectors}/tap_v0/_check_independent.py", line 42, in <module>'
    out = _scrub(line, vectors)
    assert str(vectors) not in out
    assert str(REPO) not in out
    assert "<vectors>/tap_v0/_check_independent.py" in out
    assert "line 42" in out, "scrubbing must not destroy the diagnostic"


def test_scrub_leaves_text_without_paths_alone():
    from conformance_runner import _scrub
    msg = "optional dependency not installed"
    assert _scrub(msg, REPO / "tests" / "vectors") == msg
    assert _scrub("", REPO / "tests" / "vectors") == ""


def test_canonical_bytes_are_stable_across_key_order():
    a = {"b": 1, "a": {"d": 2, "c": [3, 4]}}
    b = {"a": {"c": [3, 4], "d": 2}, "b": 1}
    assert canonical_bytes(a) == canonical_bytes(b)


def test_canonical_bytes_use_no_incidental_whitespace():
    assert canonical_bytes({"a": 1, "b": 2}) == b'{"a":1,"b":2}'


def test_seal_is_idempotent():
    """Sealing twice must not fold the first digest into the second."""
    rec = {"tool": "x", "totals": {"passed": 1}}
    once = seal(dict(rec))["record_sha256"]
    twice = seal(seal(dict(rec)))["record_sha256"]
    assert once == twice


def test_pin_corpus_skips_pycache_so_the_digest_is_reproducible(tmp_path):
    suite = tmp_path / "demo_v0"
    (suite / "cases").mkdir(parents=True)
    (suite / "_check_independent.py").write_text("# checker\n")
    (suite / "cases" / "one.json").write_text('{"a":1}')
    cache = suite / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00\x01")

    pinned = pin_corpus(tmp_path, ["demo_v0"])
    files = pinned["suites"]["demo_v0"]["files_sha256"]
    assert "cases/one.json" in files
    assert not any("__pycache__" in k for k in files)
