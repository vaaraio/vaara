"""Both trail verifiers refuse a repeated JSON member and a non-finite number.

A trail line that repeats a member (`"tool_name":"FIRST","tool_name":"T"`)
reads as T to a last-wins parser and as FIRST to a first-wins one. The hash is
over T, so the verifier printed `Verification: OK` while a second reader
holding the same file saw a different call. A bare `NaN`, `Infinity` or
`1e999` in `data` ended in an uncaught ValueError from the canonical re-encode,
with no FAILED listing. `scripts/verify_vaara_trail.py` and
`vaara.audit.verify` now refuse both on the trail lines and the manifest, as
`record N: duplicate member "x"` and `record N: non-finite number NaN`, so
"an independent party reproduces every verdict offline" keeps holding for a
file only the signer could have written.

A trail line or manifest that parses but is not a JSON object (`[1]`, `7`,
`"x"`, `null`) used to end in an AttributeError at the first `.get`, with no
FAILED listing. Both verifiers now fail the run as `record N: not a JSON
object` and `manifest.json: not a JSON object`, and read a special record
whose `data` is not an object as they read empty data.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from vaara.audit import verify as lib  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_vaara_trail.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_vaara_trail_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


# --- a synthetic signed export whose lines and manifest can be made hostile -------


def _hash(content: dict) -> str:
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _records(count: int, data: dict[int, object] | None = None,
             event_types: dict[int, str] | None = None) -> list[dict]:
    """A valid chain of `count` records; `data` and `event_types` set a record's
    data and event type by index."""
    out: list[dict] = []
    prev = ""
    for i in range(count):
        content = {
            "record_id": f"r{i}", "action_id": f"a{i}",
            "event_type": (event_types or {}).get(i, "x"),
            "timestamp": 1760000000.0 + i, "agent_id": "ag", "tool_name": "T",
            "data": (data or {}).get(i, {}), "regulatory_articles": [],
            "previous_hash": prev,
        }
        rec = {**content, "record_hash": _hash(content)}
        out.append(rec)
        prev = rec["record_hash"]
    return out


def _text(rec: dict) -> str:
    return json.dumps(rec, separators=(",", ":"))


def _export(tmp_path: Path, records: list[dict], *, mutate: dict[int, object] | None = None,
            mutate_manifest=None) -> Path:
    """Sign `records`, first rewriting the text of line i with mutate[i], and
    the manifest text with mutate_manifest. The hashes stay those of the
    last-wins reading, so the unmodified verifier calls the export clean."""
    texts = [_text(r) for r in records]
    for i, fn in (mutate or {}).items():
        texts[i] = fn(texts[i])
    trail = ("\n".join(texts) + "\n").encode()
    manifest_text = json.dumps({
        "schema_version": "1", "trail_sha256": hashlib.sha256(trail).hexdigest(),
        "record_count": len(records), "first_hash": records[0]["record_hash"],
        "last_hash": records[-1]["record_hash"], "signature_algorithm": "Ed25519",
    }, separators=(",", ":"))
    if mutate_manifest is not None:
        manifest_text = mutate_manifest(manifest_text)
    manifest = manifest_text.encode()
    key = Ed25519PrivateKey.generate()
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("trail.jsonl", trail)
        zf.writestr("manifest.json", manifest)
        zf.writestr("trail.sig", key.sign(hashlib.sha256(trail + manifest).digest()))
        zf.writestr("signer_pubkey.pem", pem)
    return path


def _run(zip_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(zip_path)],
                          capture_output=True, text=True)


# --- the hostile lines: (record 1's data, how its text is rewritten, the refusal) ---


def _swap(old: str, new: str):
    return lambda text: text.replace(old, new)


_LINES = {
    "repeated top-level member": (
        {}, _swap('"tool_name":"T"', '"tool_name":"FIRST","tool_name":"T"'),
        'record 1: duplicate member "tool_name"'),
    "repeated member inside data": (
        {"a": 2}, _swap('"a":2', '"a":1,"a":2'),
        'record 1: duplicate member "a"'),
    "NaN": (
        {}, _swap('"data":{}', '"data":{"score":NaN}'),
        "record 1: non-finite number NaN"),
    "Infinity": (
        {}, _swap('"data":{}', '"data":{"score":Infinity}'),
        "record 1: non-finite number Infinity"),
    "-Infinity": (
        {}, _swap('"data":{}', '"data":{"score":-Infinity}'),
        "record 1: non-finite number -Infinity"),
    "a literal that overflows to infinity": (
        {}, _swap('"data":{}', '"data":{"score":1e999}'),
        "record 1: non-finite number 1e999"),
}


def _hostile_export(tmp_path: Path, case: str) -> tuple[Path, str]:
    data, mutate, refusal = _LINES[case]
    records = _records(3, {1: data})
    return _export(tmp_path, records, mutate={1: mutate}), refusal


@pytest.mark.parametrize("case", list(_LINES))
def test_the_script_fails_the_export_and_names_the_record(tmp_path: Path, case: str):
    zp, refusal = _hostile_export(tmp_path, case)

    run = _run(zp)

    assert run.returncode == 1, run.stdout
    assert "Verification: FAILED" in run.stderr
    assert f"hash chain check failed: {refusal}" in run.stderr
    assert "Traceback" not in run.stderr
    assert "Verification: OK" not in run.stdout


@pytest.mark.parametrize("case", list(_LINES))
def test_the_library_gives_the_same_verdict_as_the_script(tmp_path: Path, case: str):
    zp, refusal = _hostile_export(tmp_path, case)

    result = lib.verify_signed(zp)
    ok, script_errors, _ = script.verify(zp, None)

    assert not result.ok and not ok
    assert result.errors == [f"hash chain check failed: {refusal}"]
    assert script_errors == result.errors


def test_a_hostile_last_line_fails_without_a_traceback(tmp_path: Path):
    records = _records(2)
    zp = _export(tmp_path, records, mutate={1: _swap('"data":{}', '"data":{"x":NaN}')})

    run = _run(zp)

    assert run.returncode == 1
    assert "hash chain check failed: record 1: non-finite number NaN" in run.stderr
    assert "Traceback" not in run.stderr
    assert any("record 1: non-finite number NaN" in e for e in lib.verify_signed(zp).errors)


@pytest.mark.parametrize("fn,refusal", [
    (_swap('"record_count":3', '"record_count":9,"record_count":3'),
     'manifest.json: duplicate member "record_count"'),
    (_swap('"record_count":3', '"record_count":NaN'),
     "manifest.json: non-finite number NaN"),
])
def test_both_verifiers_refuse_a_hostile_manifest(tmp_path: Path, fn, refusal: str):
    zp = _export(tmp_path, _records(3), mutate_manifest=fn)

    run = _run(zp)
    result = lib.verify_signed(zp)
    ok, script_errors, _ = script.verify(zp, None)

    assert run.returncode == 1
    assert refusal in run.stderr
    assert "Traceback" not in run.stderr
    assert not result.ok and not ok
    assert result.errors == [refusal]
    assert script_errors == [refusal]


# --- JSON that parses but is not an object ---------------------------------------

_NOT_OBJECTS = {
    "an array": "[1]",
    "an empty array": "[]",
    "a number": "7",
    "a string": '"x"',
    "null": "null",
    "a boolean": "true",
}


def _not_an_object_line(tmp_path: Path, text: str, index: int) -> Path:
    return _export(tmp_path, _records(3), mutate={index: lambda _old: text})


@pytest.mark.parametrize("text", list(_NOT_OBJECTS.values()), ids=list(_NOT_OBJECTS))
def test_the_script_fails_a_trail_line_that_is_not_an_object(tmp_path: Path, text: str):
    run = _run(_not_an_object_line(tmp_path, text, 1))

    assert run.returncode == 1, run.stdout
    assert "Verification: FAILED" in run.stderr
    assert "hash chain check failed: record 1: not a JSON object" in run.stderr
    assert "Traceback" not in run.stderr
    assert "Verification: OK" not in run.stdout


@pytest.mark.parametrize("index", [0, 1, 2], ids=["first", "middle", "last"])
@pytest.mark.parametrize("text", ["[1]", "null"], ids=["an array", "null"])
def test_both_verifiers_fail_a_non_object_line_wherever_it_sits(
        tmp_path: Path, text: str, index: int):
    zp = _not_an_object_line(tmp_path, text, index)
    refusal = f"hash chain check failed: record {index}: not a JSON object"

    run = _run(zp)
    result = lib.verify_signed(zp)
    ok, script_errors, _ = script.verify(zp, None)

    assert run.returncode == 1
    assert "Traceback" not in run.stderr
    assert not result.ok and not ok
    assert refusal in result.errors
    assert refusal in script_errors


@pytest.mark.parametrize("text", list(_NOT_OBJECTS.values()), ids=list(_NOT_OBJECTS))
def test_a_non_object_line_does_not_stop_the_other_readers(tmp_path: Path, text: str):
    zp = _not_an_object_line(tmp_path, text, 1)

    run = _run(zp)

    assert run.returncode == 1
    assert "Traceback" not in run.stderr
    assert [e["index"] for e in script._collect_erasures(zp, None)[1]] == []
    assert script._collect_key_lifecycle(zp) == []
    assert script._collect_cage_lifecycle(zp) == []


@pytest.mark.parametrize("text", list(_NOT_OBJECTS.values()), ids=list(_NOT_OBJECTS))
def test_both_verifiers_refuse_a_manifest_that_is_not_an_object(tmp_path: Path, text: str):
    refusal = "manifest.json: not a JSON object"
    zp = _export(tmp_path, _records(3), mutate_manifest=lambda _old: text)

    run = _run(zp)
    result = lib.verify_signed(zp)
    ok, script_errors, manifest = script.verify(zp, None)

    assert run.returncode == 1
    assert refusal in run.stderr
    assert "Traceback" not in run.stderr
    assert not result.ok and not ok
    assert result.errors == [refusal]
    assert script_errors == [refusal]
    assert result.manifest is None and manifest is None


@pytest.mark.parametrize("value", ["1", "[1]", "null", "true", '{"a":1}'],
                         ids=["a number", "an array", "null", "a boolean", "an object"])
def test_both_verifiers_fail_a_signature_algorithm_that_is_not_a_string(
        tmp_path: Path, value: str):
    refusal = f"unknown signature_algorithm: {json.loads(value)!r}"
    zp = _export(tmp_path, _records(3), mutate_manifest=_swap(
        '"signature_algorithm":"Ed25519"', f'"signature_algorithm":{value}'))

    run = _run(zp)
    result = lib.verify_signed(zp)
    ok, script_errors, _ = script.verify(zp, None)

    assert run.returncode == 1
    assert "Traceback" not in run.stderr
    assert not result.ok and not ok
    assert refusal in result.errors
    assert refusal in script_errors


@pytest.mark.parametrize("value", ["1", '"x"', "[1]", "true"],
                         ids=["a number", "a string", "an array", "a boolean"])
def test_the_script_fails_a_revocation_block_that_is_not_an_object(tmp_path: Path, value: str):
    refusal = "manifest revocation block is not an object"
    zp = _export(tmp_path, _records(3), mutate_manifest=_swap(
        '"signature_algorithm":"Ed25519"',
        f'"signature_algorithm":"Ed25519","revocation":{value}'))
    with zipfile.ZipFile(zp, "a") as zf:
        zf.writestr("revocation.json", b"{}")

    run = _run(zp)
    ok, script_errors, _ = script.verify(zp, None)

    assert run.returncode == 1
    assert "Traceback" not in run.stderr
    assert not ok
    assert refusal in script_errors
    lib.verify_signed(zp)  # must not raise


@pytest.mark.parametrize("bad", ['"x"', "[1]", "7"], ids=["a string", "an array", "a number"])
@pytest.mark.parametrize("event_type", ["record_erased", "key_lifecycle", "cage_lifecycle"])
def test_a_special_record_whose_data_is_not_an_object_reads_like_empty_data(
        tmp_path: Path, event_type: str, bad: str):
    hostile = _export(tmp_path, _records(3, {1: json.loads(bad)}, {1: event_type}))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty = _export(empty_dir, _records(3, {1: {}}, {1: event_type}))

    run = _run(hostile)
    result = lib.verify_signed(hostile)
    ok, script_errors, _ = script.verify(hostile, None)

    assert "Traceback" not in run.stderr
    assert run.returncode == (0 if lib.verify_signed(empty).ok else 1)
    assert result.errors == lib.verify_signed(empty).errors
    assert script_errors == script.verify(empty, None)[1] == result.errors
    assert ok == result.ok


# --- what must not change --------------------------------------------------------


def test_a_clean_export_still_verifies_on_both(tmp_path: Path):
    zp = _export(tmp_path, _records(3))

    run = _run(zp)

    assert run.returncode == 0, run.stderr
    assert "Verification: OK" in run.stdout
    assert lib.verify_signed(zp).ok


def test_the_same_name_in_different_objects_and_finite_extremes_still_verify(tmp_path: Path):
    data = {"data": {"data": 1}, "big": 1e308, "tiny": 5e-324, "zero": -0.0}
    zp = _export(tmp_path, _records(3, {1: data}))

    run = _run(zp)

    assert run.returncode == 0, run.stderr
    assert lib.verify_signed(zp).ok


# --- an independent reader: stdlib only, no import from the script or the library ---


def _last_wins(text: str):
    return json.loads(text)


def _first_wins(text: str):
    def pairs(items):
        out: dict = {}
        for k, v in items:
            out.setdefault(k, v)
        return out

    return json.loads(text, object_pairs_hook=pairs)


def _independent_refuses(text: str) -> bool:
    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("repeated member")
        return dict(items)

    def number(s):
        if float(s) in (float("inf"), float("-inf")):
            raise ValueError("overflow")
        return float(s)

    def constant(name):
        raise ValueError(name)

    try:
        json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except ValueError:
        return True
    return False


@pytest.mark.parametrize("case", list(_LINES))
def test_an_independent_reader_refuses_every_hostile_line_and_accepts_the_clean_one(
        tmp_path: Path, case: str):
    data, mutate, _ = _LINES[case]
    records = _records(3, {1: data})

    assert not _independent_refuses(_text(records[1]))
    assert _independent_refuses(mutate(_text(records[1])))


def test_the_repeated_member_line_reads_differently_to_two_readers():
    data, mutate, _ = _LINES["repeated top-level member"]
    line = mutate(_text(_records(2, {1: data})[1]))

    assert _last_wins(line)["tool_name"] == "T"
    assert _first_wins(line)["tool_name"] == "FIRST"
