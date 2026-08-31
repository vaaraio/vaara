# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``scripts/hcs27_mirror_check.py``, the off-ledger checker.

The checker's whole value is that it needs nothing: no account, no key, no
install, no Vaara. That makes it the piece a stranger runs, and the piece
whose behaviour has to be pinned down here rather than discovered against a
live network.

Three layers:

  agreement    the checker rebuilds the root top-down, splitting at the
               largest power of two below n. Vaara folds bottom-up. The two
               must land on the same bytes for every size, or an honest
               publisher looks like a tampering one.
  framing      what the checker accepts off the mirror node, and what it
               refuses to guess at.
  rules        the append-only rules, including the three defects that only
               showed up when this ran against a real topic.

The network is never touched. ``fetch_messages`` is replaced with a list of
mirror-node rows, which is also the only way to construct the dishonest
publishers these rules exist to catch.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from vaara.audit.hcs27 import b64u, root_for_entries

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "hcs27_mirror_check.py"
VECTOR = REPO / "tests" / "vectors" / "hcs27_checkpoint_v0"


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("_hcs27_mirror_check", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mc = _load_checker()


def _entries(n: int) -> list[dict[str, Any]]:
    return [{"v": 1, "recordId": f"rec-{i:04d}"} for i in range(n)]


def _root(entries: list[dict[str, Any]]) -> str:
    return b64u(root_for_entries(entries))


def _checkpoint(
    entries: list[dict[str, Any]],
    *,
    log_id: str = "log-a",
    registry: str = "0.0.900",
    prev: dict[str, str] | None = None,
    tree_size: str | None = None,
    root: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "stream": {"registry": registry, "log_id": log_id},
        "root": {
            "treeSize": str(len(entries)) if tree_size is None else tree_size,
            "rootHashB64u": _root(entries) if root is None else root,
        },
        "log": {"leaf": "sha256(0x00 || jcs(entry))"},
        "vaaraProfile": "vaara-trail-v1",
    }
    if prev is not None:
        metadata["prev"] = prev
    return {"p": "hcs-27", "op": "register", "metadata": metadata}


def _prev_of(checkpoint: dict[str, Any]) -> dict[str, str]:
    root = checkpoint["metadata"]["root"]
    return {"treeSize": root["treeSize"], "rootHashB64u": root["rootHashB64u"]}


def _wire(seq: int, message: Any, *, chunk_total: int = 1) -> dict[str, Any]:
    if isinstance(message, bytes):
        payload = message
    else:
        payload = json.dumps(message).encode("utf-8")
    return {
        "sequence_number": seq,
        "consensus_timestamp": f"1756220400.{seq:09d}",
        "message": base64.b64encode(payload).decode("ascii"),
        "chunk_info": {"number": 1, "total": chunk_total},
    }


def _run(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> int:
    monkeypatch.setattr(
        mc, "fetch_messages", lambda network, topic_id, limit: list(messages)
    )
    return mc.check_topic("testnet", "0.0.12345", entries, 100)


def _toplevel_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


# --- agreement between the two constructions ---------------------------------

def test_top_down_root_matches_the_bottom_up_fold_for_every_small_size():
    """Vaara folds bottom-up, the checker splits top-down. Same bytes, or the
    checker calls an honest publisher a liar."""
    for n in range(65):
        entries = _entries(n)
        top_down = mc.merkle_root([mc.jcs(e) for e in entries])
        assert top_down == root_for_entries(entries), f"diverged at tree size {n}"


def test_the_two_constructions_agree_at_power_of_two_boundaries():
    """The split point is the largest power of two below n, so 2^k and its
    neighbours are where an off-by-one would hide."""
    sizes = sorted(
        {s for k in range(11) for s in (2**k - 1, 2**k, 2**k + 1) if 0 <= s <= 1025}
    )
    for n in sizes:
        entries = _entries(n)
        top_down = mc.merkle_root([mc.jcs(e) for e in entries])
        assert top_down == root_for_entries(entries), f"diverged at tree size {n}"


def test_the_empty_tree_agrees():
    assert mc.merkle_root([]) == root_for_entries([])


def test_the_checker_canonicalises_identically_to_vaara():
    from vaara.audit.hcs27 import canonical_json

    entries = json.loads((VECTOR / "entries.json").read_text(encoding="utf-8"))
    for entry in entries:
        assert mc.jcs(entry) == canonical_json(entry)


def test_the_shipped_vector_root_recomputes_top_down():
    entries = json.loads((VECTOR / "entries.json").read_text(encoding="utf-8"))
    assert b64u(mc.merkle_root([mc.jcs(e) for e in entries])) == _root(entries)


def test_reordering_entries_changes_the_top_down_root():
    entries = _entries(7)
    swapped = list(entries)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert mc.merkle_root([mc.jcs(e) for e in swapped]) != mc.merkle_root(
        [mc.jcs(e) for e in entries]
    )


# --- what comes off the mirror node ------------------------------------------

def test_a_chunked_message_is_not_reassembled():
    """A Vaara checkpoint is a few hundred bytes and never chunks. Reassembling
    someone else's framing would be guessing."""
    assert mc.decode_message(_wire(1, _checkpoint(_entries(3)), chunk_total=4)) is None


def test_an_unchunked_message_is_decoded():
    checkpoint = _checkpoint(_entries(3))
    assert mc.decode_message(_wire(1, checkpoint)) == checkpoint


def test_a_message_that_is_not_json_is_skipped():
    assert mc.decode_message(_wire(1, b"\x00\x01 not json")) is None


def test_a_message_with_no_chunk_info_is_decoded():
    row = _wire(1, _checkpoint(_entries(3)))
    del row["chunk_info"]
    assert mc.decode_message(row) is not None


def test_a_topic_carrying_no_hcs27_register_fails(monkeypatch, capsys):
    other = {"p": "hcs-2", "op": "update"}
    assert _run(monkeypatch, [_wire(1, other)], _entries(3)) == 1
    assert "none an hcs-27 register" in capsys.readouterr().out


def test_an_empty_topic_fails(monkeypatch, capsys):
    assert _run(monkeypatch, [], _entries(3)) == 1
    assert "no messages on the topic" in capsys.readouterr().out


# --- the append-only rules ---------------------------------------------------

def test_a_clean_chain_passes(monkeypatch, capsys):
    first = _checkpoint(_entries(3))
    second = _checkpoint(_entries(7), prev=_prev_of(first))
    assert _run(monkeypatch, [_wire(1, first), _wire(2, second)], _entries(7)) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "root recomputes from the local entries" in out


def test_two_logs_on_one_topic_do_not_report_a_false_break(monkeypatch, capsys):
    """HCS-27 gives each log a stream.log_id, so prev chains within a log and
    says nothing across logs. Chaining per topic reported a break the first
    time a second stream shared a topic."""
    a1 = _checkpoint(_entries(3), log_id="log-a")
    b1 = _checkpoint(_entries(5), log_id="log-b")
    a2 = _checkpoint(_entries(7), log_id="log-a", prev=_prev_of(a1))
    messages = [_wire(1, a1), _wire(2, b1), _wire(3, a2)]
    assert _run(monkeypatch, messages, _entries(7)) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert out.count("first checkpoint seen for this log") == 2


def test_the_same_log_id_under_a_different_registry_is_a_different_log(
    monkeypatch, capsys
):
    a1 = _checkpoint(_entries(3), log_id="log-a", registry="0.0.900")
    b1 = _checkpoint(_entries(7), log_id="log-a", registry="0.0.901")
    assert _run(monkeypatch, [_wire(1, a1), _wire(2, b1)], _entries(7)) == 0
    assert capsys.readouterr().out.count("first checkpoint seen for this log") == 2


def test_a_shrinking_log_fails(monkeypatch, capsys):
    first = _checkpoint(_entries(7))
    second = _checkpoint(_entries(3), prev=_prev_of(first))
    assert _run(monkeypatch, [_wire(1, first), _wire(2, second)], _entries(3)) == 1
    assert "the log shrank" in capsys.readouterr().out


def test_a_log_replaced_at_the_same_size_fails(monkeypatch, capsys):
    """prev chaining alone cannot catch this: the new checkpoint honestly names
    the old one. Only comparing the pair says the contents moved underneath."""
    first = _checkpoint(_entries(7))
    replacement = _entries(7)
    replacement[2] = {"v": 1, "recordId": "rec-swapped"}
    second = _checkpoint(replacement, prev=_prev_of(first))
    assert _run(monkeypatch, [_wire(1, first), _wire(2, second)], replacement) == 1
    assert "replaced, not extended" in capsys.readouterr().out


def test_a_non_genesis_checkpoint_without_prev_fails(monkeypatch, capsys):
    first = _checkpoint(_entries(3))
    second = _checkpoint(_entries(7))
    assert _run(monkeypatch, [_wire(1, first), _wire(2, second)], _entries(7)) == 1
    assert "carries no prev" in capsys.readouterr().out


def test_a_prev_naming_the_wrong_checkpoint_fails(monkeypatch, capsys):
    first = _checkpoint(_entries(3))
    second = _checkpoint(
        _entries(7), prev={"treeSize": "4", "rootHashB64u": _root(_entries(4))}
    )
    assert _run(monkeypatch, [_wire(1, first), _wire(2, second)], _entries(7)) == 1
    assert "prev does not match the previous checkpoint" in capsys.readouterr().out


def test_a_tree_size_that_is_not_a_decimal_string_fails(monkeypatch, capsys):
    checkpoint = _checkpoint(_entries(7), tree_size="0x7")
    assert _run(monkeypatch, [_wire(1, checkpoint)], _entries(7)) == 1
    assert "treeSize is not a canonical decimal string" in capsys.readouterr().out


def test_a_tree_size_carried_as_a_number_fails(monkeypatch, capsys):
    checkpoint = _checkpoint(_entries(7))
    checkpoint["metadata"]["root"]["treeSize"] = 7
    assert _run(monkeypatch, [_wire(1, checkpoint)], _entries(7)) == 1
    assert "treeSize is not a canonical decimal string" in capsys.readouterr().out


def test_a_padded_or_non_url_safe_root_fails(monkeypatch, capsys):
    entries = _entries(7)
    padded = base64.b64encode(root_for_entries(entries)).decode("ascii")
    checkpoint = _checkpoint(entries, root=padded)
    assert _run(monkeypatch, [_wire(1, checkpoint)], entries) == 1
    assert "rootHashB64u is not unpadded base64url" in capsys.readouterr().out


def test_a_topic_that_never_publishes_the_local_root_fails(monkeypatch, capsys):
    """The checker's real question is whether the ledger carries the root of
    the entries in front of you, not merely a well-formed chain."""
    published = _checkpoint(_entries(7))
    assert _run(monkeypatch, [_wire(1, published)], _entries(5)) == 1
    assert "no checkpoint on the topic publishes the root" in capsys.readouterr().out


def test_a_matching_root_at_the_wrong_tree_size_does_not_count(monkeypatch, capsys):
    entries = _entries(7)
    checkpoint = _checkpoint(entries, tree_size="8", root=_root(entries))
    assert _run(monkeypatch, [_wire(1, checkpoint)], entries) == 1
    assert "no checkpoint on the topic publishes the root" in capsys.readouterr().out


# --- the checker needs nothing -----------------------------------------------

def test_the_checker_imports_nothing_outside_the_standard_library():
    """This is the file a stranger runs. The moment it needs an install, the
    independent check stops being independent."""
    outside = _toplevel_imports(SCRIPT) - set(sys.stdlib_module_names)
    assert outside == set(), f"{SCRIPT.name} now needs {sorted(outside)}"


def test_the_checker_imports_neither_vaara_nor_the_hedera_sdk():
    imported = _toplevel_imports(SCRIPT)
    assert "vaara" not in imported
    assert "hiero_sdk_python" not in imported


def test_the_checkpoint_module_needs_no_third_party_package():
    """vaara.audit.hcs27 builds and verifies with the standard library alone,
    so a machine with no SDK and no Hedera account can do everything except
    submit. Its own imports are stdlib plus Vaara; note this says nothing
    about what importing the vaara package as a whole drags in."""
    outside = _toplevel_imports(REPO / "src" / "vaara" / "audit" / "hcs27.py")
    outside -= set(sys.stdlib_module_names)
    assert outside == {"vaara"}, f"hcs27.py now needs {sorted(outside)}"


def test_only_the_publisher_needs_the_hedera_sdk():
    assert "hiero_sdk_python" in _toplevel_imports(REPO / "scripts" / "hcs27_publish.py")
