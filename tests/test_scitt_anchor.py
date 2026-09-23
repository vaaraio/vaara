from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")

from vaara.audit.scitt_anchor import ScittAnchor, ScittAnchorError, verify_scitt_anchor


VECTOR = (Path(__file__).resolve().parents[0]
          / "vectors/x402_settlement_v0/generic/step1/receipt.json")


@pytest.fixture
def receipt() -> dict:
    return json.loads(VECTOR.read_text())


def test_anchor_receipt_adds_scitt_entry(receipt: dict) -> None:
    anchor = ScittAnchor().anchor_receipt(receipt)
    assert anchor["method"] == "scitt"
    assert "sha256:" in anchor["anchoredDigest"]
    assert isinstance(anchor["leafIndex"], int)
    assert isinstance(anchor["treeSize"], int)
    assert isinstance(anchor["inclusionProof"], list)
    assert isinstance(anchor["rootHash"], str)
    assert anchor["leafIndex"] == 0
    assert anchor["treeSize"] == 1


def test_anchor_is_verifiable(receipt: dict) -> None:
    anchor = ScittAnchor().anchor_receipt(receipt)
    result = verify_scitt_anchor(receipt, anchor)
    assert result["verified"] is True
    assert result["status"] == "verified"
    assert result["leaf_index"] == 0


def test_verify_rejects_tampered_receipt(receipt: dict) -> None:
    anchor = ScittAnchor().anchor_receipt(receipt)
    tampered = copy.deepcopy(receipt)
    tampered["decisionDerived"]["decision"] = "block"
    result = verify_scitt_anchor(tampered, anchor)
    assert result["verified"] is False
    assert "mismatch" in result["status"]


def test_verify_rejects_wrong_method(receipt: dict) -> None:
    anchor = ScittAnchor().anchor_receipt(receipt)
    with pytest.raises(ScittAnchorError, match="not a scitt anchor"):
        verify_scitt_anchor(receipt, dict(anchor, method="rfc3161"))


def test_verify_different_logs_are_independent(receipt: dict) -> None:
    a1 = ScittAnchor(log_id="log-a").anchor_receipt(receipt)
    a2 = ScittAnchor(log_id="log-b").anchor_receipt(receipt)
    assert verify_scitt_anchor(receipt, a1)["verified"]
    assert verify_scitt_anchor(receipt, a2)["verified"]


def test_multiple_anchors_in_same_log(receipt: dict) -> None:
    log = ScittAnchor()
    a1 = log.anchor_receipt(receipt)
    a2 = log.anchor_receipt(receipt)
    assert a1["leafIndex"] == 0
    assert a2["leafIndex"] == 1
    assert a1["treeSize"] == 1
    assert a2["treeSize"] == 2
    assert verify_scitt_anchor(receipt, a1)["verified"]
    assert verify_scitt_anchor(receipt, a2)["verified"]


def test_verify_invalid_proof(receipt: dict) -> None:
    anchor = ScittAnchor().anchor_receipt(receipt)
    anchor["inclusionProof"] = [anchor["inclusionProof"][0]] if anchor["inclusionProof"] else ["AAAA"]
    result = verify_scitt_anchor(receipt, anchor)
    assert result["verified"] is False


# --- the log persists, and the root is only trusted against a held head ---


def _other(receipt: dict, n: int) -> dict:
    """A distinct receipt: same shape, different signed-payload digest."""
    out = copy.deepcopy(receipt)
    out["decisionDerived"]["decision"] = f"other-{n}"
    return out


def test_file_backed_log_grows_across_runs(receipt: dict, tmp_path: Path) -> None:
    """Two CLI-style runs share one tree instead of each making a tree of one."""
    a1 = ScittAnchor.load_or_create(tmp_path).anchor_receipt(receipt)
    a2 = ScittAnchor.load_or_create(tmp_path).anchor_receipt(receipt)
    assert (a1["leafIndex"], a1["treeSize"]) == (0, 1)
    assert (a2["leafIndex"], a2["treeSize"]) == (1, 2)
    assert (tmp_path / "vaara-scitt-log.leaves").read_text().count("\n") == 2


def test_default_log_dir_is_not_ignored(receipt: dict, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    anchorer = ScittAnchor.load_or_create()
    anchorer.anchor_receipt(receipt)
    assert anchorer.path == tmp_path / ".vaara/anchor-log/vaara-scitt-log.leaves"
    assert anchorer.path.is_file()


def test_truncated_log_line_is_refused(receipt: dict, tmp_path: Path) -> None:
    ScittAnchor.load_or_create(tmp_path).anchor_receipt(receipt)
    leaves = tmp_path / "vaara-scitt-log.leaves"
    leaves.write_text(leaves.read_text() + "abc")
    with pytest.raises(ScittAnchorError, match="line 2 is not a complete leaf"):
        ScittAnchor.load_or_create(tmp_path)


def test_bad_log_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScittAnchorError, match="log id"):
        ScittAnchor.load_or_create(tmp_path, log_id="../escape")


def test_log_and_path_together_are_refused(tmp_path: Path) -> None:
    from vaara.attestation.transparency_log import InProcessTransparencyLog
    with pytest.raises(ScittAnchorError, match="not both"):
        ScittAnchor(InProcessTransparencyLog(), path=tmp_path / "x.leaves")


def test_without_head_the_root_is_reported_as_unwitnessed(receipt: dict) -> None:
    result = verify_scitt_anchor(receipt, ScittAnchor().anchor_receipt(receipt))
    assert result["verified"] is True
    assert result["root_witnessed"] is False
    assert "not checked against an independently held tree head" in result["root"]


def test_forged_tree_of_one_fails_against_the_real_head(receipt: dict) -> None:
    """A producer can always build a one-leaf tree; a held head catches it."""
    log = ScittAnchor()
    log.anchor_receipt(_other(receipt, 1))
    log.anchor_receipt(_other(receipt, 2))
    head = log.head()
    forged = ScittAnchor().anchor_receipt(receipt)
    assert verify_scitt_anchor(receipt, forged)["verified"] is True
    result = verify_scitt_anchor(receipt, forged,
                                 trusted_head=dict(head, logId=forged["logId"]))
    assert result["verified"] is False
    assert result["root_witnessed"] is False


def test_head_of_the_same_size_witnesses_the_root(receipt: dict) -> None:
    log = ScittAnchor()
    anchor = log.anchor_receipt(receipt)
    result = verify_scitt_anchor(receipt, anchor, trusted_head=log.head())
    assert result["verified"] is True
    assert result["root_witnessed"] is True


def test_later_head_needs_and_accepts_a_consistency_proof(receipt: dict) -> None:
    log = ScittAnchor()
    log.anchor_receipt(_other(receipt, 1))
    anchor = log.anchor_receipt(receipt)
    for n in range(5):
        log.anchor_receipt(_other(receipt, 10 + n))

    bare = verify_scitt_anchor(receipt, anchor, trusted_head=log.head())
    assert bare["verified"] is False
    assert "no consistency proof" in bare["status"]

    head = log.head(consistency_from=anchor["treeSize"])
    result = verify_scitt_anchor(receipt, anchor, trusted_head=head)
    assert result["verified"] is True
    assert result["root_witnessed"] is True


def test_rewritten_history_is_inconsistent(receipt: dict) -> None:
    log = ScittAnchor()
    anchor = log.anchor_receipt(receipt)
    rewritten = ScittAnchor()
    rewritten.anchor_receipt(_other(receipt, 1))
    rewritten.anchor_receipt(_other(receipt, 2))
    head = rewritten.head(consistency_from=1)
    result = verify_scitt_anchor(receipt, anchor, trusted_head=head)
    assert result["verified"] is False
    assert "not consistent" in result["status"]


def test_head_from_another_log_is_refused(receipt: dict) -> None:
    anchor = ScittAnchor(log_id="log-a").anchor_receipt(receipt)
    other = ScittAnchor(log_id="log-b")
    other.anchor_receipt(receipt)
    result = verify_scitt_anchor(receipt, anchor, trusted_head=other.head())
    assert result["verified"] is False
    assert "different log" in result["status"]


def test_head_older_than_the_anchor_is_refused(receipt: dict) -> None:
    log = ScittAnchor()
    log.anchor_receipt(_other(receipt, 1))
    early = log.head()
    anchor = log.anchor_receipt(receipt)
    result = verify_scitt_anchor(receipt, anchor, trusted_head=early)
    assert result["verified"] is False
    assert "older" in result["status"]


def test_consistency_from_out_of_range_is_refused(receipt: dict) -> None:
    log = ScittAnchor()
    log.anchor_receipt(receipt)
    with pytest.raises(ScittAnchorError, match="consistency_from"):
        log.head(consistency_from=2)


def test_receipt_without_signed_blocks_raises_the_anchor_error() -> None:
    with pytest.raises(ScittAnchorError, match="signed-payload block"):
        ScittAnchor().anchor_receipt({"not": "a receipt"})


def test_concurrent_appends_get_distinct_positions(receipt: dict, tmp_path: Path) -> None:
    import threading

    anchors: list[dict] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        a = ScittAnchor.load_or_create(tmp_path).anchor_receipt(_other(receipt, n))
        with lock:
            anchors.append(a)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(a["leafIndex"] for a in anchors) == list(range(8))
    head = ScittAnchor.load_or_create(tmp_path).head()
    assert head["treeSize"] == 8


def test_cli_anchor_head_and_verify_round_trip(receipt: dict, tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    from vaara.cli import main

    log_dir = tmp_path / "log"
    first = tmp_path / "first.json"
    first.write_text(json.dumps(_other(receipt, 1)))
    mine = tmp_path / "receipt.json"
    mine.write_text(json.dumps(receipt))
    assert main(["receipt", "anchor-scitt", str(first), "--log-dir", str(log_dir)]) == 0
    assert main(["receipt", "anchor-scitt", str(mine), "--log-dir", str(log_dir)]) == 0
    anchor = json.loads(mine.read_text())["timestampAnchors"][-1]
    assert (anchor["leafIndex"], anchor["treeSize"]) == (1, 2)

    assert main(["receipt", "verify-scitt", str(mine)]) == 0
    assert "not checked against an independently held tree head" in capsys.readouterr().out

    later = tmp_path / "later.json"
    later.write_text(json.dumps(_other(receipt, 3)))
    assert main(["receipt", "anchor-scitt", str(later), "--log-dir", str(log_dir)]) == 0
    capsys.readouterr()
    assert main(["receipt", "anchor-scitt-head", "--log-dir", str(log_dir),
                 "--consistency-from", str(anchor["treeSize"])]) == 0
    head = tmp_path / "head.json"
    head.write_text(capsys.readouterr().out)
    assert json.loads(head.read_text())["treeSize"] == 3

    assert main(["receipt", "verify-scitt", str(mine), "--head", str(head)]) == 0
    assert "prefix of the trusted head" in capsys.readouterr().out

    forged = json.loads(mine.read_text())
    forged["timestampAnchors"][-1] = ScittAnchor().anchor_receipt(receipt)
    mine.write_text(json.dumps(forged))
    assert main(["receipt", "verify-scitt", str(mine), "--head", str(head)]) == 1


def test_cli_verify_scitt_without_anchor_exits_2(receipt: dict, tmp_path: Path) -> None:
    from vaara.cli import main

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(receipt))
    assert main(["receipt", "verify-scitt", str(bare)]) == 2


@pytest.mark.parametrize("head", [
    [], "head", 3,
    {"treeSize": 2, "rootHash": "AA==", "consistency": []},
    {"treeSize": 2, "rootHash": "AA==", "consistency": {"firstSize": 1, "hashes": "AA=="}},
])
def test_malformed_trusted_head_raises_the_anchor_error(receipt: dict, head: object) -> None:
    log = ScittAnchor()
    anchor = log.anchor_receipt(receipt)
    with pytest.raises(ScittAnchorError, match="malformed trusted head"):
        verify_scitt_anchor(receipt, anchor, trusted_head=head)  # type: ignore[arg-type]


def test_cli_verify_scitt_with_a_non_object_head_exits_2(receipt: dict, tmp_path: Path) -> None:
    from vaara.cli import main

    mine = tmp_path / "receipt.json"
    mine.write_text(json.dumps(receipt))
    assert main(["receipt", "anchor-scitt", str(mine), "--log-dir", str(tmp_path / "log")]) == 0
    head = tmp_path / "head.json"
    head.write_text("[]")
    assert main(["receipt", "verify-scitt", str(mine), "--head", str(head)]) == 2


def test_reads_do_not_see_a_half_written_leaf(receipt: dict, tmp_path: Path) -> None:
    """Readers open the log while appends run, and none of them fails."""
    import threading

    errors: list[Exception] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                ScittAnchor.load_or_create(tmp_path).head()
            except Exception as exc:  # any failure is the bug under test
                errors.append(exc)
                return

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in readers:
        t.start()
    for n in range(60):
        ScittAnchor.load_or_create(tmp_path).anchor_receipt(_other(receipt, n))
    stop.set()
    for t in readers:
        t.join()
    assert errors == []
