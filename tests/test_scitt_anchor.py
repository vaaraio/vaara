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
