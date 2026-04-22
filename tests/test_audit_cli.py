"""Tests for ``vaara-audit`` CLI."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.export import export_signed
from vaara.audit.trail import AuditTrail
from vaara.audit_cli import (
    _rule_rate_burst,
    _rule_missing_completion,
    _rule_timestamp_regression,
    _rule_unknown_spike,
    build_parser,
    main,
)
from vaara.taxonomy.actions import ActionRequest, create_default_registry

_REGISTRY = create_default_registry()
_TX_TRANSFER = _REGISTRY.get("tx.transfer")


def _build_trail(n: int = 5, agent_id: str = "agent-0") -> AuditTrail:
    trail = AuditTrail()
    for i in range(n):
        req = ActionRequest(
            agent_id=agent_id,
            tool_name="send_funds",
            action_type=_TX_TRANSFER,
            parameters={"to": f"0xabc{i}", "amount": 10 * i},
        )
        trail.record_action_requested(req)
    return trail


def _pem_keys(tmp_path: Path) -> tuple[Path, Path]:
    priv_key = Ed25519PrivateKey.generate()
    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_path = tmp_path / "signer.key.pem"
    pub_path = tmp_path / "signer.pub.pem"
    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)
    return priv_path, pub_path


@pytest.fixture
def signed_trail(tmp_path: Path) -> Path:
    trail = _build_trail(n=5)
    priv, _ = _pem_keys(tmp_path)
    out = tmp_path / "trail.zip"
    export_signed(trail, out_path=out, signer_key=priv, agent_id="test")
    return out


def test_parser_builds_without_error():
    p = build_parser()
    assert p.prog == "vaara-audit"


def test_verify_passes_on_valid_trail(signed_trail, capsys):
    rc = main(["verify", str(signed_trail)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_verify_fails_on_tampered_trail(signed_trail, tmp_path, capsys):
    # Corrupt the zip: replace one byte of trail.jsonl inside the zip
    import shutil
    bad = tmp_path / "bad.zip"
    shutil.copy(signed_trail, bad)
    # Rewrite the zip with a mutated trail.jsonl entry
    with zipfile.ZipFile(bad, "r") as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}
    data["trail.jsonl"] = data["trail.jsonl"].replace(b'"agent-0"', b'"agentX0"', 1)
    bad.unlink()
    with zipfile.ZipFile(bad, "w") as zout:
        for n, b in data.items():
            zout.writestr(n, b)
    rc = main(["verify", str(bad)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out


def test_verify_json_output(signed_trail, capsys):
    rc = main(["verify", str(signed_trail), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert "manifest" in payload
    assert payload["manifest"]["record_count"] == 5


def test_inspect_lists_all_records(signed_trail, capsys):
    rc = main(["inspect", str(signed_trail)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "matched" in out
    # 5 events in the trail
    assert "matched 5" in out or "matched 10" in out  # depends on per-request events emitted


def test_inspect_filters_by_agent(signed_trail, capsys):
    rc = main(["inspect", str(signed_trail), "--agent", "agent-does-not-exist"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no matching records" in out or "matched 0" in out


def test_inspect_json_output(signed_trail, capsys):
    rc = main(["inspect", str(signed_trail), "--json", "--limit", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "records" in payload
    assert payload["n"] <= 2


def test_stats_default_group(signed_trail, capsys):
    rc = main(["stats", str(signed_trail)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "event_type" in out
    assert "total records" in out


def test_stats_group_by_agent_json(signed_trail, capsys):
    rc = main(["stats", str(signed_trail), "--group-by", "agent", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["group_by"] == "agent"
    assert payload["total"] >= 1


def test_anomalies_clean_trail(signed_trail, capsys):
    rc = main(["anomalies", str(signed_trail)])
    out = capsys.readouterr().out
    # Clean trail should have no findings
    assert rc == 0
    assert "findings: 0" in out or "(no anomalies)" in out


def test_anomalies_bad_rule_name(signed_trail, capsys):
    rc = main(["anomalies", str(signed_trail), "--rules", "nonsense"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown rules" in err


def test_missing_completion_detects_orphan_request():
    records = [
        {"agent_id": "a", "event_type": "action_requested", "action_id": "act-1",
         "tool_name": "send", "record_id": "r1"},
        {"agent_id": "a", "event_type": "decision_emitted", "action_id": "act-1",
         "record_id": "r2"},
        {"agent_id": "a", "event_type": "action_requested", "action_id": "act-2",
         "tool_name": "send", "record_id": "r3"},
        # act-2 never got a decision — should fire
    ]
    findings = _rule_missing_completion(records)
    assert len(findings) == 1
    assert findings[0]["rule"] == "missing_completion"
    assert findings[0]["action_id"] == "act-2"


def test_timestamp_regression_rule():
    records = [
        {"agent_id": "a", "sequence_position": 0, "timestamp": "2026-01-01T00:00:00Z"},
        {"agent_id": "a", "sequence_position": 1, "timestamp": "2026-01-01T00:00:01Z"},
        {"agent_id": "a", "sequence_position": 2, "timestamp": "2025-01-01T00:00:00Z"},
    ]
    findings = _rule_timestamp_regression(records)
    assert len(findings) == 1
    assert findings[0]["rule"] == "timestamp_regression"


def test_rate_burst_rule_triggers():
    base = "2026-01-01T00:00:00Z"
    from datetime import datetime, timedelta
    t0 = datetime.fromisoformat(base.replace("Z", "+00:00"))
    # 25 records within 1 second → should trip burst with defaults (20/10s)
    records = [
        {"agent_id": "a", "sequence_position": i, "timestamp": (t0 + timedelta(milliseconds=i * 10)).isoformat()}
        for i in range(25)
    ]
    findings = _rule_rate_burst(records)
    assert len(findings) >= 1
    assert findings[0]["rule"] == "rate_burst"


def test_unknown_spike_rule_triggers():
    records = []
    for i in range(60):
        action_type = "unknown" if i < 40 else "data.read"
        records.append({
            "agent_id": "a",
            "sequence_position": i,
            "payload": {"action_type": action_type},
        })
    findings = _rule_unknown_spike(records, window=50, frac_threshold=0.25)
    assert len(findings) >= 1
    assert findings[0]["rule"] == "unknown_spike"


def test_load_records_from_zip_rejects_missing_trail(tmp_path: Path):
    from vaara.audit_cli import _load_records_from_zip as load
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("not_trail.txt", "hi")
    with pytest.raises(ValueError):
        load(bad)
