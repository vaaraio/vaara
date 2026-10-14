# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCSF export: each decision is one valid OCSF 1.9.0 API Activity event.

The required attributes, enum values and at-least-one constraints below are
copied from the OCSF 1.9.0 schema (schema.ocsf.io/api/classes/api_activity and
the objects it references), so the test fails if the export drifts from them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vaara.audit.ocsf_export import decision_events, write_ocsf_jsonl
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline

REQUIRED = ("activity_id", "category_uid", "class_uid", "metadata", "severity_id",
            "time", "type_uid", "actor", "api", "src_endpoint")
ACTOR_ONE_OF = ("process", "user", "iam_role", "invoked_by", "session",
                "application", "app_name", "app_uid")
ENDPOINT_ONE_OF = ("ip", "uid", "name", "hostname", "svc_name", "instance_uid",
                   "interface_uid", "interface_name", "domain")
ACTIVITY_IDS = {0, 1, 2, 3, 4, 99}
SEVERITY_IDS = {0, 1, 2, 3, 4, 5, 6, 99}
STATUS_IDS = {0, 1, 2, 99}
ACTION_IDS = {0, 1, 2, 3, 4, 99}
DISPOSITION_IDS = set(range(0, 28)) | {99}


def _trail():
    trail = SQLiteAuditBackend(":memory:").load_trail()
    pipe = InterceptionPipeline(trail=trail)
    pipe.intercept(agent_id="a1", tool_name="read_file",
                   parameters={"path": "/repo/README.md"}, session_id="run-1")
    trail.record_decision(action_id="act-deny", agent_id="a1", tool_name="Bash",
                          decision="deny", reason="rule: rm -rf on a home path",
                          risk_score=0.97, policy_id="deny_rule:rm",
                          violation_type="policy_rule")
    trail.record_decision(action_id="act-hold", agent_id="a1", tool_name="send_email",
                          decision="escalate", reason="held for a human", risk_score=0.6)
    return trail


def _check(event: dict) -> None:
    for key in REQUIRED:
        assert key in event, key
    assert event["class_uid"] == 6003 and event["category_uid"] == 6
    assert event["type_uid"] == event["class_uid"] * 100 + event["activity_id"]
    assert event["activity_id"] in ACTIVITY_IDS
    if event["activity_id"] == 99:
        assert event["activity_name"]
    assert event["type_name"] == f"{event['class_name']}: {event['activity_name']}"
    assert event["severity_id"] in SEVERITY_IDS
    assert event["status_id"] in STATUS_IDS
    assert event["action_id"] in ACTION_IDS
    assert event["disposition_id"] in DISPOSITION_IDS
    if event["action_id"] == 99:
        assert event["action"]
    assert isinstance(event["time"], int) and event["time"] > 1_600_000_000_000
    meta = event["metadata"]
    assert meta["version"] == "1.9.0"
    assert meta["product"]["name"] == "Vaara"
    assert any(k in event["actor"] for k in ACTOR_ONE_OF)
    assert any(k in event["src_endpoint"] for k in ENDPOINT_ONE_OF)
    assert event["api"]["operation"]
    assert any(k in event["ai_agent"] for k in ("name", "uid"))
    assert any(k in event["policy"] for k in ("name", "type", "uid"))
    assert isinstance(event["risk_score"], int)
    assert "attestation_list" not in event


def test_every_decision_is_a_valid_api_activity_event():
    trail = _trail()
    events = decision_events(trail._records, hostname="dev-box", vaara_version="0")
    assert len(events) == 3
    for event in events:
        _check(event)


def test_the_decision_maps_to_the_control_outcome():
    events = {e["api"]["operation"]: e for e in decision_events(_trail()._records, "h")}
    allow, deny, hold = events["read_file"], events["Bash"], events["send_email"]
    assert (allow["action_id"], allow["disposition_id"], allow["is_alert"]) == (1, 1, False)
    assert (deny["action_id"], deny["disposition_id"], deny["is_alert"]) == (2, 2, True)
    assert (deny["status_id"], deny["status_detail"]) == (2, "policy_rule")
    assert (hold["action_id"], hold["action"], hold["disposition_id"]) == (99, "Escalate", 14)
    assert deny["severity_id"] == 4 and deny["risk_score"] == 97
    assert deny["policy"]["name"] == "deny_rule:rm"


def test_the_event_names_the_trail_record_and_session():
    trail = _trail()
    records = {r.action_id: r for r in trail._records
               if r.event_type.value in ("decision_made", "action_blocked")}
    events = decision_events(trail._records, "h")
    allow = next(e for e in events if e["api"]["operation"] == "read_file")
    record = next(r for r in records.values() if r.tool_name == "read_file")
    assert allow["metadata"]["uid"] == record.record_id
    assert allow["metadata"]["correlation_uid"] == record.action_id
    assert allow["unmapped"]["vaara"]["record_hash"] == record.record_hash
    assert allow["ai_agent"]["instance_uid"] == "run-1"
    assert allow["actor"] == {"session": {"uid": "run-1"}}
    assert allow["policy"]["uid"].startswith("sha256:")
    # A decision without a request record still names an actor.
    deny = next(e for e in events if e["api"]["operation"] == "Bash")
    assert deny["actor"]["application"]["uid"] == "a1"


def test_cli_exports_a_trail_jsonl(tmp_path: Path):
    trail = _trail()
    src = tmp_path / "trail.jsonl"
    src.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in trail._records))
    out = tmp_path / "ocsf.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "trail", "export-ocsf",
         "--trail", str(src), "--out", str(out), "--hostname", "dev-box"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        event = json.loads(line)
        _check(event)
        assert event["src_endpoint"] == {"hostname": "dev-box"}


def test_write_without_hostname_names_an_unknown_endpoint(tmp_path: Path):
    n = write_ocsf_jsonl(_trail()._records, tmp_path / "o.jsonl", hostname="")
    assert n == 3
    event = json.loads((tmp_path / "o.jsonl").read_text().splitlines()[0])
    assert event["src_endpoint"] == {"name": "unknown"}


def test_cli_exports_straight_from_a_sqlite_trail(tmp_path: Path):
    db = tmp_path / "audit.db"
    trail = SQLiteAuditBackend(str(db)).load_trail()
    InterceptionPipeline(trail=trail).intercept(
        agent_id="a1", tool_name="read_file", parameters={"path": "/repo/x"},
        session_id="run-1")
    out = tmp_path / "ocsf.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "trail", "export-ocsf",
         "--db", str(db), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    [line] = out.read_text().splitlines()
    _check(json.loads(line))
