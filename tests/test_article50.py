"""Tests for the Article 50 transparency evidence module and CLI."""
from __future__ import annotations

import json
import zipfile

import pytest

pytest.importorskip("cryptography")

from vaara.audit.article50 import (
    AGENT_PROFILE,
    DISCLOSURE_TOOL,
    build_article50_report,
    export_article50,
    find_disclosures,
    record_agent_disclosure,
    record_disclosure,
    render_article50_md,
)
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.verify import verify_signed
from vaara.pipeline import InterceptionPipeline


def _persistent_trail(tmp_path):
    backend = SQLiteAuditBackend(tmp_path / "audit.db")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return trail


def _agent_activity(trail, session_id: str, n: int = 2) -> None:
    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    for i in range(n):
        pipeline.intercept(
            agent_id="assistant-1",
            tool_name=f"mcp__demo__tool{i}",
            parameters={"x": i},
            session_id=session_id,
        )


def test_record_disclosure_lands_in_trail(tmp_path):
    trail = _persistent_trail(tmp_path)
    action_id = record_disclosure(
        trail,
        paragraph="50(1)",
        statement="You are chatting with an AI assistant.",
        session_id="s1",
        channel="chat_ui",
    )
    assert action_id
    events = find_disclosures(trail._records)
    assert len(events) == 1
    event = events[0]
    assert event["article"] == "50(1)"
    assert event["session_id"] == "s1"
    assert event["channel"] == "chat_ui"
    assert "AI assistant" in event["statement"]


def test_record_disclosure_validates_input(tmp_path):
    trail = _persistent_trail(tmp_path)
    with pytest.raises(ValueError):
        record_disclosure(trail, paragraph="50(9)", statement="x")
    with pytest.raises(ValueError):
        record_disclosure(trail, paragraph="50(1)", statement="")


def test_report_session_coverage_and_timing(tmp_path):
    trail = _persistent_trail(tmp_path)
    # session s1: disclosure BEFORE first action (compliant timing)
    record_disclosure(
        trail, paragraph="50(1)", statement="AI notice", session_id="s1",
    )
    _agent_activity(trail, "s1")
    # session s2: activity with no disclosure at all
    _agent_activity(trail, "s2")

    report = build_article50_report(trail._records, {"record_count": 0})
    cov = report["session_coverage_50_1"]
    assert cov["sessions_with_agent_activity"] == 2
    assert cov["sessions_with_50_1_disclosure"] == 1
    assert cov["disclosed_at_or_before_first_action"] == 1
    assert report["disclosures"]["by_paragraph"]["50(1)"] == 1
    assert report["disclosures"]["by_paragraph"]["50(4)"] == 0

    md = render_article50_md(report)
    assert "does not prove" in md
    assert "Sessions with agent activity: 2" in md


def test_export_package_verifies_and_carries_report(tmp_path):
    key = tmp_path / "key"
    from vaara.cli import main as cli_main
    assert cli_main(["keygen", "--dev", "--out", str(key)]) == 0

    trail = _persistent_trail(tmp_path)
    record_disclosure(
        trail, paragraph="50(1)", statement="AI notice", session_id="s1",
    )
    _agent_activity(trail, "s1")

    out = tmp_path / "a50.zip"
    result = export_article50(trail, out, signer_key=key)
    assert result.chain_intact

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "article50_report.md" in names
        assert "article50_summary.json" in names
        summary = json.loads(zf.read("article50_summary.json"))
        assert summary["disclosures"]["total"] == 1
        assert DISCLOSURE_TOOL in zf.read("trail.jsonl").decode()

    # The signed core must still verify with the standard verifier.
    verdict = verify_signed(out)
    assert verdict.ok, verdict.errors


def test_cli_export_article50_from_db(tmp_path, capsys):
    from vaara.cli import main as cli_main

    key = tmp_path / "key"
    assert cli_main(["keygen", "--dev", "--out", str(key)]) == 0

    trail = _persistent_trail(tmp_path)
    record_disclosure(
        trail, paragraph="50(1)", statement="AI notice", session_id="s1",
    )
    _agent_activity(trail, "s1")

    out = tmp_path / "a50.zip"
    capsys.readouterr()
    rc = cli_main([
        "trail", "export-article50",
        "--db", str(tmp_path / "audit.db"),
        "--out", str(out), "--key", str(key),
    ])
    assert rc == 0
    assert "Article 50 transparency package" in capsys.readouterr().out
    assert out.is_file()


def test_record_agent_disclosure_para31_fields(tmp_path):
    trail = _persistent_trail(tmp_path)
    action_id = record_agent_disclosure(
        trail,
        statement="I am an AI agent acting for Example Oy.",
        on_behalf_of="Example Oy",
        step="first_interaction",
        session_id="s1",
        channel="chat_ui",
        authority_ref="grant-42",
    )
    assert action_id
    event = find_disclosures(trail._records)[0]
    assert event["article"] == "50(1)"
    assert event["profile"] == AGENT_PROFILE
    assert event["on_behalf_of"] == "Example Oy"
    assert event["step"] == "first_interaction"
    assert event["authority_ref"] == "grant-42"


def test_record_agent_disclosure_validates_input(tmp_path):
    trail = _persistent_trail(tmp_path)
    with pytest.raises(ValueError):
        record_agent_disclosure(
            trail, statement="x", on_behalf_of="p", step="whenever",
        )
    with pytest.raises(ValueError):
        record_agent_disclosure(
            trail, statement="x", on_behalf_of="", step="authorisation",
        )
    with pytest.raises(ValueError):
        record_agent_disclosure(
            trail, statement="", on_behalf_of="p", step="authorisation",
        )


def test_agent_disclosure_threads_delegation_edge(tmp_path):
    from vaara.audit.delegation import graph_from_trail

    trail = _persistent_trail(tmp_path)
    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    parent = pipeline.intercept(
        agent_id="orchestrator",
        tool_name="mcp__demo__spawn",
        session_id="s1",
    )
    child_id = record_agent_disclosure(
        trail,
        statement="AI agent notice",
        on_behalf_of="Example Oy",
        step="authorisation",
        session_id="s1",
        parent_action_id=parent.action_id,
    )
    graph = graph_from_trail(trail)
    assert graph.chain_for(child_id) == [parent.action_id, child_id]


def test_report_para31_section(tmp_path):
    trail = _persistent_trail(tmp_path)
    record_agent_disclosure(
        trail, statement="AI agent notice", on_behalf_of="Example Oy",
        step="first_interaction", session_id="s1",
    )
    record_agent_disclosure(
        trail, statement="AI agent notice", on_behalf_of="Example Oy",
        step="validation", session_id="s1", authority_ref="grant-42",
    )
    # A generic 50(1) disclosure must not count into the agent profile.
    record_disclosure(
        trail, paragraph="50(1)", statement="AI notice", session_id="s2",
    )
    _agent_activity(trail, "s1")

    report = build_article50_report(trail._records, {"record_count": 0})
    agent = report["agent_disclosure_para31"]
    assert agent["total"] == 2
    assert agent["by_step"]["first_interaction"] == 1
    assert agent["by_step"]["validation"] == 1
    assert agent["named_principal"] == 2
    assert agent["carried_authority_ref"] == 1
    assert agent["sessions"]["s1"] == ["first_interaction", "validation"]

    md = render_article50_md(report)
    assert "guidance para 31" in md
    assert "first_interaction, validation" in md


def test_disclosures_default_allow_and_never_blocked(tmp_path):
    """A disclosure record must never be blocked by the gate itself."""
    trail = _persistent_trail(tmp_path)
    record_disclosure(
        trail, paragraph="50(4)", statement="This image is AI-generated.",
        subject="img-123",
    )
    blocked = [
        r for r in trail._records
        if r.tool_name == DISCLOSURE_TOOL
        and r.event_type.value == "action_blocked"
    ]
    assert blocked == []
