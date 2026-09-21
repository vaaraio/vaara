"""The boundary red-team harness in conformance/redteam.

Pins the two things the harness must never get wrong: the control cases
(existing deny rules) are caught, and benign calls are not blocked. With the in-tree rules every forbidden case must be caught.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

_RUN = Path(__file__).parent.parent / "conformance" / "redteam" / "run.py"
_spec = importlib.util.spec_from_file_location("redteam_run", _RUN)
redteam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(redteam)


def test_matcher_is_fullmatch():
    ms = ["Bash|WebFetch|Task|mcp__.*"]
    assert redteam._covered(ms, "Task")
    assert redteam._covered(ms, "mcp__serena__find_symbol")
    assert not redteam._covered(ms, "TaskStop")
    assert not redteam._covered(ms, "Agent")


def test_settings_matcher_reads_only_the_vaara_hook(tmp_path):
    s = tmp_path / "settings.json"
    s.write_text(json.dumps({"hooks": {"PreToolUse": [
        {"matcher": "Bash", "hooks": [{"command": "rtk hook claude"}]},
        {"matcher": "Bash|WebFetch", "hooks": [{"command": "/x/vaara hook pre-tool-use"}]},
    ]}}))
    assert redteam._matchers_from_settings(s) == ["Bash|WebFetch"]


def test_a_source_that_misses_the_tool_is_unmediated():
    """Per-source, not any(): the live deployment gets its own verdict.

    The plugin ships `.*`, so any() over the sources was satisfied by the
    plugin for every case and a narrow settings.json matcher could never
    fail a run. That is how six tools named by deny rules scored CAUGHT on
    a box where the hook never sees them.
    """
    coverage = {"plugin": True, "settings": False}
    assert redteam._verdict("deny", coverage, "deny") == ("unmediated", "UNMEDIATED")


def test_covered_everywhere_and_denied_is_caught():
    coverage = {"plugin": True, "settings": True}
    assert redteam._verdict("deny", coverage, "deny") == ("deny", "CAUGHT")


def test_covered_everywhere_and_allowed_passes_the_fence():
    coverage = {"plugin": True, "settings": True}
    assert redteam._verdict("deny", coverage, "allow") == ("allow", "PASSED FENCE")


def test_benign_is_ok_whether_or_not_it_is_routed():
    """An allow-expected call that no matcher routes is simply not mediated.

    Nothing is being enforced on it, so there is no fence to fail.
    """
    assert redteam._verdict("allow", {"plugin": True}, "allow") == ("allow", "OK")
    assert redteam._verdict("allow", {"plugin": False}, "allow") == ("unmediated", "OK")


def test_benign_denied_is_a_false_positive():
    assert redteam._verdict("allow", {"plugin": True}, "deny") == ("deny", "FALSE POSITIVE")


def test_controls_caught_and_benign_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_BIN", "")
    monkeypatch.setattr(redteam, "shutil", type("S", (), {"which": staticmethod(lambda _: None)}))
    cases = redteam.CASES
    if not cases.exists():
        pytest.skip(
            "no case file: the harness ships without one. Point "
            "VAARA_REDTEAM_CASES at your own set to run this."
        )
    doc = json.loads(cases.read_text())
    matchers = {"plugin": redteam._matchers_from_plugin(redteam.PLUGIN_HOOKS)}
    rows = redteam.run(doc["cases"], matchers, tmp_path)
    by_id = {r["id"]: r for r in rows}
    for cid in (
        "egress_ssrf_metadata_control", "denied_etc_shadow_control",
        "denied_pipe_to_shell_control", "denied_ssh_key_write_control",
        "persist_bashrc_control", "persist_systemd_user_control",
    ):
        assert by_id[cid]["result"] == "CAUGHT", by_id[cid]
    assert [r["id"] for r in rows if r["result"] == "FALSE POSITIVE"] == []
    # With the in-tree rules nothing forbidden passes the fence.
    assert [r["id"] for r in rows if r["result"] == "PASSED FENCE"] == []
