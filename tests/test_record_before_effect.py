# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An allowed call whose record did not land does not run.

Measured 2026-10-24 on e2f75440: with the audit store refusing writes,
``InterceptionPipeline.intercept`` counted the failures, returned its
verdict, and an allowed call ran. Two calls, six refused writes, both
``allowed=True``, and the chain on disk stayed contiguous, so nothing in it
said a call had gone unrecorded. The LLM API proxy already refused in that
case (``test_llm_proxy_fail_closed.py``). These tests pin the same posture
on the MCP proxy, the shell proxy and the Claude Code hook's ``mcp__*`` path,
with ``fail_open`` as the stated way to run anyway.

The store is made to fail by replacing the backend's append, as in the LLM
proxy tests. The pipeline, the trail and the SQLite backend are real; only
the upstream MCP server is a stub.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend, TrailRepair
from vaara.integrations import claude_code_hooks as hooks
from vaara.integrations import mcp_proxy, shell_proxy
from vaara.pipeline import InterceptionPipeline


def _pipeline(tmp_path: Path, *, enforce: bool = True) -> InterceptionPipeline:
    trail = SQLiteAuditBackend(str(tmp_path / "trail.db")).load_trail()
    return InterceptionPipeline(trail=trail, enforce=enforce)


def _break_store(monkeypatch, pipeline, *, repair_result=None, heals=False):
    """Make every append fail; ``repair`` returns ``repair_result``.

    With ``heals`` the store works again once repair has run, the way a
    REINDEX clears index damage underneath a live connection.
    """
    backend = pipeline.trail._chain_backend()
    real_append = backend.append_record
    state = {"broken": True, "repairs": 0}

    def append(record, stamp):
        if state["broken"]:
            raise OSError("disk I/O error")
        return real_append(record, stamp)

    def repair():
        state["repairs"] += 1
        if heals:
            state["broken"] = False
        return repair_result

    monkeypatch.setattr(backend, "append_record", append)
    monkeypatch.setattr(backend, "repair", repair)
    return state


# --- the MCP proxy ---------------------------------------------------------

READ_CALL = {
    "jsonrpc": "2.0", "id": 5, "method": "tools/call",
    "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
}
UPSTREAM_REPLY = {
    "jsonrpc": "2.0", "id": 5, "result": {"content": [{"type": "text", "text": "ok"}]},
}


def _proxy(monkeypatch, pipeline, **kwargs):
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    proxy = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"], pipeline=pipeline, deny_rules=[], **kwargs)
    proxy._upstream = MagicMock()
    proxy._upstream.request.return_value = UPSTREAM_REPLY
    return proxy


def test_the_mcp_proxy_runs_a_call_it_could_record(monkeypatch, tmp_path):
    proxy = _proxy(monkeypatch, _pipeline(tmp_path))
    assert proxy._handle_tools_call(dict(READ_CALL)) is UPSTREAM_REPLY
    proxy._upstream.request.assert_called_once()


def test_the_mcp_proxy_does_not_forward_a_call_it_could_not_record(monkeypatch, tmp_path):
    pipeline = _pipeline(tmp_path)
    state = _break_store(monkeypatch, pipeline, repair_result=TrailRepair(
        db="x", method="failed", error="database disk image is malformed"))
    proxy = _proxy(monkeypatch, pipeline)

    response = proxy._handle_tools_call(dict(READ_CALL))

    assert response["id"] == 5
    assert response["result"]["isError"] is True
    body = json.loads(response["result"]["content"][0]["text"])
    assert body["vaara_blocked"] is True
    assert body["type"] == "vaara_trail_not_recording"
    assert "vaara trail repair" in body["reason"]
    assert "record_before_effect:refuse" in body["gates"]
    proxy._upstream.request.assert_not_called()
    assert state["repairs"] == 1


def test_a_repair_that_works_lets_the_mcp_call_through_recorded(monkeypatch, tmp_path):
    pipeline = _pipeline(tmp_path)
    _break_store(monkeypatch, pipeline, heals=True,
                 repair_result=TrailRepair(db="x", method="reindex"))
    proxy = _proxy(monkeypatch, pipeline)

    assert proxy._handle_tools_call(dict(READ_CALL)) is UPSTREAM_REPLY
    proxy._upstream.request.assert_called_once()


def test_mcp_fail_open_forwards_the_call_unrecorded(monkeypatch, tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    _break_store(monkeypatch, pipeline)
    proxy = _proxy(monkeypatch, pipeline, fail_open=True)

    assert proxy._handle_tools_call(dict(READ_CALL)) is UPSTREAM_REPLY
    proxy._upstream.request.assert_called_once()
    assert "UNRECORDED" in caplog.text


def test_a_shadow_mcp_proxy_keeps_forwarding_and_says_so(monkeypatch, tmp_path, caplog):
    """Shadow mode blocks nothing by design, so it does not start here."""
    pipeline = _pipeline(tmp_path, enforce=False)
    _break_store(monkeypatch, pipeline)
    proxy = _proxy(monkeypatch, pipeline)

    assert proxy._handle_tools_call(dict(READ_CALL)) is UPSTREAM_REPLY
    assert "UNRECORDED" in caplog.text


def test_the_mcp_proxy_cli_has_a_fail_open_flag(monkeypatch):
    captured = {}

    class _Stop(Exception):
        pass

    def fake_proxy(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(mcp_proxy, "VaaraMCPProxy", fake_proxy)
    for argv, expected in ((["--upstream", "echo"], False),
                           (["--upstream", "echo", "--fail-open"], True)):
        with pytest.raises(_Stop):
            mcp_proxy.main(argv)
        assert captured["fail_open"] is expected


# --- the shell proxy -------------------------------------------------------

def test_the_shell_proxy_does_not_run_a_command_it_could_not_record(
        monkeypatch, tmp_path, capsys):
    pipeline = _pipeline(tmp_path)
    _break_store(monkeypatch, pipeline)
    ran = tmp_path / "ran"

    code = shell_proxy._run_one(pipeline, f"touch {ran}", "/bin/sh", "t")

    assert code == 1
    assert not ran.exists()
    out = capsys.readouterr()
    assert "not recording" in (out.out + out.err)
    assert "--fail-open" in (out.out + out.err)


@pytest.mark.skipif(sys.platform == "win32", reason="the shell proxy runs /bin/sh")
def test_the_shell_proxy_runs_a_command_it_could_record(tmp_path):
    ran = tmp_path / "ran"
    assert shell_proxy._run_one(_pipeline(tmp_path), f"touch {ran}", "/bin/sh", "t") == 0
    assert ran.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the shell proxy runs /bin/sh")
def test_shell_fail_open_runs_the_command_unrecorded(monkeypatch, tmp_path, capsys):
    pipeline = _pipeline(tmp_path)
    _break_store(monkeypatch, pipeline)
    ran = tmp_path / "ran"

    code = shell_proxy._run_one(
        pipeline, f"touch {ran}", "/bin/sh", "t", fail_open=True)

    assert code == 0
    assert ran.exists()
    out = capsys.readouterr()
    assert "UNRECORDED" in (out.out + out.err)


def test_the_shell_proxy_cli_passes_fail_open_through(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(shell_proxy, "run_shell_proxy", fake_run)
    shell_proxy.main([])
    assert seen["fail_open"] is False
    shell_proxy.main(["--fail-open"])
    assert seen["fail_open"] is True


# --- the Claude Code hook --------------------------------------------------

MCP_EVENT = {
    "tool_name": "mcp__demo__write_file",
    "tool_input": {"path": "/tmp/x", "content": "hello"},
    "session_id": "s1",
}


@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    """The hook on a healthy trail, cut off from the developer's own config."""
    db = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(db))
    monkeypatch.setattr(hooks, "CONFIG_PATH", tmp_path / "absent-config.json")
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS", "0")
    monkeypatch.setenv("VAARA_PLUGIN_NOTIFY", "0")
    monkeypatch.delenv("VAARA_PLUGIN_FAIL_OPEN", raising=False)
    monkeypatch.delenv("VAARA_PLUGIN_DISABLE", raising=False)
    return db


def _feed(monkeypatch, event: dict) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))


@pytest.fixture
def store_refuses_writes(hook_env, monkeypatch):
    """The trail opens, then every append is refused underneath the scorer."""
    def refuse(self, record, stamp):
        raise OSError("disk I/O error")

    monkeypatch.setattr(SQLiteAuditBackend, "append_record", refuse)


def test_the_hook_blocks_an_mcp_call_it_could_not_record(
        store_refuses_writes, monkeypatch, capsys):
    _feed(monkeypatch, MCP_EVENT)
    assert hooks.run_pre_tool_use() == 2
    err = capsys.readouterr().err
    assert "fail-closed" in err
    assert "mcp__demo__write_file" in err
    assert "not recorded" in err


def test_hook_fail_open_passes_an_unrecorded_mcp_call(
        store_refuses_writes, monkeypatch, capsys):
    monkeypatch.setenv("VAARA_PLUGIN_FAIL_OPEN", "1")
    _feed(monkeypatch, MCP_EVENT)
    assert hooks.run_pre_tool_use() == 0
    err = capsys.readouterr().err
    assert "UNRECORDED" in err
    assert "UNSCORED" not in err, "the call was scored; only the record is missing"


def test_hook_shadow_mode_passes_an_unrecorded_mcp_call(
        store_refuses_writes, monkeypatch):
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "1")
    _feed(monkeypatch, MCP_EVENT)
    assert hooks.run_pre_tool_use() == 0


def test_the_hook_runs_an_mcp_call_it_could_record(hook_env, monkeypatch):
    _feed(monkeypatch, MCP_EVENT)
    assert hooks.run_pre_tool_use() == 0


def test_the_hook_keeps_the_regex_path_fail_open(store_refuses_writes, monkeypatch):
    """Deny rules reach a verdict without the trail; a hook that blocks every
    shell call gets uninstalled and an uninstalled hook records nothing."""
    _feed(monkeypatch, {
        "tool_name": "Bash", "tool_input": {"command": "ls -la"}, "session_id": "s1",
    })
    assert hooks.run_pre_tool_use() == 0
