"""The escalation leg, driven end to end by the real Claude Code binary.

An MCP call the scorer escalates holds the hook on a signed approval in
``~/.vaara/approvals``. Here Claude Code, with the hooks ``vaara init``
writes and a small stdio MCP server, is asked by a scripted model for two
MCP calls. Thresholds put every MCP call on escalate. A thread plays the
operator: it approves the first request and denies the second, signed with
the approval key, the way the dashboard and the MCP approval tool do. The
test reads back which calls ran, what Claude Code told the model, and the
trail's escalation records.

Skipped unless a Claude Code binary is found, like the other real-binary
tests; CI runs it in the ``claude-code-e2e`` job.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.test_claude_code_real_binary import CLAUDE, _Model, _setup

pytest.importorskip("cryptography")
pytest.importorskip("rfc8785")
pytestmark = pytest.mark.skipif(not CLAUDE, reason="no Claude Code binary")

# A stdio MCP server with one tool, ``touch``, which creates a file in its
# working directory, so a call that ran leaves a file behind.
_SERVER = r'''
import json, sys
from pathlib import Path

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    req = json.loads(line)
    method, rid = req.get("method"), req.get("id")
    if rid is None:
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": req["params"].get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "e2e", "version": "1"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [{
            "name": "touch", "description": "Create a file.",
            "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}},
                            "required": ["name"]}}]}})
    elif method == "tools/call":
        name = req["params"]["arguments"]["name"]
        Path(name).write_text("ran\n")
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": "created " + name}]}})
    else:
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
'''


def _operator(approvals: Path, answers: list[str], stop: threading.Event,
              seen: list[dict]) -> threading.Thread:
    """Answer each approval request in arrival order, signed."""
    from vaara.approvals import write_decision

    def loop() -> None:
        while not stop.is_set():
            for req in sorted(approvals.glob("*.request.json"), key=lambda p: p.stat().st_mtime):
                try:
                    request = json.loads(req.read_text())
                except (OSError, ValueError):
                    continue
                if any(r.get("action_id") == request.get("action_id") for r in seen):
                    continue
                if len(seen) >= len(answers):
                    continue
                answer = answers[len(seen)]
                if write_decision(request["action_id"], answer, approvals_dir=approvals):
                    seen.append(dict(request, answered=answer))
            stop.wait(0.1)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def test_an_escalated_mcp_call_waits_for_a_signed_human_decision(tmp_path):
    model = _Model([
        {"tool": "mcp__e2e__touch", "input": {"name": "approved.txt"}},
        {"tool": "mcp__e2e__touch", "input": {"name": "denied.txt"}},
        {"text": "done"},
    ])
    stop = threading.Event()
    seen: list[dict] = []
    try:
        env, work, trail = _setup(tmp_path, model.port)
        home = Path(env["HOME"])
        config = home / ".vaara" / "claude-code" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({
            "audit_db": str(trail),
            # Every MCP call escalates, and none is denied outright.
            "thresholds": {"escalate": 0.0, "deny": 1.0},
            "approvals_timeout": 30,
        }))
        env["VAARA_PLUGIN_APPROVALS"] = "1"
        server = tmp_path / "e2e_server.py"
        server.write_text(_SERVER)
        mcp = tmp_path / "mcp.json"
        mcp.write_text(json.dumps({"mcpServers": {"e2e": {
            "type": "stdio", "command": sys.executable, "args": [str(server)]}}}))

        approvals = home / ".vaara" / "approvals"
        approvals.mkdir(parents=True, exist_ok=True)
        _operator(approvals, ["approve", "deny"], stop, seen)
        started = time.time()
        proc = subprocess.run(
            [CLAUDE, "-p", "go", "--output-format", "json",
             "--mcp-config", str(mcp), "--strict-mcp-config",
             "--allowedTools", "mcp__e2e__touch", "--model", "claude-sonnet-5"],
            cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=240)
    finally:
        stop.set()
        model.server.shutdown()
    log = proc.stdout[-3000:] + proc.stderr[-3000:]

    # Both calls were held for a person, in order, and each was answered well
    # inside the timeout, so neither verdict is a timeout.
    assert [r["answered"] for r in seen] == ["approve", "deny"], (seen, log)
    assert {r.get("tool_name") for r in seen} == {"mcp__e2e__touch"}, seen
    assert time.time() - started < 120, log

    out = model.outputs()
    assert len(out) == 2, (out, log)
    assert (work / "approved.txt").read_text() == "ran\n", (out, log)
    assert "created approved.txt" in out[0], (out, log)
    assert not (work / "denied.txt").exists(), (out, log)
    assert "DENIED mcp__e2e__touch by human" in out[1], (out, log)

    con = sqlite3.connect(trail)
    rows = con.execute(
        "SELECT action_id, event_type, data FROM audit_records "
        "WHERE tool_name = 'mcp__e2e__touch' ORDER BY seq").fetchall()
    by_action: dict[str, list[tuple[str, dict]]] = {}
    for action_id, kind, data in rows:
        by_action.setdefault(action_id, []).append((kind, json.loads(data or "{}")))
    assert list(by_action) == [r["action_id"] for r in seen], (list(by_action), seen)

    for request, want in zip(seen, ["allow", "deny"]):
        events = by_action[request["action_id"]]
        kinds = [k for k, _ in events]
        assert "escalation_sent" in kinds, kinds
        resolved = [d for k, d in events if k == "escalation_resolved"]
        assert len(resolved) == 1, events
        assert resolved[0].get("resolution") == want, resolved
        assert resolved[0].get("approver") == "human", resolved
