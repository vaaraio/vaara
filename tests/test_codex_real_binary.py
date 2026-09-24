"""Codex is governed: the real Codex binary, Vaara's hooks, the trail.

Every other Codex test feeds the runner payloads written from Codex's
documentation. This one runs the Codex CLI itself (``codex exec``) against a
scripted local model that asks for a shell command, a destructive command,
a patch that adds a file and a patch into Codex's own hook config. The hooks
are the ones ``install_hooks`` writes, trusted with the hash
``hook_hash`` computes, so a wrong hash shows up here as a hook Codex
silently skips. The test reads back what Codex told the model, which files
exist, and what the trail recorded.

Skipped unless a Codex binary is found: ``$VAARA_CODEX_BIN`` or ``codex`` on
PATH. CI installs a pinned version in the ``codex-e2e`` job.

Destructive strings are assembled at runtime so no governed session writing
this file trips the rules under test.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from vaara.integrations import codex

pytest.importorskip("cryptography")

CODEX = os.environ.get("VAARA_CODEX_BIN") or shutil.which("codex")
pytestmark = pytest.mark.skipif(not CODEX, reason="no Codex binary")


class _Model:
    """A Responses API endpoint that plays back one scripted turn per call."""

    def __init__(self, script: list[dict]):
        self.script, self.requests = script, []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send(200, "application/json", b'{"data":[],"models":[]}')

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                model.requests.append(json.loads(body or b"{}"))
                i = len(model.requests) - 1
                turn = model.script[i] if i < len(model.script) else {"say": "done"}
                self._send(200, "text/event-stream", model._events(i, turn))

            def _send(self, code, kind, data):
                self.send_response(code)
                self.send_header("content-type", kind)
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = HTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def _events(i: int, turn: dict) -> bytes:
        if "call" in turn:
            item = {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}",
                    "name": turn["call"], "arguments": json.dumps(turn["args"]),
                    "status": "completed"}
        elif "custom" in turn:
            item = {"type": "custom_tool_call", "id": f"ct_{i}", "call_id": f"call_{i}",
                    "name": turn["custom"], "input": turn["input"], "status": "completed"}
        else:
            item = {"type": "message", "id": f"msg_{i}", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": turn["say"],
                                 "annotations": []}]}
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                 "input_tokens_details": {"cached_tokens": 0},
                 "output_tokens_details": {"reasoning_tokens": 0}}
        out = b""
        for kind, obj in (
            ("response.created", {"response": {"id": f"resp_{i}"}}),
            ("response.output_item.done", {"output_index": 0, "item": item}),
            ("response.completed", {"response": {"id": f"resp_{i}", "output": [item],
                                                 "usage": usage}}),
        ):
            out += f"event: {kind}\ndata: {json.dumps({'type': kind, **obj})}\n\n".encode()
        return out

    def outputs(self) -> dict[str, str]:
        """What Codex sent back to the model for each call, by call id."""
        seen: dict[str, str] = {}
        for body in self.requests:
            for item in body.get("input", []):
                if str(item.get("type", "")).endswith("output"):
                    seen.setdefault(item["call_id"], str(item.get("output")))
        return seen


#: Codex's metadata for the scripted model, taken from Codex 0.156's own
#: catalog shape. Without it Codex falls back to metadata that offers no
#: ``apply_patch`` tool, and every patch fails before any hook runs.
_CATALOG_ENTRY = {
    "slug": "fake-model", "display_name": "fake-model", "description": "scripted",
    "default_reasoning_level": "medium",
    "supported_reasoning_levels": [{"effort": "medium", "description": "medium"}],
    "shell_type": "unified_exec", "visibility": "list", "supported_in_api": True,
    "priority": 12, "additional_speed_tiers": [], "service_tiers": [],
    "availability_nux": None, "upgrade": None,
    "model_messages": {"instructions_template": "Scripted test model.",
                       "instructions_variables": None, "approvals": None,
                       "collaboration_modes": None, "auto_review": None,
                       "permissions": None, "multi_agent": None},
    "include_skills_usage_instructions": False, "include_plugin_usage_instructions": False,
    "include_apps_usage_instructions": False, "default_reasoning_summary": "none",
    "support_verbosity": True, "default_verbosity": "low",
    "apply_patch_tool_type": "freeform", "web_search_tool_type": "text_and_image",
    "truncation_policy": {"mode": "tokens", "limit": 10000},
    "supports_image_detail_original": True, "context_window": 272000,
    "max_context_window": 272000, "comp_hash": "2911",
    "effective_context_window_percent": 95, "experimental_supported_tools": [],
    "input_modalities": ["text"], "supports_search_tool": False,
    "supports_experimental_context": False, "use_responses_lite": False,
    "node_repl_auto_review_required": False, "node_repl_disabled": False,
}


def _setup(tmp: Path, port: int) -> tuple[dict, Path, Path]:
    home, codex_home, work = tmp / "home", tmp / "codexhome", tmp / "work"
    for d in (home, codex_home, work):
        d.mkdir()
    shim = tmp / "bin" / "vaara"
    shim.parent.mkdir()
    shim.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" -c '
        '"import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))" "$@"\n')
    shim.chmod(0o755)

    codex.install_hooks(str(shim), codex_home)
    hooks = json.loads(codex.hooks_path(codex_home).read_text())["hooks"]
    path = codex.hooks_path(codex_home).resolve()
    trust = ""
    for event, label in (("PreToolUse", "pre_tool_use"), ("PostToolUse", "post_tool_use")):
        group = hooks[event][0]
        digest = codex.hook_hash(label, group.get("matcher"), group["hooks"][0])
        trust += f'\n[hooks.state."{path}:{label}:0:0"]\ntrusted_hash = "{digest}"\n'
    catalog = codex_home / "catalog.json"
    catalog.write_text(json.dumps({"models": [_CATALOG_ENTRY]}))
    (codex_home / "config.toml").write_text(f"""\
model = "fake-model"
model_catalog_json = "{catalog}"
model_provider = "fake"
approval_policy = "never"
sandbox_mode = "workspace-write"

[model_providers.fake]
name = "fake"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
env_key = "FAKE_KEY"
{trust}""")
    assert codex.trust_status(codex_home) == "trusted"

    env = {"HOME": str(home), "CODEX_HOME": str(codex_home), "FAKE_KEY": "x",
           "PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(sys.path),
           "VAARA_PLUGIN_SHADOW": "0", "VAARA_PLUGIN_APPROVALS": "0",
           "VAARA_PLUGIN_NOTIFY": "0"}
    return env, work, home / ".vaara" / "trail" / "audit.db"


def test_codex_calls_are_decided_by_vaara_and_recorded(tmp_path):
    root_wipe = " ".join(["rm", "-rf", "/"])
    hooks_file = str((tmp_path / "codexhome" / "hooks.json").resolve())
    model = _Model([
        {"call": "exec_command", "args": {"cmd": "echo governed"}},
        {"call": "exec_command", "args": {"cmd": root_wipe}},
        {"custom": "apply_patch",
         "input": "*** Begin Patch\n*** Add File: ok.txt\n+fine\n*** End Patch\n"},
        {"custom": "apply_patch",
         "input": f"*** Begin Patch\n*** Add File: note.txt\n+x\n"
                  f"*** Update File: {hooks_file}\n@@\n-a\n+b\n*** End Patch\n"},
        {"say": "done"},
    ])
    try:
        env, work, trail = _setup(tmp_path, model.port)
        proc = subprocess.run(
            [CODEX, "exec", "--skip-git-repo-check", "-C", str(work), "go"],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
    finally:
        model.server.shutdown()
    log = proc.stdout[-3000:] + proc.stderr[-3000:]

    out = model.outputs()
    # Allowed means Vaara's hook let it through to Codex. Whether the
    # command then runs is Codex's sandbox, which cannot start in some
    # containers (no user namespaces); either way no hook blocked it.
    assert "call_0" in out and "PreToolUse" not in out["call_0"], log
    assert "rm_rf_root" in out.get("call_1", ""), log
    assert "call_2" in out and "PreToolUse" not in out["call_2"], log
    if "bwrap: No permissions" not in log:
        assert (work / "ok.txt").exists(), log
    assert "harness_config_write" in out.get("call_3", ""), log
    assert not (work / "note.txt").exists(), "a blocked patch wrote part of itself"

    rows = sqlite3.connect(trail).execute(
        "SELECT agent_id, tool_name, event_type, data FROM audit_records "
        "WHERE event_type IN ('decision_made', 'action_blocked') ORDER BY seq").fetchall()
    verdicts = [(tool, kind, json.loads(data).get("decision")) for _a, tool, kind, data in rows]
    assert {a for a, *_ in rows} == {"codex"}
    assert verdicts == [
        ("Bash", "decision_made", "allow"),
        ("Bash", "action_blocked", "deny"),
        ("Write", "decision_made", "allow"),
        ("Write", "action_blocked", "deny"),
    ], log
