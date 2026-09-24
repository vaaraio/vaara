"""The llm-proxy refuses a tool call for an agent that has no Vaara hook.

The real Codex binary, with none of Vaara's hooks installed, talks to a
scripted model through ``vaara llm-proxy`` under enforcement. The model asks
for a harmless command and then for ``rm -rf /``. The proxy lets the first
through and takes the second out of the streamed reply, so Codex never gets
a call to run: it shows the refusal as the model's text and ends the turn.
This is the chokepoint that does not depend on a per-agent adapter.

Skipped unless a Codex binary is found, as in test_codex_real_binary.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")
uvicorn = pytest.importorskip("uvicorn", reason="proxy deps not installed: no uvicorn")

from tests.test_codex_real_binary import _CATALOG_ENTRY, CODEX, _Model  # noqa: E402
from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402

pytestmark = pytest.mark.skipif(not CODEX, reason="no Codex binary")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Proxy:
    def __init__(self, upstream: str, db: Path):
        self.pipeline = _build_pipeline(db)
        app = build_app(upstream=upstream, api_key=None, api_key_header="authorization",
                        pipeline=self.pipeline, enforce=True)
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error"))
        threading.Thread(target=self.server.run, daemon=True).start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("proxy did not start")

    def stop(self):
        self.server.should_exit = True


def test_the_proxy_refuses_a_tool_call_codex_has_no_hook_for(tmp_path):
    wipe = " ".join(["rm", "-rf", "/"])
    model = _Model([
        {"call": "exec_command", "args": {"cmd": "echo governed"}},
        {"call": "exec_command", "args": {"cmd": wipe}},
        {"say": "done"},
    ])
    proxy = _Proxy(f"http://127.0.0.1:{model.port}", tmp_path / "proxy.db")
    home, codex_home, work = tmp_path / "home", tmp_path / "codexhome", tmp_path / "work"
    for d in (home, codex_home, work):
        d.mkdir()
    catalog = codex_home / "catalog.json"
    catalog.write_text(json.dumps({"models": [_CATALOG_ENTRY]}))
    (codex_home / "config.toml").write_text(f"""\
model = "fake-model"
model_catalog_json = "{catalog}"
model_provider = "viaproxy"
approval_policy = "never"
sandbox_mode = "workspace-write"

[model_providers.viaproxy]
name = "viaproxy"
base_url = "http://127.0.0.1:{proxy.port}/v1"
wire_api = "responses"
env_key = "FAKE_KEY"
""")
    env = {"HOME": str(home), "CODEX_HOME": str(codex_home), "FAKE_KEY": "x",
           "PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(sys.path)}
    try:
        proc = subprocess.run(
            [CODEX, "exec", "--skip-git-repo-check", "-C", str(work), "go"],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
    finally:
        proxy.stop()
        model.server.shutdown()
    log = proc.stdout[-4000:] + proc.stderr[-3000:]

    assert not (codex_home / "hooks.json").exists()
    out = model.outputs()
    assert "call_0" in out, log                   # the allowed call ran
    assert "call_1" not in out, log               # the refused one never did
    assert "Vaara refused the tool call exec_command (rule rm_rf_root)" in log, log

    trail = proxy.pipeline.trail
    blocked = [r for r in trail.get_records_by_type(EventType.ACTION_BLOCKED)
               if r.tool_name == "llm.tool_call"]
    assert len(blocked) == 1, log
    calls = [r for r in trail.get_records_by_type(EventType.ACTION_REQUESTED)
             if r.tool_name == "llm.tool_call"]
    assert [r.data["parameters"]["tool"] for r in calls] == ["exec_command", "exec_command"]
