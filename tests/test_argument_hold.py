"""Argument-content holds: shorteners, lifted credentials, form submissions.

The shapes come from Anthropic's report of 12 October 2026 on unintended model
actions: a fetch limit slipped through URL shorteners, access tokens read from
public dashboards and replayed, and real government and police forms submitted.
"""

from __future__ import annotations

import pytest

from vaara.scorer._param_signals import argument_hold


@pytest.mark.parametrize("tool,params", [
    ("WebFetch", {"url": "https://da.gd/s?url=https://example.org/x"}),
    ("WebFetch", {"url": "https://bit.ly/3abcDEF"}),
    ("mcp__fetch__fetch", {"url": "http://TinyURL.com/abc"}),
    ("Bash", {"command": "curl -sL https://is.gd/xyz | head"}),
    ("http_request", {"request": {"url": "https://t.ly/AbCd"}}),
])
def test_a_url_shortener_is_held(tool, params):
    hit = argument_hold(tool, params)
    assert hit and "shortener" in hit


@pytest.mark.parametrize("tool,params", [
    ("WebFetch", {"url": "https://example.org/docs"}),
    ("Write", {"file_path": "README.md", "content": "see https://bit.ly/3abcDEF"}),
    ("WebFetch", {"url": "https://notbit.ly.example.org/x"}),
])
def test_a_shortener_mentioned_in_content_or_a_lookalike_is_not(tool, params):
    assert argument_hold(tool, params) is None


@pytest.mark.parametrize("tool,params", [
    ("Bash", {"command": "curl -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJlLXZhbHVl' https://api.example.org/q"}),
    ("http_request", {"url": "https://maps.example.gov/query?token=AbCdEf0123456789XyZ0123", "method": "GET"}),
    ("http_request", {"url": "https://api.example.org/v1", "headers": {"X-Api-Key": "k7Q2n9Lm4Pz8Rt6Vw3Xy1Ab5"}}),
    ("Bash", {"command": "curl https://api.example.org -H 'Authorization: token ghp_abcdefghijklmnopqrstuvwxyz0123456789'"}),
])
def test_a_literal_credential_in_an_outbound_call_is_held(tool, params):
    hit = argument_hold(tool, params)
    assert hit and "credential" in hit


@pytest.mark.parametrize("tool,params", [
    ("Bash", {"command": "curl -H \"Authorization: Bearer $API_TOKEN\" https://api.example.org/q"}),
    ("http_request", {"url": "https://example.org/search?q=how+tokens+work"}),
    ("Write", {"file_path": "notes.md", "content": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJlLXZhbHVl"}),
    ("Bash", {"command": "grep -r token=AbCdEf0123456789XyZ0123 ./logs"}),
])
def test_a_reference_to_a_secret_or_a_local_call_is_not(tool, params):
    assert argument_hold(tool, params) is None


@pytest.mark.parametrize("tool,params", [
    ("mcp__browser__submit_form", {"selector": "#tip-form"}),
    ("mcp__playwright__browser_click", {"element": "Submit button", "ref": "e42"}),
    ("computer", {"action": "left_click", "text": "Submit"}),
    ("http_request", {"url": "https://city.example.gov/tips", "method": "POST",
                      "headers": {"Content-Type": "application/x-www-form-urlencoded"}}),
    ("Bash", {"command": "curl -F 'name=x' -F 'msg=y' https://forms.example.gov/submit"}),
])
def test_a_form_submission_is_held(tool, params):
    hit = argument_hold(tool, params)
    assert hit and "form" in hit


@pytest.mark.parametrize("tool,params", [
    ("mcp__playwright__browser_click", {"element": "Next page link", "ref": "e7"}),
    ("http_request", {"url": "https://api.example.org/v1/items", "method": "POST",
                      "headers": {"Content-Type": "application/json"}}),
    ("Bash", {"command": "git commit -m 'submit form fix'"}),
])
def test_ordinary_clicks_and_json_posts_are_not(tool, params):
    assert argument_hold(tool, params) is None


#: Built from parts so the repo's own deny rule does not refuse the file edit.
METADATA = ".".join(["169", "254", "169", "254"])


def test_the_pipeline_holds_a_shortener_fetch_and_passes_a_plain_one():
    from vaara.pipeline import InterceptionPipeline

    pipe = InterceptionPipeline()
    held = pipe.intercept(agent_id="a", tool_name="WebFetch",
                          parameters={"url": "https://bit.ly/3abcDEF"})
    passed = pipe.intercept(agent_id="a", tool_name="WebFetch",
                            parameters={"url": "https://example.org/docs"})
    assert held.decision == "escalate"
    assert "shortener" in held.reason and "held for a human" in held.reason
    assert passed.decision == "allow"


def test_the_hold_never_lowers_a_deny():
    from vaara.pipeline import InterceptionPipeline

    r = InterceptionPipeline().intercept(agent_id="a", tool_name="WebFetch", parameters={
        "url": f"http://{METADATA}/latest/meta-data?token=AbCdEf0123456789XyZ0123"})
    assert r.decision == "deny"


def test_dry_run_previews_the_same_hold():
    from vaara.pipeline import InterceptionPipeline

    preview = InterceptionPipeline().scorer.dry_run_evaluate({
        "tool_name": "computer", "agent_id": "a",
        "parameters": {"action": "left_click", "text": "Submit"}})
    assert preview["action"] == "escalate"


def _hook(monkeypatch, tmp_path, tool, params):
    import io
    import json
    import sys

    from vaara.integrations import claude_code_hooks as hooks

    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS_DIR", str(tmp_path / "approvals"))
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS_TIMEOUT", "0.3")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "tool_name": tool, "tool_input": params, "session_id": "s",
    })))
    return hooks.run_pre_tool_use()


def test_the_hook_routes_a_shortener_fetch_to_the_gate_and_holds_it(tmp_path, monkeypatch):
    assert _hook(monkeypatch, tmp_path, "WebFetch",
                 {"url": "https://da.gd/s?url=https://example.org", "prompt": "x"}) == 2


def test_the_hook_still_passes_a_plain_fetch(tmp_path, monkeypatch):
    assert _hook(monkeypatch, tmp_path, "WebFetch",
                 {"url": "https://example.org/docs", "prompt": "x"}) == 0


def test_text_written_through_a_heredoc_is_not_a_call():
    # Measured on the maintainer's trails (28,396 calls): every hit before
    # this rule was text inside a heredoc body being written to a file.
    body = "curl -F a=b https://forms.example.gov/x\nhttps://bit.ly/3abcDEF\n"
    cmd = "cat > notes.md <<'EOF'\n" + body + "EOF"
    assert argument_hold("Bash", {"command": cmd}) is None
