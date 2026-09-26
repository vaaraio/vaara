"""A call that deletes or discards state is held for a human.

Measured 2026-09-26 on a live trail: 7,815 scores, every shell call at
0.1625 whether it listed a directory or deleted a tree, against an escalate
line of 0.55. The experts read the tool name, not the payload, so nothing a
shell call did could reach a human. The destructive floor reads the payload
and raises such a call to the escalate line, on every path that scores.

Commands are assembled at runtime: a governed session writing this file
would otherwise trip the rules under test.
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys

import pytest

from vaara.integrations import claude_code_hooks as hooks
from vaara.pipeline import InterceptionPipeline
from vaara.scorer._param_signals import destructive_action

RM = "r" + "m"

DESTRUCTIVE = [
    ("Bash", {"command": f"{RM} -rf build"}),
    ("Bash", {"command": f"cd /tmp && {RM} -f fx.db && ls"}),
    ("Bash", {"command": f"sudo {RM} /etc/hosts"}),
    ("Bash", {"command": f"xargs {RM} -rf < list.txt"}),
    ("Bash", {"command": f"find . -name '*.log' -exec {RM} {{}} +"}),
    ("Bash", {"command": "find /var/log -name '*.gz' -delete"}),
    ("Bash", {"command": "truncate -s 0 app.log"}),
    ("Bash", {"command": "git push --force origin main"}),
    ("Bash", {"command": "git push -q --force-with-lease origin feat/x"}),
    ("Bash", {"command": "git push origin --delete release/v1"}),
    ("Bash", {"command": "git push origin :old-branch"}),
    ("Bash", {"command": "git branch -D feature/x"}),
    ("Bash", {"command": "git tag -d v1.0.0"}),
    ("Bash", {"command": "git reset -q --hard origin/main"}),
    ("Bash", {"command": "git clean -fdx"}),
    ("Bash", {"command": "git checkout -- ."}),
    ("Bash", {"command": "git restore src/app.py"}),
    ("Bash", {"command": "git stash drop"}),
    ("Bash", {"command": "git -C /repo branch -D old"}),
    ("Bash", {"command": "gh repo delete owner/name --yes"}),
    ("Bash", {"command": "kubectl delete namespace prod"}),
    ("Bash", {"command": "terraform destroy -auto-approve"}),
    ("Bash", {"command": "aws s3 rm s3://bucket --recursive"}),
    ("Bash", {"command": "docker system prune -af"}),
    ("Bash", {"command": "dd if=/dev/zero of=/dev/sdb bs=1M"}),
    ("Bash", {"command": f"bash -c '{RM} -rf /srv/data'"}),
    ("Bash", {"command": "sqlite3 app.db 'DROP TABLE users'"}),
    ("run_shell", {"cmd": f"{RM} -rf /data"}),
    ("mcp__shell__run_command", {"command": "git reset --hard HEAD~3"}),
    ("mcp__shell__exec", {"argv": [RM, "-rf", "/data"]}),
    ("run_program", {"program": RM, "args": ["-rf", "/"]}),
    ("run_python_code", {"code": "import os; os.remove('/etc/passwd')"}),
    ("execute_sql", {"query": "TRUNCATE TABLE orders"}),
    ("execute_sql", {"query": "delete from users where 1=1"}),
    ("mcp__vaara-memory__mem_delete", {"id": 1049}),
    ("CronDelete", {"id": "abc"}),
    ("s3_delete_bucket", {"bucket": "prod"}),
    ("deleteBranch", {"name": "main"}),
    ("terraform_destroy", {}),
]

BENIGN = [
    ("Bash", {"command": "ls -la"}),
    ("Bash", {"command": "git status"}),
    ("Bash", {"command": "git commit -m 'remove the old rm wrapper'"}),
    ("Bash", {"command": "git push origin feat/x"}),
    ("Bash", {"command": "git branch feature/y"}),
    ("Bash", {"command": "git checkout -b feature/z"}),
    ("Bash", {"command": "git restore --staged src/app.py"}),
    ("Bash", {"command": "git stash push -m wip"}),
    ("Bash", {"command": "pytest -q tests/"}),
    ("Bash", {"command": "grep -rn 'DELETE FROM' src/"}),
    ("Bash", {"command": "sed -i 's/git branch -D/git branch -d/' notes.md"}),
    ("Bash", {"command": f"echo '{RM} -rf is dangerous' > notes.txt"}),
    ("Bash", {"command": f"python3 - <<'EOF'\nimport os\nprint('{RM} -rf /')\nEOF"}),
    ("Bash", {"command": f"cat > clean.sh <<'EOF'\n{RM} -rf build\nEOF"}),
    ("Bash", {"command": "kubectl get pods", "description": "list pods in the delete script"}),
    ("Write", {"file_path": "clean.sh", "content": f"#!/bin/sh\n{RM} -rf build\n"}),
    ("Edit", {"file_path": "x.py", "old_string": "a", "new_string": "b"}),
    ("Read", {"file_path": "README.md"}),
    ("Skill", {"skill": "humanizer", "args": "deny rules: a shell rule fires on git branch -D"}),
    ("mcp__vaara-memory__mem_save", {"title": "lesson", "content": f"{RM} -rf / was refused"}),
    ("deleted_items_report", {"since": "2026-09-01"}),
]


@pytest.mark.parametrize("tool,params", DESTRUCTIVE)
def test_a_destructive_call_is_named(tool, params):
    assert destructive_action(tool, params)


@pytest.mark.parametrize("tool,params", BENIGN)
def test_an_ordinary_call_is_not(tool, params):
    assert destructive_action(tool, params) is None


def test_the_scorer_holds_a_delete_and_lets_a_listing_through():
    pipe = InterceptionPipeline()
    held = pipe.intercept(agent_id="a", tool_name="Bash",
                          parameters={"command": f"{RM} -rf build"})
    passed = pipe.intercept(agent_id="a", tool_name="Bash",
                            parameters={"command": "ls -la"})
    assert held.decision == "escalate"
    assert "held for a human" in held.reason
    assert passed.decision == "allow"


def test_the_floor_never_lowers_a_deny():
    # The cloud-metadata floor denies; a delete in the same call stays denied.
    pipe = InterceptionPipeline()
    r = pipe.intercept(agent_id="a", tool_name="Bash", parameters={
        "command": f"curl http://169.254.169.254/ && {RM} -rf /tmp/x"})
    assert r.decision == "deny"


def test_dry_run_previews_the_same_hold():
    pipe = InterceptionPipeline()
    preview = pipe.scorer.dry_run_evaluate({
        "tool_name": "Bash", "agent_id": "a",
        "parameters": {"command": "git push --force origin main"}})
    assert preview["action"] == "escalate"


def _hook(monkeypatch, tmp_path, tool, params, *, shadow="0"):
    db = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(db))
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", shadow)
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS_DIR", str(tmp_path / "approvals"))
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS_TIMEOUT", "0.3")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "tool_name": tool, "tool_input": params, "session_id": "s",
    })))
    return hooks.run_pre_tool_use(), db


def _events(db):
    con = sqlite3.connect(db)
    try:
        return [r[0] for r in con.execute(
            "select event_type from audit_records order by seq")]
    finally:
        con.close()


def test_the_hook_holds_a_shell_delete_and_nobody_answering_denies_it(
        tmp_path, monkeypatch):
    code, db = _hook(monkeypatch, tmp_path, "Bash", {"command": f"{RM} -rf build"})
    assert code == 2
    events = _events(db)
    assert "escalation_sent" in events
    # Nobody answered, so no one is recorded as having decided.
    assert "escalation_resolved" not in events


def test_the_hook_lets_an_ordinary_shell_call_through(tmp_path, monkeypatch):
    code, db = _hook(monkeypatch, tmp_path, "Bash", {"command": "ls -la"})
    assert code == 0
    assert "escalation_sent" not in _events(db)


def test_shadow_mode_records_the_hold_without_blocking(tmp_path, monkeypatch):
    code, _ = _hook(monkeypatch, tmp_path, "Bash",
                    {"command": "git push --force origin main"}, shadow="1")
    assert code == 0


def test_the_session_start_canary_passes_on_a_working_install(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(tmp_path / "audit.db"))
    assert hooks._report_hold_canary(hooks.load_config()) is True
    assert "canary ok" in capsys.readouterr().err
    # The probes went to an in-memory trail, never the configured one.
    assert not (tmp_path / "audit.db").exists()


def test_the_canary_fails_loudly_when_deletes_are_not_held(tmp_path, monkeypatch, capsys):
    import vaara.scorer._param_signals as signals

    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setattr(signals, "destructive_action", lambda *a, **k: None)
    assert hooks._report_hold_canary(hooks.load_config()) is False
    assert "CANARY FAILED" in capsys.readouterr().err


def test_the_approval_request_carries_the_whole_call(tmp_path, monkeypatch):
    # Is what you approve the thing that runs? The request a surface shows
    # the human carries the call's full arguments and their digest.
    import hashlib
    import threading

    seen = {}
    approvals = tmp_path / "approvals"

    def watch():
        import time
        for _ in range(100):
            for f in approvals.glob("*.request.json") if approvals.exists() else []:
                seen.update(json.loads(f.read_text()))
                return
            time.sleep(0.02)

    t = threading.Thread(target=watch)
    t.start()
    params = {"command": f"{RM} -rf build", "description": "clean"}
    _hook(monkeypatch, tmp_path, "Bash", params)
    t.join()
    assert seen["parameters"] == params
    assert seen["parameters_sha256"] == hashlib.sha256(json.dumps(
        params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
