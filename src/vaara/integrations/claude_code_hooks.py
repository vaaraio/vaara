# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Claude Code hook runner: the plugin's governance logic, in the package.

``vaara hook pre-tool-use|post-tool-use|session-start`` reads the hook
event JSON on stdin and returns the plugin's exit-code contract (exit 2
blocks a PreToolUse call; anything else passes through). The Claude Code
plugin's hook entries shell out to the ``vaara`` binary on PATH, so
whatever installed the CLI — pip, pipx, Homebrew — is a complete engine
install. The historical failure mode this kills: the plugin's hooks ran
``python3``, and if the vaara package lived in a different interpreter
(brew's sealed virtualenv, a venv), governance silently never engaged.

Configuration is the plugin's: ``~/.vaara/claude-code/config.json`` plus
the same environment variables. Deny patterns resolve in order:
``--deny-patterns`` flag, ``VAARA_PLUGIN_DENY_PATTERNS_FILE``,
``$CLAUDE_PLUGIN_ROOT/policies/default_deny.json``, then the copy
bundled with the package.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path.home() / ".vaara" / "claude-code" / "config.json"

# ---------------------------------------------------------------------------
# config (mirrors the plugin's hooks/_config.py; the package is now the
# source of truth and the plugin shims to it)

def load_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def plugin_disabled(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_DISABLE") == "1":
        return True
    return cfg.get("mode") == "off"


def shadow_mode(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_SHADOW") == "1":
        return True
    return cfg.get("mode") == "watch"


def agent_id(cfg: dict) -> str:
    return os.environ.get("VAARA_PLUGIN_AGENT_ID") or cfg.get("agent_id") or "claude-code"


def audit_db_path(cfg: dict) -> Path:
    override = os.environ.get("VAARA_PLUGIN_AUDIT_DB") or cfg.get("audit_db")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "claude-code" / "audit.db"


def notifications_enabled(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_NOTIFY") == "0":
        return False
    return cfg.get("notifications", True) is not False


def fail_open(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_FAIL_OPEN") == "1":
        return True
    return cfg.get("fail_open") is True


def approvals_enabled(cfg: dict) -> bool:
    if os.environ.get("VAARA_PLUGIN_APPROVALS") == "0":
        return False
    return cfg.get("approvals", True) is not False


def approvals_dir(cfg: dict) -> Path:
    override = os.environ.get("VAARA_PLUGIN_APPROVALS_DIR") or cfg.get("approvals_dir")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "approvals"


def approvals_timeout(cfg: dict) -> float:
    raw = os.environ.get("VAARA_PLUGIN_APPROVALS_TIMEOUT") or cfg.get("approvals_timeout")
    try:
        timeout = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 60.0
    return timeout if timeout > 0 else 60.0


def protection_preset(cfg: dict) -> Optional[str]:
    preset = os.environ.get("VAARA_PLUGIN_PROTECTION") or cfg.get("protection")
    return preset if isinstance(preset, str) and preset else None


def article50_statement(cfg: dict) -> Optional[str]:
    statement = (
        os.environ.get("VAARA_PLUGIN_ARTICLE50_STATEMENT")
        or cfg.get("article50_statement")
    )
    return statement if isinstance(statement, str) and statement.strip() else None


def article50_on_behalf_of(cfg: dict) -> Optional[str]:
    """Principal the agent acts for (guidance C(2026) 5054 para 31).

    When set, the session-start disclosure upgrades from a generic 50(1)
    record to the agent-profile receipt naming the principal.
    """
    principal = (
        os.environ.get("VAARA_PLUGIN_ARTICLE50_ON_BEHALF_OF")
        or cfg.get("article50_on_behalf_of")
    )
    return principal if isinstance(principal, str) and principal.strip() else None


def custom_thresholds(cfg: dict) -> Optional[tuple[float, float]]:
    raw = cfg.get("thresholds")
    if not isinstance(raw, dict):
        return None
    escalate, deny = raw.get("escalate"), raw.get("deny")
    if not isinstance(escalate, (int, float)) or not isinstance(deny, (int, float)):
        return None
    if isinstance(escalate, bool) or isinstance(deny, bool):
        return None
    # Strictly below, as the policy schema requires. An equal pair used to
    # pass this check, fail in apply_policy, and take the preset down with
    # it, so the hook ran on the defaults while the operator's config named
    # strict. Treated as malformed here, the preset still applies.
    if not (0 <= escalate < deny <= 1):
        return None
    return float(escalate), float(deny)


# ---------------------------------------------------------------------------
# notify (fire-and-forget; must never break a hook)

def notify(cfg: dict, verdict: str, tool_name: str, detail: str) -> None:
    if not notifications_enabled(cfg):
        return
    try:
        clean = lambda text, limit: (  # noqa: E731
            text.replace('"', "'").replace("\\", "/").replace("\n", " ")[:limit]
        )
        title = clean(f"Vaara: {verdict}", 60)
        subtitle = clean(tool_name, 80)
        body = clean(detail, 180)
        if sys.platform == "darwin":
            script = (
                f'display notification "{body}" '
                f'with title "{title}" subtitle "{subtitle}" '
                f'sound name "Funk"'
            )
            cmd = ["osascript", "-e", script]
        elif shutil.which("notify-send"):
            cmd = ["notify-send", "--app-name=Vaara", f"{title} {subtitle}", body]
        else:
            return
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# deny patterns

from vaara.deny_rules import (  # noqa: E402
    deny_rules_path as _deny_rules_path,
    load_deny_rules as _load_deny_rules,
    match_deny_rule as _match_deny_rule,
)


def deny_patterns_path(explicit: Optional[str] = None) -> Optional[Path]:
    return _deny_rules_path(explicit)


def load_deny_rules(explicit: Optional[str] = None) -> list[dict]:
    return _load_deny_rules(explicit)


def match_deny_rule(
    rules: list[dict], tool_name: str, tool_input: dict
) -> Optional[tuple[str, str]]:
    return _match_deny_rule(rules, tool_name, tool_input)


# ---------------------------------------------------------------------------
# runners

def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_event() -> dict:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}
    return event if isinstance(event, dict) else {}


def _open_trail(cfg: dict):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    db_path = audit_db_path(cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteAuditBackend(db_path)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return trail


def _event_name(event_type: object) -> str:
    """Normalise an audit event type to its uppercase name.

    ``record.event_type`` is an ``EventType`` enum whose value is the
    lowercase string, so comparing it directly against
    ``"ACTION_REQUESTED"`` is always False.
    """
    return str(getattr(event_type, "value", event_type)).upper()


def _record_tool_name(record: object) -> str:
    """Tool name from the column when present, else from the payload."""
    direct = getattr(record, "tool_name", None)
    if direct:
        return str(direct)
    data = getattr(record, "data", None) or {}
    return str(data.get("tool_name", ""))


def _record_call(cfg: dict, agent: str, tool_name: str, tool_input: dict,
                 context: dict, session_id: str = "") -> None:
    """Write an ACTION_REQUESTED record for a call on the regex path.

    Every matched call is recorded, not only the denied ones. Recording
    denials alone left allowed shell, web and file calls out of the trail
    entirely, so it could answer "what was blocked" but not "what did the
    agent do", and PostToolUse then had no ACTION_REQUESTED to correlate
    its outcome against.

    ``enforce=False``: the verdict on this path belongs to the deny
    rules, and scoring here is for the record only. Roughly 5 ms to
    construct and under a millisecond per call, which is the SQLite
    append. The ML classifier stays on the ``mcp__*`` path.
    """
    try:
        from vaara.pipeline import InterceptionPipeline

        pipeline = InterceptionPipeline(trail=_open_trail(cfg), enforce=False)
        pipeline.intercept(
            agent_id=agent, tool_name=tool_name, parameters=tool_input,
            context=context, session_id=session_id,
        )
    except Exception as exc:
        _note_trail_failure(cfg, exc, stage="record_call")


def _note_trail_failure(cfg: dict, exc: BaseException, *, stage: str) -> None:
    """Say the trail is not recording, in those words, without blocking.

    The call still goes through: fail-open is the right default, because a
    governance hook that stops the session gets uninstalled and an
    uninstalled hook records nothing at all. What changes is that the
    operator finds out. Rate-limited by the marker, so a persistent outage
    stays one visible line a minute instead of the per-call traceback that
    hid this failure mode for thirteen days.
    """
    try:
        from vaara.audit.write_failure import failure_banner, record_failure

        state = record_failure(audit_db_path(cfg), exc, stage=stage)
        if state is None:
            _emit(f"vaara-governance: trail write failed ({exc!r}); NOT recording.")
            return
        if state.get("notify"):
            _emit(failure_banner(state))
            notify(
                cfg, "TRAIL NOT RECORDING", "audit trail",
                f"{state.get('count', '?')} failed writes since "
                f"{state.get('first_failure_utc', 'an unknown time')}",
            )
    except Exception:
        # Reporting a failure must never become the failure.
        pass


def _ungovernable(cfg: dict, tool_name: str, reason: str) -> int:
    """Verdict for an ``mcp__*`` call that cannot be scored or recorded.

    The posture is not new. A missing ``vaara`` package already fails
    closed on this path, with ``"fail_open": true`` as the documented
    escape hatch, because a gate that waves everything through when its
    engine is gone is not a gate. A trail that will not open is the same
    condition reached by a different route, and the backend says so in
    the exception it raises: the evidence chain is the product, so it
    refuses rather than starting a fresh trail with a silent gap. That
    refusal used to stop here, at a bare ``return 0``.

    Measured 2026-09-22: 162 failed writes in a seventeen-minute window,
    a loud marker on disk the whole time, and the calls in that window
    still ran unscored. The marker was never the missing part.

    Only the ``mcp__*`` path fails closed. Deny rules reach a verdict
    without the trail, so a broken trail on the regex path costs evidence
    and not enforcement; blocking every shell call there is how the hook
    gets uninstalled, and an uninstalled hook records nothing at all.
    """
    if shadow_mode(cfg) or fail_open(cfg):
        _emit(
            f"vaara-governance: {reason}; passing {tool_name} through UNSCORED "
            f"and UNRECORDED."
        )
        return 0
    _emit(
        f"vaara-governance: BLOCKED {tool_name} (fail-closed): {reason}, so "
        f"this MCP call cannot be scored or recorded. Repair the trail "
        f"(`vaara trail repair --db <db>` keeps every readable record and "
        f"declares the rest; keep the damaged file, it is the evidence), or set \"fail_open\": true in "
        f"~/.vaara/claude-code/config.json to pass through unscored."
    )
    notify(cfg, "BLOCKED", tool_name, f"cannot govern this call: {reason}")
    return 2


def run_pre_tool_use(deny_patterns: Optional[str] = None) -> int:
    """PreToolUse: exit 0 allows, exit 2 blocks or holds for review."""
    cfg = load_config()
    if plugin_disabled(cfg):
        return 0
    event = _read_event()
    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {}) or {}
    if not isinstance(tool_input, dict):
        tool_input = {"_raw": tool_input}
    session_id = event.get("session_id", "")
    agent = agent_id(cfg)
    shadow = shadow_mode(cfg)

    rules = load_deny_rules(deny_patterns)
    match = match_deny_rule(rules, tool_name, tool_input)
    matched_by = "tool"
    if match is None and tool_name.startswith("mcp__"):
        # An MCP server names its tools whatever it likes, so no rule's
        # tool list can name them and the by-name match above never fires.
        # The same payload a rule catches on Bash or Write reached the
        # classifier here with nothing in front of it, and the classifier
        # scores the tool-name taxonomy, not the payload (measured
        # 2026-09-22: an upload of .env to a remote host through
        # mcp__shell__run_command scored exactly as `ls`). Match by
        # content instead, the way the MCP proxy already does.
        from vaara.deny_rules import match_deny_rule_any_field

        try:
            match = match_deny_rule_any_field(rules, tool_input)
        except Exception as exc:  # a broken rule must not take the hook down
            _emit(f"vaara-governance: content deny rules failed ({exc!r}); skipping.")
            match = None
        matched_by = "content"
    if match is not None:
        rule_id, message = match
        _record_call(
            cfg, agent, tool_name, tool_input,
            {
                "vaara_governance_layer": "deny_pattern",
                "rule_id": rule_id, "rule_message": message,
                "matched_by": matched_by,
            },
            session_id,
        )
        if shadow:
            _emit(f"vaara-governance: SHADOW deny on {tool_name} (rule={rule_id}): {message}")
            notify(cfg, "SHADOW deny", tool_name, message)
            return 0
        _emit(f"vaara-governance: BLOCKED {tool_name} (rule={rule_id}). {message}")
        notify(cfg, "BLOCKED", tool_name, message)
        return 2

    if not tool_name.startswith("mcp__"):
        # Passed the deny rules. Record it anyway: a trail holding only
        # the blocked calls cannot answer "what did the agent do", which
        # is the question it exists for, and PostToolUse needs an
        # ACTION_REQUESTED to correlate its outcome against.
        _record_call(
            cfg, agent, tool_name, tool_input,
            {"vaara_governance_layer": "regex_pass"}, session_id,
        )
        return 0

    from vaara.pipeline import InterceptionPipeline

    try:
        trail = _open_trail(cfg)
    except Exception as exc:
        # A trail that will not open used to take the hook down with it, with
        # a traceback and no statement of what that meant. Record the outage
        # durably, say plainly that nothing is being recorded, and do not let
        # an MCP call through unscored on the strength of a warning.
        _note_trail_failure(cfg, exc, stage="open")
        return _ungovernable(cfg, tool_name, "the audit trail cannot be opened")
    pipeline = InterceptionPipeline(trail=trail, enforce=not shadow)

    preset = protection_preset(cfg)
    custom = custom_thresholds(cfg)
    if preset or custom:
        try:
            from vaara.policy import from_dict
            from vaara.policy.modes import get_mode, to_policy_dict

            policy = to_policy_dict(get_mode(preset or "balanced"))
            if custom:
                escalate, deny = custom
                policy["thresholds"]["default"] = {"escalate": escalate, "deny": deny}
            pipeline.scorer.apply_policy(from_dict(policy))
        except Exception as exc:
            _emit(
                f"vaara-governance: policy (preset={preset!r}, "
                f"custom_thresholds={custom!r}) not applied ({exc}); "
                f"using default thresholds."
            )

    try:
        result = pipeline.intercept(
            agent_id=agent, tool_name=tool_name,
            parameters=tool_input, session_id=session_id,
        )
    except Exception as exc:
        # The trail opened and the append failed underneath the scorer: the
        # shape of the 05:08 window on 2026-09-22, where `load_trail` read
        # the file and `intercept` raised on the write. Same outcome as a
        # trail that never opened, so the same verdict.
        _note_trail_failure(cfg, exc, stage="intercept")
        return _ungovernable(cfg, tool_name, f"the classifier failed ({exc!r})")

    if result.allowed:
        return 0

    # `allowed` is `decision == "allow"`, so escalate never reached the
    # branch above; the handshake below is what actually handles it.
    if result.decision == "escalate":
        return _handle_escalation(cfg, pipeline, result, tool_name)

    _emit(
        f"vaara-governance: BLOCKED {tool_name} "
        f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
        f"Reason: {result.reason}"
    )
    notify(cfg, "BLOCKED", tool_name, f"risk {result.risk_score:.2f}: {result.reason}")
    return 2


def _handle_escalation(cfg: dict, pipeline, result, tool_name: str) -> int:
    """Block on the file-based approval handshake for an escalated action.

    The approvals directory is watched by whatever surface fronts the
    human. Approve is the only way through:
    deny and timeout both keep the escalate fail-closed, so an unattended
    machine behaves exactly as before this handshake existed.
    """
    detail = f"risk {result.risk_score:.2f}: {result.reason}"
    if approvals_enabled(cfg):
        notify(cfg, "APPROVAL NEEDED", tool_name, detail)
        try:
            from vaara.approvals import request_approval

            human = request_approval(
                result.action_id, tool_name, detail,
                approvals_dir=approvals_dir(cfg),
                timeout=approvals_timeout(cfg),
            )
        except Exception as exc:
            _emit(f"vaara-governance: approval handshake failed ({exc!r}); "
                  f"treating as unanswered.")
            human = "timeout"
        if human in ("approve", "deny"):
            resolution = "allow" if human == "approve" else "deny"
            try:
                pipeline.resolve_escalation(
                    result.action_id, resolution,
                    reviewer="approvals-handshake",
                    justification="human decision via ~/.vaara/approvals",
                    # A person wrote the decision file. Only approve and deny
                    # reach here; timeout and an unreadable file both fall
                    # through without resolving, so this branch is the one
                    # place a human demonstrably acted.
                    approver="human",
                    human_disposed=True,
                )
            except Exception as exc:
                _emit(f"vaara-governance: could not record resolution ({exc!r}).")
        if human == "approve":
            _emit(
                f"vaara-governance: APPROVED {tool_name} by human "
                f"(action_id={result.action_id})."
            )
            return 0
        if human == "deny":
            _emit(
                f"vaara-governance: DENIED {tool_name} by human "
                f"(action_id={result.action_id})."
            )
            notify(cfg, "DENIED", tool_name, detail)
            return 2
    _emit(
        f"vaara-governance: ESCALATE {tool_name} blocked pending review "
        f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
        f"Reason: {result.reason}"
    )
    notify(cfg, "ESCALATE", tool_name, detail)
    return 2


def _outcome_severity(tool_response: object) -> float:
    if not isinstance(tool_response, dict):
        return 0.0
    if tool_response.get("interrupted") is True:
        return 1.0
    if tool_response.get("isError") is True:
        return 0.7
    stderr = tool_response.get("stderr") or ""
    if isinstance(stderr, str) and stderr.strip():
        return 0.3
    return 0.0


def run_post_tool_use() -> int:
    """PostToolUse: append the outcome, feed the online learner. Never blocks."""
    cfg = load_config()
    if plugin_disabled(cfg):
        return 0
    event = _read_event()
    tool_name = event.get("tool_name", "")
    severity = _outcome_severity(event.get("tool_response", {}))

    db_path = audit_db_path(cfg)
    if not db_path.exists():
        return 0
    try:
        trail = _open_trail(cfg)
        agent = agent_id(cfg)
        target_action_id = None
        for record in reversed(trail._records):
            if record.agent_id != agent:
                continue
            if _record_tool_name(record) != tool_name:
                continue
            if _event_name(record.event_type) == "ACTION_REQUESTED":
                target_action_id = record.action_id
                break
        if target_action_id is None:
            return 0
        from vaara.pipeline import InterceptionPipeline

        pipeline = InterceptionPipeline(trail=trail)
        pipeline._pending_outcomes[target_action_id] = (0.5, {})
        pipeline.report_outcome(target_action_id, outcome_severity=severity)
    except Exception as exc:
        _note_trail_failure(cfg, exc, stage="post_tool_use")
        return 0
    return 0


def run_session_start() -> int:
    """SessionStart: status line + optional Article 50(1) auto-disclosure."""
    import vaara

    cfg = load_config()
    if plugin_disabled(cfg):
        _emit("vaara-governance: off (config.json mode or VAARA_PLUGIN_DISABLE=1).")
        return 0
    event = _read_event()
    session_id = event.get("session_id", "")

    mode = "watch (nothing blocked, all recorded)" if shadow_mode(cfg) else "protect"
    preset = protection_preset(cfg) or "balanced"
    notif = "on" if notifications_enabled(cfg) else "off"
    db_path = audit_db_path(cfg)
    existed = db_path.exists()
    reported = False
    try:
        _open_trail(cfg)
        db_state = "existing" if existed else "created"
    except Exception as exc:
        db_state = f"unavailable ({exc!r})"
        _note_trail_failure(cfg, exc, stage="open")
        # That call already printed the banner and fired the notification for
        # this exact marker. Letting the health report print it again would
        # make session start say the same thing three times, which is the
        # noise this whole change exists to remove.
        reported = True

    disclosure = ""
    statement = article50_statement(cfg)
    if statement:
        try:
            principal = article50_on_behalf_of(cfg)
            if principal:
                from vaara.audit.article50 import record_agent_disclosure

                record_agent_disclosure(
                    _open_trail(cfg), statement=statement,
                    on_behalf_of=principal, step="first_interaction",
                    agent_id=agent_id(cfg), session_id=session_id,
                    channel="claude-code-session",
                )
            else:
                from vaara.audit.article50 import record_disclosure

                record_disclosure(
                    _open_trail(cfg), paragraph="50(1)", statement=statement,
                    agent_id=agent_id(cfg), session_id=session_id,
                    channel="claude-code-session",
                )
            status = "recorded"
        except Exception as exc:
            status = f"failed ({exc!r})"
        disclosure = f", article50_disclosure={status}"

    _emit(
        f"vaara-governance (vaara {vaara.__version__} engine, mode={mode}, "
        f"protection={preset}, notifications={notif}, audit_db={db_path} "
        f"[{db_state}]{disclosure}). Settings: /vaara-setup"
    )
    _report_hook_registration()
    if not reported:
        _report_trail_health(cfg, db_path, existed)
    return 0


def _report_hook_registration() -> None:
    """Say what is registered to govern this session, when it is not one layer.

    Silent on a healthy install. Session start is the only place this can
    be said: the registration is fixed for the whole session, and from
    inside a tool call neither a second layer nor a stale matcher is
    visible. A doubled decision looks like two ordinary actions, and a
    narrow matcher looks like a quiet day.
    """
    try:
        from vaara.integrations.hook_registration import inspect_registration

        for finding in inspect_registration():
            _emit(finding.render())
    except Exception:
        # Reporting the wiring must never become the reason a session fails.
        pass


def _report_trail_health(cfg: dict, db_path: Path, existed: bool) -> None:
    """Once per session, on the line the operator already reads.

    Three questions, in order of how bad the answer is. Has this trail been
    failing to record — the marker knows, across processes, which is the
    thing nothing knew on 2026-08-22. Does the file still read clean —
    which catches a trail damaged while nothing was writing to it, where
    the first symptom would otherwise be the next record nobody is
    watching. And is it running in a journal mode its filesystem can
    survive — the one that has not gone wrong yet.
    """
    try:
        from vaara.audit.sqlite_backend import journal_mode_warning
        from vaara.audit.write_failure import active_failure, failure_banner, quick_check

        state = active_failure(db_path)
        if state is not None:
            _emit(failure_banner(state))
            notify(
                cfg, "TRAIL NOT RECORDING", "audit trail",
                f"{state.get('count', '?')} failed writes since "
                f"{state.get('first_failure_utc', 'an unknown time')}",
            )
            return
        if not existed:
            return
        problem = quick_check(db_path)
        if problem is not None:
            _emit(
                f"vaara-governance: the audit trail at {db_path} does not read "
                f"clean ({problem}). Records may not be persisting. Repair it "
                f"with `vaara trail repair --db {db_path}`, which keeps every "
                "readable record and declares any it cannot keep. Do not "
                "delete the file, it is the evidence."
            )
            notify(cfg, "TRAIL DAMAGED", "audit trail", problem)
            return
        unsafe = journal_mode_warning(db_path)
        if unsafe is not None:
            _emit(
                f"vaara-governance: the audit trail at {db_path} is intact, but "
                f"{unsafe}. Vaara asks for DELETE journal mode on these mounts "
                "and the switch needs a brief exclusive lock it did not get. "
                "Close other Vaara processes and reopen the trail, or set "
                "VAARA_TRAIL_JOURNAL_MODE=delete. If that variable is set to "
                "wal, unset it."
            )
            notify(cfg, "TRAIL AT RISK", "audit trail", unsafe)
    except Exception:
        # Reporting on the trail's health must never break the session it is
        # reporting to. There is nowhere left to escalate to from here: the
        # thing that would carry the error is the trail itself.
        pass
