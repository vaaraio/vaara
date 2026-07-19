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
import re
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


def custom_thresholds(cfg: dict) -> Optional[tuple[float, float]]:
    raw = cfg.get("thresholds")
    if not isinstance(raw, dict):
        return None
    escalate, deny = raw.get("escalate"), raw.get("deny")
    if not isinstance(escalate, (int, float)) or not isinstance(deny, (int, float)):
        return None
    if isinstance(escalate, bool) or isinstance(deny, bool):
        return None
    if not (0 <= escalate <= deny <= 1):
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

def deny_patterns_path(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser()
    override = os.environ.get("VAARA_PLUGIN_DENY_PATTERNS_FILE")
    if override:
        return Path(override).expanduser()
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if plugin_root:
        candidate = Path(plugin_root) / "policies" / "default_deny.json"
        if candidate.exists():
            return candidate
    bundled = Path(__file__).parent / "claude_code_deny.json"
    return bundled if bundled.exists() else None


def load_deny_rules(explicit: Optional[str] = None) -> list[dict]:
    path = deny_patterns_path(explicit)
    if path is None or not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _emit(f"vaara-governance: deny-patterns load failed ({exc!r}); skipping layer 1.")
        return []
    return doc.get("rules", [])


def match_deny_rule(
    rules: list[dict], tool_name: str, tool_input: dict
) -> Optional[tuple[str, str]]:
    for rule in rules:
        if tool_name not in rule.get("tools", []):
            continue
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        for field in rule.get("fields", []):
            value = tool_input.get(field, "")
            if not isinstance(value, str):
                continue
            if regex.search(value):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None


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


def run_pre_tool_use(deny_patterns: Optional[str] = None) -> int:
    """PreToolUse: exit 0 allows/escalates, exit 2 blocks."""
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

    match = match_deny_rule(load_deny_rules(deny_patterns), tool_name, tool_input)
    if match is not None:
        rule_id, message = match
        try:
            from vaara.pipeline import InterceptionPipeline

            pipeline = InterceptionPipeline(trail=_open_trail(cfg), enforce=False)
            pipeline.intercept(
                agent_id=agent, tool_name=tool_name, parameters=tool_input,
                context={
                    "vaara_governance_layer": "deny_pattern",
                    "rule_id": rule_id, "rule_message": message,
                },
            )
        except Exception:
            pass
        if shadow:
            _emit(f"vaara-governance: SHADOW deny on {tool_name} (rule={rule_id}): {message}")
            notify(cfg, "SHADOW deny", tool_name, message)
            return 0
        _emit(f"vaara-governance: BLOCKED {tool_name} (rule={rule_id}). {message}")
        notify(cfg, "BLOCKED", tool_name, message)
        return 2

    if not tool_name.startswith("mcp__"):
        return 0

    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=_open_trail(cfg), enforce=not shadow)

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
        _emit(f"vaara-governance: classifier failed ({exc!r}); passing through.")
        return 0

    if result.allowed:
        if result.decision == "escalate":
            _emit(
                f"vaara-governance: ESCALATE on {tool_name} "
                f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
                f"Reason: {result.reason}"
            )
            notify(cfg, "ESCALATE", tool_name,
                   f"risk {result.risk_score:.2f}: {result.reason}")
        return 0

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
            if record.data.get("tool_name") != tool_name:
                continue
            if record.event_type == "ACTION_REQUESTED":
                target_action_id = record.action_id
                break
        if target_action_id is None:
            return 0
        from vaara.pipeline import InterceptionPipeline

        pipeline = InterceptionPipeline(trail=trail)
        pipeline._pending_outcomes[target_action_id] = (0.5, {})
        pipeline.report_outcome(target_action_id, outcome_severity=severity)
    except Exception:
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
    try:
        _open_trail(cfg)
        db_state = "existing" if existed else "created"
    except Exception as exc:
        db_state = f"unavailable ({exc!r})"

    disclosure = ""
    statement = article50_statement(cfg)
    if statement:
        try:
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
    return 0
