# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CrewAI integration — Vaara as task-level and tool-level governance.

CrewAI orchestrates multi-agent crews where agents have roles, goals, and
tools.  Vaara integrates at two levels:

1. **Tool-level** — wraps individual tools with risk scoring (same as
   LangChain since CrewAI uses LangChain tools internally).

2. **Task-level** — a pre-execution hook that scores the entire task
   context (agent role + task description + tools) before the crew
   starts working.  This catches dangerous task assignments before
   any tool calls happen.

3. **Completeness-level** — ``VaaraGovernance`` registers CrewAI's
   ``before_tool_call`` / ``after_tool_call`` hooks and turns every tool
   call into a gap-evident authorization record. A per-run monotonic
   ``seq`` plus ``running_count`` ride inside a key-free, hash-chained
   Vaara receipt, so a dropped tool-call record is a provable gap from the
   held records alone. See ``VaaraGovernance`` below for the contract and
   wiring.

Integration:

    from crewai import Crew, Agent, Task
    from vaara.integrations.crewai import VaaraCrewGovernance

    pipeline = InterceptionPipeline()
    gov = VaaraCrewGovernance(pipeline)

    # Wrap tools
    safe_tools = gov.wrap_tools(agent.tools, agent_id="research-agent")

    # Or wrap entire crew execution
    result = gov.governed_kickoff(crew)

Requires: crewai >= 0.30 (not a hard dependency — imported at runtime)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypedDict
from uuid import uuid4

from vaara.audit.receipts import CommitPayload, OutcomePayload, Receipt
from vaara.credential._contiguity import ContiguityReport, verify_contiguity
from vaara.integrations.langchain import (
    ToolExecutionBlocked,
)
from vaara.pipeline import InterceptionPipeline

logger = logging.getLogger(__name__)


class VaaraCrewGovernance:
    """Governance layer for CrewAI crews.

    Provides tool wrapping, task-level pre-screening, and crew-level
    execution governance.
    """

    def __init__(
        self,
        pipeline: InterceptionPipeline,
        block_on_escalate: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.block_on_escalate = block_on_escalate

    def wrap_tools(
        self,
        tools: list[Any],
        agent_id: str = "crewai-agent",
    ) -> list[Any]:
        """Wrap a list of CrewAI/LangChain tools with Vaara interception.

        CrewAI uses LangChain tools internally, so we reuse the
        LangChain tool wrapper.
        """
        from vaara.integrations.langchain import vaara_wrap_tool

        wrapped = []
        for tool in tools:
            wrapped.append(
                vaara_wrap_tool(
                    tool, self.pipeline, agent_id=agent_id,
                    block_on_escalate=self.block_on_escalate,
                )
            )
        return wrapped

    def screen_task(
        self,
        task_description: str,
        agent_role: str,
        tools: list[str],
        agent_id: str = "crewai-agent",
    ) -> dict:
        """Pre-screen a task before execution.

        Scores the task's risk based on what tools it might use and
        the agent's role.  Returns a dict with the assessment.

        This is a heuristic pre-check — the real scoring happens when
        tools are actually called.  But catching obviously dangerous
        task assignments early saves compute and reduces attack surface.
        """
        # Read-only scoring: hitting pipeline.intercept here would pollute
        # the audit trail with hypothetical events and bias the sequence
        # detector (a pre-screen that checks read to export would trip the
        # data_exfiltration pattern before any real action). Use the
        # scorer directly so no audit record is written.
        assessments = []
        for tool_name in tools:
            action_type = self.pipeline.registry.classify(tool_name)
            context = {
                "tool_name": tool_name,
                "agent_id": agent_id,
                "base_risk_score": action_type.base_risk_score,
                "reversibility": action_type.reversibility.value,
                "blast_radius": action_type.blast_radius.value,
                "framework": "crewai",
                "task_description": task_description[:500],
                "agent_role": agent_role,
                "prescreen": True,
            }
            # dry_run_evaluate is read-only: no profile mutation, no
            # sequence-detector warnings, no global history append.
            if hasattr(self.pipeline.scorer, "dry_run_evaluate"):
                scorer_result = self.pipeline.scorer.dry_run_evaluate(context)
            else:
                scorer_result = self.pipeline.scorer.evaluate(context)
            raw = scorer_result.get("raw_result", {})
            decision = scorer_result.get("action", "escalate")
            risk_score = raw.get("point_estimate", 0.5)
            assessments.append({
                "tool": tool_name,
                "decision": decision,
                "risk_score": risk_score,
                "action_id": None,  # no audit record for pre-screen
            })

        max_risk = max(a["risk_score"] for a in assessments) if assessments else 0.0
        any_blocked = any(a["decision"] == "deny" for a in assessments)
        any(a["decision"] == "escalate" for a in assessments)

        return {
            "task_allowed": not any_blocked,
            "max_risk_score": max_risk,
            "blocked_tools": [a["tool"] for a in assessments if a["decision"] == "deny"],
            "escalated_tools": [a["tool"] for a in assessments if a["decision"] == "escalate"],
            "assessments": assessments,
        }

    def governed_kickoff(
        self,
        crew: Any,
        pre_screen: bool = True,
    ) -> Any:
        """Execute a CrewAI crew with Vaara governance.

        Optionally pre-screens all tasks, then wraps all agent tools.
        Falls back gracefully if CrewAI is not installed.

        Args:
            crew: A CrewAI Crew instance.
            pre_screen: If True, pre-screen all tasks before execution.

        Returns:
            The crew's kickoff result.

        Raises:
            ToolExecutionBlocked: If a task's tools are all blocked.
        """
        # Pre-screen tasks
        if pre_screen and hasattr(crew, "tasks"):
            for task in crew.tasks:
                agent = getattr(task, "agent", None)
                agent_id = getattr(agent, "role", "crewai-agent") if agent else "crewai-agent"
                # Pick task.tools first; fall back to agent.tools only when
                # task.tools is absent (None). Using truthiness (`or`) would
                # treat tools=[] (explicitly no tools allowed) as absent,
                # silently inheriting the agent's full tool set — a governance
                # bypass for tasks intentionally scoped with no tools.
                task_tools = getattr(task, "tools", None)
                source_tools = (
                    task_tools if task_tools is not None
                    else (getattr(agent, "tools", []) if agent else [])
                ) or []
                tool_names = [getattr(t, "name", str(t)) for t in source_tools]
                if tool_names:
                    screening = self.screen_task(
                        task_description=getattr(task, "description", ""),
                        agent_role=agent_id,
                        tools=tool_names,
                        agent_id=agent_id,
                    )
                    if not screening["task_allowed"]:
                        raise ToolExecutionBlocked(
                            tool_name=", ".join(screening["blocked_tools"]),
                            action_id=screening["assessments"][0]["action_id"] or "prescreen",
                            risk_score=screening["max_risk_score"],
                            risk_interval=(0.0, 1.0),
                            reason=f"Task pre-screening blocked: {screening['blocked_tools']}",
                        )

        # Wrap all agent tools
        if hasattr(crew, "agents"):
            for agent in crew.agents:
                if hasattr(agent, "tools") and agent.tools:
                    agent_id = getattr(agent, "role", "crewai-agent")
                    agent.tools = self.wrap_tools(agent.tools, agent_id=agent_id)

        return crew.kickoff()


# ---------------------------------------------------------------------------
# Completeness-level integration: gap-evident tool-call receipts via hooks.
# ---------------------------------------------------------------------------

_ALLOW = "allow"
_DEFAULT_BOUNDARY = "crew-run"
_V0_REASON = "recorded (v0 always-allow completeness mode)"
_RECEIPT_VERSION = "1.0"


class GovernanceDecision(TypedDict, total=False):
    """Pre-execution authorization record, mirroring crewAI PR #6030.

    Defined structurally so the adapter ships without importing
    ``crewai.governance`` (not in a released CrewAI yet). All keys are optional,
    matching the upstream ``total=False`` contract. ``seq`` / ``running_count``
    are surfaced at the top level on purpose: omission-detection only works if
    the completeness envelope is framework-neutral and uniform across providers,
    while the cryptographic evidence stays per-vendor under ``extensions``.
    """

    decision_id: str
    agent_id: str
    agent_role: str
    tool: str
    request_id: str
    params_hash: str
    decision: str
    reason: str
    issued_at: str
    seq: int
    running_count: int
    extensions: dict[str, Any]


class GovernanceOutcome(TypedDict, total=False):
    """Post-execution result record linked to a ``GovernanceDecision``."""

    decision_id: str
    outcome: str
    seq: int
    extensions: dict[str, Any]


class VaaraGovernance:
    """Records a contiguous, receipt-backed decision stream for a crew run.

    One instance per crew run (or per process, if you scope boundaries with
    ``boundary_id_for``). Thread-safe: CrewAI may execute tools concurrently, so
    the sequence counter and the ledgers are guarded by a single lock.

        from vaara.integrations.crewai import VaaraGovernance, register

        gov = VaaraGovernance()
        register(gov)                  # registers both tool-call hooks
        crew.kickoff(...)
        assert gov.verify_run().ok     # no tool-call record was dropped

    The boundary a sequence is scoped to is the crew run: by default the crew's
    id, overridable via ``boundary_id_for``. v0 runs in recording mode (every
    call allowed and recorded) and the before hook fails open, because a
    recorder must never halt the crew it observes.
    """

    def __init__(
        self,
        *,
        boundary_id_for: Optional[Callable[[Any], Optional[str]]] = None,
        threshold_allow: float = 0.4,
        threshold_deny: float = 0.7,
    ) -> None:
        self._boundary_id_for = boundary_id_for
        self._threshold_allow = threshold_allow
        self._threshold_deny = threshold_deny
        self._lock = threading.Lock()
        # Per-boundary monotonic sequence, gap-free by construction.
        self._counters: dict[str, int] = {}
        # Ledgers, keyed by boundary id.
        self._decisions: dict[str, list[GovernanceDecision]] = {}
        self._outcomes: dict[str, list[GovernanceOutcome]] = {}
        # Per-decision lookups for the after hook and receipt assembly.
        self._commits: dict[str, CommitPayload] = {}
        self._completeness_by_id: dict[str, dict[str, Any]] = {}
        self._receipts_by_id: dict[str, Receipt] = {}
        # FIFO pairing of before -> after by (boundary, tool, params_hash). The
        # decision sequence is the completeness proof and never depends on this;
        # pairing only attaches outcomes. A context-carried request id (or the
        # PR #6030 decision_id round-trip) would make it exact.
        self._pending: dict[tuple[str, str, str], deque[str]] = {}
        # Per-boundary sealing record, set by ``finalize_run``. Optional: a run
        # that is never finalized verifies exactly as before, minus tail-drop
        # detection.
        self._seals: dict[str, dict[str, Any]] = {}

    # -- hooks ---------------------------------------------------------------

    def before_tool_call(self, context: Any) -> None:
        """Record a pre-execution decision. Registerable as a before hook.

        Returns ``None`` (allow). Failures are logged and swallowed: a recorder
        must not block the call it observes.
        """
        try:
            boundary_id = self._boundary_for(context)
            tool_name = _opt_str(getattr(context, "tool_name", ""))
            params_hash = _params_hash(getattr(context, "tool_input", None) or {})
            agent = getattr(context, "agent", None)
            agent_id = _opt_str(getattr(agent, "id", ""))
            agent_role = _opt_str(getattr(agent, "role", ""))

            with self._lock:
                seq = self._counters.get(boundary_id, 0)
                self._counters[boundary_id] = seq + 1
                running_count = seq + 1
                decided_at = time.time()
                decision_id = f"{boundary_id}#{seq}"
                commit = CommitPayload(
                    action_id=decision_id,
                    decision=_ALLOW,
                    risk_score=0.0,
                    threshold_allow=self._threshold_allow,
                    threshold_deny=self._threshold_deny,
                    decided_at=decided_at,
                )
                completeness = {
                    "boundaryId": boundary_id,
                    "seq": seq,
                    "runningCount": running_count,
                }
                decision: GovernanceDecision = {
                    "decision_id": decision_id,
                    "agent_id": agent_id,
                    "agent_role": agent_role,
                    "tool": tool_name,
                    "request_id": uuid4().hex,
                    "params_hash": "sha256:" + params_hash,
                    "decision": _ALLOW,
                    "reason": _V0_REASON,
                    "issued_at": _iso8601(decided_at),
                    "seq": seq,
                    "running_count": running_count,
                    "extensions": {
                        "vaara": {
                            "receiptVersion": _RECEIPT_VERSION,
                            "commitHash": commit.hash(),
                            "completeness": completeness,
                        }
                    },
                }
                self._decisions.setdefault(boundary_id, []).append(decision)
                self._commits[decision_id] = commit
                self._completeness_by_id[decision_id] = completeness
                self._pending.setdefault(
                    (boundary_id, tool_name, params_hash), deque()
                ).append(decision_id)
        except Exception:
            logger.exception("Vaara before_tool_call hook failed; allowing call")
        return None

    def after_tool_call(self, context: Any) -> None:
        """Record a post-execution outcome paired to its decision.

        Registerable as an after hook. Returns ``None`` so the tool result is
        left untouched. Failures are logged and swallowed.
        """
        try:
            boundary_id = self._boundary_for(context)
            tool_name = _opt_str(getattr(context, "tool_name", ""))
            params_hash = _params_hash(getattr(context, "tool_input", None) or {})
            errored = _looks_errored(context)

            with self._lock:
                queue = self._pending.get((boundary_id, tool_name, params_hash))
                decision_id = queue.popleft() if queue else None
                if decision_id is None:
                    # No matching pre-record (hook registered mid-run, or the
                    # pairing heuristic missed). The decision sequence, which is
                    # the completeness proof, is unaffected; skip the outcome.
                    return None
                commit = self._commits.get(decision_id)
                completeness = self._completeness_by_id.get(decision_id, {})
                outcome_payload = OutcomePayload(
                    action_id=decision_id,
                    commit_hash=commit.hash() if commit is not None else "",
                    outcome_severity=1.0 if errored else 0.0,
                    recorded_at=time.time(),
                )
                outcome: GovernanceOutcome = {
                    "decision_id": decision_id,
                    "outcome": "error" if errored else "executed",
                    "seq": int(completeness.get("seq", -1)),
                    "extensions": {
                        "vaara": {
                            "outcomeHash": outcome_payload.hash(),
                            "completeness": completeness,
                        }
                    },
                }
                self._outcomes.setdefault(boundary_id, []).append(outcome)
                if commit is not None:
                    self._receipts_by_id[decision_id] = Receipt(
                        commit=commit, outcome=outcome_payload
                    )
        except Exception:
            logger.exception("Vaara after_tool_call hook failed")
        return None

    # -- reads ---------------------------------------------------------------

    def decisions(self, boundary_id: Optional[str] = None) -> list[GovernanceDecision]:
        with self._lock:
            if boundary_id is None:
                return [d for ds in self._decisions.values() for d in ds]
            return list(self._decisions.get(boundary_id, []))

    def outcomes(self, boundary_id: Optional[str] = None) -> list[GovernanceOutcome]:
        with self._lock:
            if boundary_id is None:
                return [o for outs in self._outcomes.values() for o in outs]
            return list(self._outcomes.get(boundary_id, []))

    def receipts(self, boundary_id: Optional[str] = None) -> list[Receipt]:
        """Key-free Vaara receipt pairs, for ``verify_receipt`` recomputation."""
        decisions = self.decisions(boundary_id)
        with self._lock:
            out: list[Receipt] = []
            for d in decisions:
                rid = d.get("decision_id")
                if rid is not None and rid in self._receipts_by_id:
                    out.append(self._receipts_by_id[rid])
            return out

    def verify_run(self, boundary_id: Optional[str] = None) -> ContiguityReport:
        """Check the held decision stream for gaps under one boundary.

        Builds the ``evidence`` shape ``verify_contiguity`` expects from the
        completeness blocks the decisions carry. With ``boundary_id`` omitted,
        the boundary is inferred when the records name exactly one (and
        ``verify_contiguity`` raises if they span more than one).
        """
        evidence_records = [
            {"completeness": d["extensions"]["vaara"]["completeness"]}
            for d in self.decisions(boundary_id)
            if "vaara" in d.get("extensions", {})
        ]
        with self._lock:
            if boundary_id is None:
                seals = list(self._seals.values())
            else:
                seal = self._seals.get(boundary_id)
                seals = [seal] if seal is not None else []
        for seal in seals:
            evidence_records.append({"completeness": dict(seal)})
        return verify_contiguity(evidence_records, boundary_id)

    def finalize_run(
        self,
        boundary_id: Optional[str] = None,
        max_class: Optional[str] = None,
    ) -> dict[str, Any]:
        """Seal a run: emit a terminal record pinning the final decision count.

        ``running_count`` is per-record (``seq + 1``), so a truncated stream
        still looks internally consistent and a dropped tail is invisible from
        the held set alone. A sealing record carries the boundary's final total;
        ``verify_run`` then flags a short set even when the missing records took
        their own ``seq`` with them.

        ``max_class`` optionally pins the highest action class the boundary
        authorized. It rides on the seal as ``maxClass`` so a reader bounds a
        gap's worst case (a dropped record could have authorized an action of at
        most this class) from the held set alone. Omitted, the seal is unchanged.

        The irreducible residual: a suffix drop that *also* removes this seal
        stays invisible from the held set alone. An external anchor (an rfc3161
        timestamp minted over the run) closes that, and is recorded separately.

        Returns the sealing completeness block so a holder can carry it beside
        the decision stream. Re-sealing updates the total to the current count.
        """
        with self._lock:
            if boundary_id is None:
                known = list(self._counters)
                if len(known) == 1:
                    boundary_id = known[0]
                elif not known:
                    boundary_id = _DEFAULT_BOUNDARY
                else:
                    raise ValueError(
                        "finalize_run needs an explicit boundary_id when the "
                        "recorder spans more than one boundary"
                    )
            total = self._counters.get(boundary_id, 0)
            seal: dict[str, Any] = {
                "boundaryId": boundary_id,
                "sealed": True,
                "total": total,
            }
            if max_class is not None:
                seal["maxClass"] = str(max_class)
            self._seals[boundary_id] = seal
            return dict(seal)

    # -- internals -----------------------------------------------------------

    def _boundary_for(self, context: Any) -> str:
        if self._boundary_id_for is not None:
            try:
                bid = self._boundary_id_for(context)
            except Exception:
                logger.exception("boundary_id_for callable raised; using default")
                bid = None
            if bid:
                return str(bid)
        crew = getattr(context, "crew", None)
        cid = getattr(crew, "id", None)
        if cid:
            return str(cid)
        return _DEFAULT_BOUNDARY


def register(
    governance: VaaraGovernance, *, before: bool = True, after: bool = True
) -> None:
    """Register a ``VaaraGovernance`` instance's hooks with CrewAI.

    Thin convenience over ``crewai.hooks``. Raises ``ImportError`` with an
    install hint if CrewAI is not present.
    """
    try:
        from crewai.hooks import (
            register_after_tool_call_hook,
            register_before_tool_call_hook,
        )
    except ImportError as exc:
        raise ImportError(
            "vaara.integrations.crewai.register requires CrewAI. "
            'Install with: pip install "vaara[crewai]"'
        ) from exc
    if before:
        register_before_tool_call_hook(governance.before_tool_call)
    if after:
        register_after_tool_call_hook(governance.after_tool_call)


def _params_hash(tool_input: Any) -> str:
    """SHA-256 over the canonical JSON of the tool input.

    Uses the same canonicalization as Vaara receipts (sorted keys, compact
    separators, no NaN). Non-serializable inputs fall back to their ``repr`` so
    the hook never raises on an exotic argument.
    """
    try:
        canonical = json.dumps(
            tool_input, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        canonical = json.dumps(repr(tool_input), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _looks_errored(context: Any) -> bool:
    """Best-effort: a raw tool result that is an exception means the call failed."""
    raw = getattr(context, "raw_tool_result", None)
    return isinstance(raw, BaseException)


def _opt_str(value: Any) -> str:
    return str(value) if value else ""


def _iso8601(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
