#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the ``decision_vocabulary_v0`` vectors from a real Vaara run.

The vectors are audit records Vaara actually produced, not records hand-written
to match a checker. Every scenario is driven through
``InterceptionPipeline.intercept`` with a stub scorer that returns one fixed
policy decision, and the resulting chain is exported verbatim.

Record ids and timestamps are the only things this script controls: ``uuid4``
and ``time.time`` are replaced with deterministic sequences so re-running
produces byte-identical fixtures. Everything else, including every record hash
and every chain link, is computed by Vaara exactly as it is in production. That
is what makes the committed hashes worth checking: an outside party recomputing
them is reproducing the real hashing path.

    python scripts/build_decision_vocabulary_vectors.py          # write vectors
    python scripts/build_decision_vocabulary_vectors.py --check  # fail on drift

``--check`` is what CI runs: it regenerates into a temporary directory and
compares, so an accidental change to the record shape shows up as a diff rather
than as silently updated evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "tests" / "vectors" / "decision_vocabulary_v0" / "cases"

# A fixed clock and a fixed id sequence. Both exist so the fixtures are
# reproducible; neither touches the hashing path itself.
EPOCH = 1_780_000_000.0
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-0000000000de")


class _FixedScorer:
    """Returns one fixed policy decision, whatever it is asked."""

    def __init__(self, action: str, **extra) -> None:
        self._payload = {
            "action": action,
            "reason": f"vector: policy returned {action}",
            "raw_result": {
                "point_estimate": 0.5,
                "conformal_interval": [0.4, 0.6],
                "signals": {},
            },
            **extra,
        }

    def evaluate(self, context):
        return dict(self._payload)


def _deterministic_env():
    """Patch the two sources of nondeterminism in the record path."""
    counter = {"n": 0}

    def _uuid4():
        counter["n"] += 1
        return uuid.uuid5(_NAMESPACE, str(counter["n"]))

    def _now():
        return EPOCH + counter["n"]

    return _uuid4, _now


def _build(out: Path) -> None:
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.pipeline import InterceptionPipeline

    _uuid4, _now = _deterministic_env()

    def new_pipeline(scorer):
        backend = SQLiteAuditBackend(":memory:")
        trail = backend.load_trail()
        return InterceptionPipeline(trail=trail, scorer=scorer)

    cases: dict[str, object] = {}

    with mock.patch("vaara.audit.trail.uuid.uuid4", _uuid4), \
            mock.patch("vaara.audit.trail.time.time", _now):

        for name, action in (
            ("allow_plain", "allow"),
            ("deny_plain", "deny"),
            ("escalate_plain", "escalate"),
            ("step_up_hold", "step_up"),
            ("defer_hold", "defer"),
        ):
            pipeline = new_pipeline(_FixedScorer(action))
            pipeline.intercept(
                agent_id="agent:vector", tool_name="tx.transfer",
                parameters={"amount": 50000, "to": "acct:9"},
            )
            cases[name] = pipeline.trail

        # The re-decision loop. A modify blocks the arguments it was asked
        # about and hands back the ones it will permit; the caller resubmits
        # and that resubmission is decided on its own.
        pipeline = new_pipeline(_FixedScorer(
            "modify", modified_parameters={"amount": 5000, "to": "acct:9"},
        ))
        first = pipeline.intercept(
            agent_id="agent:vector", tool_name="tx.transfer",
            parameters={"amount": 50000, "to": "acct:9"},
        )
        pipeline.scorer = _FixedScorer("allow")
        pipeline.intercept(
            agent_id="agent:vector", tool_name="tx.transfer",
            parameters=first.modified_parameters,
            parent_action_id=first.action_id,
        )
        cases["modify_then_retry"] = pipeline.trail

        # A policy that says modify and supplies nothing is malformed. The
        # action is blocked either way; what must not happen is a refinement
        # on the chain that the policy never supplied.
        pipeline = new_pipeline(_FixedScorer("modify"))
        pipeline.intercept(
            agent_id="agent:vector", tool_name="tx.transfer",
            parameters={"amount": 50000, "to": "acct:9"},
        )
        cases["modify_without_modification"] = pipeline.trail

        # record_decision is reachable from custom policy code. A refinement
        # that contradicts the decision it claims to explain is dropped.
        pipeline = new_pipeline(_FixedScorer("allow"))
        pipeline.trail.record_decision(
            action_id="vector-contradiction", agent_id="agent:vector",
            tool_name="tx.transfer", decision="allow",
            reason="vector: allow carrying a modify refinement",
            risk_score=0.1, decision_detail="modify",
            modified_parameters={"amount": 1},
        )
        cases["contradictory_refinement_dropped"] = pipeline.trail

    out.mkdir(parents=True, exist_ok=True)
    for name, trail in cases.items():
        # One flat file per case: that is what the aggregate runner counts, and
        # a suite that reports zero cases reads as empty coverage.
        # export_json is the regulator-facing export path, so the vectors are
        # the bytes a deployment would actually hand over.
        trail.export_json(out / f"{name}.json")

    _build_adversarial(out)


def _rechain(records: list) -> list:
    """Recompute every hash and link so the chain verifies again.

    Used only for the adversarial cases. Vaara's own ``compute_hash`` does the
    work, so these fixtures model an implementation that never had the guard
    rather than a chain someone tampered with after the fact. Without this the
    chain check would fire on every adversarial case and mask the rule each one
    exists to test.
    """
    from vaara.audit.trail import AuditRecord

    previous = ""
    out = []
    for raw in records:
        record = dict(raw)
        record["previous_hash"] = previous
        record["record_hash"] = ""
        rebuilt = AuditRecord.from_dict(record)
        record["record_hash"] = rebuilt.compute_hash()
        previous = record["record_hash"]
        out.append(record)
    return out


def _decision_index(records: list) -> int:
    for i, record in enumerate(records):
        if record["event_type"] in ("decision_made", "action_blocked"):
            return i
    raise AssertionError("no decision record in case")


def _build_adversarial(out: Path) -> None:
    """Cases no correct implementation produces, so the checker has teeth.

    Every one of these is what a gate WITHOUT the projection would write. They
    are committed as non-conforming so an outside party can confirm their own
    checker rejects them, rather than only confirming it accepts good records.
    """
    def load(name):
        return json.loads((out / f"{name}.json").read_text())

    def write(name, records):
        (out / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")

    # Route B, the road not taken: the refinement used as the verdict itself.
    # This is what renaming escalate to step_up would have put on the chain.
    records = load("step_up_hold")
    idx = _decision_index(records)
    records[idx]["data"]["decision"] = "step_up"
    records[idx]["data"].pop("decision_detail", None)
    write("verdict_outside_the_enum", _rechain(records))

    # A refinement that does not agree with the verdict beside it. `modify`
    # means the arguments do not run, so `allow` cannot stand next to it.
    records = load("modify_then_retry")
    idx = _decision_index(records)
    records[idx]["data"]["decision"] = "allow"
    write("refinement_contradicts_verdict", _rechain(records))

    # The failure the whole design exists to prevent: a retry that claims the
    # modify as its parent but carries the arguments the modify refused.
    records = load("modify_then_retry")
    for record in records:
        if (record["event_type"] == "action_requested"
                and record["data"].get("parent_action_id")):
            record["data"]["parameters"] = {"amount": 50000, "to": "acct:9"}
    write("retry_bound_to_other_arguments", _rechain(records))

    # A modify whose caller never retried. Legitimate, so this is advisory and
    # the case still conforms: a caller is allowed to give up.
    records = load("modify_then_retry")
    keep = records[0]["action_id"]
    write("modify_without_retry",
          _rechain([r for r in records if r["action_id"] == keep]))

    # And a plain tamper, left unchained, so the hash rule is exercised too.
    records = load("allow_plain")
    records[_decision_index(records)]["data"]["reason"] = "vector: edited after the fact"
    write("chain_break", records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temp dir and fail on any diff")
    args = parser.parse_args()

    if not args.check:
        _build(OUT)
        print(f"wrote {OUT}")
        return 0

    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "cases"
        _build(fresh)
        drift = []
        for path in sorted(fresh.rglob("*.json")):
            committed = OUT / path.relative_to(fresh)
            if not committed.is_file() or committed.read_bytes() != path.read_bytes():
                drift.append(str(path.relative_to(fresh)))
        for path in sorted(OUT.rglob("*.json")):
            if not (fresh / path.relative_to(OUT)).is_file():
                drift.append(f"{path.relative_to(OUT)} (committed but not generated)")
        shutil.rmtree(fresh, ignore_errors=True)

    if drift:
        print("decision_vocabulary_v0 vectors drifted:")
        for d in drift:
            print(f"  {d}")
        print("\nRegenerate with: python scripts/build_decision_vocabulary_vectors.py")
        return 1
    print("decision_vocabulary_v0 vectors match the committed bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
