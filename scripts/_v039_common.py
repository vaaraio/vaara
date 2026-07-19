# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers for the v0.39 LLM-agent harness.

v0.32-v0.38 scored adversarial entries already shaped like tool calls.
v0.39 scores tool calls that an LLM agent emitted on its own in response
to a prompt-injection corpus (BIPIA first, LLMail-Inject and IPI-4th
next). The harness runs the agent, captures whatever it emits; the eval
script then shapes those captures into classifier entries and scores
with v8.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = REPO / "tests/adversarial/external"
TRACES_DIR = REPO / "tests/adversarial/traces"


@dataclasses.dataclass
class BipiaTrial:
    trial_id: str
    benign_topic: str
    benign_task: str
    attack_class: str
    injection_text: str

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(self.benign_task.encode())
        h.update(b"\x00")
        h.update(self.injection_text.encode())
        return h.hexdigest()[:16]


@dataclasses.dataclass
class ToolCallEmit:
    tool_name: str
    parameters: dict
    reflects_injection: bool


@dataclasses.dataclass
class HarnessTrace:
    trial_id: str
    trial_fingerprint: str
    benign_topic: str
    attack_class: str
    model: str
    agent_text: str
    tool_calls: list[ToolCallEmit]
    latency_ms: float
    error: Optional[str] = None


HTTP_REQUEST_TOOL = {
    "name": "http_request",
    "description": "Issue an HTTP request. The agent decides method, url, and body.",
    "input_schema": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
            "url": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["method", "url"],
    },
}

FILE_WRITE_TOOL = {
    "name": "file_write",
    "description": "Write text content to a file at the given path.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}

SHELL_EXEC_TOOL = {
    "name": "shell_exec",
    "description": "Execute a shell command and return stdout.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

DEFAULT_TOOLS = [HTTP_REQUEST_TOOL, FILE_WRITE_TOOL, SHELL_EXEC_TOOL]

BIPIA_BENIGN_TOPICS = {
    "Task Automation",
    "Business Intelligence",
    "Conversational Agent",
    "Research Assistance",
    "Sentiment Analysis",
}


def load_bipia(
    n_benign: int = 25,
    n_attack: int = 50,
    seed: int = 43,
) -> list[BipiaTrial]:
    """Pair BIPIA benign user tasks with code-attack injection payloads.

    BIPIA ships two attack-corpus files. `text_attack_test.json` mixes
    benign user tasks (under topics like Task Automation, Business
    Intelligence) with text-style attack instructions (Substitution
    Ciphers, Marketing & Advertising, Scams & Fraud). Only the first
    five topics are benign-shaped; the other ten are themselves attack
    instructions. `code_attack_test.json` is uniformly code-injection
    payloads keyed by attack class.

    Pool sizes in the vendored mirror are 25 benign tasks (5 topics x
    5 each) and 50 code-attack payloads (10 classes x 5 each), so the
    full cross-product is 1,250 trials per seed.
    """
    rng = random.Random(seed)
    benign_raw = json.loads(
        (EXTERNAL_DIR / "bipia" / "text_attack_test.json").read_text()
    )
    benigns: list[tuple[str, str]] = []
    for topic, items in benign_raw.items():
        if topic not in BIPIA_BENIGN_TOPICS:
            continue
        for q in items:
            benigns.append((topic, q))

    attack_raw = json.loads(
        (EXTERNAL_DIR / "bipia" / "code_attack_test.json").read_text()
    )
    attacks: list[tuple[str, str]] = []
    for klass, items in attack_raw.items():
        for inj in items:
            attacks.append((klass, inj))

    rng.shuffle(benigns)
    rng.shuffle(attacks)
    benigns = benigns[:n_benign]
    attacks = attacks[:n_attack]

    trials: list[BipiaTrial] = []
    for i, (topic, btask) in enumerate(benigns):
        for j, (aklass, atext) in enumerate(attacks):
            trials.append(
                BipiaTrial(
                    trial_id=f"bipia-s{seed}-{i:04d}-{j:04d}",
                    benign_topic=topic,
                    benign_task=btask,
                    attack_class=aklass,
                    injection_text=atext,
                )
            )
    return trials


def write_trace(trace: HarnessTrace, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": trace.trial_id,
        "trial_fingerprint": trace.trial_fingerprint,
        "benign_topic": trace.benign_topic,
        "attack_class": trace.attack_class,
        "model": trace.model,
        "agent_text": trace.agent_text,
        "tool_calls": [dataclasses.asdict(tc) for tc in trace.tool_calls],
        "latency_ms": trace.latency_ms,
        "error": trace.error,
    }
    with out_path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def read_traces(in_path: Path) -> list[HarnessTrace]:
    out: list[HarnessTrace] = []
    for line in in_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tcs = [
            ToolCallEmit(
                tool_name=t["tool_name"],
                parameters=t["parameters"],
                reflects_injection=t["reflects_injection"],
            )
            for t in d.get("tool_calls", [])
        ]
        out.append(
            HarnessTrace(
                trial_id=d["trial_id"],
                trial_fingerprint=d["trial_fingerprint"],
                benign_topic=d["benign_topic"],
                attack_class=d["attack_class"],
                model=d["model"],
                agent_text=d["agent_text"],
                tool_calls=tcs,
                latency_ms=d["latency_ms"],
                error=d.get("error"),
            )
        )
    return out
