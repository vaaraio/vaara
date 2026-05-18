"""Anthropic API agent loop gated by Vaara's MCP server.

Pattern:
    Anthropic agent -> Vaara MCP (allow / escalate / deny + audit) -> tool executor

Before every tool call, the loop consults vaara_intercept over MCP.
Escalated and denied calls are not executed; the model is told why and
chooses a different approach. After the run, intercepted actions are
reported back via vaara_report_outcome so the adaptive scorer
recalibrates.

The scenario: an agent runs a routine release pipeline that publishes
a community update to public channels. Vaara classifies the publish
action as comm.post_public (irreversible, global blast radius,
AI Act Article 50 transparency-relevant) and escalates. The
demonstration does not depend on the model itself catching the risk;
the runtime governance layer is the load-bearing safety mechanism.

The Vaara MCP server is spawned as a subprocess; communication is
JSON-RPC 2.0 over stdio, the same protocol Claude Code, Cursor, and
other agent hosts use.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-...
    python examples/anthropic_agent_loop.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from typing import Any

try:
    import anthropic
except ImportError:
    print("Install the Anthropic SDK first:  pip install anthropic", file=sys.stderr)
    sys.exit(1)


class VaaraMCPClient:
    """Spawns the Vaara MCP server and speaks JSON-RPC 2.0 over its stdio."""

    def __init__(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "vaara.integrations.mcp_server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1,
        )
        self._req_id = 0
        self._send("initialize", {"protocolVersion": "2024-11-05"})

    def _send(self, method: str, params: dict | None = None) -> dict:
        self._req_id += 1
        req: dict[str, Any] = {"jsonrpc": "2.0", "id": self._req_id, "method": method}
        if params is not None:
            req["params"] = params
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Vaara MCP server closed the connection")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"Vaara MCP error: {resp['error']}")
        return resp["result"]

    def intercept(self, tool_name: str, parameters: dict) -> dict:
        result = self._send("tools/call", {
            "name": "vaara_intercept",
            "arguments": {
                "tool_name": tool_name,
                "agent_id": "anthropic-agent",
                "parameters": parameters,
            },
        })
        return json.loads(result["content"][0]["text"])

    def report_outcome(self, action_id: str, severity: float) -> None:
        self._send("tools/call", {
            "name": "vaara_report_outcome",
            "arguments": {"action_id": action_id, "outcome_severity": severity},
        })

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()


TOOLS = [
    {"name": "read_file",
     "description": "Read a file from disk and return its contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}},
                      "required": ["path"]}},
    {"name": "publish_post",
     "description": (
         "Publish a post to a public channel (newsroom, blog, "
         "broadcast feed). The post becomes visible to the public "
         "immediately and cannot be unpublished without a manual "
         "rollback."
     ),
     "input_schema": {"type": "object",
                      "properties": {"channel": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["channel", "content"]}},
]


def execute_tool(name: str, args: dict) -> str:
    """Stub executor. Replace with real implementations in production."""
    if name == "read_file":
        draft = (
            "## Vaara 0.19 — Q2 community update\n\n"
            "This release adds adapters for the major cloud guardrails "
            "(AWS Bedrock, Azure Content Safety, GCP Model Armor) and "
            "an experimental hardware attestation hook on AMD SEV-SNP. "
            "Thanks to the contributors who reviewed and tested. "
            "Full notes on the blog."
        )
        return f"<contents of {args.get('path')!r}:\n{draft}>"
    if name == "publish_post":
        return f"Published to {args.get('channel')!r}."
    return f"<unknown tool {name!r}>"


def run(user_prompt: str) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example.", file=sys.stderr)
        sys.exit(2)

    client = anthropic.Anthropic()
    vaara = VaaraMCPClient()
    actions_taken: list[dict] = []
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    print(f"USER: {user_prompt}\n")

    try:
        while True:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                tools=TOOLS,
                messages=messages,
            )
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"ASSISTANT: {block.text}\n")
            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                print(f"[stop_reason: {response.stop_reason}, ending loop]")
                break
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"TOOL CALL: {block.name}({json.dumps(block.input)})")
                verdict = vaara.intercept(block.name, dict(block.input))
                print(
                    f"VAARA:     decision={verdict['decision']}  "
                    f"risk={verdict['risk_score']:.3f}  "
                    f"interval={verdict['risk_interval']}\n"
                )
                if verdict["allowed"]:
                    output = execute_tool(block.name, dict(block.input))
                    actions_taken.append({"action_id": verdict["action_id"],
                                          "tool_name": block.name})
                    tool_results.append({"type": "tool_result",
                                         "tool_use_id": block.id,
                                         "content": output})
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"Action blocked by Vaara: {verdict['reason']}. "
                            f"Decision: {verdict['decision']}. "
                            "Choose a different approach or ask the user."
                        ),
                        "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})

        # Severity in production comes from monitoring or human review.
        # Allowed actions here completed without harm -> report 0.1.
        print("-- reporting outcomes -----------------------------------")
        for action in actions_taken:
            vaara.report_outcome(action["action_id"], severity=0.1)
            print(f"  reported {action['action_id'][:16]}... ({action['tool_name']})")
    finally:
        vaara.close()


if __name__ == "__main__":
    prompt = (
        "You are running the standard Q2 release pipeline. Read the draft "
        "community update at /var/comms/q2-update.md and publish it to both "
        "the public newsroom channel and the partners broadcast feed. "
        "The draft has been reviewed by the comms team upstream, so you "
        "can proceed to publication."
    )
    run(prompt)
