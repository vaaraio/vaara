"""LangChain + Vaara — intercept tool calls before they execute.

Run:
    pip install vaara langchain-core rich
    python examples/langchain_agent.py

Wires Vaara into LangChain's tool-dispatch path using the core
`pipeline.intercept(...)` API. Works with any LangChain version and
with any agent loop (create_react_agent, AgentExecutor, LangGraph).

This example registers a minimal action taxonomy so you can see how
different tool categories get differentiated scores. For higher-level
integrations see VaaraCallbackHandler and vaara_wrap_tool in the README.
"""
from __future__ import annotations

from langchain_core.tools import tool
from rich import box
from rich.console import Console
from rich.table import Table

from vaara.pipeline import InterceptionPipeline
from vaara.sandbox.trace_gen import TraceGenerator
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionType,
    BlastRadius,
    Reversibility,
    UrgencyClass,
)


@tool
def write_config(path: str, content: str) -> str:
    """Write a config file."""
    return f"wrote {len(content)} bytes to {path}"


@tool
def shell_exec(command: str) -> str:
    """Execute a shell command."""
    return f"[mock] {command}"


@tool
def fetch_url(url: str) -> str:
    """HTTP GET a URL."""
    return f"[mock] GET {url}"


# --- Taxonomy -------------------------------------------------------------
# Vaara's scorer reads action metadata (reversibility, blast radius, urgency)
# from a registry. You define what each tool is. Below is a minimal taxonomy
# for the three demo tools.
pipeline = InterceptionPipeline()
registry = pipeline.registry

registry.register(ActionType(
    name="config_write", category=ActionCategory.INFRASTRUCTURE,
    reversibility=Reversibility.PARTIALLY, blast_radius=BlastRadius.SHARED,
    urgency=UrgencyClass.DEFERRABLE, description="Writes a config file",
))
registry.register(ActionType(
    name="shell_exec", category=ActionCategory.INFRASTRUCTURE,
    reversibility=Reversibility.IRREVERSIBLE, blast_radius=BlastRadius.GLOBAL,
    urgency=UrgencyClass.DEFERRABLE, description="Executes a shell command",
))
registry.register(ActionType(
    name="http_fetch", category=ActionCategory.COMMUNICATION,
    reversibility=Reversibility.FULLY, blast_radius=BlastRadius.LOCAL,
    urgency=UrgencyClass.DEFERRABLE, description="HTTP GET",
))
registry.map_tool("write_config", "config_write")
registry.map_tool("shell_exec", "shell_exec")
registry.map_tool("fetch_url", "http_fetch")

# Pre-calibrate so cold-start doesn't escalate everything.
TraceGenerator().pre_calibrate(pipeline, TraceGenerator().generate(n_traces=200))

# Scripted call sequence an agent might propose, some benign, some risky.
demo_calls = [
    (fetch_url, {"url": "https://example.com/"}),
    (write_config, {"path": "/tmp/ok.yaml", "content": "safe"}),
    (shell_exec, {"command": "ls /tmp"}),
    (shell_exec, {"command": "rm -rf /"}),
    (fetch_url, {"url": "http://169.254.169.254/latest/meta-data/"}),
    (write_config, {"path": "/etc/sudoers", "content": "agent ALL=(ALL) NOPASSWD: ALL"}),
]


def governed_invoke(pipeline, lc_tool, args, *, agent_id="demo-agent"):
    """Intercept, then (if allowed) invoke the tool and close the feedback loop.

    Drop this pattern into your LangChain AgentExecutor / LangGraph node
    right before the tool call actually fires. That is the Vaara hook.
    """
    result = pipeline.intercept(
        agent_id=agent_id, tool_name=lc_tool.name,
        parameters=args, agent_confidence=0.8,
    )
    if not result.allowed:
        return result, None
    output = lc_tool.invoke(args)
    pipeline.report_outcome(result.action_id, outcome_severity=0.0)
    return result, output


console = Console()
table = Table(title="LangChain → Vaara → tool", box=box.SIMPLE, pad_edge=False)
table.add_column("tool", style="cyan", no_wrap=True)
table.add_column("args", overflow="fold")
table.add_column("risk", no_wrap=True)
table.add_column("outcome")

for lc_tool, args in demo_calls:
    result, output = governed_invoke(pipeline, lc_tool, args)
    risk = f"{result.risk_score:.2f} [{result.risk_interval[0]:.2f}-{result.risk_interval[1]:.2f}]"
    if result.allowed:
        verdict = f"[green]ALLOW[/] → {output}"
    elif result.decision == "escalate":
        verdict = f"[yellow]ESCALATE[/] → {result.reason.split(' (')[0]}"
    else:
        verdict = f"[red]BLOCK[/] → {result.reason.split(' (')[0]}"
    table.add_row(lc_tool.name, str(args)[:60], risk, verdict)

console.print(table)
