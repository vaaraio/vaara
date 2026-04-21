from rich import print

from vaara import Pipeline

pipeline = Pipeline()

result = pipeline.intercept(
    agent_id="agent-1",
    tool_name="transfer",
)

print()
color = {"allow": "bold green", "escalate": "bold yellow", "deny": "bold red"}[result.decision]
print(f"  [dim]decision[/]   [{color}]{result.decision.upper()}[/]")
print(f"  [dim]risk[/]       [cyan]{result.risk_score:.3f}[/]")
print()
