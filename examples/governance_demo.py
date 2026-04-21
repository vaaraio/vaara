"""Regulator-facing demo — Article 14 action gate in action."""
import time
from vaara import Pipeline
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
pipeline = Pipeline()

console.print()
console.print("  [dim]incoming agent action[/]")
console.print("  [bold]agent-007[/]  \u2192  [yellow]fs.delete_recursive[/]  {'path': '/'}")
console.print()
time.sleep(0.9)

result = pipeline.intercept(
    agent_id="agent-007",
    tool_name="fs.delete_recursive",
    parameters={"path": "/"},
    agent_confidence=0.95,
)

v_color = {"allow": "bold green", "escalate": "bold yellow", "deny": "bold red"}[result.decision]

body = Text()
body.append("  decision      ", style="dim")
body.append(f"{result.decision.upper()}\n", style=v_color)
body.append("  risk          ", style="dim")
body.append(f"{result.risk_score:.3f}", style="white")
body.append(f"   ci [{result.risk_interval[0]:.3f}, {result.risk_interval[1]:.3f}]\n", style="bright_black")
body.append("  reason        ", style="dim")
body.append(f"{result.reason.split(' (threshold')[0]}\n", style="white")
body.append("  action_id     ", style="dim")
body.append(f"{result.action_id[:16]}\u2026\n", style="cyan")
body.append("  evaluation    ", style="dim")
body.append(f"{result.evaluation_ms:.1f} ms", style="bright_black")

console.print(Panel(
    body,
    title="[bold #78a08a]vaara[/]  [dim]Article 14 action gate[/]",
    border_style="#78a08a",
    padding=(1, 2),
    width=72,
))
console.print()
