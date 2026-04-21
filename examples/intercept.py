"""Minimal Vaara demo — risky action intercepted."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from vaara import InterceptionPipeline

pipeline = InterceptionPipeline()

result = pipeline.intercept(
    agent_id="agent-007",
    tool_name="fs.delete_recursive",
    parameters={"path": "/"},
    agent_confidence=0.95,
)

console = Console()

verdict = "ALLOW" if result.allowed else result.decision.upper()
verdict_color = "sea_green3" if result.allowed else (
    "light_goldenrod3" if result.decision == "escalate" else "indian_red"
)

table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
table.add_column(style="grey66", no_wrap=True)
table.add_column(style="white")
table.add_row("agent", "agent-007")
table.add_row("tool", "fs.delete_recursive")
table.add_row("risk", f"{result.risk_score:.3f}  "
             f"[grey50][{result.risk_interval[0]:.3f}, "
             f"{result.risk_interval[1]:.3f}][/]")
table.add_row("verdict", f"[{verdict_color}]{verdict}[/]")

console.print(Panel(table, title="[bold]vaara[/]", border_style="grey50",
                    padding=(0, 2)))
