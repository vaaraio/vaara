"""The 60-second Vaara quickstart. Zero config, zero dependencies.

`@vaara.govern` is the whole integration: every call to a governed function
is classified, risk-scored, and decided allow/escalate/deny before the body
runs. A blocked call raises `vaara.Blocked`; every decision lands in the
hash-chained audit trail either way.

Run:  python examples/quickstart.py

Want the decision object in hand instead of an exception? The explicit
pipeline behind the decorator is in examples/intercept.py.
"""

import vaara


@vaara.govern
def read_file(path: str) -> str:
    return f"(pretend contents of {path})"


@vaara.govern
def shell_exec(command: str) -> str:
    return "(pretend the shell ran)"


# A boring read is allowed and the function body runs.
print(read_file(path="README.md"))

# A destructive command never reaches the body.
try:
    shell_exec(command="rm -rf /")
except vaara.Blocked as blocked:
    print(f"blocked: {blocked}")

print()
print("Both decisions were scored and recorded in the process-wide trail")
print("(in-memory by default; wire a signing SQLite trail via")
print("vaara.govern.set_default_pipeline for a persistent, verifiable record).")
print()
print("Next steps:")
print("  examples/intercept.py               # the explicit pipeline API")
print("  examples/github-mcp-proxy-demo/     # persistent trail in front of a real MCP server")
print("  examples/policies/                  # write your own policy")
