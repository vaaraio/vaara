<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/vaara-wordmark-dark.png">
    <img src="docs/vaara-wordmark-light.png" alt="Vaara" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/vaara/"><img src="https://img.shields.io/pypi/v/vaara.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/vaara/"><img src="https://img.shields.io/pypi/pyversions/vaara.svg" alt="Python"></a>
  <a href="https://github.com/vaaraio/vaara/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/vaara.svg" alt="License"></a>
  <a href="https://github.com/vaaraio/vaara/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/vaaraio/vaara/ci.yml?branch=main&label=tests" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/vaaraio/vaara"><img src="https://api.scorecard.dev/projects/github.com/vaaraio/vaara/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/12612"><img src="https://www.bestpractices.dev/projects/12612/badge" alt="OpenSSF Best Practices"></a>
</p>

Vaara intercepts agent tool calls, scores each one with a conformal risk interval, and writes a hash-chained audit record. Online learning across five expert signals via Multiplicative Weight Update. Distribution-free conformal coverage on the score.

For broader agent governance (zero-trust identity, capability-based access control, multi-language SDKs) see Microsoft's [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit). Vaara is narrower.

## Install

```bash
pip install vaara
```

Python 3.10+. Zero runtime deps. Optional XGBoost classifier: `pip install vaara[ml]`.

## Quick start

```python
from vaara.pipeline import InterceptionPipeline

pipeline = InterceptionPipeline()
result = pipeline.intercept(
    agent_id="agent-007",
    tool_name="fs.write_file",
    parameters={"path": "/etc/service.yaml", "content": "..."},
    agent_confidence=0.8,
)
if result.allowed:
    pipeline.report_outcome(result.action_id, outcome_severity=0.0)
else:
    print(result.reason)
```

`report_outcome` closes the loop. MWU reweights signals based on which ones predicted the outcome.

## Where things live

- [docs/formal_specification.md](docs/formal_specification.md): math. MWU regret bound O(sqrt(T log N)), conformal coverage guarantees, security properties.
- [COMPLIANCE.md](COMPLIANCE.md): Article-level evidence mapping for EU AI Act (Articles 9, 11 to 15, 61) and DORA (Articles 10, 12, 13). Eval numbers, threshold sweeps, PAIR adversarial calibration.
- `src/vaara/integrations/`: LangChain, OpenAI Agents SDK, CrewAI, MCP server.
- `src/vaara/audit/`: hash-chain trail, SQLite backend, append-only WAL.
- `src/vaara/sandbox/`: synthetic-trace cold-start calibration.

> Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

## License

[LICENSE](LICENSE)
