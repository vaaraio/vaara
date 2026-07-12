"""Base rule scorer blocks cloud-metadata SSRF without the ML extra.

A zero-config install previously allowed a network fetch to
169.254.169.254; only the opt-in ML classifier caught it. The rule scorer
now applies a deterministic decision floor for cloud-metadata endpoints in
call parameters, mirroring the egress guard's "no legitimate reason to dial"
boundary. Legitimate private hosts are unaffected.
"""

from __future__ import annotations

import pytest

from vaara.scorer._param_signals import metadata_endpoint_risk


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://[fe80::a9fe:a9fe]/latest/meta-data/",
    "http://2852039166/latest/meta-data/",          # dotless decimal
    "http://0xa9fea9fe/latest/meta-data/",           # hex
])
def test_metadata_urls_flagged(url):
    assert metadata_endpoint_risk({"url": url}) >= 0.9


@pytest.mark.parametrize("params", [
    {"url": "https://api.github.com/repos/x/y"},
    {"url": "http://10.0.0.5/internal"},             # RFC1918: legitimate, not floored
    {"url": "http://localhost:8080/health"},         # loopback: legitimate
    {"path": "README.md"},
    {"command": "ls -la"},
    {},
])
def test_benign_params_not_flagged(params):
    assert metadata_endpoint_risk(params) == 0.0


def test_nested_and_listed_params_scanned():
    params = {"requests": [{"target": "http://169.254.169.254/"}]}
    assert metadata_endpoint_risk(params) >= 0.9


def test_scorer_blocks_metadata_fetch_zero_config():
    from vaara.scorer.adaptive import AdaptiveScorer
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    registry = create_default_registry()
    scorer = AdaptiveScorer()
    req = ActionRequest(
        agent_id="agent-x",
        tool_name="http_get",
        action_type=registry.classify("http_get"),
        parameters={"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"},
    )
    decision = scorer.evaluate(req.to_policy_context())
    assert decision["action"] in ("deny", "escalate")


def test_scorer_allows_benign_fetch():
    from vaara.scorer.adaptive import AdaptiveScorer
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    registry = create_default_registry()
    scorer = AdaptiveScorer()
    req = ActionRequest(
        agent_id="agent-x",
        tool_name="http_get",
        action_type=registry.classify("http_get"),
        parameters={"url": "https://api.github.com/repos/x/y"},
    )
    decision = scorer.evaluate(req.to_policy_context())
    assert decision["action"] == "allow"
