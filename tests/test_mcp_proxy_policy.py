"""--policy on the MCP proxy: a policy file drives the proxy's pipeline.

Before this, the proxy always ran the default scorer thresholds; the only
operator controls were the exact-name --allow-tool/--deny-tool perimeter.
"""

from __future__ import annotations

import json

import pytest

from vaara.integrations._mcp_overt import policy_hash_from_perimeter
from vaara.integrations.mcp_proxy import VaaraMCPProxy, main
from vaara.policy import from_dict

_POLICY_DICT = {
    "version": "0.1",
    "domains": ["eu_ai_act"],
    "action_classes": {},
    "thresholds": {"default": {"escalate": 0.15, "deny": 0.35}},
}


def test_proxy_applies_policy_to_pipeline_scorer(tmp_path):
    policy = from_dict(_POLICY_DICT)
    proxy = VaaraMCPProxy(
        upstream_command=["true"],
        db_path=tmp_path / "audit.db",
        policy=policy,
    )
    assert proxy._pipeline.scorer._threshold_allow == 0.15
    assert proxy._pipeline.scorer._threshold_deny == 0.35


def test_main_rejects_malformed_policy(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": "0.1", "thresholds": {"default": {"escalate": 2.0, "deny": 0.1}}}))
    with pytest.raises(SystemExit) as exc:
        main([
            "--upstream", "up=true",
            "--db", str(tmp_path / "audit.db"),
            "--policy", str(bad),
        ])
    assert exc.value.code == 2
    err = capsys.readouterr().err.lower()
    # Must be a validation failure, not argparse choking on an unknown flag.
    assert "unrecognized arguments" not in err
    assert "policy" in err and "failed" in err


def test_policy_source_changes_perimeter_hash():
    base = dict(
        tool_allow=None, tool_deny=set(),
        resource_allow=None, resource_deny=set(),
        prompt_allow=None, prompt_deny=set(),
    )
    without = policy_hash_from_perimeter(**base)
    with_policy = policy_hash_from_perimeter(**base, policy_source=b'{"thresholds": {}}')
    assert without != with_policy
    # No policy source keeps the historical hash payload byte-identical.
    assert without == policy_hash_from_perimeter(**base)
