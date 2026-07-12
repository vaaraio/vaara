"""--shadow on the MCP proxy: observe-first onboarding.

Shadow mode classifies, scores, and audits every call but blocks nothing,
so an operator can put the proxy in front of live traffic on day one and
read the shadow report before flipping to enforcement.
"""

from __future__ import annotations


def test_proxy_shadow_flag_disables_enforcement(tmp_path):
    from vaara.integrations.mcp_proxy import VaaraMCPProxy

    proxy = VaaraMCPProxy(
        upstream_command=["true"],
        db_path=tmp_path / "audit.db",
        shadow=True,
    )
    assert proxy._pipeline._enforce is False


def test_proxy_default_is_enforcing(tmp_path):
    from vaara.integrations.mcp_proxy import VaaraMCPProxy

    proxy = VaaraMCPProxy(
        upstream_command=["true"],
        db_path=tmp_path / "audit.db",
    )
    assert proxy._pipeline._enforce is True
