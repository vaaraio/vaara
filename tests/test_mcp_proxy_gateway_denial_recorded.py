"""A credential-gateway denial must land in the trail, not just on the wire.

Regression cover for the boundary an external reviewer (VATE) flagged: the
MCP proxy takes its local policy decision (``InterceptionPipeline.intercept``)
and writes it to the hash chain BEFORE the constrained-grant
``CredentialGateway.authorize`` check runs. For a constrained tool the policy
decision can be ``allow`` while the gateway then refuses — runtime arguments
that no longer match the digest the grant was minted for are the common case.

Without an outcome record for that refusal the trail claims the call was
allowed and carries nothing saying it never executed. Record and behaviour
would disagree, which is the same failure mode the prior-approval path is
written to avoid.
"""

from unittest.mock import MagicMock


def _proxy_with_denying_gateway(monkeypatch, *, reason="args digest mismatch"):
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)

    upstream = MagicMock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream = upstream

    # Constrained tool whose gateway refuses.
    verdict = MagicMock()
    verdict.ok = False
    verdict.reason = reason

    attest = MagicMock()
    attest.gateway = MagicMock()
    attest.gateway.authorize.return_value = verdict
    attest.is_constrained.return_value = True
    # emit_attestation returns an (attestation, counter) pair; emit_grant
    # returns None here so the request carries no credential and the
    # gateway is the thing that refuses.
    attest.emit_attestation.return_value = (MagicMock(), 1)
    attest.emit_grant.return_value = None
    p._attest = attest
    p._mint_credentials = True
    p._emit_authorization_receipts = False

    return p, pipeline, trail, upstream


def _call(p, tool="read_file", args=None):
    return p._handle_request({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args or {"path": "/tmp/x"}},
    })


def test_gateway_denial_is_reported_as_a_blocked_outcome(monkeypatch):
    p, pipeline, trail, upstream = _proxy_with_denying_gateway(monkeypatch)

    reported = []
    real = pipeline.report_outcome

    def spy(action_id, outcome_severity, description=""):
        reported.append((action_id, outcome_severity, description))
        return real(action_id, outcome_severity, description)

    monkeypatch.setattr(pipeline, "report_outcome", spy)

    resp = _call(p)

    # The caller still gets the denial.
    assert "error" in resp, resp
    assert "grant required but not valid" in resp["error"]["message"]

    # ...and the refusal is on the record, at full severity, tied to the
    # same action_id the policy decision used.
    assert reported, "gateway denial was not reported as an outcome"
    action_id, severity, description = reported[0]
    assert action_id, "outcome reported without an action_id"
    assert severity == 1.0
    assert "gateway" in description.lower()
    assert "args digest mismatch" in description


def test_gateway_denial_does_not_reach_upstream(monkeypatch):
    p, pipeline, trail, upstream = _proxy_with_denying_gateway(monkeypatch)
    _call(p)
    upstream.request.assert_not_called()


def test_outcome_failure_never_masks_the_denial(monkeypatch):
    """If recording the outcome blows up, the call is still refused."""
    p, pipeline, trail, upstream = _proxy_with_denying_gateway(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("trail unavailable")

    monkeypatch.setattr(pipeline, "report_outcome", boom)

    resp = _call(p)
    assert "error" in resp
    assert "grant required but not valid" in resp["error"]["message"]
    upstream.request.assert_not_called()
