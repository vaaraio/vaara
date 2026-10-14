# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A proxy that owns its database leaves a signed receipt per decision.

The proxy built its trail as a bare AuditTrail, so the database filled with
decisions while the receipts directory beside it stayed empty. Every other
entry point opens the trail with load_trail, which wires the receipt sink.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("rfc8785")


def test_each_proxied_decision_leaves_a_receipt_that_verifies(monkeypatch, tmp_path):
    from vaara.audit import decision_receipts as dr
    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    db = tmp_path / "trail" / "audit.db"
    db.parent.mkdir()
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], db_path=db)
    upstream = MagicMock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream = upstream

    for req_id, tool in ((1, "read_file"), (2, "list_dir")):
        p._handle_request({
            "jsonrpc": "2.0", "id": req_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": {"path": "/tmp/x"}},
        })

    with sqlite3.connect(db) as con:
        hashes = dict(con.execute("SELECT record_id, record_hash FROM audit_records"))
    receipts = sorted((db.parent / dr.RECEIPTS_DIRNAME).rglob("*.json"))
    assert len(receipts) == 2
    assert all(dr.verify_receipt_file(f, trail_hashes=hashes).ok for f in receipts)
