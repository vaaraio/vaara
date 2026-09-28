# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dashboard answers only to names DNS rebinding cannot hand an attacker.

It binds 127.0.0.1 on a fixed port and guards writes with a token that
GET /api/config returns to the page. A page at http://evil.test:7517 whose
name is re-resolved to 127.0.0.1 is same-origin with the dashboard, reads
the token, then posts policy thresholds and OS-layer changes. Refusing any
request whose Host is not a loopback name or address closes that.
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from vaara import dashboard


@pytest.fixture
def port(tmp_path):
    dashboard._Handler.token = "t0ken"
    dashboard._Handler.policy_path = tmp_path / "policy.yaml"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _call(port, method, path, host, body=None, token=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host}
    if token:
        headers["X-Vaara-Token"] = token
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_a_rebound_name_cannot_read_the_write_token(port):
    status, data = _call(port, "GET", "/api/config", f"evil.test:{port}")
    assert status == 403
    assert b"t0ken" not in data


def test_a_rebound_name_cannot_change_policy(port):
    status, _ = _call(port, "POST", "/api/policy", f"evil.test:{port}",
                      body={"escalate": 0.99}, token="t0ken")
    assert status == 403


@pytest.mark.parametrize("host", ["127.0.0.1:{p}", "localhost:{p}", "[::1]:{p}"])
def test_loopback_names_still_reach_the_dashboard(port, host):
    status, data = _call(port, "GET", "/api/config", host.format(p=port))
    assert status == 200
    assert json.loads(data)["token"] == "t0ken"


@pytest.mark.parametrize("action_id", ["../escape", "a/b", "..", ".hidden", "x\\y", ""])
def test_a_decision_id_names_a_file_in_the_approvals_dir_only(tmp_path, action_id):
    from vaara.approvals import _approval_key, write_decision

    approvals = tmp_path / "approvals"
    approvals.mkdir()
    (tmp_path / "escape.request.json").write_text('{"nonce": "n"}')
    _approval_key(approvals, create=True)
    assert write_decision(action_id, "approve", approvals_dir=approvals) is False
    assert not (tmp_path / "escape.decision.json").exists()
