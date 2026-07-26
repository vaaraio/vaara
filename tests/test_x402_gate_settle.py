# SPDX-License-Identifier: AGPL-3.0-or-later
"""The x402 gate's settlement handshake: each facilitator step must confirm
with its own flags. A verify-shaped answer on /settle is not settlement."""
import io
import json
from unittest import mock

from vaara.server.x402 import X402Config, X402Gate


def _gate() -> X402Gate:
    return X402Gate(
        X402Config(
            enabled=True,
            pay_to="0xabc",
            network="base",
            asset="usdc",
            price="10",
            facilitator="https://facilitator.test",
        )
    )


def _urlopen_returning(responses):
    """urlopen stub yielding one canned JSON body per called path, in order."""
    bodies = iter(responses)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        body = json.dumps(next(bodies)).encode()
        resp = io.BytesIO(body)
        resp.__enter__ = lambda *a: resp
        resp.__exit__ = lambda *a: False
        return resp

    return fake_urlopen


def _settle_with(responses) -> bool:
    with mock.patch(
        "vaara.server.x402.urllib.request.urlopen",
        _urlopen_returning(responses),
    ):
        return _gate()._settle("hdr", "/resource", "10")


def test_settled_when_both_steps_confirm():
    assert _settle_with([{"isValid": True}, {"settled": True}]) is True


def test_settle_success_flag_accepted():
    assert _settle_with([{"isValid": True}, {"success": True}]) is True


def test_verify_shaped_settle_answer_is_refused():
    # the audited leniency: isValid alone on /settle must NOT admit the call
    assert _settle_with([{"isValid": True}, {"isValid": True}]) is False


def test_failed_verify_refused():
    assert _settle_with([{"isValid": False}, {"settled": True}]) is False


def test_no_facilitator_refuses():
    gate = X402Gate(
        X402Config(
            enabled=True, pay_to="0xabc", network="base",
            asset="usdc", price="10", facilitator=None,
        )
    )
    assert gate._settle("hdr", "/resource", "10") is False
