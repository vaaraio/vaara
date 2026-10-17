# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Value origin, slice 2: the MCP proxy, more value shapes, strict mode.

The proxy sees both ends of the case without any host hook: the result of a
tools/call that carried an attacker's IBAN, and a later tools/call that puts
the same IBAN into a payment. Phone numbers and wallet addresses are
authority values too. Strict mode passes only values on the trusted list.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaara.audit.trail import AuditTrail
from vaara.integrations import claude_code_hooks as hooks
from vaara.pipeline import InterceptionPipeline, _value_origin
from vaara.scorer.value_origin import (
    OriginStore, authority_args, authority_tool, describe, extract, origin_hold,
)

ATTACKER_IBAN = "FI2112345600000785"
ETH = "0x52908400098527886E0F7030069857D2E4169EE7"
BTC = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
INBOX = (f"From: billing@acme.example\nPlease pay to FI21 1234 5600 0007 85.\n"
         f"Questions: call +358 40 123 4567 or wallet {ETH}")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OriginStore:
    path = tmp_path / "value_origin.json"
    monkeypatch.setenv("VAARA_VALUE_ORIGIN_PATH", str(path))
    return OriginStore(path)


class TestMoreShapes:
    def test_phone_numbers_in_international_form(self):
        assert ("phone", "+358401234567") in extract("call +358 40 123 4567 today")
        assert ("phone", "+14155550123") in extract("US line: +1 (415) 555-0123")

    def test_a_number_without_a_plus_is_not_a_phone(self):
        assert not [v for k, v in extract("order 0401234567 shipped") if k == "phone"]

    def test_wallet_addresses(self):
        found = extract(f"send to {ETH} or {BTC}")
        assert ("wallet", ETH.lower()) in found
        assert ("wallet", BTC) in found

    def test_a_hex_hash_of_the_wrong_length_is_not_a_wallet(self):
        assert not [v for k, v in extract("commit 0x" + "a" * 64) if k == "wallet"]

    def test_sms_and_wallet_arguments_carry_authority(self):
        assert authority_tool("send_sms")
        assert set(authority_args("send_sms", {"phone": "+358401234567", "text": "x"})) == {"phone"}
        assert set(authority_args("transfer_funds", {"wallet_address": ETH, "memo": "x"})) == {"wallet_address"}

    def test_a_phone_seen_only_in_a_result_is_held(self, store):
        store.note_external(extract(INBOX), tool_name="read_inbox", action_id="a1")
        held = origin_hold("send_sms", {"phone": "+358 40 123 4567", "text": "hi"}, store)
        assert held == {"phone": {"seen_in": ["a1"], "source": "read_inbox"}}


class TestStrict:
    def test_strict_holds_a_value_never_seen(self, store):
        held = origin_hold("create_payment", {"iban": ATTACKER_IBAN}, store, strict=True)
        assert held == {"iban": {"seen_in": [], "source": "", "rule": "strict"}}

    def test_strict_holds_a_value_the_user_typed(self, store):
        store.note_user(extract(f"pay {ATTACKER_IBAN}"))
        assert origin_hold("create_payment", {"iban": ATTACKER_IBAN}, store, strict=True)

    def test_strict_passes_a_trusted_value(self, store):
        assert origin_hold("create_payment", {"iban": ATTACKER_IBAN}, store,
                           strict=True, trusted=["FI21 1234 5600 0007 85"]) is None

    def test_strict_still_names_the_result_a_value_came_from(self, store):
        store.note_external(extract(INBOX), tool_name="read_inbox", action_id="a1")
        held = origin_hold("create_payment", {"iban": ATTACKER_IBAN}, store, strict=True)
        assert held == {"iban": {"seen_in": ["a1"], "source": "read_inbox", "rule": "strict"}}

    def test_strict_holds_an_account_in_no_known_shape(self, store):
        held = origin_hold("create_payment", {"account": "12345-678"}, store, strict=True)
        assert held == {"account": {"seen_in": [], "source": "", "rule": "strict"}}
        assert origin_hold("create_payment", {"account": "12345-678"}, store,
                           strict=True, trusted=["12345-678"]) is None

    def test_strict_leaves_non_authority_arguments_alone(self, store):
        assert origin_hold("create_payment", {"memo": "FI2112345600000785"}, store, strict=True) is None

    def test_describe_says_the_value_is_not_trusted(self):
        line = describe({"iban": {"seen_in": [], "source": "", "rule": "strict"}})
        assert "iban is not on the trusted list (strict)" in line

    def test_the_record_keeps_the_strict_rule(self):
        kept = _value_origin({"iban": {"seen_in": [], "source": "", "rule": "strict"}})
        assert kept == {"iban": {"seen_in": [], "source": "", "rule": "strict"}}

    def test_the_record_drops_an_unknown_rule(self):
        kept = _value_origin({"iban": {"seen_in": [], "source": "", "rule": "anything"}})
        assert kept == {"iban": {"seen_in": [], "source": ""}}

    def test_hook_reads_strict_from_config(self, store):
        cfg = {"value_origin": True, "value_origin_strict": True}
        assert hooks._value_origin(cfg, "create_payment", {"iban": ATTACKER_IBAN}) == {
            "iban": {"seen_in": [], "source": "", "rule": "strict"}}


def _proxy(monkeypatch, store, **kw):
    from vaara.integrations import mcp_proxy

    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **_kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline,
                                deny_rules=[], value_origin_store=store, **kw)
    upstream = MagicMock()
    p._upstream = upstream
    return p, upstream, trail


def _call(p, tool, args, rid=1):
    return p._handle_request({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                              "params": {"name": tool, "arguments": args}})


def _result(text, rid=1):
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}


class TestProxy:
    def test_an_iban_read_through_the_proxy_holds_the_payment(self, monkeypatch, store):
        p, upstream, trail = _proxy(monkeypatch, store, value_origin=True)
        upstream.request.return_value = _result(INBOX)
        _call(p, "read_inbox", {"folder": "inbox"})
        upstream.request.reset_mock()
        resp = _call(p, "create_payment", {"iban": ATTACKER_IBAN, "amount": 100}, rid=2)
        assert resp["result"]["isError"] is True
        assert '"decision": "escalate"' in resp["result"]["content"][0]["text"]
        assert "iban first seen in the result of read_inbox" in resp["result"]["content"][0]["text"]
        upstream.request.assert_not_called()
        decided = [r for r in trail._records if r.event_type.value == "decision_made"
                   and r.data.get("value_origin")]
        assert decided and decided[-1].data["value_origin"]["iban"]["source"] == "read_inbox"

    def test_the_gate_list_names_the_value_origin_step(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store, value_origin=True)
        upstream.request.return_value = _result(INBOX)
        _call(p, "read_inbox", {"folder": "inbox"})
        resp = _call(p, "create_payment", {"iban": ATTACKER_IBAN}, rid=2)
        assert "value_origin:hold" in resp["result"]["content"][0]["text"]

    def test_a_value_the_user_typed_passes(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store, value_origin=True)
        upstream.request.return_value = _result(INBOX)
        _call(p, "read_inbox", {"folder": "inbox"})
        store.note_user(extract(f"pay {ATTACKER_IBAN}"))
        upstream.request.return_value = _result("paid", rid=2)
        resp = _call(p, "create_payment", {"iban": ATTACKER_IBAN}, rid=2)
        assert resp == _result("paid", rid=2)

    def test_off_by_default(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store)
        upstream.request.return_value = _result(INBOX)
        _call(p, "read_inbox", {"folder": "inbox"})
        assert store.entries()["external"] == {}
        upstream.request.return_value = _result("paid", rid=2)
        assert _call(p, "create_payment", {"iban": ATTACKER_IBAN}, rid=2) == _result("paid", rid=2)

    def test_strict_on_the_proxy_holds_an_unseen_wallet(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store, value_origin=True, value_origin_strict=True)
        resp = _call(p, "transfer_funds", {"wallet_address": ETH, "amount": 1})
        assert resp["result"]["isError"] is True
        assert "wallet_address is not on the trusted list (strict)" in resp["result"]["content"][0]["text"]
        upstream.request.assert_not_called()

    def test_trusted_values_pass_strict(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store, value_origin=True,
                                value_origin_strict=True, trusted_values=[ETH])
        upstream.request.return_value = _result("sent")
        assert _call(p, "transfer_funds", {"wallet_address": ETH}) == _result("sent")

    def test_a_resource_read_is_harvested(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store, value_origin=True)
        upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {
            "contents": [{"uri": "mail://1", "text": INBOX}]}}
        p._handle_request({"jsonrpc": "2.0", "id": 1, "method": "resources/read",
                           "params": {"uri": "mail://1"}})
        assert origin_hold("create_payment", {"iban": ATTACKER_IBAN}, store)

    def test_a_broken_store_never_blocks_the_call(self, monkeypatch, store):
        p, upstream, _ = _proxy(monkeypatch, store, value_origin=True)
        monkeypatch.setattr(store, "entries", MagicMock(side_effect=OSError("disk")))
        upstream.request.return_value = _result("paid")
        assert _call(p, "create_payment", {"iban": ATTACKER_IBAN}) == _result("paid")


class TestProxyCli:
    def test_flags_reach_the_proxy(self, monkeypatch):
        from vaara.integrations import mcp_proxy

        seen = {}

        class Stop(Exception):
            pass

        def fake(**kw):
            seen.update(kw)
            raise Stop

        monkeypatch.setattr(mcp_proxy, "VaaraMCPProxy", fake)
        with pytest.raises(Stop):
            mcp_proxy.main(["--value-origin-strict", "--trusted-value", ETH,
                            "--authority-arg", "pay=payee", "--upstream", "echo"])
        assert seen["value_origin_strict"] is True
        assert seen["trusted_values"] == [ETH]
        assert seen["authority_args"] == {"pay": ["payee"]}

    def test_a_malformed_authority_arg_is_refused(self, monkeypatch):
        from vaara.integrations import mcp_proxy

        with pytest.raises(SystemExit):
            mcp_proxy.main(["--authority-arg", "payee", "--upstream", "echo"])
