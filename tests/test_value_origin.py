# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Value origin: an authority argument whose value came only from content.

The case it exists for: "pay the invoice in my inbox", the agent reads the
mail, the payee IBAN in it was edited by an attacker, and the payment call
is in scope like every other call. The hold names the result the IBAN came
from. A value the user typed, a trusted value, and an argument that names no
recipient, payee or endpoint are never held.
"""

from __future__ import annotations

import io
import json
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from vaara.audit.trail import AuditTrail
from vaara.integrations import claude_code_hooks as hooks
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.value_origin import (
    OriginStore, authority_args, describe, extract, extract_any, origin_hold,
)

ATTACKER_IBAN = "FI21 1234 5600 0007 85"
ATTACKER_FLAT = "FI2112345600000785"
USER_IBAN = "DE89 3704 0044 0532 0130 00"
INVOICE = (f"Invoice 2026-114\nPayee: Acme Oy\nIBAN: {ATTACKER_IBAN}\n"
           "Questions: billing@acme.example\nPortal https://pay.acme.example/i/114")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OriginStore:
    path = tmp_path / "value_origin.json"
    monkeypatch.setenv("VAARA_VALUE_ORIGIN_PATH", str(path))
    return OriginStore(path)


class TestExtract:
    def test_email_iban_and_host(self):
        found = extract(INVOICE)
        assert ("iban", ATTACKER_FLAT) in found
        assert ("email", "billing@acme.example") in found
        assert ("host", "pay.acme.example") in found

    def test_an_iban_with_a_bad_checksum_is_not_one(self):
        assert not [v for k, v in extract("IBAN FI21 1234 5600 0007 86") if k == "iban"]

    def test_an_iban_followed_by_words_stops_at_the_account(self):
        assert ("iban", ATTACKER_FLAT) in extract(f"pay {ATTACKER_IBAN} TODAY PLEASE")

    def test_email_is_case_folded(self):
        assert ("email", "bob@example.org") in extract("Bob <Bob@Example.ORG>")

    def test_nested_tool_results_are_read(self):
        result = {"content": [{"type": "text", "text": INVOICE}]}
        assert ("iban", ATTACKER_FLAT) in extract_any(result)


class TestAuthorityArgs:
    def test_a_payment_tool_names_its_payee(self):
        args = authority_args("mcp__bank__pay_invoice",
                              {"iban": ATTACKER_IBAN, "amount": 120, "memo": "x"})
        assert set(args) == {"iban"}

    @pytest.mark.parametrize("tool", [
        "mcp__mail__read_email", "mcp__bank__list_payments", "mcp__mail__search_mail",
        "mcp__crm__getMessages", "mcp__mail__get_draft", "Read", "WebFetch",
    ])
    def test_a_read_tool_has_none(self, tool):
        assert authority_args(tool, {"to": "a@b.example", "url": "https://x.example"}) == {}

    @pytest.mark.parametrize("tool", [
        "mcp__mail__send_email", "mcp__bank__create_payment", "mcp__slack__postMessage",
        "mcp__mail__reply", "mcp__stripe__create_payout", "mcp__mail__email_user",
    ])
    def test_a_sending_tool_has_them(self, tool):
        assert authority_args(tool, {"to": "a@b.example"}) == {"to": "a@b.example"}

    def test_the_operator_declares_more(self):
        args = authority_args("mcp__crm__update", {"owner": "a@b.example"},
                              {"update": ["owner"]})
        assert set(args) == {"owner"}


class TestStore:
    def test_it_keeps_digests_not_values(self, store):
        store.note_external(extract(INVOICE), tool_name="mcp__mail__read_email",
                            action_id="act-1")
        text = store.path.read_text()
        assert ATTACKER_FLAT not in text and "billing@acme.example" not in text
        assert "act-1" in text

    def test_the_key_is_private_and_stable(self, store):
        first = store.digest("email", "a@b.example")
        key = store.path.with_name("value_origin.key")
        if sys.platform != "win32":  # Windows has no POSIX mode bits to set
            assert stat.S_IMODE(key.stat().st_mode) == 0o600
        assert OriginStore(store.path).digest("email", "a@b.example") == first

    def test_capped(self, tmp_path):
        small = OriginStore(tmp_path / "o.json", max_entries=3)
        for i in range(5):
            small.note_external({("email", f"u{i}@x.example")}, tool_name="WebFetch")
        assert len(small.entries()["external"]) == 3

    def test_corrupt_file_reads_as_empty(self, store):
        store.path.write_text("{not json")
        assert store.entries() == {"external": {}, "user": {}}


class TestOriginHold:
    PAY = "mcp__bank__pay_invoice"

    def _read_invoice(self, store):
        store.note_external(extract(INVOICE), tool_name="mcp__mail__read_email",
                            action_id="act-mail")

    def test_the_tampered_invoice_is_held(self, store):
        self._read_invoice(store)
        held = origin_hold(self.PAY, {"iban": ATTACKER_IBAN, "amount": 120}, store)
        assert held == {"iban": {"seen_in": ["act-mail"],
                                 "source": "mcp__mail__read_email"}}
        assert "iban first seen in the result of mcp__mail__read_email (action act-mail)" \
            in describe(held)

    def test_an_iban_the_user_typed_passes(self, store):
        self._read_invoice(store)
        store.note_user(extract(f"pay {ATTACKER_IBAN} for the invoice"))
        assert origin_hold(self.PAY, {"iban": ATTACKER_IBAN}, store) is None

    def test_a_trusted_value_passes(self, store):
        self._read_invoice(store)
        assert origin_hold(self.PAY, {"iban": ATTACKER_IBAN}, store,
                           trusted=[ATTACKER_IBAN]) is None

    def test_a_value_never_seen_passes(self, store):
        self._read_invoice(store)
        assert origin_hold(self.PAY, {"iban": USER_IBAN}, store) is None

    def test_an_unrelated_argument_is_untouched(self, store):
        self._read_invoice(store)
        assert origin_hold(self.PAY, {"memo": f"see {ATTACKER_IBAN}"}, store) is None

    def test_a_reply_to_a_sender_read_from_mail_is_held(self, store):
        self._read_invoice(store)
        held = origin_hold("mcp__mail__send_email",
                           {"to": "Billing <billing@acme.example>", "body": "paid"}, store)
        assert held is not None and set(held) == {"to"}


class TestPipeline:
    HELD = {"iban": {"seen_in": ["act-mail"], "source": "mcp__mail__read_email"}}

    def _decisions(self, trail):
        return [r for r in trail._records if r.event_type.value == "decision_made"]

    def test_an_allow_becomes_an_escalate_naming_the_result(self):
        trail = AuditTrail()
        result = InterceptionPipeline(trail=trail).intercept(
            agent_id="a", tool_name="mcp__bank__pay_invoice",
            parameters={"iban": ATTACKER_FLAT}, value_origin=self.HELD)
        assert result.decision == "escalate" and not result.allowed
        assert "iban first seen in the result of mcp__mail__read_email" in result.reason
        record = self._decisions(trail)[-1]
        assert record.data["value_origin"] == self.HELD

    def test_without_it_the_record_gains_no_key(self):
        trail = AuditTrail()
        InterceptionPipeline(trail=trail).intercept(
            agent_id="a", tool_name="mcp__bank__pay_invoice",
            parameters={"iban": ATTACKER_FLAT})
        assert "value_origin" not in self._decisions(trail)[-1].data

    def test_a_malformed_mapping_is_not_recorded(self):
        trail = AuditTrail()
        result = InterceptionPipeline(trail=trail).intercept(
            agent_id="a", tool_name="t", parameters={}, value_origin={"x": "not a dict"})
        assert result.decision == "allow"


def _feed(monkeypatch: pytest.MonkeyPatch, event: dict) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))


class TestHooks:
    """Read the mail, then pay: through the installed hook path."""

    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch, store):
        monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(tmp_path / "audit.db"))
        monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
        monkeypatch.setattr(hooks, "load_config", lambda: {"value_origin": True})
        self.db = tmp_path / "audit.db"
        self.store = store

    def _read_mail(self, monkeypatch):
        _feed(monkeypatch, {"tool_name": "mcp__mail__read_email",
                            "tool_input": {"id": "114"}, "session_id": "s1"})
        hooks.run_pre_tool_use()
        _feed(monkeypatch, {"tool_name": "mcp__mail__read_email",
                            "tool_input": {"id": "114"}, "session_id": "s1",
                            "tool_response": {"content": [{"type": "text", "text": INVOICE}]}})
        assert hooks.run_post_tool_use() == 0

    def _pay(self, monkeypatch, iban):
        _feed(monkeypatch, {"tool_name": "mcp__bank__pay_invoice",
                            "tool_input": {"iban": iban, "amount": 120},
                            "session_id": "s1"})
        return hooks.run_pre_tool_use()

    def _last_decision(self):
        con = sqlite3.connect(self.db)
        try:
            row = con.execute(
                "select data from audit_records where event_type='decision_made' "
                "order by seq desc limit 1").fetchone()
        finally:
            con.close()
        return json.loads(row[0])

    def test_the_payment_is_held_and_the_record_names_the_mail(self, monkeypatch):
        self._read_mail(monkeypatch)
        assert self._pay(monkeypatch, ATTACKER_IBAN) != 0
        data = self._last_decision()
        assert data["decision"] == "escalate"
        origin = data["value_origin"]["iban"]
        assert origin["source"] == "mcp__mail__read_email" and origin["seen_in"]

    def test_the_users_own_iban_is_not_held(self, monkeypatch):
        self._read_mail(monkeypatch)
        _feed(monkeypatch, {"prompt": f"pay the invoice to {ATTACKER_IBAN}"})
        assert hooks.run_user_prompt_submit() == 0
        self._pay(monkeypatch, ATTACKER_IBAN)
        assert "value_origin" not in self._last_decision()

    def test_the_prompt_hook_prints_nothing(self, monkeypatch, capsys):
        _feed(monkeypatch, {"prompt": "mail bob@example.org"})
        hooks.run_user_prompt_submit()
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""

    @pytest.mark.parametrize("cfg", [{}, {"value_origin": False}, {"value_origin": "yes"}])
    def test_off_unless_switched_on(self, monkeypatch, cfg):
        monkeypatch.setattr(hooks, "load_config", lambda: cfg)
        self._read_mail(monkeypatch)
        assert self.store.entries()["external"] == {}
        assert self._pay(monkeypatch, ATTACKER_IBAN) == 0

    def test_a_file_in_the_project_is_not_external(self, tmp_path):
        assert not hooks._external_result(
            "Read", {"file_path": str(tmp_path / "a.txt")}, str(tmp_path))
        assert hooks._external_result(
            "Read", {"file_path": "/srv/downloads/invoice.txt"}, str(tmp_path))
        assert not hooks._external_result(
            "Read", {"file_path": str(Path.home() / ".claude" / "x.md")}, str(tmp_path))
