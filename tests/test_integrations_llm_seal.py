"""Tests for reversible sealing on the egress path.

The property that matters: a secret named in advance never reaches the
provider, and the caller still reads a coherent response.  Everything fails
open, so a sealing fault costs nothing but the sealing.
"""

from __future__ import annotations

import json

from vaara.integrations.llm_seal import (
    PLACEHOLDER_LEN,
    SealRegistry,
    StreamUnsealer,
    placeholder_for,
)


class TestPlaceholder:
    def test_is_stable_across_calls(self):
        assert placeholder_for("northern lights") == placeholder_for("northern lights")

    def test_differs_per_secret(self):
        assert placeholder_for("a") != placeholder_for("b")

    def test_length_matches_the_published_constant(self):
        assert len(placeholder_for("anything")) == PLACEHOLDER_LEN


class TestRoundTrip:
    def test_secret_does_not_survive_sealing(self):
        reg = SealRegistry({"concept": "anti-note"})
        sealed = reg.seal_bytes(b'{"messages":[{"content":"the anti-note idea"}]}')
        assert b"anti-note" not in sealed
        assert placeholder_for("anti-note").encode() in sealed

    def test_unseal_restores_the_original(self):
        reg = SealRegistry({"concept": "anti-note"})
        raw = b'{"content":"the anti-note idea"}'
        assert reg.unseal_bytes(reg.seal_bytes(raw)) == raw

    def test_overlapping_secrets_seal_longest_first(self):
        reg = SealRegistry({"short": "mesh", "long": "lightmesh atom"})
        sealed = reg.seal_bytes(b"the lightmesh atom runs")
        assert b"lightmesh atom" not in sealed
        assert reg.unseal_bytes(sealed) == b"the lightmesh atom runs"

    def test_json_escaped_form_is_also_sealed(self):
        secret = 'he said "go"'
        reg = SealRegistry({"quoted": secret})
        body = json.dumps({"content": f"and then {secret} loudly"}).encode()
        sealed = reg.seal_bytes(body)
        assert json.dumps(secret)[1:-1].encode() not in sealed

    def test_inactive_registry_is_a_passthrough(self):
        reg = SealRegistry()
        assert not reg.active
        assert reg.seal_bytes(b"anything") == b"anything"
        assert reg.unseal_bytes(b"anything") == b"anything"


class TestFailOpen:
    def test_undecodable_bytes_pass_through_unchanged(self):
        reg = SealRegistry({"concept": "anti-note"})
        raw = b"\xff\xfe not utf-8 at all"
        assert reg.seal_bytes(raw) == raw

    def test_missing_registry_file_yields_inactive(self, tmp_path):
        assert not SealRegistry.from_file(tmp_path / "absent.json").active

    def test_unreadable_registry_file_yields_inactive(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        assert not SealRegistry.from_file(p).active

    def test_non_object_registry_file_yields_inactive(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text('["a","b"]', encoding="utf-8")
        assert not SealRegistry.from_file(p).active

    def test_empty_secret_is_skipped_rather_than_sealing_everything(self):
        reg = SealRegistry({"blank": ""})
        assert not reg.active
        assert reg.seal_bytes(b"untouched") == b"untouched"


class TestReceiptInputs:
    def test_counts_placeholders_in_a_sealed_body(self):
        reg = SealRegistry({"a": "alpha", "b": "beta"})
        sealed = reg.seal_bytes(b"alpha and beta and alpha")
        assert reg.count_sealed(sealed) == 3

    def test_unmapped_placeholder_is_reported(self):
        reg = SealRegistry({"a": "alpha"})
        foreign = placeholder_for("a secret this process never held")
        assert reg.unmapped_placeholders(foreign.encode()) == [foreign]

    def test_own_placeholders_are_not_reported_as_unmapped(self):
        reg = SealRegistry({"a": "alpha"})
        assert reg.unmapped_placeholders(reg.seal_bytes(b"alpha")) == []


class TestStreamUnsealer:
    def test_restores_a_placeholder_split_across_chunks(self):
        reg = SealRegistry({"concept": "anti-note"})
        token = placeholder_for("anti-note")
        whole = f"the {token} idea".encode()
        cut = len(b"the ") + 5  # mid-placeholder
        un = StreamUnsealer(reg)
        out = un.feed(whole[:cut]) + un.feed(whole[cut:]) + un.flush()
        assert out == b"the anti-note idea"

    def test_byte_at_a_time_still_restores(self):
        reg = SealRegistry({"concept": "anti-note"})
        token = placeholder_for("anti-note")
        whole = f"x{token}y".encode()
        un = StreamUnsealer(reg)
        out = b"".join(un.feed(whole[i:i + 1]) for i in range(len(whole)))
        assert out + un.flush() == b"xanti-notey"

    def test_inactive_registry_streams_unchanged(self):
        un = StreamUnsealer(SealRegistry())
        assert un.feed(b"chunk") == b"chunk"
        assert un.flush() == b""

    def test_nothing_is_lost_when_no_placeholder_is_present(self):
        reg = SealRegistry({"concept": "anti-note"})
        un = StreamUnsealer(reg)
        parts = [b"hello ", b"there ", b"friend"]
        out = b"".join(un.feed(p) for p in parts) + un.flush()
        assert out == b"hello there friend"
