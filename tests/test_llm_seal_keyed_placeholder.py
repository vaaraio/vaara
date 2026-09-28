# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""A placeholder is a keyed digest of its secret, not a plain one.

The placeholder used to be the first 48 bits of sha256(secret). The provider
sees every placeholder, and a named secret is often a phrase or a concept
name rather than a random token, so anyone holding the prompt could confirm
a guess at the secret offline by hashing candidates. The placeholder is now
HMAC-SHA256 under a key that lives beside the seal file, so it is as stable
as before across restarts and says nothing about the secret without the key.
"""
from __future__ import annotations

import hashlib
import os
import stat

from vaara.integrations.llm_seal import (
    PLACEHOLDER_LEN,
    SealRegistry,
    placeholder_for,
)

KEY_A = bytes(range(32))
KEY_B = bytes(reversed(range(32)))


def test_placeholder_is_keyed_and_says_nothing_without_the_key():
    secret = "northern lights"
    a = placeholder_for(secret, KEY_A)
    assert a == placeholder_for(secret, KEY_A)
    assert a != placeholder_for(secret, KEY_B)
    assert len(a) == PLACEHOLDER_LEN
    unkeyed = hashlib.sha256(secret.encode()).hexdigest()[:12]
    assert unkeyed not in a


def test_a_registry_seals_with_its_own_key_and_restores():
    reg = SealRegistry({"concept": "northern lights"}, key=KEY_A)
    sealed = reg.seal_bytes(b'{"m": "the northern lights plan"}')
    assert b"northern lights" not in sealed
    assert reg.placeholder("northern lights").encode() in sealed
    assert placeholder_for("northern lights", KEY_B).encode() not in sealed
    assert reg.unseal_bytes(sealed) == b'{"m": "the northern lights plan"}'


def test_two_registries_without_a_shared_key_do_not_agree():
    one = SealRegistry({"c": "northern lights"})
    two = SealRegistry({"c": "northern lights"})
    assert one.placeholder("northern lights") != two.placeholder("northern lights")


def test_a_file_backed_registry_keeps_its_placeholders_across_restarts(tmp_path):
    seal = tmp_path / "seal.json"
    seal.write_text('{"concept": "northern lights"}')
    first = SealRegistry.from_file(seal)
    token = first.placeholder("northern lights")
    key_file = tmp_path / "keys" / "seal-hmac.key"
    assert key_file.is_file()
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    second = SealRegistry.from_file(seal)
    assert second.placeholder("northern lights") == token
    assert second.seal_bytes(b"northern lights") == token.encode()


def test_learned_values_use_the_same_key(tmp_path):
    seal = tmp_path / "seal.json"
    seal.write_text("{}")
    reg = SealRegistry.from_file(seal)
    reg.known_formats = True
    key = "sk-ant-" + "a1B2c3D4" * 4
    sealed = reg.seal_bytes(key.encode())
    assert sealed == reg.placeholder(key).encode()
    assert reg.unseal_bytes(sealed) == key.encode()


def test_the_deny_rules_keep_an_agent_away_from_the_seal_key():
    """The key lands under keys/ so the shipped rules that guard the receipt
    and approval keys guard it too, on the read and on the shell path."""
    from vaara.deny_rules import load_deny_rules, match_deny_rule

    rules = load_deny_rules()
    path = "/home/h/.vaara/llm-proxy/keys/seal-hmac.key"
    assert match_deny_rule(rules, "Read", {"file_path": path}) is not None
    assert match_deny_rule(rules, "Bash", {"command": f"cat {path}"}) is not None
    assert match_deny_rule(rules, "Write", {"file_path": path}) is not None
