"""Credentials in no published format are sealed by where they sit.

``KNOWN_SECRET_FORMATS`` catches keys whose issuer publishes a prefix. An
internal service token, a vendor with no prefix, or a rotated secret with a
shape nobody listed went out as written. With ``known_formats`` on, a
high-entropy value assigned to a secret-named key, or sent as a bearer token,
is sealed too. The other half matters as much: hex digests, UUIDs, ids under
ordinary keys, token counts and template filler stay as written.
"""
from __future__ import annotations

import pytest

from vaara.integrations.llm_seal import (
    CONTEXT_SECRET_RULES,
    SealRegistry,
)

# Generated with secrets.token_urlsafe / token_hex; none has a known prefix.
OPAQUE = "q7Rz2LmX9vKp4TnW8bYc1HdF"
HEXTOKEN = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b"
B64 = "dGhpcyBpcyBub3QgYSByZWFsIGtleQ3kZ9=="


def _body(text: str) -> bytes:
    return ('{"messages":[{"role":"user","content":"' + text + '"}]}').encode()


@pytest.mark.parametrize("text,value,kind", [
    (f"API_KEY={OPAQUE}", OPAQUE, "assigned_secret"),
    (f"export INTERNAL_SERVICE_TOKEN={HEXTOKEN}", HEXTOKEN, "assigned_secret"),
    (f"client_secret: {B64}", B64, "assigned_secret"),
    (f"x-api-key: {OPAQUE}", OPAQUE, "assigned_secret"),
    (f'config {{\\"db_password\\": \\"{OPAQUE}\\"}}', OPAQUE, "assigned_secret"),
    (f"curl -H 'Authorization: Bearer {OPAQUE}' https://x", OPAQUE, "bearer_token"),
])
def test_an_unknown_token_is_sealed_and_restored(text, value, kind):
    reg = SealRegistry(known_formats=True)
    raw = _body(text)
    out = reg.seal_bytes(raw)
    assert value.encode() not in out
    assert reg.placeholder(value).encode() in out
    assert reg.last_kinds == {kind: 1}
    assert reg.unseal_bytes(out) == raw


def test_only_the_value_is_sealed_not_the_key():
    reg = SealRegistry(known_formats=True)
    out = reg.seal_bytes(_body(f"STRIPE_WEBHOOK_SECRET={OPAQUE}")).decode()
    assert "STRIPE_WEBHOOK_SECRET=" + reg.placeholder(OPAQUE) in out


@pytest.mark.parametrize("text", [
    # Hex and ids under ordinary keys.
    f"commit {HEXTOKEN}",
    f"sha256: {HEXTOKEN}",
    f"request_id: {OPAQUE}",
    "id: 550e8400-e29b-41d4-a716-446655440000",
    f"{OPAQUE} appears in the log",
    # Secret-ish words in the key that do not end it.
    "max_tokens: 4096000000000000000",
    f"token_count={OPAQUE}",
    "secret_name: prod-database-password-rotation",
    f"tokenizer: {OPAQUE}",
    "author: AbcdefghIjklmnop1",
    # Right key, value is not a generated token.
    "password: changeme",
    "API_KEY=YOUR_API_KEY_HERE_PLEASE",
    "API_KEY=${OPENAI_API_KEY}",
    "api_key: aaaaaaaaaaaaaaaaAAAA1111",
    "token: correct-horse-battery-staple",
    "PWD=/home/runner/work/Proj3ct/Proj3ct",
    # Prose.
    "the password is stored in the vault, token rotation is weekly",
    "Bearer tokens expire after an hour",
])
def test_ordinary_text_is_left_alone(text):
    reg = SealRegistry(known_formats=True)
    raw = _body(text)
    assert reg.seal_bytes(raw) == raw, reg.last_kinds
    assert reg.last_kinds == {}


def test_a_known_format_is_counted_once_not_twice():
    key = "ghp_" + "Ab1" * 12
    reg = SealRegistry(known_formats=True)
    out = reg.seal_bytes(_body(f"GITHUB_TOKEN={key}"))
    assert reg.last_kinds == {"github_token": 1}
    assert out.count(b"VAARA_SEAL_") == 1


def test_off_without_known_formats():
    reg = SealRegistry()
    raw = _body(f"API_KEY={OPAQUE}")
    assert reg.seal_bytes(raw) == raw


def test_rule_names_are_distinct():
    names = [k for k, _ in CONTEXT_SECRET_RULES]
    assert len(names) == len(set(names))
