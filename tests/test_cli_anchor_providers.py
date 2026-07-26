# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""`vaara anchor-providers --country CC` lists the EU qualified timestamping
providers from the official trusted list so an operator can pick one. It
endorses none and sets no default.
"""
import json

import vaara.audit.eu_trusted_list as etl
from vaara.audit.eu_trusted_list import QualifiedTSA
from vaara.cli import main

_SAMPLE = [
    QualifiedTSA(
        territory="AT",
        provider="ACME QTSP",
        service_name="ACME Qualified Timestamping",
        endpoint="https://tsa.example.at/tsa",
    ),
    QualifiedTSA(
        territory="AT",
        provider="Beta Trust",
        service_name="Beta Qualified TSA",
        endpoint="https://tsa.example.at/beta",
    ),
]


def _use_config(monkeypatch, tmp_path):
    """Point the shared plugin config at a temp file for both cli and menu."""
    import vaara.menu as menu

    cfg = tmp_path / "config.json"
    monkeypatch.setattr(menu, "CONFIG_PATH", cfg)
    return cfg


def test_anchor_providers_lists_qualified_tsas(monkeypatch, capsys):
    seen = {}

    def fake(country, fetch):
        seen["country"] = country
        return _SAMPLE

    monkeypatch.setattr(etl, "providers_for_country", fake)

    rc = main(["anchor-providers", "--country", "AT"])

    out = capsys.readouterr().out
    assert rc == 0
    assert seen["country"] == "AT"
    assert "ACME QTSP" in out
    assert "https://tsa.example.at/tsa" in out


def test_anchor_providers_json_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)

    rc = main(["anchor-providers", "--country", "AT", "--json"])

    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data[0]["endpoint"] == "https://tsa.example.at/tsa"
    assert data[0]["provider"] == "ACME QTSP"


def test_anchor_providers_numbered_so_a_choice_can_be_named(monkeypatch, capsys):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)

    rc = main(["anchor-providers", "--country", "AT"])

    out = capsys.readouterr().out
    assert rc == 0
    # Each provider is numbered so `--set N` and the app can reference one.
    assert "1)" in out and "2)" in out
    assert "Beta Trust" in out


def test_anchor_providers_set_writes_choice_to_shared_config(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)
    cfg = _use_config(monkeypatch, tmp_path)

    rc = main(["anchor-providers", "--country", "AT", "--set", "2"])

    assert rc == 0
    saved = json.loads(cfg.read_text())
    # The sample carries endpoints, so the listed one is used directly.
    assert saved["anchor_tsa_url"] == "https://tsa.example.at/beta"
    assert saved["anchor_provider"] == "Beta Trust"
    assert saved["anchor_service"] == "Beta Qualified TSA"
    assert saved["anchor_country"] == "AT"
    assert "Beta Trust" in capsys.readouterr().out


# The real EU trusted list almost never carries the RFC3161 request URL, so
# a provider with an empty endpoint is the common case, not an edge case.
_NO_ENDPOINT = [
    QualifiedTSA(
        territory="AT",
        provider="A-Trust GmbH",
        service_name="a-sign-premium-timestamping-10",
        endpoint="",
    )
]


def test_anchor_providers_set_without_endpoint_is_refused(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        etl, "providers_for_country", lambda country, fetch: _NO_ENDPOINT
    )
    cfg = _use_config(monkeypatch, tmp_path)

    rc = main(["anchor-providers", "--country", "AT", "--set", "1"])

    assert rc == 1
    assert not cfg.exists()
    assert "--endpoint" in capsys.readouterr().err


def test_anchor_providers_set_with_explicit_endpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        etl, "providers_for_country", lambda country, fetch: _NO_ENDPOINT
    )
    cfg = _use_config(monkeypatch, tmp_path)

    rc = main([
        "anchor-providers", "--country", "AT", "--set", "1",
        "--endpoint", "https://tsa.a-trust.at/tsp",
    ])

    assert rc == 0
    saved = json.loads(cfg.read_text())
    assert saved["anchor_tsa_url"] == "https://tsa.a-trust.at/tsp"
    assert saved["anchor_provider"] == "A-Trust GmbH"


def test_anchor_providers_set_preserves_other_config_keys(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)
    cfg = _use_config(monkeypatch, tmp_path)
    cfg.write_text(json.dumps({"user_level": "enterprise", "mode": "protect"}))

    rc = main(["anchor-providers", "--country", "AT", "--set", "1"])

    assert rc == 0
    saved = json.loads(cfg.read_text())
    assert saved["user_level"] == "enterprise"
    assert saved["mode"] == "protect"
    assert saved["anchor_tsa_url"] == "https://tsa.example.at/tsa"


def test_anchor_providers_set_out_of_range_is_an_error(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)
    _use_config(monkeypatch, tmp_path)

    rc = main(["anchor-providers", "--country", "AT", "--set", "9"])

    assert rc == 1
    assert "9" in capsys.readouterr().err


def test_anchor_providers_interactive_pick_sets_config(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)
    cfg = _use_config(monkeypatch, tmp_path)
    # Behave like a TTY and answer the prompt with "2".
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "2")

    rc = main(["anchor-providers", "--country", "AT"])

    assert rc == 0
    saved = json.loads(cfg.read_text())
    assert saved["anchor_tsa_url"] == "https://tsa.example.at/beta"


def test_anchor_providers_interactive_blank_skips_without_writing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(etl, "providers_for_country", lambda country, fetch: _SAMPLE)
    cfg = _use_config(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "")

    rc = main(["anchor-providers", "--country", "AT"])

    assert rc == 0
    assert not cfg.exists()


def test_anchor_providers_interactive_prompts_for_missing_url(
    monkeypatch, tmp_path
):
    """When the list carries no endpoint, picking a provider prompts for the
    RFC3161 URL, then writes it."""
    monkeypatch.setattr(
        etl, "providers_for_country", lambda country, fetch: _NO_ENDPOINT
    )
    cfg = _use_config(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["1", "https://tsa.a-trust.at/tsp"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    rc = main(["anchor-providers", "--country", "AT"])

    assert rc == 0
    saved = json.loads(cfg.read_text())
    assert saved["anchor_tsa_url"] == "https://tsa.a-trust.at/tsp"


def test_anchoring_falls_back_to_the_configured_provider(monkeypatch, tmp_path):
    """With no --anchor-tsa, the trail anchors against the provider the
    operator set via the picker, so the choice actually takes effect."""
    import types

    import vaara.audit.timeanchor as ta
    from vaara.cli import _obtain_time_anchor

    cfg = _use_config(monkeypatch, tmp_path)
    cfg.write_text(json.dumps({"anchor_tsa_url": "https://tsa.example.at/beta"}))

    seen = {}

    class FakeClient:
        def __init__(self, url):
            seen["url"] = url

        def anchor(self, position, head_hash):
            return "ANCHOR"

    monkeypatch.setattr(ta, "RFC3161TimeAnchorClient", FakeClient)

    rec = types.SimpleNamespace(record_hash="deadbeef")
    trail = types.SimpleNamespace(_records=[rec])
    args = types.SimpleNamespace(anchor_tsa=None, anchor_file=None)

    result = _obtain_time_anchor(args, trail)

    assert result == "ANCHOR"
    assert seen["url"] == "https://tsa.example.at/beta"
