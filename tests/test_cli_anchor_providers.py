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
    )
]


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
