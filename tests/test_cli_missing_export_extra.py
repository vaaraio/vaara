"""`vaara trail export` and `vaara trail verify` on an install without the
`[export]` extra must print the install hint and exit 2, like `vaara keygen`
does, instead of dumping an ImportError traceback.

Found by a clean PyPI install on 2026-09-23: `pip install vaara` then the
README's own `vaara trail export` crashed with a traceback.
"""
from __future__ import annotations

import pytest

from vaara import cli
from vaara.audit import export as export_mod
from vaara.audit import verify as verify_mod


def _main(argv):
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return exc.code


def test_trail_export_without_crypto_exits_clean(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_mod, "_HAS_CRYPTO", False)
    trail = tmp_path / "t.jsonl"
    trail.write_text("")
    rc = _main(["trail", "export", "--trail", str(trail),
                "--out", str(tmp_path / "x.zip"), "--key", str(tmp_path / "k.pem")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "vaara[export]" in err
    assert "Traceback" not in err


def test_trail_verify_without_crypto_exits_clean(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(verify_mod, "_HAS_CRYPTO", False)
    bogus = tmp_path / "any.zip"
    bogus.write_bytes(b"crypto check fires first")
    rc = _main(["trail", "verify", "--zip", str(bogus)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "vaara[export]" in err
    assert "Traceback" not in err
