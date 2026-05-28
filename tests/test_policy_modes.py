"""Tests for `vaara.policy.modes` preset bundles and the `vaara mode` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.cli import main
from vaara.policy import from_dict, from_json, from_yaml
from vaara.policy.modes import (
    Mode,
    available_modes,
    emit_json,
    emit_yaml,
    get_mode,
    to_policy_dict,
)


CANONICAL_NAMES = ("eco", "balanced", "performance", "strict")


class TestModeRegistry:
    def test_available_modes_canonical_order(self) -> None:
        assert available_modes() == CANONICAL_NAMES

    @pytest.mark.parametrize("name", CANONICAL_NAMES)
    def test_get_mode_returns_dataclass(self, name: str) -> None:
        m = get_mode(name)
        assert isinstance(m, Mode)
        assert m.name == name
        assert 0.0 <= m.escalate < m.deny <= 1.0
        assert m.description
        assert m.watt_profile

    def test_get_mode_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError) as exc:
            get_mode("turbo")
        msg = str(exc.value)
        for name in CANONICAL_NAMES:
            assert repr(name) in msg


class TestPolicyRoundTrip:
    @pytest.mark.parametrize("name", CANONICAL_NAMES)
    def test_to_policy_dict_round_trips_through_from_dict(
        self, name: str,
    ) -> None:
        m = get_mode(name)
        data = to_policy_dict(m)
        policy = from_dict(data)
        assert policy.thresholds_default.escalate == m.escalate
        assert policy.thresholds_default.deny == m.deny

    @pytest.mark.parametrize("name", CANONICAL_NAMES)
    def test_emit_json_round_trips_through_from_json(
        self, name: str, tmp_path: Path,
    ) -> None:
        text = emit_json(name)
        assert text.endswith("\n")
        # Indented for human readers.
        assert "  " in text
        # Round-trip through the loader.
        path = tmp_path / f"{name}.json"
        path.write_text(text, encoding="utf-8")
        policy = from_json(path)
        m = get_mode(name)
        assert policy.thresholds_default.escalate == m.escalate
        assert policy.thresholds_default.deny == m.deny

    @pytest.mark.parametrize("name", CANONICAL_NAMES)
    def test_emit_yaml_round_trips_through_from_yaml(
        self, name: str, tmp_path: Path,
    ) -> None:
        pytest.importorskip("yaml")
        text = emit_yaml(name)
        path = tmp_path / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        policy = from_yaml(path)
        m = get_mode(name)
        assert policy.thresholds_default.escalate == m.escalate
        assert policy.thresholds_default.deny == m.deny

    def test_emit_yaml_without_extra_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == "yaml":
                raise ImportError("simulated missing extra")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError) as exc:
            emit_yaml("balanced")
        assert "vaara[yaml]" in str(exc.value)


class TestModeCLI:
    def test_mode_list_prints_all_modes(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["mode", "list"])
        out = capsys.readouterr().out
        assert rc == 0
        for name in CANONICAL_NAMES:
            assert name in out

    def test_mode_show_prints_thresholds(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["mode", "show", "balanced"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mode:" in out
        assert "0.55" in out
        assert "0.85" in out

    def test_mode_show_unknown_exits_nonzero(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["mode", "show", "turbo"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "turbo" in err

    def test_mode_emit_json_to_stdout_round_trips(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["mode", "emit", "eco"])
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        policy = from_dict(data)
        assert policy.thresholds_default.escalate == 0.40
        assert policy.thresholds_default.deny == 0.60

    def test_mode_emit_yaml_to_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        pytest.importorskip("yaml")
        out_path = tmp_path / "strict.yaml"
        rc = main([
            "mode", "emit", "strict",
            "--format", "yaml",
            "--output", str(out_path),
        ])
        assert rc == 0
        # stdout stays clean when writing to file.
        assert capsys.readouterr().out == ""
        policy = from_yaml(out_path)
        m = get_mode("strict")
        assert policy.thresholds_default.escalate == m.escalate
        assert policy.thresholds_default.deny == m.deny

    def test_mode_emit_unknown_exits_nonzero(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["mode", "emit", "turbo"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "turbo" in err
