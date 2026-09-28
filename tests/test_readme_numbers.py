"""Numbers the READMEs state are read from the code, and fail when it moves.

The README said its quick-start example raises on a fresh install. That was
true under the 0.40 / 0.70 defaults and stopped being true in 1.94.0, when
the defaults became the balanced mode's 0.55 / 0.85, and nothing noticed.
The plugin README listed an enumerated matcher two releases after the
matcher became a catch-all. Each claim here is recomputed from the code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
PLUGIN = ROOT / "plugins" / "claude-code-vaara-governance"
PLUGIN_README = (PLUGIN / "README.md").read_text()


def test_quickstart_example_runs_as_the_readme_says(tmp_path, monkeypatch):
    import importlib
    import sqlite3

    import vaara

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The bare decorator uses one process-wide default pipeline. An earlier
    # test may have built it on another trail, so start from none, which is
    # what a fresh install has.
    monkeypatch.setattr(importlib.import_module("vaara.govern"),
                        "_default_pipeline", None)

    @vaara.govern
    def transfer_funds(to: str, amount: float) -> str:
        return "sent"

    assert transfer_funds("x", 1.0) == "sent"
    db = tmp_path / ".vaara" / "trail" / "audit.db"
    (data,) = sqlite3.connect(db).execute(
        "select data from audit_records where event_type='decision_made'"
    ).fetchone()
    reason = json.loads(data)["reason"]
    m = re.search(r"risk=([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", reason)
    assert m, reason
    point, lo, hi = m.groups()
    assert f"scores {point}" in README
    assert f"[{lo}, {hi}]" in README
    assert "it will raise" not in README


def test_plugin_readme_counts_match_the_rule_file():
    rules = json.loads((PLUGIN / "policies" / "default_deny.json").read_text())
    rules = rules if isinstance(rules, list) else rules["rules"]
    tools: set[str] = set()
    for r in rules:
        for key in ("tools", "match_any"):
            if isinstance(r.get(key), list):
                tools.update(r[key])
    assert f"{len(rules)} rules over {len(tools)} tools" in PLUGIN_README


def test_plugin_readme_matchers_match_hooks_json():
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    for event in ("PreToolUse", "PostToolUse"):
        matchers = {m.get("matcher") for m in hooks[event]}
        assert matchers == {".*"}, event
        row = re.search(rf"^\| `{event}` \| ([^|]+)\|", PLUGIN_README, re.M)
        assert row and "every tool" in row.group(1), event



def test_dependency_claims_match_pyproject():
    """The base install carries what pyproject says, and the docs say the same.

    2.2.0 moved cbor2, cryptography and rfc8785 into the base install so a
    plain pip install signs receipts. The README and the platforms page still
    said the package had no runtime dependencies.
    """
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    names = sorted(re.split(r"[<>=!~ ]", d, 1)[0] for d in project["dependencies"])
    platforms = (ROOT / "docs" / "supported-platforms.md").read_text()
    for text in (README, platforms):
        assert "zero runtime dependencies" not in text
        assert "no runtime dependencies" not in text
        assert "zero dependencies" not in text
        assert "none for the base install" not in text
    if names:
        for name in names:
            assert name in README, f"README does not name the base dependency {name}"
            assert name in platforms, f"supported-platforms.md does not name {name}"
    assert "vaara[export]" not in README, "the export extra is in the base install now"
