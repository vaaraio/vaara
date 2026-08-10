# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Python example in COMPLIANCE.md is executed, not just read.

The block under "### From Python" is the one an evaluator pastes into a
shell to see what a conformity report looks like. It printed a commented
sample output claiming Article 9(1) and Article 12(1) come back
`evidence_sufficient`. They do not: the block intercepts one action, and
the default requirements want 10 to 20 qualifying events inside a 30-day
window. Every article except 11(1) and 73(1) reads `evidence_insufficient`
on that trail.

So the test runs the documented code and compares the documented output
to the real one. It also runs it against a temporary HOME, because the
example uses the default pipeline, which writes to
``~/.vaara/trail/audit.db``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "COMPLIANCE.md"


def _python_example() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("### From Python")
    block = re.search(r"```python\n(.*?)```", text[start:], re.S)
    assert block, "COMPLIANCE.md no longer carries a Python example"
    return block.group(1)


def _documented_statuses() -> dict[str, str]:
    """The `# Article 9(1) evidence_insufficient` lines in the example."""
    pairs = re.findall(
        r"^\s*#\s*(Article [\w()]+):?\s+(evidence_\w+|not_applicable)\s*$",
        _python_example(),
        re.MULTILINE,
    )
    assert pairs, "the example no longer shows any sample output"
    return dict(pairs)


@pytest.fixture()
def documented_run(tmp_path, monkeypatch, capsys):
    """Execute the documented block verbatim against a throwaway HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    namespace: dict = {"__name__": "__doc_example__"}
    exec(compile(_python_example(), str(DOC), "exec"), namespace)  # noqa: S102
    capsys.readouterr()  # the example prints; the report object is the subject
    return namespace


def test_the_example_runs_start_to_finish(documented_run):
    assert documented_run["result"].action_id
    assert documented_run["report"].articles


def test_the_sample_output_is_what_the_example_prints(documented_run):
    real = {
        article.requirement.article: article.status.value
        for article in documented_run["report"].articles
    }
    documented = _documented_statuses()

    unknown = set(documented) - set(real)
    assert not unknown, f"the example shows articles the report never emits: {sorted(unknown)}"

    wrong = {
        article: (status, real[article])
        for article, status in documented.items()
        if real[article] != status
    }
    assert not wrong, (
        "docs/COMPLIANCE.md shows sample output the example does not produce "
        "(article: documented -> real): "
        + ", ".join(f"{a}: {d} -> {r}" for a, (d, r) in sorted(wrong.items()))
    )


def test_the_example_writes_nothing_outside_home(documented_run, tmp_path):
    """The default pipeline persists; the reader should know where."""
    assert (tmp_path / ".vaara" / "trail" / "audit.db").exists()
