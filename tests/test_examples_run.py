"""The standalone examples run, the way the docs tell a stranger to run them.

`examples/data_locality_demo.py` imported a module removed when the SEP-2787
code was renamed, so it crashed on the verification step it exists to show,
and no test ran it. Each example here is started as a subprocess with an empty
home directory, the state of a first run, and must exit 0 without a traceback.
Examples that need an API key, a model download or a live MCP server are not
in this list.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (path, modules it needs beyond the base install, skip reason)
EXAMPLES = [
    ("examples/prove-it-yourself/prove_it.py", [], ""),
    ("examples/quickstart.py", [], ""),
    ("examples/intercept.py", ["rich"], "examples need rich"),
    ("examples/governance_demo.py", ["rich"], "examples need rich"),
    ("examples/data_locality_demo.py", ["rich", "rfc8785"], "examples need rich"),
]


@pytest.mark.parametrize("path, needs, reason", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_example_runs(path, needs, reason, tmp_path):
    for module in needs:
        if module == "rfc8785":
            pytest.importorskip(
                module, reason="attestation extra not installed (pip install 'vaara[attestation]')")
        else:
            pytest.importorskip(module, reason=reason)
    env = {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "PATH": "/usr/bin:/bin",
           # Windows: OpenSSL draws entropy through the system directory.
           **{k: os.environ[k] for k in ("SYSTEMROOT", "SYSTEMDRIVE") if k in os.environ}}
    # Run the example against the code under test, not whatever vaara the
    # interpreter has installed (a worktree run sets PYTHONPATH).
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(p) for p in os.environ["PYTHONPATH"].split(os.pathsep) if p
        )
    proc = subprocess.run(
        [sys.executable, str(ROOT / path)],
        cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=180,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-2000:]
    assert "Traceback" not in output, output[-2000:]


def _run_prove_it(home: Path) -> str:
    env = {"HOME": str(home), "USERPROFILE": str(home), "PATH": "/usr/bin:/bin",
           **{k: os.environ[k] for k in ("SYSTEMROOT", "SYSTEMDRIVE") if k in os.environ}}
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(p) for p in os.environ["PYTHONPATH"].split(os.pathsep) if p
        )
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples/prove-it-yourself/prove_it.py")],
        cwd=home, env=env, capture_output=True, text=True, timeout=180,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-2000:]
    return output


def test_prove_it_bundles_only_its_own_run_on_a_machine_that_already_has_a_trail(tmp_path):
    """The demo ran on the machine's own trail, so where Vaara already ran, the
    bundle carried that whole history, and a break anywhere in it failed the
    honest bundle while the text under it said the signature checks out. Run it
    twice in one home: the first run leaves a trail behind, as Vaara in use
    would, and the second bundle must still hold only the second run's records."""
    import json
    import zipfile

    bundle = ROOT / "examples/prove-it-yourself/evidence.zip"
    counts = []
    for _ in range(2):
        output = _run_prove_it(tmp_path)
        assert "evidence.zip)  ->  ok=True" in output, output[-2000:]
        with zipfile.ZipFile(bundle) as z:
            counts.append(json.loads(z.read("manifest.json"))["record_count"])
    assert counts[0] == counts[1], counts
