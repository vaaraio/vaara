# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The deployment chart has to agree with the software it deploys.

A Helm chart is a second copy of facts that already live in the package: the
version, the UID the image runs as, the path the trail is written to, the
flags the CLI accepts, the route a probe polls. Every one of those can drift
without anything failing until an operator installs the chart and reads a
crash loop.

Some of these checks are plain text and run everywhere. The rest need pyyaml
(an optional extra) or the helm binary, and skip without them. The
``helm-chart`` CI job installs both and fails if either is missing, so the
render is verified on every pull request rather than skipped into a green
build.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import vaara

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "vaara"
TEMPLATES = CHART / "templates"
DOCKERFILE = ROOT / "Dockerfile"

#: The image creates this user and the chart runs as it. One number, three
#: files, and a mismatch makes an existing volume unwritable.
EXPECTED_UID = 10001

#: Keys the templates read that Helm supplies or that only appear via
#: --set at install time.
BUILTIN_PREFIXES = ("Values.nameOverride", "Values.fullnameOverride")


def _yaml():
    """pyyaml is an optional extra, so the main test job does not have it.

    The dedicated `helm-chart` CI job installs it alongside helm, which is
    where these checks are guaranteed to run rather than skip.
    """
    return pytest.importorskip("yaml")


def _chart() -> dict:
    return _yaml().safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))


def _values() -> dict:
    return _yaml().safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))


def _template_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(TEMPLATES.rglob("*"))
        if path.is_file()
    )


def _lookup(values: dict, dotted: str) -> bool:
    node = values
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


# --- the chart agrees with the package -------------------------------------


def test_chart_appversion_matches_the_package():
    """A release that bumps vaara without bumping the chart ships a chart
    whose default image tag points at the previous version."""
    assert _chart()["appVersion"] == vaara.__version__


def test_every_value_a_template_reads_is_declared():
    """Catches `.Values.proxy.allowed` when the key is `allow`.

    Helm renders an undefined value as empty and carries on, so the typo
    reaches the cluster as a missing flag rather than an error.
    """
    values = _values()
    referenced = set(re.findall(r"\.Values\.([a-zA-Z0-9_.]+)", _template_text()))
    missing = sorted(
        ref for ref in referenced
        if not _lookup(values, ref) and not ref.startswith(BUILTIN_PREFIXES)
    )
    assert not missing, f"templates read values.yaml does not define: {missing}"


def test_the_uid_is_the_same_in_the_image_and_the_chart():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert f"--uid {EXPECTED_UID}" in dockerfile
    assert f"USER {EXPECTED_UID}:{EXPECTED_UID}" in dockerfile
    security = _values()["podSecurityContext"]
    assert security["runAsUser"] == EXPECTED_UID
    assert security["fsGroup"] == EXPECTED_UID, (
        "fsGroup must match the image UID or the mounted volume is not "
        "writable and the pod crash-loops on the first write"
    )


def test_the_trail_is_written_onto_the_mounted_volume():
    """A trail under / would be lost on restart, and the root filesystem is
    read-only, so it would not even be written."""
    values = _values()
    mount = "/var/lib/vaara"
    assert values["trail"]["path"].startswith(mount + "/")
    assert values["signing"]["receiptsDir"].startswith(mount)
    assert values["proxy"]["approvals"]["path"].startswith(mount)
    assert f"mountPath: {mount}" in (TEMPLATES / "statefulset.yaml").read_text()


def test_the_chart_does_not_offer_to_scale_the_chain():
    """One hash chain has one writer. There must be no replicaCount knob and
    the replica count must be literal."""
    statefulset = (TEMPLATES / "statefulset.yaml").read_text(encoding="utf-8")
    assert "kind: StatefulSet" in statefulset
    assert re.search(r"^\s+replicas: 1$", statefulset, re.M)
    # The prose in these files explains why there is no such knob, so look
    # for the knob itself rather than the word.
    assert ".Values.replicaCount" not in _template_text()
    assert "replicaCount" not in _values()


# --- the chart agrees with the CLI and the app -----------------------------


def test_probe_path_is_a_route_the_proxy_answers_itself():
    """The probe must not be forwarded to the model server."""
    app_source = (
        ROOT / "src" / "vaara" / "integrations" / "_infer_proxy_app.py"
    ).read_text(encoding="utf-8")
    assert '@app.get("/healthz")' in app_source
    assert "path: /healthz" in (TEMPLATES / "statefulset.yaml").read_text()


def test_chart_flags_exist_in_the_cli():
    """Every `--flag` the templates pass has to be one `vaara proxy` accepts."""
    script = Path(__import__("sys").executable).parent / "vaara"
    if not script.exists():
        script_path = shutil.which("vaara")
        if not script_path:
            pytest.skip("vaara console script is not installed")
        script = Path(script_path)
    help_text = subprocess.run(
        [str(script), "proxy", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    accepted = set(re.findall(r"(--[a-z][\w-]+)", help_text.stdout + help_text.stderr))
    assert accepted, "vaara proxy --help produced no flags"
    used = set(re.findall(r"- (--[a-z][\w-]+)", (TEMPLATES / "statefulset.yaml").read_text()))
    assert used, "the statefulset passes no flags at all"
    assert used <= accepted, f"chart passes flags vaara proxy rejects: {sorted(used - accepted)}"


# --- the documented platforms agree with the chart -------------------------


def test_supported_platforms_names_the_image_and_the_chart_version():
    """SUSE Ready certification requires published documentation naming the
    supported versions. If the page and the chart disagree, the published
    claim is the wrong one."""
    page = (ROOT / "docs" / "supported-platforms.md").read_text(encoding="utf-8")
    chart = _chart()
    assert vaara.__version__ in page
    assert f"chart {chart['version']}" in page or f"chart version {chart['version']}" in page
    assert _values()["image"]["repository"] in page


# --- the real thing, when helm is available --------------------------------


def _helm() -> str:
    path = shutil.which("helm") or str(Path.home() / ".local" / "bin" / "helm")
    if not Path(path).exists():
        pytest.skip("helm is not installed")
    return path


def test_helm_lint_passes():
    done = subprocess.run(
        [_helm(), "lint", str(CHART)], capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_default_render_is_valid_yaml_and_holds_one_replica():
    done = subprocess.run(
        [_helm(), "template", "release", str(CHART)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    docs = [doc for doc in _yaml().safe_load_all(done.stdout) if doc]
    kinds = {doc["kind"] for doc in docs}
    assert {"StatefulSet", "Service", "ServiceAccount"} <= kinds
    sts = next(doc for doc in docs if doc["kind"] == "StatefulSet")
    assert sts["spec"]["replicas"] == 1
    container = sts["spec"]["template"]["spec"]["containers"][0]
    assert container["args"][0] == "proxy"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"


def test_enforce_without_an_allow_list_refuses_to_render():
    """The CLI refuses this at startup. The chart refuses it at install, so
    the operator reads the reason instead of a crash loop."""
    done = subprocess.run(
        [_helm(), "template", "release", str(CHART), "--set", "proxy.mode=enforce"],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode != 0
    assert "proxy.allow" in done.stderr


def test_signing_without_a_secret_refuses_to_render():
    done = subprocess.run(
        [_helm(), "template", "release", str(CHART), "--set", "signing.enabled=true"],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode != 0
    assert "signing.existingSecret" in done.stderr


def test_signing_mounts_the_named_secret_read_only():
    done = subprocess.run(
        [
            _helm(), "template", "release", str(CHART),
            "--set", "signing.enabled=true",
            "--set", "signing.existingSecret=vaara-signing",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    sts = next(
        doc for doc in _yaml().safe_load_all(done.stdout)
        if doc and doc["kind"] == "StatefulSet"
    )
    spec = sts["spec"]["template"]["spec"]
    volume = next(v for v in spec["volumes"] if v["name"] == "signing-key")
    assert volume["secret"]["secretName"] == "vaara-signing"
    mount = next(
        m for m in spec["containers"][0]["volumeMounts"] if m["name"] == "signing-key"
    )
    assert mount["readOnly"] is True


def test_the_helm_test_pod_checks_vaara_not_the_model():
    done = subprocess.run(
        [_helm(), "template", "release", str(CHART)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    pod = next(
        doc for doc in _yaml().safe_load_all(done.stdout)
        if doc and doc["kind"] == "Pod"
    )
    assert pod["metadata"]["annotations"]["helm.sh/hook"] == "test"
    command = json.dumps(pod["spec"]["containers"][0]["command"])
    assert "/healthz" in command
