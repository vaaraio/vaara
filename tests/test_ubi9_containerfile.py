# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The UBI 9 image has to pass Red Hat's preflight container checks.

preflight (github.com/redhat-openshift-ecosystem/openshift-preflight) runs
against a built image, which CI does not build. These are the same checks
read off the Containerfile as text, so a change that would fail
certification fails here first: a UBI base in every stage
(BasedOnUbi), a non-root user (RunAsNonRoot), the required labels with no
Red Hat mark in them (HasRequiredLabel), licence texts under /licenses
(HasLicense) and fewer than 40 layers (LayerCountAcceptable).

OpenShift's restricted-v2 SCC runs a pod under an arbitrary UID with GID 0,
so the trail directory also has to be group 0 and group writable, or the
proxy cannot open its database there.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINERFILE = ROOT / "packaging" / "ubi9" / "Containerfile"

#: Same UID as Dockerfile and the Helm chart (see test_helm_chart.py).
EXPECTED_UID = 10001

#: preflight HasRequiredLabel.
REQUIRED_LABELS = (
    "name",
    "maintainer",
    "vendor",
    "version",
    "release",
    "summary",
    "description",
)

#: Instructions that add a filesystem layer to the final image.
LAYER_INSTRUCTIONS = ("RUN", "COPY", "ADD")


def _instructions() -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) pairs with continuations joined."""
    text = CONTAINERFILE.read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", " ", text)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, rest = line.partition(" ")
        out.append((word.upper(), rest.strip()))
    return out


def _final_stage() -> list[tuple[str, str]]:
    ins = _instructions()
    last_from = max(i for i, (word, _) in enumerate(ins) if word == "FROM")
    return ins[last_from:]


def _labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for word, rest in _final_stage():
        if word != "LABEL":
            continue
        for key, value in re.findall(r'([\w.\-]+)="([^"]*)"', rest):
            labels[key] = value
    return labels


def test_every_stage_is_built_on_ubi9():
    bases = [rest.split()[0] for word, rest in _instructions() if word == "FROM"]
    assert bases, "no FROM line"
    for base in bases:
        assert base.startswith("registry.access.redhat.com/ubi9/"), base


def test_the_image_runs_as_the_shared_non_root_uid():
    users = [rest for word, rest in _final_stage() if word == "USER"]
    assert users, "final stage sets no USER, so it runs as root"
    uid = users[-1].split(":")[0]
    assert uid.isdigit(), "preflight needs a numeric UID to prove non-root"
    assert int(uid) == EXPECTED_UID
    assert int(uid) != 0


def test_required_labels_are_present_and_not_empty():
    labels = _labels()
    for key in REQUIRED_LABELS:
        assert labels.get(key, "").strip(), f"label {key} missing or empty"


def test_no_label_starts_with_a_red_hat_mark():
    for key, value in _labels().items():
        assert not value.lower().startswith("red hat"), (key, value)
        assert not value.lower().startswith("redhat"), (key, value)


def test_licence_texts_land_in_slash_licenses():
    copies = [rest for word, rest in _final_stage() if word == "COPY"]
    assert any("LICENSE" in c and "/licenses/" in c for c in copies), copies


def test_final_image_adds_fewer_than_forty_layers():
    added = [w for w, _ in _final_stage() if w in LAYER_INSTRUCTIONS]
    # The UBI base brings its own one or two layers on top of these.
    assert len(added) < 35, len(added)


def test_trail_directory_is_writable_by_an_arbitrary_uid_in_group_zero():
    runs = " ".join(rest for word, rest in _final_stage() if word == "RUN")
    assert re.search(r"chgrp -R 0 /var/lib/vaara", runs), "trail dir not group 0"
    assert re.search(r"chmod -R g=u /var/lib/vaara", runs), "group cannot write"


def test_no_kernel_packages_are_installed():
    runs = " ".join(rest for word, rest in _instructions() if word == "RUN")
    assert not re.search(r"\bkernel[\w-]*\b", runs)


def test_image_build_context_excludes_the_private_exchange():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".shared/" in ignore
