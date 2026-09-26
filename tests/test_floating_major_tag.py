"""The floating major tag the docs pin to moves with each release.

The docs tell consumers `uses: vaaraio/vaara@v<major>`, and the 1.66 changelog
said the tag moves with each release. Nothing moved it: v1 stayed on the commit
it was created at while 1.67 through 1.99 shipped. The release job now moves
it, and the container build ignores it so the move cannot re-push an image.
"""
from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _job(text: str, name: str) -> str:
    match = re.search(rf"^  {name}:\n(.*?)(?=^  [a-z][a-z_-]*:\n|\Z)", text, re.M | re.S)
    assert match, f"job {name} not found"
    return match.group(1)


def test_the_release_job_moves_the_major_tag_on_a_tag_push_only():
    job = _job((WORKFLOWS / "release.yml").read_text(encoding="utf-8"), "github-release")
    step = job[job.index("Move the floating major tag"):]
    assert "if: github.event_name == 'push'" in step
    assert 'MAJOR="${GITHUB_REF_NAME%%.*}"' in step
    assert 'git push -f origin "refs/tags/${MAJOR}"' in step
    assert "contents: write" in job


def test_the_major_tag_matches_neither_workflow_trigger():
    for name in ("release.yml", "container.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        tags = re.search(r"^  push:\n    tags:\n((?:      .*\n)+)", text, re.M)
        assert tags, name
        patterns = re.findall(r"^      - ['\"]?([^'\"\n]+)['\"]?$", tags.group(1), re.M)
        assert patterns == ["v[0-9]+.[0-9]+.[0-9]+"], (name, patterns)
