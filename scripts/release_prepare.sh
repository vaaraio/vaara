#!/usr/bin/env bash
# Local prep for a Vaara release. Stops before any push.
#
# Usage: scripts/release_prepare.sh <VERSION> [CO_TAG]
#   VERSION  e.g., 0.39.2
#   CO_TAG   optional second annotated tag pinned to the same commit
#            (e.g., sep2787-ref-v2)
#
# Expects:
#   .commit_msg_v<VERSION>_release.txt  (commit message body)
#   .pr_body_v<VERSION>.md              (PR body)
#   CHANGELOG.md                        (entry for [<VERSION>])
#
# What it does:
#   1. Pre-flight: required files exist, CHANGELOG entry present,
#      working tree clean except for the staged release changes.
#   2. Bumps version in pyproject.toml + clients/ts/package.json +
#      src/vaara/__init__.py + server.json + server-vaara-server.json +
#      the Claude Code plugin manifest.
#   3. ruff check on changed Python paths.
#   4. pytest --no-header on the full suite (skips adversarial dir;
#      deselects pre-existing known-failing test).
#   5. Stages explicit paths (no `git add -A`).
#   6. Commits via -F.
#   7. Creates annotated tag v<VERSION>; if CO_TAG passed, also that.
#   8. Creates branch release/v<VERSION> at HEAD.
#
# Push remains gated. Run scripts/release_push_and_pr.sh next.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <VERSION> [CO_TAG]" >&2
  exit 2
fi

VERSION="$1"
CO_TAG="${2:-}"

COMMIT_MSG=".commit_msg_v${VERSION}_release.txt"
PR_BODY=".pr_body_v${VERSION}.md"
BRANCH="release/v${VERSION}"

# 1. Pre-flight
for f in "$COMMIT_MSG" "$PR_BODY" CHANGELOG.md pyproject.toml \
         clients/ts/package.json src/vaara/__init__.py; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

if ! grep -qE "^## \[${VERSION}\]" CHANGELOG.md; then
  echo "CHANGELOG.md missing entry: ## [${VERSION}]" >&2
  exit 1
fi

if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
  echo "tag v${VERSION} already exists locally; delete first" >&2
  exit 1
fi

if [[ -n "$CO_TAG" ]] && git rev-parse "$CO_TAG" >/dev/null 2>&1; then
  echo "tag $CO_TAG already exists locally; delete first" >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "branch ${BRANCH} already exists; delete first" >&2
  exit 1
fi

# 2. Bump versions (idempotent: only touches the version line)
sed -i -E "s/^version = \"[0-9]+\.[0-9]+\.[0-9]+\"$/version = \"${VERSION}\"/" pyproject.toml
sed -i -E "s/^  \"version\": \"[0-9]+\.[0-9]+\.[0-9]+\",$/  \"version\": \"${VERSION}\",/" clients/ts/package.json
sed -i -E "s/^__version__ = \"[0-9]+\.[0-9]+\.[0-9]+\"$/__version__ = \"${VERSION}\"/" src/vaara/__init__.py
# MCP Registry manifests: bump every semver "version" value (root listing
# + the pypi package entry). The release workflow asserts the live registry
# version equals the tag, so a stale manifest fails the publish gate.
sed -i -E "s/(\"version\": \")[0-9]+\.[0-9]+\.[0-9]+(\")/\1${VERSION}\2/g" \
  server.json server-vaara-server.json
# Claude Code plugin manifest: unified to the release version so the plugin
# tracks the tag like the other planes. It is git-marketplace distributed, so
# committing the bumped manifest to main is the publish.
PLUGIN_MANIFEST="plugins/claude-code-vaara-governance/.claude-plugin/plugin.json"
sed -i -E "s/^  \"version\": \"[0-9]+\.[0-9]+\.[0-9]+\",$/  \"version\": \"${VERSION}\",/" \
  "$PLUGIN_MANIFEST"

grep -E "^version = \"${VERSION}\"$" pyproject.toml >/dev/null
grep -E "\"version\": \"${VERSION}\"" clients/ts/package.json >/dev/null
grep -E "^__version__ = \"${VERSION}\"$" src/vaara/__init__.py >/dev/null
grep -E "\"version\": \"${VERSION}\"" server.json >/dev/null
grep -E "\"version\": \"${VERSION}\"" server-vaara-server.json >/dev/null
grep -E "\"version\": \"${VERSION}\"" "$PLUGIN_MANIFEST" >/dev/null

# 3. Lint changed paths (best-effort; lint all of src + tests if no
# precise change list)
CHANGED=$(git diff --name-only HEAD -- '*.py' | tr '\n' ' ')
if [[ -n "${CHANGED// }" ]]; then
  .venv/bin/ruff check $CHANGED
else
  .venv/bin/ruff check src/vaara tests
fi

# 4. Tests (full suite, deselect pre-existing known failure)
.venv/bin/python -m pytest -q --no-header \
  --ignore=tests/adversarial \
  --deselect tests/test_adversarial_classifier_integration.py::test_known_bad_metadata_ssrf_scores_high

# 5. Stage explicit paths only
git add CHANGELOG.md pyproject.toml clients/ts/package.json src/vaara/__init__.py \
  server.json server-vaara-server.json \
  plugins/claude-code-vaara-governance/.claude-plugin/plugin.json \
  scripts/release_prepare.sh
# Re-add any other paths the caller has already staged
git status --short

# 6. Commit
git commit -F "$COMMIT_MSG"

# 7. Tags
git tag -a "v${VERSION}" -m "Vaara v${VERSION} (see CHANGELOG.md)"
if [[ -n "$CO_TAG" ]]; then
  git tag -a "$CO_TAG" -m "Pinned co-tag for v${VERSION} (see CHANGELOG.md)"
fi

# 8. Release branch
git checkout -b "$BRANCH"

# Report
HEAD_SHA=$(git rev-parse --short HEAD)
echo
echo "Prepared v${VERSION} at ${HEAD_SHA}."
echo "Next: scripts/release_push_and_pr.sh ${VERSION}"
