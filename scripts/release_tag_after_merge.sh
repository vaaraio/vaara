#!/usr/bin/env bash
# For when the release PR was merged via the GH UI (or any path that
# bypassed scripts/release_merge_and_tag.sh). Tags origin/main at the
# merged SHA and prints the gated push command.
#
# Usage: scripts/release_tag_after_merge.sh <VERSION> [CO_TAG]
#   VERSION    e.g., 0.40.4
#   CO_TAG     optional second annotated tag
#
# Differs from release_merge_and_tag.sh by skipping the gh pr checks
# watch and gh pr merge steps. Useful when the PR is already merged.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <VERSION> [CO_TAG]" >&2
  exit 2
fi

VERSION="$1"
CO_TAG="${2:-}"

# 1. Fetch the merged commit
git fetch origin main
MERGED_SHA=$(git rev-parse --short origin/main)
echo "Tagging origin/main at: $MERGED_SHA"

# 2. Move tags to the merged SHA
if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
  git tag -d "v${VERSION}"
fi
git tag -a "v${VERSION}" origin/main -m "Vaara v${VERSION} (see CHANGELOG.md)"

if [[ -n "$CO_TAG" ]]; then
  if git rev-parse "$CO_TAG" >/dev/null 2>&1; then
    git tag -d "$CO_TAG"
  fi
  git tag -a "$CO_TAG" origin/main -m "Pinned co-tag for v${VERSION} (see CHANGELOG.md)"
fi

# 3. Print the push command (gated)
echo
if [[ -n "$CO_TAG" ]]; then
  echo "Tags ready at ${MERGED_SHA}. To publish (fires Release workflow):"
  echo "  git push origin v${VERSION} ${CO_TAG}"
else
  echo "Tag ready at ${MERGED_SHA}. To publish (fires Release workflow):"
  echo "  git push origin v${VERSION}"
fi
echo
echo "If the Release workflow fails or is misconfigured, fallback:"
echo "  scripts/release_publish_manual.sh ${VERSION}"
echo "(Only when GH Actions confirmed broken, not for transient noise.)"
