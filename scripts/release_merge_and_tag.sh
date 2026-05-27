#!/usr/bin/env bash
# After CI is green, squash-merge the PR and re-tag at the merged SHA.
# Tag push remains gated.
#
# Usage: scripts/release_merge_and_tag.sh <PR_NUMBER> <VERSION> [CO_TAG]
#   PR_NUMBER  the open release PR
#   VERSION    e.g., 0.39.2
#   CO_TAG     optional second annotated tag
#
# Why re-tag: squash-merge writes a NEW commit on main (different SHA
# from the local pre-merge commit). The pre-merge local tags would
# point at a commit that no longer exists on main. We move them to the
# merged remote SHA.
#
# Why we do NOT sync local main: squash-merge diverges the histories
# (local pre-merge commit and the remote squash commit share the prior
# release as parent but have different SHAs). Syncing would require
# `git reset --hard origin/main`, which is on the destructive-ops list
# and needs explicit approval. Tagging against `origin/main` directly
# avoids the issue. Reconcile local main yourself afterwards if you
# care; the published tags are independent of local-main state.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <PR_NUMBER> <VERSION> [CO_TAG]" >&2
  exit 2
fi

PR_NUM="$1"
VERSION="$2"
CO_TAG="${3:-}"

# 1. Wait for CI to land
echo "Watching CI for PR #${PR_NUM}..."
gh pr checks "$PR_NUM" --watch --required

# 2. Squash-merge (matches v0.39.1 / v0.39.2 release pattern)
gh pr merge "$PR_NUM" --squash --delete-branch

# 3. Fetch the merged commit (no local main checkout, no reset)
git fetch origin main
MERGED_SHA=$(git rev-parse --short origin/main)
echo "Merged commit on origin/main: $MERGED_SHA"

# 4. Move tags to the merged SHA (point them at origin/main directly)
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

# 5. Print the push command (gated)
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
