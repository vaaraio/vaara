#!/usr/bin/env bash
# Push the release branch and open the PR. Push requires your keystroke
# (gated by your shell auth); this script bundles the commands you'd
# otherwise paste.
#
# Usage: scripts/release_push_and_pr.sh <VERSION>
#   VERSION  e.g., 0.39.2
#
# Expects scripts/release_prepare.sh has already run:
#   - branch release/v<VERSION> exists locally
#   - .pr_body_v<VERSION>.md exists
#   - HEAD is the release commit
#
# What it does:
#   1. git push -u origin release/v<VERSION>
#   2. gh pr create --title <subject> --body-file .pr_body_v<VERSION>.md
#                   --base main --head release/v<VERSION>
#   3. Prints the PR URL and the next-step command.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <VERSION>" >&2
  exit 2
fi

VERSION="$1"
BRANCH="release/v${VERSION}"
PR_BODY=".pr_body_v${VERSION}.md"

[[ -f "$PR_BODY" ]] || { echo "missing: $PR_BODY" >&2; exit 1; }
[[ "$(git branch --show-current)" == "$BRANCH" ]] || \
  { echo "not on $BRANCH (currently $(git branch --show-current))" >&2; exit 1; }

# Derive PR title from the commit subject line so the PR matches the
# commit history one-to-one.
SUBJECT=$(git log -1 --pretty=%s)

# 1. Push the branch
git push -u origin "$BRANCH"

# 2. Open the PR
PR_URL=$(gh pr create \
  --title "$SUBJECT" \
  --body-file "$PR_BODY" \
  --base main \
  --head "$BRANCH")

echo
echo "PR opened: $PR_URL"
PR_NUM=$(echo "$PR_URL" | grep -oE '[0-9]+$')
echo "Next: scripts/release_merge_and_tag.sh ${PR_NUM} ${VERSION} [CO_TAG]"
