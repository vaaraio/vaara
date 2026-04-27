#!/usr/bin/env bash
# Vaara pre-push lint sweep.
# Runs the static checks that gate merge AND the ones we use to catch
# CodeRabbit-class findings before pushing. Designed to keep iteration
# tight — what fails locally would otherwise come back as a review-bot
# round-trip on the PR.
#
# Tools (ordered cheapest first):
#   1. ruff check        — style + correctness lint (fast)
#   2. bandit            — security-focused static analysis
#   3. mypy              — type checking (strict on vaara.policy, lenient on legacy)
#   4. pytest            — full test suite
#
# Usage: scripts/lint_full.sh
# Runtime: ~10s if no surprises.
#
# Setup: requires the [dev] extras. From repo root:
#   pip install -e '.[dev]'
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Use the venv's tools when available, fall back to PATH.
if [[ -x .venv/bin/ruff ]]; then
    RUFF=.venv/bin/ruff
    BANDIT=.venv/bin/bandit
    MYPY=.venv/bin/mypy
    PYTEST=.venv/bin/pytest
else
    RUFF=ruff
    BANDIT=bandit
    MYPY=mypy
    PYTEST=pytest
fi

echo "[1/4] ruff check..."
"$RUFF" check src/ scripts/ tests/

echo "[2/4] bandit (security, src/ only)..."
"$BANDIT" -c pyproject.toml -q -r src/

echo "[3/4] mypy (strict on vaara.policy, lenient on legacy)..."
"$MYPY" src/vaara/policy/

echo "[4/4] pytest..."
"$PYTEST" -q

echo
echo "[lint_full] PASS"
