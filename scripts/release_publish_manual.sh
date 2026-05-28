#!/usr/bin/env bash
# Manual publish fallback for PyPI + npm when the GH Actions Release
# workflow is broken (workflow misconfig, OIDC trust failure, lapsed
# token). Use ONLY in that case.
#
# Per repo policy: transient GH Actions infra noise, then wait + rerun the
# workflow, do NOT bypass. This script exists for the case where the
# workflow itself cannot run.
#
# Usage: scripts/release_publish_manual.sh <VERSION>
#   VERSION  e.g., 0.39.2
#
# Requires:
#   - tag v<VERSION> exists locally and points at a pushed commit on main
#   - PyPI: VAARA_PYPI_TOKEN env var (API token starting with pypi-)
#   - npm:  npm login state for @vaara scope (run `npm login` first)
#
# What it does:
#   PyPI:
#     1. Build sdist + wheel via python -m build
#     2. twine check dist/*
#     3. twine upload dist/* (no Sigstore signing here; Sigstore needs
#        the GH OIDC flow. Manual publish ships without Sigstore.)
#   npm:
#     1. cd clients/ts; npm ci; npm run build; npm test
#     2. npm publish --access public --no-provenance
#        (Provenance also needs the GH OIDC flow.)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <VERSION>" >&2
  exit 2
fi

VERSION="$1"

git rev-parse "v${VERSION}" >/dev/null 2>&1 || \
  { echo "tag v${VERSION} does not exist locally" >&2; exit 1; }

# Confirm intent (manual publish is a deliberate exception)
echo "Manual publish for v${VERSION}. This bypasses the GH Actions"
echo "Release workflow and ships WITHOUT Sigstore signatures / SLSA"
echo "provenance / npm provenance. Use only when GH Actions itself is"
echo "broken, not for transient infra noise."
read -r -p "Confirm by typing the version (${VERSION}): " CONFIRM
[[ "$CONFIRM" == "$VERSION" ]] || { echo "aborted" >&2; exit 1; }

# Clean dist/
rm -rf dist build *.egg-info

# --- PyPI ---
if [[ -z "${VAARA_PYPI_TOKEN:-}" ]]; then
  echo "VAARA_PYPI_TOKEN not set; skipping PyPI publish."
else
  echo "Building Python distributions..."
  .venv/bin/python -m pip install --quiet --upgrade build twine
  .venv/bin/python -m build
  .venv/bin/python -m twine check dist/*
  echo "Uploading to PyPI..."
  TWINE_USERNAME="__token__" \
    TWINE_PASSWORD="$VAARA_PYPI_TOKEN" \
    .venv/bin/python -m twine upload dist/*
  echo "PyPI publish: done."
fi

# --- npm ---
pushd clients/ts >/dev/null
if ! npm whoami >/dev/null 2>&1; then
  echo "Not logged in to npm. Run: npm login"
  echo "Then re-run this script (PyPI step is idempotent on already-published versions)."
  exit 1
fi

echo "Building npm client..."
npm ci --no-audit --no-fund
npm run build
node --test test/*.test.mjs

echo "Publishing @vaara/client@${VERSION} to npm..."
npm publish --access public --no-provenance
popd >/dev/null

echo
echo "Manual publish for v${VERSION}: done."
echo "Follow-up:"
echo "  - Verify: pip install vaara==${VERSION}; npm view @vaara/client@${VERSION}"
echo "  - Open an incident note explaining why the GH Actions workflow"
echo "    was bypassed (workflow misconfig, OIDC issue, etc.)."
echo "  - Restore the workflow before the next release."
