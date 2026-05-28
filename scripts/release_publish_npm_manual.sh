#!/usr/bin/env bash
# npm-only manual publish for when the GH Actions Release workflow npm
# step is broken (the PyPI step usually works since trusted publishing
# is rock solid; npm provenance via OIDC is the chunkier path).
#
# Token-based: no interactive `npm login`. Mirrors VAARA_PYPI_TOKEN.
#
# Usage: scripts/release_publish_npm_manual.sh <VERSION>
#   VERSION  e.g., 0.40.4
#
# Requires:
#   - tag v<VERSION> exists locally and points at a pushed commit
#   - VAARA_NPM_TOKEN env var (npmjs.org > Access Tokens > Automation)
#
# Ships @vaara/client@<VERSION> WITHOUT npm provenance (provenance
# needs the GH OIDC flow). Restore the workflow before the next release.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <VERSION>" >&2
  exit 2
fi

VERSION="$1"

git rev-parse "v${VERSION}" >/dev/null 2>&1 || \
  { echo "tag v${VERSION} does not exist locally" >&2; exit 1; }

if [[ -z "${VAARA_NPM_TOKEN:-}" ]]; then
  echo "VAARA_NPM_TOKEN not set." >&2
  echo "Create one at npmjs.org > Access Tokens > Generate Automation token." >&2
  exit 1
fi

PKG_VERSION=$(node -p "require('./clients/ts/package.json').version")
if [[ "$PKG_VERSION" != "$VERSION" ]]; then
  echo "clients/ts/package.json is at ${PKG_VERSION}, expected ${VERSION}." >&2
  echo "Bump it first (release_prepare.sh does this)." >&2
  exit 1
fi

pushd clients/ts >/dev/null

NPMRC="$PWD/.npmrc.publish"
trap 'rm -f "$NPMRC"' EXIT
printf '//registry.npmjs.org/:_authToken=%s\nregistry=https://registry.npmjs.org/\n' \
  "$VAARA_NPM_TOKEN" > "$NPMRC"

echo "Building @vaara/client@${VERSION}..."
npm ci --no-audit --no-fund
npm run build
node --test test/*.test.mjs

echo "Publishing @vaara/client@${VERSION} to npm (no provenance)..."
NPM_CONFIG_USERCONFIG="$NPMRC" npm publish --access public --no-provenance

popd >/dev/null

echo
echo "@vaara/client@${VERSION}: published."
echo "Verify: npm view @vaara/client@${VERSION}"
echo "Restore the GH Actions npm step before the next release so provenance is back."
