#!/bin/bash
# Build script for Vaara macOS app + WebKit Governance extension.
# Requires Xcode 16+ and an Apple Developer account.
#
# Usage:
#   ./build.sh              # Debug build
#   ./build.sh release      # Release build + codesign
#   ./build.sh install      # Build and install to /Applications

set -euo pipefail

# Run from this script's own directory. Every path below is relative to it,
# so calling it as ./clients/macos/build.sh from the repo root used to fail on
# a project file it was looking for in the wrong place.
cd "$(dirname "${BASH_SOURCE[0]}")"

PROJECT="VaaraMenuBar"
SCHEME="VaaraMenuBar"
EXTENSION="WebKitGovernance"

# xcodegen writes VaaraMenuBar.xcodeproj from project.yml, and the generated
# project is not tracked, so it does not exist in a fresh clone. This script
# went straight to xcodebuild and failed with "'VaaraMenuBar.xcodeproj' does
# not exist", which reads as a broken checkout rather than a missing step. CI
# has always run `xcodegen generate` first; now so does this.
if ! command -v xcodegen > /dev/null 2>&1; then
    echo "xcodegen not found. Install it with: brew install xcodegen" >&2
    exit 1
fi

echo "==> Generating $PROJECT.xcodeproj from project.yml..."
xcodegen generate

echo "==> Building $PROJECT with $EXTENSION extension..."

if [ "${1:-}" = "release" ]; then
    CONFIG="release"
else
    CONFIG="debug"
fi

# Build the main app and extension together
xcodebuild -project "$PROJECT.xcodeproj" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -derivedDataPath ".build" \
    CODE_SIGN_STYLE="Manual" \
    CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-Apple Development}" \
    DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-}" \
    build

echo "==> Build complete."

if [ "${1:-}" = "install" ]; then
    APP_PATH=".build/Build/Products/${CONFIG}/$PROJECT.app"
    echo "==> Installing to /Applications..."
    cp -R "$APP_PATH" /Applications/
    echo "==> Installed. Enable the extension in System Settings → Network → Content Filter."
fi
