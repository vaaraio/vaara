#!/bin/bash
# Build script for Vaara macOS app + WebKit Governance extension.
# Requires Xcode 16+ and an Apple Developer account.
#
# Usage:
#   ./build.sh              # Debug build
#   ./build.sh release      # Release build + codesign
#   ./build.sh install      # Build and install to /Applications

set -euo pipefail

PROJECT="VaaraMenuBar"
SCHEME="VaaraMenuBar"
EXTENSION="WebKitGovernance"

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
