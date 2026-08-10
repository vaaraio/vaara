#!/bin/bash
# Build script for Vaara macOS app + WebKit Governance extension.
# Requires Xcode 16+. The default and release builds additionally require a
# paid Apple Developer Program membership, because the app requests the
# Network Extension and system-extension entitlements. `./build.sh local`
# needs neither.
#
# Usage:
#   ./build.sh              # Debug build (needs a paid Apple Developer
#                           # provisioning profile for the Network Extension)
#   ./build.sh local        # Debug build, ad-hoc signed, no entitlements.
#                           # Everything but the network filter works.
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

# `local` builds the app without the restricted entitlements.
#
# VaaraMenuBar.entitlements asks for
# com.apple.developer.networking.networkextension and
# com.apple.developer.system-extension.install. Both are restricted: Xcode
# refuses to sign without a provisioning profile that carries them, and only a
# paid Apple Developer Program membership can issue one. A free personal team
# cannot. So the default build fails on any machine without that membership,
# with "requires a provisioning profile with the Network Extensions and System
# Extension features", and that is Apple's rule rather than anything this
# project can code around.
#
# Everything other than the network filter is unaffected: the menu bar, the
# approval window, Setup, the trail views and the Accessibility observer need
# no entitlement at all. That is also why the Homebrew formula produces a
# working app, since it compiles Sources/VaaraMenuBar and Sources/Shared with a
# plain swiftc and never touches the extension.
#
# This mode ad-hoc signs and drops the entitlements file, so the app builds and
# runs and only the filter is missing. Use it to work on everything else.
EXTRA_ARGS=()
if [ "${1:-}" = "local" ]; then
    echo "==> Local mode: ad-hoc signed, no Network Extension entitlements."
    echo "    The network filter will not load. Everything else runs."
    CODE_SIGN_IDENTITY="-"
    EXTRA_ARGS=(
        CODE_SIGN_ENTITLEMENTS=
        CODE_SIGNING_REQUIRED=NO
    )
fi

# Build the main app and extension together
xcodebuild -project "$PROJECT.xcodeproj" \
    -scheme "$SCHEME" \
    -configuration "$CONFIG" \
    -derivedDataPath ".build" \
    CODE_SIGN_STYLE="Manual" \
    CODE_SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-Apple Development}" \
    DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    build

echo "==> Build complete."

if [ "${1:-}" = "install" ]; then
    APP_PATH=".build/Build/Products/${CONFIG}/$PROJECT.app"
    echo "==> Installing to /Applications..."
    cp -R "$APP_PATH" /Applications/
    echo "==> Installed. Enable the extension in System Settings → Network → Content Filter."
fi
