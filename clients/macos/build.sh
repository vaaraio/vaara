#!/bin/bash
# Build Vaara.app from the Swift package. Needs Xcode Command Line Tools
# (xcode-select --install); no full Xcode required.
#   ./build.sh        build dist/Vaara.app
#   ./build.sh dmg    also produce dist/Vaara.dmg (drag-to-Applications)
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release

APP=dist/Vaara.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp .build/release/VaaraMenuBar "$APP/Contents/MacOS/Vaara"
# SPM puts the .copy resources in a bundle next to the binary.
BUNDLE=$(find .build/release -maxdepth 1 -name "*VaaraMenuBar*.bundle" | head -1)
if [ -n "$BUNDLE" ]; then
  cp -R "$BUNDLE" "$APP/Contents/MacOS/"
fi

cp AppIcon.icns "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Vaara</string>
  <key>CFBundleDisplayName</key><string>Vaara</string>
  <key>CFBundleIdentifier</key><string>io.vaara.menubar</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>Vaara</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>CFBundleIconFile</key><string>AppIcon</string>
</dict>
</plist>
PLIST

# Ad-hoc signature: enough for the machine it was built on.
codesign --force --deep --sign - "$APP"

echo
echo "Built $APP"
echo "Run it:   open $APP"
echo "Install:  mv $APP /Applications/"

if [ "${1:-}" = "dmg" ]; then
  rm -f dist/Vaara.dmg
  if command -v create-dmg >/dev/null; then
    # The classic installer window: branded background, app left,
    # Applications right. brew install create-dmg
    create-dmg \
      --volname "Vaara" \
      --volicon AppIcon.icns \
      --background dmg-background.png \
      --window-size 660 400 \
      --icon-size 110 \
      --icon "Vaara.app" 165 230 \
      --app-drop-link 495 230 \
      --hide-extension "Vaara.app" \
      dist/Vaara.dmg "$APP"
  else
    echo "create-dmg not found (brew install create-dmg); building a plain dmg."
    STAGE=$(mktemp -d)
    cp -R "$APP" "$STAGE/"
    ln -s /Applications "$STAGE/Applications"
    hdiutil create -volname Vaara -srcfolder "$STAGE" -ov -format UDZO dist/Vaara.dmg
    rm -rf "$STAGE"
  fi
  echo
  echo "Built dist/Vaara.dmg (open it, drag Vaara to Applications)"
fi
