#!/usr/bin/env bash
# Runtime verification for the Vaara macOS client. Run on a real Mac.
#
# CI can compile and link this client, but it cannot install a system
# extension or watch a network flow. Those need a machine with a user who
# can approve them. Run this, paste the output.
#
#   bash clients/macos/verify.sh
#
# Nothing here modifies the repo. The install step is the only one that
# changes system state, and it prompts before doing so.

set -uo pipefail
cd "$(dirname "$0")"

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
info() { printf '  ....  %s\n' "$1"; }
FAILED=0

echo
echo "Vaara macOS client verification"
echo "==============================="

echo
echo "Toolchain"
sw_vers -productVersion >/dev/null 2>&1 && pass "macOS $(sw_vers -productVersion)" || fail "not macOS"
xcodebuild -version >/dev/null 2>&1 && pass "$(xcodebuild -version | head -1)" || fail "xcodebuild missing (install Xcode)"
command -v xcodegen >/dev/null 2>&1 && pass "xcodegen $(xcodegen --version 2>/dev/null)" || fail "xcodegen missing (brew install xcodegen)"

echo
echo "Build"
if xcodegen generate >/dev/null 2>&1; then
  pass "project generated"
else
  fail "xcodegen generate failed"
fi

if xcodebuild build -project VaaraMenuBar.xcodeproj -scheme VaaraMenuBar \
     -configuration Release -destination 'platform=macOS' >/tmp/vaara-build.log 2>&1; then
  pass "build succeeded"
else
  fail "build failed, see /tmp/vaara-build.log"
  tail -20 /tmp/vaara-build.log
fi

APP=$(find ~/Library/Developer/Xcode/DerivedData -type d -name 'VaaraMenuBar.app' 2>/dev/null | head -1)
EXT=$(find ~/Library/Developer/Xcode/DerivedData -type d -name 'WebKitGovernance.systemextension' 2>/dev/null | head -1)
[ -n "$APP" ] && pass "app bundle: $APP" || fail "no VaaraMenuBar.app produced"
[ -n "$EXT" ] && pass "extension bundle: $EXT" || fail "no WebKitGovernance.systemextension produced"

echo
echo "Signing and entitlements"
if [ -n "$EXT" ]; then
  if codesign -dv "$EXT" 2>&1 | grep -q "Signature"; then
    pass "extension is signed"
  else
    info "extension unsigned. A system extension must be signed with a Developer ID"
    info "and the Network Extension capability before macOS will load it."
  fi
  ENT=$(codesign -d --entitlements - "$EXT" 2>/dev/null)
  echo "$ENT" | grep -q "group.io.vaara" \
    && pass "app group entitlement present" \
    || fail "app group group.io.vaara missing, XPC will not resolve"
  echo "$ENT" | grep -q "content-filter-provider-system" \
    && pass "content filter entitlement present" \
    || fail "content-filter-provider-system entitlement missing"
fi

echo
echo "Runtime (needs approval in System Settings)"
if systemextensionsctl list 2>/dev/null | grep -q "io.vaara.webkit-governance"; then
  STATE=$(systemextensionsctl list 2>/dev/null | grep "io.vaara.webkit-governance" | head -1)
  pass "extension registered: $STATE"
  echo "$STATE" | grep -q "activated enabled" \
    && pass "extension is active" \
    || info "registered but not active. Approve it in System Settings > General > Login Items & Extensions"
else
  info "extension not installed yet. Launch the app and enable the filter, then rerun."
fi

echo
echo "Trail"
DB=~/.vaara/trail/audit.db
if [ -f "$DB" ]; then
  pass "trail present at $DB"
  if command -v vaara >/dev/null 2>&1; then
    COUNT=$(vaara audit count --db "$DB" 2>/dev/null || echo "?")
    info "records: $COUNT"
    info "After visiting a governed host in Safari, rerun and confirm this grew."
  else
    info "vaara CLI not on PATH; the app shells out to it for decisions"
  fi
else
  info "no trail yet at $DB"
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "All checks passed."
else
  echo "Some checks failed. Paste this output."
fi
exit "$FAILED"
