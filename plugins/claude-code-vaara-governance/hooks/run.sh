#!/bin/sh
# Hook shim: prefer the vaara binary on PATH (whatever installed the CLI
# is a complete engine install: pip, pipx, Homebrew), fall back to the
# bundled python3 scripts for installs that predate `vaara hook`.
#
# Usage (from hooks.json): run.sh pre-tool-use|post-tool-use|session-start
set -eu
kind="$1"

if command -v vaara >/dev/null 2>&1 && vaara hook --help >/dev/null 2>&1; then
  exec vaara hook "$kind"
fi

if ! command -v python3 >/dev/null 2>&1; then
  msg="vaara-governance: neither the vaara binary nor python3 is on PATH; \
governance is NOT active. Tool calls run unchecked and unrecorded. \
Install with: pip install vaara"
  echo "$msg" >&2
  # SessionStart stdout is injected into the model's context, so the
  # session itself learns governance is off and can tell the user.
  # Silence here is the one failure mode this plugin must never have.
  if [ "$kind" = "session-start" ]; then
    echo "$msg"
  fi
  exit 0
fi

dir="$(dirname "$0")"
case "$kind" in
  pre-tool-use)   exec python3 "$dir/pre_tool_use.py" ;;
  post-tool-use)  exec python3 "$dir/post_tool_use.py" ;;
  session-start)  exec python3 "$dir/session_start.py" ;;
  *) echo "vaara-governance: unknown hook kind: $kind" >&2; exit 0 ;;
esac
