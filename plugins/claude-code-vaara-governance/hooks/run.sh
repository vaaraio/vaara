#!/bin/sh
# Hook shim: prefer the vaara binary on PATH (whatever installed the CLI
# is a complete engine install: pip, pipx, Homebrew), fall back to the
# bundled python3 scripts for installs that predate `vaara hook`.
#
# pre-tool-use fails closed. Claude Code lets a call run when its hook
# exits with anything but 0 or 2, or outlives the hook's timeout, so the
# hook runs under a watchdog that stops it at the deadline (80 s, inside
# the 90 s hooks.json gives it), Vaara's own verdicts pass through, and
# every other ending blocks the call. "fail_open": true in
# ~/.vaara/claude-code/config.json or VAARA_PLUGIN_FAIL_OPEN=1 lets it run
# instead. Same gate as the one `vaara init` writes, in
# vaara/integrations/_hook_gate.py.
#
# Usage (from hooks.json): run.sh pre-tool-use|post-tool-use|session-start
set -eu
kind="$1"
deadline=80

# The config's top-level "fail_open" read as a value: a text match also
# fires on the same words inside a string. Without python3 to read it, only
# VAARA_PLUGIN_FAIL_OPEN=1 opts out.
fail_open() {
  [ "${VAARA_PLUGIN_FAIL_OPEN:-}" = 1 ] && return 0
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if isinstance(d,dict) and d.get('fail_open') is True else 1)" \
    "$HOME/.vaara/claude-code/config.json" 2>/dev/null
}

refuse() {
  echo "vaara-governance: BLOCKED (fail-closed): the Vaara hook $1. Reinstall vaara and re-run vaara init, or set \"fail_open\": true in ~/.vaara/claude-code/config.json." >&2
  exit 2
}

# gate CMD...: run the pre-tool-use hook CMD and exit with its verdict.
# Its output goes to two private files, replayed once it ends, so a child
# it left running cannot hold Claude Code's pipes open past the deadline.
gate() {
  d=${VAARA_HOOK_DEADLINE:-$deadline}
  case $d in ''|*[!0-9]*) d=$deadline;; esac
  [ "$d" -gt "$deadline" ] && d=$deadline
  o=$(mktemp 2>/dev/null) && e=$(mktemp 2>/dev/null) || { rm -f "${o:-}"; o=; }
  rc=125
  if [ -n "$o" ]; then
    exec 3<&0
    "$@" <&3 3<&- >"$o" 2>"$e" & p=$!
    exec 3<&-
    ( i=0; while [ $i -lt "$d" ]; do sleep 1; i=$((i+1)); done; kill -9 $p ) </dev/null >/dev/null 2>&1 & w=$!
    rc=0; wait $p || rc=$?
    kill $w 2>/dev/null || true
    cat "$o"; cat "$e" >&2; rm -f "$o" "$e"
  fi
  if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then exit $rc; fi
  if fail_open; then exit $rc; fi
  if [ $rc -gt 128 ]; then refuse "did not answer within $d s"; fi
  refuse "could not run (exit $rc)"
}

dir="$(dirname "$0")"

# pre-tool-use picks its engine inside the gate: the `vaara hook --help`
# probe starts an interpreter, and a probe that hangs outside the gate would
# run into Claude Code's timeout instead of the deadline.
if [ "$kind" = pre-tool-use ] && { command -v vaara || command -v python3; } >/dev/null 2>&1; then
  gate sh -c 'if command -v vaara >/dev/null 2>&1 && vaara hook --help >/dev/null 2>&1; then
  exec vaara hook pre-tool-use
fi
exec python3 "$1/pre_tool_use.py"' sh "$dir"
fi

if command -v vaara >/dev/null 2>&1 && vaara hook --help >/dev/null 2>&1; then
  exec vaara hook "$kind"
fi

if ! command -v python3 >/dev/null 2>&1; then
  if fail_open; then
    msg="vaara-governance: neither the vaara binary nor python3 is on PATH; \
governance is NOT active and fail_open is set, so tool calls run unchecked \
and unrecorded. Install with: pip install vaara"
  else
    msg="vaara-governance: neither the vaara binary nor python3 is on PATH; \
governance cannot run, so every tool call is blocked. Install with: pip \
install vaara, or set \"fail_open\": true in ~/.vaara/claude-code/config.json."
  fi
  echo "$msg" >&2
  # SessionStart stdout is injected into the model's context, so the
  # session itself learns governance is off and can tell the user.
  # Silence here is the one failure mode this plugin must never have.
  if [ "$kind" = "session-start" ]; then
    echo "$msg"
  fi
  if [ "$kind" = pre-tool-use ] && ! fail_open; then
    refuse "could not run: neither the vaara binary nor python3 is on PATH"
  fi
  exit 0
fi

case "$kind" in
  post-tool-use)  exec python3 "$dir/post_tool_use.py" ;;
  session-start)  exec python3 "$dir/session_start.py" ;;
  *) echo "vaara-governance: unknown hook kind: $kind" >&2; exit 0 ;;
esac
