# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The command in front of ``vaara hook pre-tool-use``: a hook that cannot
answer blocks the call.

Codex and Gemini CLI have no fail-closed switch, and Claude Code has none
either. A hook that exits with anything but 0 or 2, cannot be found, or
outlives the host's timeout lets the call run, so a gate that decides
correctly whenever it runs still passes everything the moment it does not
run. Vaara's own engine already fails closed when it runs and cannot
decide; this covers the cases where it never gets that far: an
interpreter that will not start, a package half removed, a binary that is
gone, an engine that hangs.

The gate is inlined in the host's own config as one ``sh -c`` line, not
kept in a file of its own, because a gate file that can go missing is the
same hole one step further out. It runs the hook under a watchdog that
stops it at :data:`DEADLINE`, inside every host's timeout, passes exit 0
and exit 2 through, and turns every other ending into exit 2 with the
reason on stderr, which each host hands to the model. ``"fail_open":
true`` in ``~/.vaara/claude-code/config.json`` or
``VAARA_PLUGIN_FAIL_OPEN=1`` keeps the old behaviour, as for OpenCode.

``VAARA_HOOK_DEADLINE`` can shorten the deadline, never lengthen it: past
the host's timeout the host would kill the hook first and let the call
through.

A call refused here has no record on the trail. The engine that writes
records is the thing that did not answer.
"""

from __future__ import annotations

import shlex
from typing import Optional

#: The timeout Vaara writes for its pre-tool-use hook in every host config.
HOST_TIMEOUT = 90

#: Seconds the gate waits for a verdict. Below :data:`HOST_TIMEOUT` so the
#: gate, not the host, decides what a slow hook means.
DEADLINE = 80

#: The hook writes into two private files, replayed once it ends, rather
#: than into the host's pipes: a child the hook left running would hold
#: those pipes open, and the host would wait on it past the deadline. The
#: watchdog is given no host pipe for the same reason, and counts down in
#: one-second sleeps so that stopping it leaves no long sleep behind.
_SCRIPT = """\
d=${{VAARA_HOOK_DEADLINE:-{deadline}}}
case $d in ''|*[!0-9]*) d={deadline};; esac
[ "$d" -gt {deadline} ] && d={deadline}
o=$(mktemp 2>/dev/null) && e=$(mktemp 2>/dev/null) || {{ rm -f "$o"; o=; }}
if [ -n "$o" ]; then
exec 3<&0
{hook} <&3 3<&- >"$o" 2>"$e" & p=$!
exec 3<&-
( i=0; while [ $i -lt "$d" ]; do sleep 1; i=$((i+1)); done; kill -9 $p ) </dev/null >/dev/null 2>&1 & w=$!
wait $p; rc=$?
kill $w 2>/dev/null
cat "$o"; cat "$e" >&2; rm -f "$o" "$e"
else rc=125
fi
[ $rc -eq 0 ] || [ $rc -eq 2 ] && exit $rc
[ "${{VAARA_PLUGIN_FAIL_OPEN:-}}" = 1 ] && exit $rc
grep -Eqs '"fail_open"[[:space:]]*:[[:space:]]*true' "$HOME/.vaara/claude-code/config.json" && exit $rc
if [ $rc -gt 128 ]; then why="did not answer within $d s"; else why="could not run (exit $rc)"; fi
echo "vaara-governance: BLOCKED (fail-closed): the Vaara hook $why. Reinstall vaara and re-run vaara init, or set \\"fail_open\\": true in ~/.vaara/claude-code/config.json." >&2
exit 2"""


def pre_command(vaara_bin: str, client: Optional[str] = None) -> str:
    """The pre-tool-use hook command for a host config, gated.

    It contains ``<vaara_bin> hook pre-tool-use`` verbatim, so every
    detector that finds Vaara's hooks by that string still finds it.
    """
    hook = f"{shlex.quote(vaara_bin)} hook pre-tool-use"
    if client:
        hook += f" --client {client}"
    return "sh -c " + shlex.quote(_SCRIPT.format(deadline=DEADLINE, hook=hook))
