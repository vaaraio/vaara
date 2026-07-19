# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Desktop notification for governance decisions. Fire-and-forget.

Sends a native notification when the gate blocks or escalates a tool
call, so the decision is visible even when the terminal is buried.
macOS uses osascript; Linux uses notify-send when present. Anything
missing or failing is silently ignored: a notification must never be
able to break the hook that sends it.

Env vars:
  VAARA_PLUGIN_NOTIFY=0   disable notifications (default: enabled)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _clean(text: str, limit: int) -> str:
    # osascript receives the strings inside AppleScript double quotes;
    # strip characters that would terminate or escape the literal.
    text = text.replace('"', "'").replace("\\", "/").replace("\n", " ")
    return text[:limit]


def notify(verdict: str, tool_name: str, detail: str) -> None:
    """Show '<verdict> <tool>' with the reason underneath. Never raises."""
    if os.environ.get("VAARA_PLUGIN_NOTIFY") == "0":
        return
    try:
        title = _clean(f"Vaara: {verdict}", 60)
        subtitle = _clean(tool_name, 80)
        body = _clean(detail, 180)
        if sys.platform == "darwin":
            script = (
                f'display notification "{body}" '
                f'with title "{title}" subtitle "{subtitle}" '
                f'sound name "Funk"'
            )
            cmd = ["osascript", "-e", script]
        elif shutil.which("notify-send"):
            cmd = ["notify-send", "--app-name=Vaara", f"{title} {subtitle}", body]
        else:
            return
        subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass
