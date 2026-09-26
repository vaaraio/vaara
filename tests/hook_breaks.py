"""Break the engine behind installed hooks, for the real-binary tests."""
from __future__ import annotations

from pathlib import Path


#: The ways a hook can fail to answer: the engine crashes, the binary is
#: gone, the engine hangs. Codex lets the call run on each unless the gate
#: Vaara writes turns it into a refusal.
BREAKS = {
    "crash": "#!/bin/sh\necho 'Traceback: engine broke' >&2\nexit 1\n",
    "missing": None,
    "hang": "#!/bin/sh\nexec sleep 60\n",
}


def break_engine(shim: Path, how: str) -> dict:
    """Break the engine behind installed hooks; the env the run needs."""
    body = BREAKS[how]
    if body is None:
        shim.unlink()
    else:
        shim.write_text(body)
    return {"VAARA_HOOK_DEADLINE": "3"} if how == "hang" else {}
