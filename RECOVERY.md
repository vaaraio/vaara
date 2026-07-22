# Box recovery — read this first after a crash

Written 2026-07-21. Everything here is on the persistent `/workspace` mount.

## The one root cause
The box that keeps coming up vanilla was launched from a **bare image with no
`/home/claude` mount**. Verified: `/home/claude` is ephemeral in the bad box, so
uv/python/plugins/MCPs die every restart. Your saved launchers all DO mount it —
the bad box just wasn't started with one.

## What is SAFE (never lost — all on /workspace)
- All code + the `feat/x402-anchor-gate` branch work
- Memory DATA: `/workspace/.shared/Obsidian/Memory/` (curated, up to mem-484, current)
  and `/workspace/.shared/Obsidian/.raw/memory/` (clean store export, but STALE at
  mem-414 / Jul 19 — missing 415-484)
- `restore.sh` (this dir)

## What is GONE (rebuildable, not re-typed)
- vaara-memory SQLite store (`memory.db`) + its node server code (was in ephemeral home)
- uv / python 3.13 / .venv / plugins  (ephemeral home)

## Step 1 — launch the RIGHT box (on your Mac)
Two 5-second checks decide rebuild vs no-rebuild:
    ~/.rd/bin/nerdctl images | grep claude-box     # is claude-box:test still there?
    ls ~/claude-box/claude-home/.local/bin/uv       # is the toolchain in the host mount?
Then launch with the `box` script (image = claude-box:test, mounts /home/claude).
If `claude-box:test` exists -> non-vanilla box is back, NO rebuild.

## Step 2 — if still bare, rebuild the toolchain (inside the box)
    bash /workspace/restore.sh
NOTE: restore.sh's `vaara-memory` line points at the wrong binary — that binary is
the AUDIT MCP, not the memory store. Fix pending: rebuild the node memory server,
then re-ingest the curated notes into a fresh memory.db.

## Open question left for next session
Does the curated-note format re-ingest into a rebuilt store? If yes, only memories
saved after mem-484 are truly lost (likely a handful). Not yet proven.
