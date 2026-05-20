#!/usr/bin/env python3
"""Archive working-tree clutter that maps to shipped work.

Default mode: --dry-run. Use --apply to move files to .shipped/YYYY-MM/.
Undo with --undo <manifest>.

Patterns handled (repo root only, dotfiles only):
  .pr_body_v<X.Y.Z>.md   -> match against git tag -l
  .commit_msg_<slug>.txt -> match against git log subjects
  .pr_body_<slug>.md     -> match against merged PR titles
  .pr_comment_<slug>.md  -> match against merged PR titles
  .tag_payload.json      -> match its tag_name against git tag -l

Never touches: .outbound_*, .application_*, .recruiter_*, .research_*,
.proposal_*, .tier1_*, .tmp_*, dot-config files, or any subdirectory.
"""
from __future__ import annotations
import argparse, datetime as dt, json, re, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHIPPED = REPO / ".shipped"
MANIFESTS = SHIPPED / "manifests"

EXCLUDE_PREFIXES = (".outbound_", ".application_", ".recruiter_",
                    ".research_", ".proposal_", ".tier1_", ".tmp_",
                    ".bash", ".git")

VERSION_RE = re.compile(r"^\.pr_body_v(\d+\.\d+\.\d+)\.md$")
COMMIT_MSG_RE = re.compile(r"^\.commit_msg_(.+)\.txt$")
PR_BODY_SLUG_RE = re.compile(r"^\.pr_body_(.+)\.md$")
PR_COMMENT_RE = re.compile(r"^\.pr_comment_(.+)\.md$")


def sh(cmd):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def slug_match(slug, haystack):
    needle = slug.replace("_", " ").replace("-", " ").lower()
    needle_tight = re.sub(r"[^a-z0-9]", "", slug.lower())
    for item in haystack:
        item_lower = item.lower()
        item_tight = re.sub(r"[^a-z0-9]", "", item_lower)
        if needle in item_lower or (needle_tight and needle_tight in item_tight):
            return item
    return None


def classify(path, tags, commits, merged_prs):
    name = path.name
    if any(name.startswith(p) for p in EXCLUDE_PREFIXES):
        return False, "excluded prefix"

    m = VERSION_RE.match(name)
    if m:
        tag = f"v{m.group(1)}"
        return (tag in tags, f"git tag {tag} {'exists' if tag in tags else 'missing'}")

    m = COMMIT_MSG_RE.match(name)
    if m:
        hit = slug_match(m.group(1), commits)
        return (bool(hit), f"commit match: {hit[:70]}" if hit else f"no commit for '{m.group(1)}'")

    m = PR_COMMENT_RE.match(name)
    if m:
        hit = slug_match(m.group(1), merged_prs)
        return (bool(hit), f"PR match: {hit[:70]}" if hit else f"no merged PR for '{m.group(1)}'")

    m = PR_BODY_SLUG_RE.match(name)
    if m and not VERSION_RE.match(name):
        hit = slug_match(m.group(1), merged_prs)
        return (bool(hit), f"PR match: {hit[:70]}" if hit else f"no merged PR for '{m.group(1)}'")

    if name == ".tag_payload.json":
        try:
            tag = json.loads(path.read_text()).get("tag_name", "")
            return (tag in tags, f"payload tag '{tag}' {'exists' if tag in tags else 'missing'}")
        except Exception:
            return False, "unreadable payload"

    return False, "no rule matched"


def candidates():
    return sorted(p for p in REPO.iterdir() if p.is_file()
                  and p.name.startswith((".pr_body_", ".commit_msg_",
                                          ".pr_comment_", ".tag_payload")))


def plan():
    tags = {t.strip() for t in sh(["git", "tag", "-l"]).splitlines() if t.strip()}
    commits = [l.strip() for l in sh(["git", "log", "--format=%s", "-500"]).splitlines() if l.strip()]
    merged = [l.strip() for l in sh(["gh", "pr", "list", "--state", "merged", "--limit", "200",
                                      "--json", "title", "--jq", ".[].title"]).splitlines() if l.strip()]
    archive, keep = [], []
    for p in candidates():
        ok, reason = classify(p, tags, commits, merged)
        (archive if ok else keep).append({"file": p.name, "reason": reason})
    return archive, keep


def apply_moves(archive):
    now = dt.datetime.now()
    bucket = SHIPPED / now.strftime("%Y-%m")
    bucket.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    moves = []
    for e in archive:
        src, dst = REPO / e["file"], bucket / e["file"]
        if src.exists():
            shutil.move(str(src), str(dst))
            moves.append({"from": str(src.relative_to(REPO)),
                          "to": str(dst.relative_to(REPO)),
                          "reason": e["reason"]})
    mp = MANIFESTS / f"{now.strftime('%Y-%m-%d-%H%M')}.json"
    mp.write_text(json.dumps({"timestamp": now.isoformat(), "moves": moves}, indent=2))
    return mp


def undo_manifest(mp):
    data = json.loads(mp.read_text())
    n = 0
    for m in data["moves"]:
        src, dst = REPO / m["to"], REPO / m["from"]
        if src.exists():
            shutil.move(str(src), str(dst))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", type=Path, metavar="MANIFEST")
    args = ap.parse_args()

    if args.undo:
        if not args.undo.exists():
            print(f"Manifest not found: {args.undo}", file=sys.stderr)
            return 1
        print(f"Restored {undo_manifest(args.undo)} files from {args.undo}")
        return 0

    archive, keep = plan()
    print(f"=== consolidate.py plan ({REPO.name}) ===\n")
    print(f"WILL ARCHIVE ({len(archive)} files) -> .shipped/")
    for e in archive:
        print(f"  + {e['file']:55s} {e['reason']}")
    print(f"\nWILL KEEP ({len(keep)} files)")
    for e in keep:
        print(f"  . {e['file']:55s} {e['reason']}")

    if args.apply:
        if archive:
            mp = apply_moves(archive)
            print(f"\nApplied. Manifest: {mp.relative_to(REPO)}")
            print(f"Undo: python scripts/consolidate.py --undo {mp.relative_to(REPO)}")
        else:
            print("\nNothing to archive.")
    else:
        print(f"\nDry-run. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
