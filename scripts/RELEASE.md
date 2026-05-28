# Release flow

Four scripts under `scripts/`, run in order. Push remains gated; you
keep the keystrokes that hit the network.

## Pre-flight (you do this by hand)

Per release, write three files in the repo root:

- `.commit_msg_v<VERSION>_release.txt`: commit message body.
- `.pr_body_v<VERSION>.md`: PR body, send-ready-prose audited.
- `CHANGELOG.md`: add a `## [<VERSION>] - YYYY-MM-DD` entry.

These stay local (gitignored by convention) and feed the scripts via
`-F` / `--body-file` so you never paste prose into the terminal.

## 1. Prepare the commit + tags + branch

```
scripts/release_prepare.sh <VERSION> [CO_TAG]
```

Example:

```
scripts/release_prepare.sh 0.39.2 sep2787-ref-v2
```

What runs:

1. Verifies the three pre-flight files exist and the CHANGELOG entry is
   present.
2. Bumps `pyproject.toml`, `clients/ts/package.json`,
   `src/vaara/__init__.py` to `<VERSION>`.
3. `ruff check` on changed Python paths.
4. Full `pytest` (skips `tests/adversarial`, deselects the pre-existing
   SSRF distribution-shift test).
5. Stages explicit paths only (no `git add -A`).
6. Commits via `-F`.
7. Creates annotated tags `v<VERSION>` and (if passed) `<CO_TAG>` at
   HEAD.
8. Creates branch `release/v<VERSION>` at HEAD.

Stops before any push.

## 2. Push the branch and open the PR

```
scripts/release_push_and_pr.sh <VERSION>
```

This pushes `release/v<VERSION>` and opens a PR against `main` using
the commit subject as PR title and `.pr_body_v<VERSION>.md` as body.
Prints the PR URL and the next-step command.

## 3. After CI is green, merge and re-tag

```
scripts/release_merge_and_tag.sh <PR_NUMBER> <VERSION> [CO_TAG]
```

1. `gh pr checks --watch --required` blocks until all required checks
   pass (or fails the script if a required check fails).
2. `gh pr merge --squash --delete-branch`.
3. `git checkout main && git pull --ff-only`.
4. Re-creates `v<VERSION>` (and `<CO_TAG>` if passed) at the new merged
   SHA. The pre-merge local tags pointed at the unmerged commit; squash
   creates a new SHA so the tags need to move.
5. Prints the gated `git push origin v<VERSION> <CO_TAG>` command for
   you to paste.

Tag push fires `.github/workflows/release.yml`: build with SLSA
provenance, publish to PyPI via trusted publishing, sign with Sigstore,
publish `@vaara/client` to npm with provenance, create the GitHub
Release.

## 3b. PR already merged via GH UI

```
scripts/release_tag_after_merge.sh <VERSION> [CO_TAG]
```

Use when the release PR was squash-merged through the GH UI (or any
path that bypassed `release_merge_and_tag.sh`). Fetches `origin/main`,
moves `v<VERSION>` (and `<CO_TAG>` if passed) to the merged SHA, prints
the gated push command. Skips the `gh pr checks --watch` and
`gh pr merge` steps from script #3.

## 4. Manual publish fallback (only when GH Actions is broken)

```
scripts/release_publish_manual.sh <VERSION>
```

For when the workflow itself cannot run (misconfig, OIDC trust
failure, expired credential). NOT for transient infra noise; that is
a wait-and-rerun situation, not a bypass.

Requires:

- `VAARA_PYPI_TOKEN` env var (PyPI API token starting with `pypi-`).
- npm login state for `@vaara` scope (run `npm login` first).

Limitations: ships PyPI wheels without Sigstore signatures / SLSA
provenance, and ships npm without provenance. Those features require
the GH OIDC flow. Manual publish trades them for being able to ship.
Open an incident note explaining the bypass and restore the workflow
before the next release.

## 4b. npm-only manual publish

```
scripts/release_publish_npm_manual.sh <VERSION>
```

For the common case where the GH Actions npm step fails but PyPI
already shipped (PyPI trusted publishing is rock solid; npm
provenance via OIDC is the chunkier path). Token-based, no
interactive `npm login`. Requires `VAARA_NPM_TOKEN` env var
(npmjs.org → Access Tokens → Automation). Ships without provenance;
restore the workflow before the next release.

## Cross-repo follow-up

The release scripts do not touch cross-repo work. After v0.39.2-style
releases that ship a SEP-2787 reference impl, the follow-up is:

- Regen `vaaraio/modelcontextprotocol#2789` v0 vectors against the new
  envelope shape using `vaara.attestation.sep2787`.
- Comment under `modelcontextprotocol/modelcontextprotocol#2787` with
  cross-repo provenance: `vaaraio/vaara@<merged-sha>` and
  `vaaraio/vaara#<pr-number>`.
