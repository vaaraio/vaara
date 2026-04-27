<!-- Keep this short. Detail belongs in the diff and the CHANGELOG. -->

## Summary

<!-- 1–3 bullets. What changes, why now. -->

-

## Test plan

<!-- Check off what you ran locally. Add new boxes for anything specific. -->

- [ ] `pytest -q` passes
- [ ] `scripts/lint_full.sh` passes (ruff + bandit + mypy + pytest)
- [ ] CHANGELOG.md updated if this changes the public API or shipped behaviour
- [ ] No secrets, audit DBs, or large binaries staged

## Risk / blast radius

<!-- One line. Examples: "test-only", "internal refactor, no API change",
     "audit DB schema bump (migration tested)", "CLI flag added", "breaking change". -->

-
