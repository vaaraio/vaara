# Contributing to Vaara

Thanks for considering a contribution.

## How to contribute

1. **Open an issue first** for anything non-trivial (bug reports, feature ideas, design questions). This avoids duplicated work.
2. **Fork and branch** from `main`. Use a descriptive branch name (`fix/<short-desc>` or `feat/<short-desc>`).
3. **Open a pull request** against `main`. One logical change per PR.

## Requirements for accepted contributions

- **Tests pass.** Run `pytest -q` locally before pushing. CI must be green.
- **Code style.** Python code follows PEP 8. No enforced formatter, but keep diffs small and consistent with surrounding code.
- **No secrets.** Never commit credentials, tokens, `.env` files, or real audit database contents.
- **Sign-off preferred.** Use `git commit -s` to add a Developer Certificate of Origin sign-off.
- **Public interface changes.** If you change the public API, update `docs/formal_specification.md` and `CHANGELOG.md` in the same PR.
- **Security-sensitive changes.** Follow `SECURITY.md` for private disclosure of vulnerabilities.

## Licensing

By contributing you agree that your contributions will be licensed under the Apache License 2.0, the same license that covers the project (see `LICENSE`).
