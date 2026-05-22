"""Atheris fuzz target: policy JSON/YAML loader.

The policy loader is what turns regulator/operator-supplied YAML or JSON
into the in-memory `Policy` that gates every action at runtime. A loader
that crashes on hostile input is a denial-of-service surface; a loader
that silently produces a malformed `Policy` is worse.

This target fuzzes the `from_json` text path (which is what file loads
funnel into) plus the `from_yaml` text path when PyYAML is importable.
Expected exceptions: `PolicyError`, `TypeError`, `ValueError`, `KeyError`.
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from vaara.policy.loader import from_json, from_yaml
    from vaara.policy.schema import PolicyError

try:
    import yaml as _yaml  # noqa: F401

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    choice = fdp.ConsumeIntInRange(0, 1) if _HAS_YAML else 0
    text = fdp.ConsumeUnicode(sys.maxsize)

    if choice == 0:
        try:
            from_json(text)
        except (PolicyError, TypeError, ValueError, KeyError, RecursionError):
            return
    else:
        try:
            from_yaml(text)
        except (PolicyError, TypeError, ValueError, KeyError, RecursionError):
            return
        except Exception as exc:
            # PyYAML raises subclasses of yaml.YAMLError on parse failure.
            # The loader is supposed to wrap those into PolicyError; anything
            # else leaking through is a finding.
            import yaml

            if isinstance(exc, yaml.YAMLError):
                return
            raise


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
