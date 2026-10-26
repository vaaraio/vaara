# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Digest-only capture: a declared argument field is kept as a digest, never as content.

By default the trail, the review queue copy and the approval request hold the
tool arguments exactly as the host sent them. An operator who must not copy
some of them (patient data, an account number, a document body) declares those
fields in ``~/.vaara/capture.json``, or in the file ``VAARA_CAPTURE_POLICY``
names::

    {"version": 1,
     "rules": [
       {"tool": "mcp__ehr__*", "fields": ["patient.id", "rows.*.note"],
        "reference": "ehr"},
       {"tool": "shell.exec", "fields": ["*"]}]}

``tool`` is a glob over the tool name (default ``*``). A field is a dotted path
through objects; ``*`` as one segment takes every key of an object or every
item of a list, and ``*`` alone takes every top-level value (the keys stay, so
the record still says which fields were passed). ``reference`` is free text
naming where the real value lives, for the person who later has to find it.

A declared value is stored as::

    {"captured": "digest-only", "digest": "hmac-sha256:<hex>", "reference": "ehr"}

    digest = HMAC-SHA256(key, "vaara.capture.v1\\n" || path || "\\n" || JCS(value))

``path`` is the concrete position (``rows.2.note``), so two fields holding the
same value do not share a digest. The key is 32 random bytes, hex, created on
first use with mode 0600 in ``keys/capture-hmac.key`` beside the policy file,
and never written to the trail. That is the difference from a salt stored next
to the digest: a holder of the trail alone cannot test a guessed identifier
against it, and the operator, who holds the key, can show that a given value
is the one that was passed.

What it does not promise. A digest of a personal identifier is pseudonymised
data while anyone holds the key, not anonymous data. Only the arguments are
masked: the engine's reason text, tool results, outcome feedback and the
free-text fields a reviewer types are not. The engine and the caller still see
the real arguments; only what the three stores keep changes. Whatever a
decision receipt commits to (``argumentsCommitment``) is the arguments as the
trail recorded them, so with capture on it commits to the masked form.

Failure always goes toward not copying. A policy file that exists but cannot be
read or understood withholds every value; a key that cannot be read or made
withholds the value and writes no digest; a value that cannot be canonicalised
is withheld. None of them writes the clear value or stops the call.
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

POLICY_ENV = "VAARA_CAPTURE_POLICY"
KEY_NAME = "capture-hmac.key"
DOMAIN = b"vaara.capture.v1\n"
DIGEST_PREFIX = "hmac-sha256:"
_MAX_REFERENCE = 128
_MIN_KEY_BYTES = 16

_UNREADABLE = object()
_warned: set[tuple[str, str]] = set()


class _Rule(NamedTuple):
    tool: str
    fields: tuple[tuple[str, ...], ...]
    reference: str


def policy_path() -> Path:
    """The capture policy file: ``VAARA_CAPTURE_POLICY``, else ``~/.vaara/capture.json``."""
    env = os.environ.get(POLICY_ENV)
    return Path(env) if env else Path.home() / ".vaara" / "capture.json"


def key_path(policy: Path) -> Path:
    """The digest key for ``policy``: ``keys/`` beside the policy file."""
    return policy.parent / "keys" / KEY_NAME


def mask(tool_name: str, parameters: Any) -> Any:
    """``parameters`` with every declared field replaced by its digest marker.

    Returns ``parameters`` itself when there is no policy file or no rule
    names ``tool_name``. Never changes the object it is given.
    """
    if parameters is None:
        return None
    path = policy_path()
    rules = _load(path)
    if rules is None:
        return parameters
    if rules is _UNREADABLE:
        return _withhold_all(parameters)
    matched = [r for r in rules if fnmatch.fnmatchcase(tool_name, r.tool)]
    if not matched:
        return parameters
    key = _read_key(key_path(path))
    if not isinstance(parameters, dict):
        # nothing to address a field by, so the whole value goes
        return _marker(parameters, "", matched[0].reference, key)
    masked = parameters
    for rule in matched:
        for steps in rule.fields:
            masked = _walk(masked, steps, (), rule.reference, key)
    return masked


def _load(path: Path):
    """The rules, None when no policy file exists, ``_UNREADABLE`` when it cannot be used."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        _warn(path, f"cannot read it ({exc.__class__.__name__})")
        return _UNREADABLE
    try:
        return _parse(json.loads(text))
    except (ValueError, TypeError) as exc:
        _warn(path, str(exc) or exc.__class__.__name__)
        return _UNREADABLE


def _parse(doc: Any) -> list[_Rule]:
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise ValueError('expected an object with "version": 1')
    raw_rules = doc.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError('"rules" must be a list')
    rules = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise ValueError("a rule must be an object")
        tool = raw.get("tool", "*")
        fields = raw.get("fields")
        reference = raw.get("reference", "")
        if not isinstance(tool, str) or not tool:
            raise ValueError('"tool" must be a non-empty string')
        if (not isinstance(fields, list) or not fields
                or not all(isinstance(f, str) and f for f in fields)):
            raise ValueError('"fields" must be a list of non-empty strings')
        if not isinstance(reference, str):
            raise ValueError('"reference" must be a string')
        rules.append(_Rule(
            tool,
            tuple(tuple(f.split(".")) for f in fields),
            reference[:_MAX_REFERENCE],
        ))
    return rules


def _warn(path: Path, why: str) -> None:
    if (str(path), why) in _warned:
        return
    _warned.add((str(path), why))
    logger.warning(
        "capture policy %s is unusable (%s): every argument value is withheld "
        "from the trail, the review queue and the approval files until it is fixed",
        path, why,
    )


def _walk(node: Any, steps: tuple[str, ...], at: tuple[str, ...],
          reference: str, key: Optional[bytes]) -> Any:
    if not steps:
        return _marker(node, ".".join(at), reference, key)
    head, rest = steps[0], steps[1:]
    if isinstance(node, dict):
        names = list(node) if head == "*" else ([head] if head in node else [])
        if not names:
            return node
        out = dict(node)
        for name in names:
            out[name] = _walk(node[name], rest, at + (str(name),), reference, key)
        return out
    if isinstance(node, list) and head == "*":
        return [_walk(item, rest, at + (str(i),), reference, key)
                for i, item in enumerate(node)]
    return node


def _marker(value: Any, at: str, reference: str, key: Optional[bytes]) -> dict:
    digest = _digest(key, at, value) if key else None
    out: dict = ({"captured": "digest-only", "digest": digest} if digest
                 else {"captured": "withheld"})
    if reference:
        out["reference"] = reference
    return out


def _digest(key: bytes, at: str, value: Any) -> Optional[str]:
    try:
        import rfc8785

        canonical = rfc8785.dumps(value)
    except Exception:
        return None
    message = DOMAIN + at.encode() + b"\n" + canonical
    return DIGEST_PREFIX + hmac.new(key, message, hashlib.sha256).hexdigest()


def _withhold_all(parameters: Any) -> Any:
    if isinstance(parameters, dict):
        return {k: {"captured": "withheld"} for k in parameters}
    return {"captured": "withheld"}


def _read_key(path: Path) -> Optional[bytes]:
    """The digest key, made with mode 0600 on first use; None when it cannot be had."""
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass  # another process made it first; read theirs
            else:
                with os.fdopen(fd, "w") as fh:
                    fh.write(secrets.token_hex(32))
        key = bytes.fromhex(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return key if len(key) >= _MIN_KEY_BYTES else None
