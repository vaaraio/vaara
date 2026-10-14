# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where a value in an authority-carrying argument came from.

    user:   "pay the invoice in my inbox"
    agent:  read_email      -> invoice, payee IBAN (edited by an attacker)
    agent:  pay(iban=...)   <- every call in scope; the payee came from content

Each call passes on its own. What makes the payment wrong is the origin of one
argument: the IBAN entered the session in a tool result and nowhere else. This
module keeps that origin. Values shaped like an address of authority (email
addresses, IBANs, hosts in URLs) are harvested from external tool results and
from the user's own prompts, and a later call that puts one of them into an
argument naming a recipient, payee or endpoint is checked against both sets:

    in the trusted list ......................... passes
    typed by the user ........................... passes
    seen only in an external result ............. held for a human
    never seen .................................. passes (origin unknown)

Vaara does not see inside the model, so this is not taint tracking. It is a
record of where each value first appeared on either side of the model, and a
hold when the only appearance was in content the agent read.

The store keeps HMAC-SHA256 digests under a per-install key, never the values,
so it holds no addresses or account numbers. It is one JSON file, rewritten
whole under the lineage store's lock and replaced atomically, capped per set
(oldest drop first). An unreadable file reads as empty.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union
from urllib.parse import urlsplit

from vaara.scorer.lineage import _exclusive

DEFAULT_MAX_ENTRIES = 5000

#: Action ids kept per value: enough to name where it came from.
_MAX_IDS = 5

#: Characters of a result read for values. A page past this is cut, not skipped.
_MAX_SCAN = 1_000_000

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}\b")
# Two letters, two check digits, then 11 to 30 alphanumerics in groups of up
# to four, single spaces allowed between groups (the printed form).
_IBAN = re.compile(r"\b[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{1,4}){3,8}\b")
_URL = re.compile(r"\b(?:https?|wss?|ftp)://[^\s<>\"'`]+", re.IGNORECASE)

#: IBAN lengths by country (ISO 13616 registry), for the countries most
#: likely to appear; others are accepted on the checksum alone.
_IBAN_LEN = {
    "AT": 20, "BE": 16, "CH": 21, "CZ": 24, "DE": 22, "DK": 18, "EE": 20,
    "ES": 24, "FI": 18, "FR": 27, "GB": 22, "IE": 22, "IT": 27, "LT": 20,
    "LU": 20, "LV": 21, "NL": 18, "NO": 15, "PL": 28, "PT": 25, "SE": 24,
}

#: Tool-name words that send, pay or post on their own...
_SEND_VERBS = frozenset({
    "send", "reply", "forward", "pay", "transfer", "wire", "remit", "payout",
    "refund", "publish", "upload", "share", "invite", "post", "notify", "email",
    "mail",
})
#: ...and words that do when they create one of these.
_MAKE_VERBS = frozenset({"create", "make", "initiate", "submit", "schedule", "new", "add"})
_SENT_THINGS = frozenset({
    "payment", "payments", "transfer", "transfers", "invoice", "payout",
    "email", "mail", "message", "draft", "webhook", "subscription",
})
#: A leading word that makes the call a read whatever follows.
_READ_VERBS = frozenset({
    "get", "list", "read", "search", "find", "fetch", "query", "lookup",
    "view", "show", "describe", "count",
})


def _words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def authority_tool(tool_name: str) -> bool:
    """True when the tool's own name says it sends, pays or posts.

    Words, not substrings: ``read_email`` and ``list_payments`` read, while
    ``send_email``, ``email_user`` and ``create_payment`` act.
    """
    words = _words(_bare_tool(tool_name))
    if not words or words[0] in _READ_VERBS:
        return False
    found = set(words)
    return bool(found & _SEND_VERBS) or bool(found & _MAKE_VERBS and found & _SENT_THINGS)


#: Argument names that carry a recipient, a payee or an endpoint.
PRESET_AUTHORITY_ARGS = frozenset({
    "to", "cc", "bcc", "recipient", "recipients", "email", "emails",
    "email_address", "address", "reply_to", "iban", "account",
    "account_number", "bank_account", "payee", "beneficiary", "creditor",
    "destination", "url", "webhook", "webhook_url", "endpoint",
    "callback_url", "target_url",
})


def _iban_ok(candidate: str) -> bool:
    want = _IBAN_LEN.get(candidate[:2])
    if want is not None and len(candidate) != want:
        return False
    if not 15 <= len(candidate) <= 34:
        return False
    moved = candidate[4:] + candidate[:4]
    digits = "".join(str(int(c, 36)) for c in moved)
    return int(digits) % 97 == 1


def _iban(raw: str) -> Optional[str]:
    """The longest valid IBAN at the start of ``raw``, spaces removed."""
    flat = raw.replace(" ", "")
    for end in range(min(len(flat), 34), 14, -1):
        if _iban_ok(flat[:end]):
            return flat[:end]
    return None


def _host(url: str) -> Optional[str]:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.rstrip(".").lower()
    return host if "." in host or ":" in host else None


def extract(text: str) -> set[tuple[str, str]]:
    """Every authority-shaped value in ``text``, as (kind, normalised value)."""
    if not isinstance(text, str) or not text:
        return set()
    text = text[:_MAX_SCAN]
    found: set[tuple[str, str]] = set()
    for m in _EMAIL.finditer(text):
        found.add(("email", m.group(0).lower()))
    for m in _IBAN.finditer(text):
        value = _iban(m.group(0))
        if value:
            found.add(("iban", value))
    for m in _URL.finditer(text):
        host = _host(m.group(0))
        if host:
            found.add(("host", host))
    return found


def _strings(value: Any, depth: int = 0) -> Iterator[str]:
    """Every string inside ``value`` (dicts, lists, nesting capped)."""
    if depth > 6:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v, depth + 1)


def extract_any(value: Any) -> set[tuple[str, str]]:
    """``extract`` over every string inside a JSON-shaped value."""
    found: set[tuple[str, str]] = set()
    budget = _MAX_SCAN
    for s in _strings(value):
        if budget <= 0:
            break
        found |= extract(s[:budget])
        budget -= len(s)
    return found


def _bare_tool(tool_name: str) -> str:
    """``mcp__mail__send_email`` -> ``send_email``."""
    return tool_name.rsplit("__", 1)[-1] if tool_name.startswith("mcp__") else tool_name


def authority_args(tool_name: str, parameters: Any,
                   extra: Optional[dict[str, Iterable[str]]] = None) -> dict[str, Any]:
    """The arguments of this call that carry authority, by name.

    The preset covers tools whose name sends, pays or posts, and only their
    recipient, payee and endpoint arguments. ``extra`` maps a tool name (full
    or bare) to argument names the operator declares on top. Undeclared
    arguments are never checked.
    """
    if not isinstance(parameters, dict):
        return {}
    names: set[str] = set()
    if authority_tool(tool_name):
        names |= PRESET_AUTHORITY_ARGS
    for key in (tool_name, _bare_tool(tool_name)):
        names |= {str(a).lower() for a in (extra or {}).get(key, ())}
    return {k: v for k, v in parameters.items()
            if isinstance(k, str) and k.lower() in names}


def _default_path() -> Path:
    return Path(os.environ.get("VAARA_VALUE_ORIGIN_PATH")
                or Path.home() / ".vaara" / "value_origin.json")


class OriginStore:
    """Digests of values seen in external results and in the user's prompts."""

    def __init__(self, path: Union[str, Path, None] = None,
                 max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self.max_entries = max_entries
        self._key: Optional[bytes] = None

    def _load_key(self) -> bytes:
        if self._key is not None:
            return self._key
        key_path = self.path.with_name(self.path.stem + ".key")
        try:
            key = key_path.read_bytes()
        except FileNotFoundError:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                key = key_path.read_bytes()
            else:
                key = secrets.token_bytes(32)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(key)
        if len(key) < 32:
            raise ValueError(f"value-origin key {key_path} is shorter than 32 bytes")
        self._key = key
        return key

    def digest(self, kind: str, value: str) -> str:
        return hmac.new(self._load_key(), f"{kind}:{value}".encode(),
                        hashlib.sha256).hexdigest()

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        out: dict[str, dict[str, Any]] = {}
        for side in ("external", "user"):
            got = data.get(side) if isinstance(data, dict) else None
            out[side] = ({k: v for k, v in got.items()
                          if isinstance(k, str) and isinstance(v, dict)}
                         if isinstance(got, dict) else {})
        return out

    def _write(self, sets: dict[str, dict[str, Any]]) -> None:
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"version": 1, **sets}, indent=1),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path.with_name(self.path.name + ".lock")):
            yield

    def _add(self, side: str, values: set[tuple[str, str]],
             tool_name: str = "", action_id: str = "") -> int:
        if not values:
            return 0
        digests = [(kind, self.digest(kind, v)) for kind, v in values]
        now = time.time()
        with self._locked():
            sets = self._read()
            entries = sets[side]
            for kind, d in digests:
                entry = entries.pop(d, None) or {"kind": kind, "first_seen": now}
                entry["last_seen"] = now
                if side == "external":
                    if tool_name and not entry.get("tool"):
                        entry["tool"] = tool_name
                    ids = [i for i in entry.get("action_ids", []) if isinstance(i, str)]
                    if action_id and action_id not in ids and len(ids) < _MAX_IDS:
                        ids.append(action_id)
                    entry["action_ids"] = ids
                entries[d] = entry
            while len(entries) > self.max_entries:
                entries.pop(next(iter(entries)))
            self._write(sets)
        return len(digests)

    def note_external(self, values: set[tuple[str, str]], *, tool_name: str,
                      action_id: str = "") -> int:
        """Remember values read in the result of ``tool_name``."""
        return self._add("external", values, tool_name, action_id)

    def note_user(self, values: set[tuple[str, str]]) -> int:
        """Remember values the user typed."""
        return self._add("user", values)

    def entries(self) -> dict[str, dict[str, Any]]:
        return self._read()


def origin_hold(tool_name: str, parameters: Any, store: OriginStore, *,
                trusted: Iterable[str] = (),
                extra_args: Optional[dict[str, Iterable[str]]] = None,
                ) -> Optional[dict[str, dict[str, Any]]]:
    """The authority arguments whose value came only from external content.

    Maps each such argument to ``{"seen_in": [action ids], "source": tool}``,
    or None when every value is trusted, was typed by the user, or was never
    seen. Checks only declared authority arguments (see ``authority_args``).
    """
    args = authority_args(tool_name, parameters, extra_args)
    if not args:
        return None
    trusted_values = set()
    for t in trusted:
        trusted_values |= {v for _, v in extract(str(t))} or {str(t).strip().lower()}
    sets = store.entries()
    held: dict[str, dict[str, Any]] = {}
    for name, value in args.items():
        for kind, v in sorted(extract_any(value)):
            if v in trusted_values:
                continue
            d = store.digest(kind, v)
            if d in sets["user"]:
                continue
            seen = sets["external"].get(d)
            if seen is None:
                continue
            slot = held.setdefault(name, {"seen_in": [], "source": seen.get("tool", "")})
            for i in seen.get("action_ids", []):
                if isinstance(i, str) and i not in slot["seen_in"]:
                    slot["seen_in"].append(i)
    return held or None


def describe(held: dict[str, dict[str, Any]]) -> str:
    """One reason line: which argument, and the result it came from."""
    parts = []
    for name, info in held.items():
        where = f"the result of {info.get('source') or 'an earlier tool call'}"
        ids = info.get("seen_in") or []
        if ids:
            where += f" (action {', '.join(ids[:3])})"
        parts.append(f"{name} first seen in {where}")
    return "value origin: " + "; ".join(parts) + ", held for a human"
