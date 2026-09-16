# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Reversible sealing of named secrets on the way out to a model provider.

The redaction in ``_llm_proxy_shape`` replaces matches with ``***``.  That is
right for the audit trail and wrong for egress: a model handed ``***`` cannot
answer, so the conversation breaks.  Sealing is the reversible form.  A named
secret leaves as a stable placeholder and is restored in the response before
the caller sees it.

What this does NOT do, stated here because the gap matters more than the
feature: it only covers what can be named in advance.  Novel thinking written
as prose cannot be pre-registered, so it travels in the clear.  This bounds and
records the channel rather than closing it.

Placeholders are derived from the secret's own digest, so they are stable
across restarts.  That is deliberate: a placeholder that changed per run would
alter the prompt prefix on every turn and defeat provider-side prompt caching.

Everything here fails open.  A sealing bug must not cost a working session, so
every entry point returns its input unchanged when anything goes wrong.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vaara.llm_seal")

#: Placeholder shape. ASCII, no punctuation a tokenizer will split oddly, and
#: distinctive enough that a collision with real text is not a live concern.
_PREFIX = "VAARA_SEAL_"
_DIGEST_CHARS = 12
_PLACEHOLDER_RE = re.compile(_PREFIX + r"[0-9a-f]{%d}" % _DIGEST_CHARS)

#: Longest placeholder a stream carry-buffer must be able to hold back.
PLACEHOLDER_LEN = len(_PREFIX) + _DIGEST_CHARS


def placeholder_for(secret: str) -> str:
    """Stable placeholder for one secret, derived from its own digest."""
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return _PREFIX + digest[:_DIGEST_CHARS]


def _json_inner(value: str) -> str:
    """The JSON-escaped form of a string, without the surrounding quotes.

    Request bodies are JSON, so a secret containing a quote, a backslash or a
    newline appears on the wire in escaped form.  Substituting on the raw bytes
    needs the escaped spelling as well as the plain one.
    """
    return json.dumps(value)[1:-1]


class SealRegistry:
    """Named secrets, and the substitution in both directions.

    Operates on raw bytes rather than on a parsed body.  Re-serialising JSON
    would rewrite byte layout the caller never asked to change, and any such
    rewrite risks the provider's prompt-cache prefix.
    """

    def __init__(self, secrets: Optional[dict[str, str]] = None) -> None:
        self._pairs: list[tuple[str, str]] = []
        self._reverse: dict[str, str] = {}
        #: Where the entries came from, if a file. ``refresh`` re-reads it.
        self._path: Optional[Path] = None
        self._mtime: Optional[float] = None
        self._load(secrets)

    def _load(self, secrets: Optional[dict[str, str]]) -> None:
        self._pairs = []
        self._reverse = {}
        skipped = 0
        for name, secret in (secrets or {}).items():
            if not secret:
                # The name is not logged either. A seal name describes what it
                # seals, so "northern_lights_concept" leaks the thing the entry
                # exists to hide. Count them and say how many.
                skipped += 1
                continue
            self.add(name, secret)
        if skipped:
            logger.warning("%d seal entr%s had an empty secret and were "
                           "skipped", skipped, "y" if skipped == 1 else "ies")

    def add(self, name: str, secret: str) -> None:
        token = placeholder_for(secret)
        self._pairs.append((secret, token))
        self._reverse[token] = secret
        # Longest first, so a secret that contains another is sealed whole.
        self._pairs.sort(key=lambda p: len(p[0]), reverse=True)

    @staticmethod
    def _read(p: Path) -> Optional[dict[str, str]]:
        """Parse the file, or None when it cannot be used. Never raises."""
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("seal registry %s unreadable (%s), sealing off", p, exc)
            return None
        if not isinstance(data, dict):
            logger.warning("seal registry %s is not an object, sealing off", p)
            return None
        return {str(k): str(v) for k, v in data.items()}

    @staticmethod
    def _stat_mtime(p: Path) -> Optional[float]:
        try:
            return p.stat().st_mtime
        except OSError:
            return None

    @classmethod
    def from_file(cls, path: str | Path) -> "SealRegistry":
        """Load ``{"name": "secret"}`` from JSON. Missing file means inactive.

        The registry remembers the path. ``refresh`` re-reads the file when
        its mtime changes, so an edit reaches the running proxy instead of
        waiting for a restart while the monitor already reports the new state.
        """
        p = Path(path)
        reg = cls()
        reg._path = p
        reg._mtime = cls._stat_mtime(p)
        if reg._mtime is None:
            return reg
        data = cls._read(p)
        if data is not None:
            reg._load(data)
        return reg

    def refresh(self) -> bool:
        """Re-read the backing file if it changed. True when reloaded.

        One stat per call. A file that vanished empties the registry, which
        is what the operator asked for by deleting it. A file that became
        unreadable also empties it: the proxy reports the state it can prove,
        not the last good one.
        """
        if self._path is None:
            return False
        mtime = self._stat_mtime(self._path)
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        data = self._read(self._path) if mtime is not None else None
        self._load(data or {})
        logger.info("seal registry %s reloaded, %d secret(s)",
                    self._path, len(self._pairs))
        return True

    @property
    def active(self) -> bool:
        return bool(self._pairs)

    def __len__(self) -> int:
        """How many secrets are registered. For startup reporting."""
        return len(self._pairs)

    def seal_text(self, text: str) -> str:
        for secret, token in self._pairs:
            text = text.replace(secret, token)
            escaped = _json_inner(secret)
            if escaped != secret:
                text = text.replace(escaped, token)
        return text

    def unseal_text(self, text: str) -> str:
        return self.unseal_text_counted(text)[0]

    def unseal_text_counted(self, text: str) -> tuple[str, int]:
        """Restore placeholders and say how many were restored."""
        restored = 0
        for token, secret in self._reverse.items():
            n = text.count(token)
            if n:
                restored += n
                text = text.replace(token, secret)
        return text, restored

    def seal_bytes(self, raw: bytes) -> bytes:
        if not self._pairs:
            return raw
        try:
            return self.seal_text(raw.decode("utf-8")).encode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return raw

    def unseal_bytes(self, raw: bytes) -> bytes:
        if not self._reverse:
            return raw
        try:
            return self.unseal_text(raw.decode("utf-8")).encode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return raw

    def count_sealed(self, raw: bytes) -> int:
        """How many placeholders a sealed body carries. For the receipt."""
        try:
            return len(_PLACEHOLDER_RE.findall(raw.decode("utf-8", "replace")))
        except Exception:  # pragma: no cover - counting must never raise
            return 0

    def unmapped_placeholders(self, raw: bytes) -> list[str]:
        """Placeholders present that this registry cannot restore.

        A body carrying a seal token with no mapping means a secret was sealed
        by a registry this process does not have.  Forwarding it is harmless,
        but the caller may want to refuse, so the question is answerable.
        """
        try:
            found = set(_PLACEHOLDER_RE.findall(raw.decode("utf-8", "replace")))
        except Exception:  # pragma: no cover
            return []
        return sorted(found - set(self._reverse))


class StreamUnsealer:
    """Restore placeholders in a streamed response.

    A placeholder can be split across two SSE events, so the tail of each chunk
    is held back until the next one arrives.  Only as many bytes are withheld
    as a placeholder could occupy, so nothing is buffered for long.
    """

    def __init__(self, registry: SealRegistry) -> None:
        self._registry = registry
        self._carry = ""
        #: How many placeholders were restored so far. For the receipt.
        self.restored = 0
        self._unmapped: set[str] = set()

    @property
    def unmapped(self) -> list[str]:
        """Placeholders that passed through unrestored, sorted."""
        return sorted(self._unmapped)

    def feed(self, chunk: bytes) -> bytes:
        if not self._registry.active:
            return chunk
        try:
            text = self._carry + chunk.decode("utf-8", "replace")
        except Exception:  # pragma: no cover
            return chunk
        # Restore before splitting. Unsealing only the emitted slice would let
        # a placeholder that straddles the boundary have its head emitted raw
        # while its tail is still in the carry, which is what the first version
        # of this did and what the byte-at-a-time test caught.
        text, n = self._registry.unseal_text_counted(text)
        self.restored += n
        keep = max(0, PLACEHOLDER_LEN - 1)
        if len(text) > keep:
            emit, self._carry = text[:-keep], text[-keep:]
        else:
            emit, self._carry = "", text
        # A placeholder still present after unsealing is one nobody here can
        # restore. Count a match the first time its start falls inside the
        # emitted slice. The carry is one byte shorter than a placeholder, so
        # a match starting in the emitted slice is complete in `text`, and a
        # match starting in the carry is seen again next feed with the same
        # bytes ahead of it. Each is counted exactly once.
        for m in _PLACEHOLDER_RE.finditer(text):
            if m.start() < len(emit):
                self._unmapped.add(m.group(0))
        return emit.encode("utf-8")

    def flush(self) -> bytes:
        if not self._carry:
            return b""
        out, n = self._registry.unseal_text_counted(self._carry)
        self.restored += n
        self._unmapped.update(_PLACEHOLDER_RE.findall(out))
        self._carry = ""
        return out.encode("utf-8")
