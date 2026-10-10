# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Reversible sealing of named secrets on the way out to a model provider.

The redaction in ``_llm_proxy_shape`` replaces matches with ``***``.  That is
right for the audit trail and wrong for egress: a model handed ``***`` cannot
answer, so the conversation breaks.  Sealing is the reversible form.  A named
secret leaves as a stable placeholder and is restored in the response before
the caller sees it.

What this does NOT do, stated here because the gap matters more than the
feature: it only covers what can be named in advance, plus, when
``known_formats`` is on, values matching a published credential format
(``KNOWN_SECRET_FORMATS``) and machine-generated tokens of any format that sit
where a credential sits (``CONTEXT_SECRET_RULES``). A short or dictionary-word
password is not caught by the context rule. Emails, names and other personal
data are not detected.  Novel thinking written
as prose cannot be pre-registered, so it travels in the clear.  This bounds and
records the channel rather than closing it.

Placeholders are a keyed digest of the secret (HMAC-SHA256 under a key that
lives beside the seal file), so they are stable across restarts.  That is
deliberate: a placeholder that changed per run would alter the prompt prefix
on every turn and defeat provider-side prompt caching.  The key is what keeps
the placeholder from saying anything about the secret: the provider sees every
placeholder, and a named secret is often a phrase or a name rather than a
random token, so a plain digest would let anyone holding the prompt confirm a
guess at it offline.  A registry with no file uses a key drawn for the process.

Everything here fails open.  A sealing bug must not cost a working session, so
every entry point returns its input unchanged when anything goes wrong.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets as _secrets
from collections import Counter
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


_EDGE = r"(?<![A-Za-z0-9_-])"

#: Credential formats sealed when ``known_formats`` is on. Each is a format
#: its issuer publishes, chosen so an ordinary word or identifier does not
#: match. Order matters where one prefix contains another (``sk-ant-`` before
#: ``sk-``). Matching runs on the JSON text as it goes out, so a PEM block is
#: matched in its escaped form and restored in the same form.
KNOWN_SECRET_FORMATS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:\\n|\n|[A-Za-z0-9+/=\s])+?"
        r"-----END [A-Z ]*PRIVATE KEY-----")),
    ("anthropic_key", re.compile(_EDGE + r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(
        _EDGE + r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(
        _EDGE + r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("aws_access_key_id", re.compile(_EDGE + r"(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])")),
    ("google_api_key", re.compile(_EDGE + r"AIza[0-9A-Za-z_-]{35}")),
    ("slack_token", re.compile(_EDGE + r"xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("stripe_key", re.compile(_EDGE + r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("huggingface_token", re.compile(_EDGE + r"hf_[A-Za-z0-9]{34,}")),
    ("jwt", re.compile(
        _EDGE + r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
)


#: Key names that hold a credential. The key has to END in one of these, so
#: ``max_tokens``, ``token_count`` and ``secret_name`` are not secret-named.
#: ``pwd`` is left out: ``PWD=`` is the shell's working directory.
_SECRET_KEY_WORD = (r"(?:api[_-]?key|apikey|secret|token|password|passwd|"
                    r"access[_-]?key|private[_-]?key|credentials?|auth)")
#: An optional quote around a key or value, JSON-escaped or not: the rules run
#: on the wire text, where a config pasted into a prompt is escaped once more.
_Q = r"""(?:\\?["'])?"""
_TOKEN = r"(?P<val>[A-Za-z0-9_\-+/=.~]{16,})"

#: Credentials in no published format, caught by where they sit: a value
#: assigned to a secret-named key (``API_KEY=...``, ``"client_secret": "..."``,
#: ``x-api-key: ...``) or sent as a bearer token. Only the value is sealed, and
#: only when ``_looks_generated`` holds, so ``password: changeme``, template
#: text like ``YOUR_API_KEY_HERE`` and an id under an ordinary key stay as
#: written. Runs after ``KNOWN_SECRET_FORMATS``.
CONTEXT_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("assigned_secret", re.compile(
        r"(?i)(?<![A-Za-z0-9])[A-Za-z0-9_.-]*" + _SECRET_KEY_WORD
        + _Q + r"\s*[:=]\s*" + _Q + _TOKEN)),
    ("bearer_token", re.compile(r"(?i)(?<![A-Za-z0-9])Bearer\s+" + _TOKEN)),
)

#: Bits per character a value needs before the context rule seals it. Random
#: base64 and hex sit near 4 and above at this length; words and repeated
#: filler sit well under.
_MIN_ENTROPY = 3.0


def _looks_generated(value: str) -> bool:
    """True when a value reads as a generated token, not a word or filler."""
    if _PLACEHOLDER_RE.fullmatch(value):
        return False
    classes = (any(c.islower() for c in value) + any(c.isupper() for c in value)
               + any(c.isdigit() for c in value))
    if classes < 2:
        return False
    n = len(value)
    entropy = -sum(k / n * math.log2(k / n) for k in Counter(value).values())
    return entropy >= _MIN_ENTROPY


#: The key file beside a seal file, under ``keys/`` so the deny rules that
#: keep a governed agent away from Vaara's other signing keys cover it too.
KEY_RELPATH = Path("keys") / "seal-hmac.key"


def placeholder_for(secret: str, key: bytes) -> str:
    """Stable placeholder for one secret under ``key``.

    HMAC-SHA256, truncated to the placeholder's digest field. Stable for a
    given key, and unrelated to the secret for anyone without it.
    """
    digest = hmac.new(key, secret.encode("utf-8"), hashlib.sha256).hexdigest()
    return _PREFIX + digest[:_DIGEST_CHARS]


def _load_or_create_key(path: Path) -> Optional[bytes]:
    """The key at ``path`` (hex text), made with mode 0600 when missing.

    Returns None when the key can neither be read nor created, so the caller
    can fall back to a process key and say so: sealing must keep working.
    """
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass  # another proxy made it first; read theirs
        except OSError as exc:
            logger.warning("the seal key beside the seal file could not be "
                           "created (%s)", type(exc).__name__)
            return None
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(_secrets.token_hex(32))
    try:
        key = bytes.fromhex(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        logger.warning("the seal key beside the seal file is unreadable (%s)",
                       type(exc).__name__)
        return None
    return key or None


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

    def __init__(self, secrets: Optional[dict[str, str]] = None, *,
                 known_formats: bool = False, key: Optional[bytes] = None) -> None:
        #: The placeholder key. ``from_file`` reads it from beside the seal
        #: file; a registry built in memory draws one for the process.
        self._key: bytes = key if key else _secrets.token_bytes(32)
        self._pairs: list[tuple[str, str]] = []
        self._reverse: dict[str, str] = {}
        #: Seal values matching ``KNOWN_SECRET_FORMATS`` as well as the named
        #: ones. A matched value is learned for the life of the process so
        #: the reply can be restored, and it survives a file reload.
        self.known_formats = known_formats
        self._learned: dict[str, str] = {}
        #: Per-format counts from the most recent ``seal_bytes`` call.
        self.last_kinds: dict[str, int] = {}
        #: Where the entries came from, if a file. ``refresh`` re-reads it.
        self._path: Optional[Path] = None
        self._mtime: Optional[float] = None
        self._load(secrets)

    def _load(self, secrets: Optional[dict[str, str]]) -> None:
        self._pairs = []
        # Learned values outlive a reload: a reply to a request sealed before
        # the file changed still has to be restored.
        self._reverse = {token: value for value, token
                         in getattr(self, "_learned", {}).items()}
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

    def placeholder(self, secret: str) -> str:
        """The placeholder this registry gives ``secret``."""
        return placeholder_for(secret, self._key)

    def add(self, name: str, secret: str) -> None:
        token = self.placeholder(secret)
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
        key = _load_or_create_key(p.parent / KEY_RELPATH)
        if key is None:
            logger.warning(
                "the seal key beside the seal file is unavailable; placeholders "
                "are stable for this process only and change on restart",
            )
        reg = cls(key=key)
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
        return bool(self._pairs) or self.known_formats

    @property
    def named(self) -> int:
        """How many secrets the operator named. Learned values excluded."""
        return len(self._pairs)

    def __len__(self) -> int:
        """How many secrets are registered. For startup reporting."""
        return len(self._pairs)

    def _learn(self, value: str, kind: str, kinds: dict[str, int]) -> str:
        token = self.placeholder(value)
        self._learned[value] = token
        self._reverse[token] = value
        kinds[kind] = kinds.get(kind, 0) + 1
        return token

    def _seal_known(self, text: str) -> str:
        kinds: dict[str, int] = {}
        for kind, pattern in KNOWN_SECRET_FORMATS:
            def known(m: re.Match[str], kind: str = kind) -> str:
                return self._learn(m.group(0), kind, kinds)
            text = pattern.sub(known, text)
        for kind, pattern in CONTEXT_SECRET_RULES:
            def sub(m: re.Match[str], kind: str = kind) -> str:
                value = m.group("val")
                if not _looks_generated(value):
                    return m.group(0)
                head = m.group(0)[:m.start("val") - m.start()]
                return head + self._learn(value, kind, kinds)
            text = pattern.sub(sub, text)
        self.last_kinds = kinds
        return text

    def seal_text(self, text: str) -> str:
        self.last_kinds = {}
        for secret, token in self._pairs:
            text = text.replace(secret, token)
            escaped = _json_inner(secret)
            if escaped != secret:
                text = text.replace(escaped, token)
        if self.known_formats:
            text = self._seal_known(text)
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
        if not self.active:
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


#: Delta fields whose text a model writes and a placeholder can appear in.
_DELTA_TEXT_FIELDS = ("text", "partial_json", "thinking")


def _partial_placeholder_tail(text: str) -> int:
    """Length of the longest suffix of ``text`` that could begin a placeholder.

    A placeholder split across two events leaves its head at the end of one
    delta and its tail at the start of the next. Only that head needs holding
    back; everything before it can be emitted at once. Returns 0 when no
    suffix could be the start of a placeholder.
    """
    limit = min(len(text), PLACEHOLDER_LEN - 1)
    for n in range(limit, 0, -1):
        tail = text[-n:]
        if n <= len(_PREFIX):
            if _PREFIX.startswith(tail):
                return n
            continue
        if tail.startswith(_PREFIX) and all(
            c in "0123456789abcdef" for c in tail[len(_PREFIX):]
        ):
            return n
    return 0


class SseUnsealer:
    """Restore placeholders in a streamed SSE response, across event frames.

    ``StreamUnsealer`` runs a regex over the raw bytes with a short carry, which
    handles a placeholder split by an HTTP chunk boundary. It cannot handle a
    placeholder split by an SSE EVENT boundary, because the two halves then
    have event framing between them, and a client streaming tool input emits
    many small ``input_json_delta`` events. MEASURED 2026-09-19 on the live
    trail: 1,881 outcomes, a placeholder restored once, an unrestored one
    reported never, while every request sealed 30 to 45 of them. The
    placeholders went back to the caller literal and the record said clean.

    This class parses the frames. For each ``content_block_delta`` it decodes
    the delta's text field, prepends the held head of the previous delta,
    restores placeholders in the decoded text, holds back only a suffix that
    could begin a placeholder, and re-emits the event with the new text. Any
    other event first flushes the held text as one extra delta of the same
    shape, so the client sees the same concatenation it would have seen.

    Bytes that are not SSE frames, or frames that do not parse, pass through
    the byte-level regex instead, so nothing is worse than before.
    """

    def __init__(self, registry: SealRegistry) -> None:
        self._registry = registry
        self._buf = b""
        self._held = ""
        self._held_shape: Optional[tuple[int, str, str]] = None
        self.restored = 0
        self._unmapped: set[str] = set()

    @property
    def unmapped(self) -> list[str]:
        return sorted(self._unmapped)

    def _note_unmapped(self, text: str) -> None:
        for m in _PLACEHOLDER_RE.finditer(text):
            self._unmapped.add(m.group(0))

    def _raw(self, data: bytes) -> bytes:
        text, n = self._registry.unseal_text_counted(data.decode("utf-8", "replace"))
        self.restored += n
        self._note_unmapped(text)
        return text.encode("utf-8")

    def _flush_held(self) -> bytes:
        if not self._held or self._held_shape is None:
            self._held = ""
            return b""
        index, dtype, field = self._held_shape
        text, n = self._registry.unseal_text_counted(self._held)
        self.restored += n
        self._note_unmapped(text)
        self._held = ""
        event = {"type": "content_block_delta", "index": index,
                 "delta": {"type": dtype, field: text}}
        return ("event: content_block_delta\ndata: "
                + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8")

    def _event(self, frame: bytes) -> bytes:
        try:
            text = frame.decode("utf-8")
        except UnicodeDecodeError:
            return self._flush_held() + self._raw(frame + b"\n\n")
        data_lines = [ln[5:].lstrip() for ln in text.split("\n") if ln.startswith("data:")]
        if len(data_lines) != 1:
            return self._flush_held() + self._raw(frame + b"\n\n")
        try:
            payload = json.loads(data_lines[0])
        except json.JSONDecodeError:
            return self._flush_held() + self._raw(frame + b"\n\n")
        delta = payload.get("delta") if isinstance(payload, dict) else None
        if (payload.get("type") != "content_block_delta"
                or not isinstance(delta, dict)):
            return self._flush_held() + self._raw(frame + b"\n\n")
        field = next((f for f in _DELTA_TEXT_FIELDS
                      if isinstance(delta.get(f), str)), None)
        if field is None:
            return self._flush_held() + self._raw(frame + b"\n\n")
        shape = (payload.get("index", 0), str(delta.get("type", "")), field)
        out = b""
        if self._held and self._held_shape != shape:
            out += self._flush_held()
        combined = self._held + delta[field]
        self._held = ""
        combined, n = self._registry.unseal_text_counted(combined)
        self.restored += n
        keep = _partial_placeholder_tail(combined)
        emit = combined[:len(combined) - keep] if keep else combined
        self._note_unmapped(emit)
        if keep:
            self._held = combined[len(combined) - keep:]
            self._held_shape = shape
        delta[field] = emit
        rebuilt = json.dumps(payload, ensure_ascii=False)
        head = text[:text.index("data:")]
        return out + (head + "data: " + rebuilt + "\n\n").encode("utf-8")

    def feed(self, chunk: bytes) -> bytes:
        if not self._registry.active:
            return chunk
        self._buf += chunk
        out = b""
        while True:
            cut = self._buf.find(b"\n\n")
            if cut < 0:
                break
            frame, self._buf = self._buf[:cut], self._buf[cut + 2:]
            try:
                out += self._event(frame)
            except Exception as exc:  # pragma: no cover - guard, not a path
                logger.warning("sse unseal failed on one frame: %s", exc)
                out += self._flush_held() + self._raw(frame + b"\n\n")
        return out

    def flush(self) -> bytes:
        out = self._flush_held()
        if self._buf:
            out += self._raw(self._buf)
            self._buf = b""
        return out
