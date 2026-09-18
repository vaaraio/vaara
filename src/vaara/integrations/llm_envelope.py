"""What left, by size and by marker. Never by content.

The llm-proxy record already says that a prompt left, for which model, and
whether sealing ran. This module adds two things a reader can check without
seeing a word of the prompt:

* the **envelope**: how many bytes were forwarded, and how they split between
  the system prompt, the tool definitions, the conversation, and the last user
  message. The last user message is the ask. Everything else is what the
  client chose to send along with it, and the ratio is the number an operator
  watches when trimming what leaves the machine.

* the **marker watch**: a private list of unique strings, each under a short
  id. The proxy records which ids were inside the bytes that actually left.
  The strings themselves never reach the trail. An operator who plants one
  marker per channel can later tell, from the trail alone, which channel a
  leaked marker travelled through and when.

Both are measured on the outbound bytes, after sealing, so a marker the seal
replaced is correctly reported as not having left.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _text_len(content: Any) -> int:
    """UTF-8 byte length of the text inside a message content field."""
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += len(str(block.get("text", "")).encode("utf-8"))
                elif block.get("type") in ("tool_result", "tool_use"):
                    total += len(json.dumps(block, ensure_ascii=False)
                                 .encode("utf-8"))
        return total
    if isinstance(content, dict):
        return len(json.dumps(content, ensure_ascii=False).encode("utf-8"))
    return 0


def measure_envelope(body: dict[str, Any], outbound: bytes) -> dict[str, int]:
    """Split the forwarded request into the parts a reader cares about.

    Works for the Anthropic ``/v1/messages`` shape (top-level ``system`` and
    ``tools``) and the OpenAI ``/v1/chat/completions`` shape (a ``system``
    role inside ``messages``). Byte counts are of the text, not of the JSON
    framing around it, except for ``tools`` which is counted whole because
    its structure is the payload.
    """
    system = body.get("system")
    bytes_system = _text_len(system) if system is not None else 0

    tools = body.get("tools")
    bytes_tools = len(json.dumps(tools, ensure_ascii=False).encode("utf-8")) \
        if tools else 0

    bytes_messages = 0
    bytes_last_user = 0
    message_count = 0
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        size = _text_len(msg.get("content"))
        if role == "system":
            bytes_system += size
            continue
        message_count += 1
        bytes_messages += size
        if role == "user":
            bytes_last_user = size

    return {
        "bytes_out": len(outbound),
        "bytes_system": bytes_system,
        "bytes_tools": bytes_tools,
        "bytes_messages": bytes_messages,
        "bytes_last_user": bytes_last_user,
        "message_count": message_count,
    }


class MarkerWatch:
    """Which watched strings are inside a request body. Reported by id only.

    ``markers`` maps a short id to the string to watch for. Load from a JSON
    file of the same shape with :meth:`from_file`; :meth:`refresh` re-reads
    it, so markers can be added while the proxy runs. Neither the path nor
    any marker string is logged.
    """

    def __init__(self, markers: Optional[dict[str, str]] = None,
                 path: Optional[str] = None) -> None:
        self._path = path
        self._markers: dict[str, bytes] = {}
        self._load(markers or {})

    def _load(self, markers: dict[str, str]) -> None:
        self._markers = {
            str(k): str(v).encode("utf-8")
            for k, v in markers.items() if str(v)
        }

    @classmethod
    def from_file(cls, path: str) -> "MarkerWatch":
        w = cls(path=path)
        w.refresh()
        return w

    def refresh(self) -> None:
        if not self._path:
            return
        try:
            raw = Path(self._path).expanduser().read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except FileNotFoundError:
            data = {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("marker file unreadable: %s", type(exc).__name__)
            return
        if isinstance(data, dict):
            self._load(data)

    @property
    def active(self) -> bool:
        return bool(self._markers)

    def __len__(self) -> int:
        return len(self._markers)

    def present(self, outbound: bytes) -> list[str]:
        """Ids of the markers found in ``outbound``, in id order."""
        return sorted(k for k, v in self._markers.items() if v in outbound)
