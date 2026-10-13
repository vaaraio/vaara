# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recover audit records from a damaged trail file page by page.

``vaara trail repair`` reads a damaged file through SQLite, so it keeps every
row the table's b-tree still reaches. A row on a leaf page that no interior
page points at any more is invisible to it, though the bytes are still on
disk. This module reads the file without SQLite: it walks every page, decodes
the cells of every table leaf page it finds, follows overflow chains, and
keeps a row only when it is an ``audit_records`` row whose ``record_hash``
recomputes from its own content. A row that passes is the record as written;
nothing is inferred or filled in.

Read-only. The damaged file is opened for reading and never written.

The result says, for each recovered record, whether its ``previous_hash``
names the record before it, so a reader can see which stretches chain end to
end and where a link is still missing.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

_HEADER_MAGIC = b"SQLite format 3\x00"
_LEAF_TABLE = 0x0D


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode an SQLite varint at ``pos``; return (value, next position)."""
    value = 0
    for i in range(8):
        if pos + i >= len(buf):
            raise ValueError("varint runs past the buffer")
        b = buf[pos + i]
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos + i + 1
    if pos + 8 >= len(buf):
        raise ValueError("varint runs past the buffer")
    return (value << 8) | buf[pos + 8], pos + 9


def _decode_record(payload: bytes) -> list[Any]:
    header_len, pos = _varint(payload, 0)
    if header_len > len(payload) or header_len < 1:
        raise ValueError("record header longer than its payload")
    types: list[int] = []
    while pos < header_len:
        t, pos = _varint(payload, pos)
        types.append(t)
    body = header_len
    values: list[Any] = []
    for t in types:
        if t == 0:
            values.append(None)
        elif 1 <= t <= 6:
            n = (1, 2, 3, 4, 6, 8)[t - 1]
            values.append(int.from_bytes(payload[body:body + n], "big", signed=True))
            body += n
        elif t == 7:
            values.append(struct.unpack(">d", payload[body:body + 8])[0])
            body += 8
        elif t == 8:
            values.append(0)
        elif t == 9:
            values.append(1)
        elif t >= 12:
            n = (t - 12) // 2
            raw = payload[body:body + n]
            values.append(raw if t % 2 == 0 else raw.decode("utf-8"))
            body += n
        else:
            raise ValueError(f"reserved serial type {t}")
        if body > len(payload):
            raise ValueError("record body runs past its payload")
    return values


class _File:
    def __init__(self, path: Path) -> None:
        self.data = path.read_bytes()
        if self.data[:16] != _HEADER_MAGIC:
            raise ValueError("not an SQLite database file")
        size = int.from_bytes(self.data[16:18], "big")
        self.page_size = 65536 if size == 1 else size
        self.usable = self.page_size - self.data[20]
        if self.data[56:60] not in (b"\x00\x00\x00\x00", b"\x00\x00\x00\x01"):
            raise ValueError("only UTF-8 trail files can be carved")
        self.pages = len(self.data) // self.page_size

    def page(self, n: int) -> bytes:
        start = (n - 1) * self.page_size
        return self.data[start:start + self.page_size]

    def _payload(self, page: bytes, pos: int, total: int) -> bytes:
        u = self.usable
        x = u - 35
        if total <= x:
            return page[pos:pos + total]
        m = ((u - 12) * 32 // 255) - 23
        k = m + ((total - m) % (u - 4))
        local = k if k <= x else m
        out = bytearray(page[pos:pos + local])
        nxt = int.from_bytes(page[pos + local:pos + local + 4], "big")
        seen: set[int] = set()
        while len(out) < total:
            if nxt < 1 or nxt > self.pages or nxt in seen:
                raise ValueError("overflow chain leaves the file or loops")
            seen.add(nxt)
            ov = self.page(nxt)
            nxt = int.from_bytes(ov[:4], "big")
            out += ov[4:4 + min(u - 4, total - len(out))]
        return bytes(out)

    def leaf_cells(self) -> Iterator[tuple[int, int, list[Any]]]:
        """(page number, rowid, column values) for every decodable leaf cell."""
        for n in range(1, self.pages + 1):
            page = self.page(n)
            hdr = 100 if n == 1 else 0
            if page[hdr] != _LEAF_TABLE:
                continue
            count = int.from_bytes(page[hdr + 3:hdr + 5], "big")
            for i in range(count):
                ptr_at = hdr + 8 + 2 * i
                ptr = int.from_bytes(page[ptr_at:ptr_at + 2], "big")
                if ptr < hdr + 8 or ptr >= self.usable:
                    continue
                try:
                    total, pos = _varint(page, ptr)
                    rowid, pos = _varint(page, pos)
                    values = _decode_record(self._payload(page, pos, total))
                except (ValueError, UnicodeDecodeError, struct.error):
                    continue
                yield n, rowid, values


def _columns(path: Path) -> list[str]:
    """audit_records column names in on-disk order, from the damaged file."""
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        # A file shorter than its header's page count is refused before any
        # read unless the schema check is skipped. The connection is read-only.
        conn.execute("PRAGMA writable_schema=ON")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(audit_records)")]
    finally:
        conn.close()
    if "record_hash" not in cols or "seq" not in cols:
        raise ValueError("audit_records schema is unreadable in this file")
    return cols


def _record_hash_holds(row: dict) -> bool:
    from vaara.audit.trail import AuditRecord, EventType
    try:
        rec = AuditRecord(
            record_id=row["record_id"], action_id=row["action_id"],
            event_type=EventType(row["event_type"]), timestamp=row["timestamp"],
            agent_id=row["agent_id"], tool_name=row["tool_name"],
            data=json.loads(row["data"]),
            regulatory_articles=json.loads(row["regulatory"]),
            previous_hash=row["previous_hash"], record_hash=row["record_hash"],
            tenant_id=row.get("tenant_id") or "",
            chain_version=row.get("chain_version") or 1,
        )
        return bool(row["record_hash"]) and rec.compute_hash() == row["record_hash"]
    except Exception:
        return False


@dataclass
class CarveResult:
    """What a page-by-page read of a damaged trail recovered."""

    source: str
    pages: int
    cells_read: int = 0
    records: list[dict] = field(default_factory=list)
    rejected: int = 0
    # seq -> True when previous_hash names the record at seq - 1 in this set.
    linked: dict[int, bool] = field(default_factory=dict)
    conflicts: list[int] = field(default_factory=list)

    def seqs(self) -> list[int]:
        return [r["seq"] for r in self.records]

    def summary(self, want: Optional[range] = None) -> dict:
        seqs = set(self.seqs())
        out: dict[str, Any] = {
            "source": self.source, "pages": self.pages,
            "cells_read": self.cells_read, "records_recovered": len(self.records),
            "cells_rejected": self.rejected,
            "unlinked_seqs": sorted(s for s, ok in self.linked.items() if not ok),
            "conflicting_seqs": list(self.conflicts),
        }
        if want is not None:
            got = [s for s in want if s in seqs]
            out["wanted"] = [want.start, want.stop - 1]
            out["wanted_recovered"] = len(got)
            out["wanted_missing"] = [s for s in want if s not in seqs]
        return out


def carve_trail(path: str | Path) -> CarveResult:
    """Read every page of ``path`` and keep the audit records that verify."""
    src = Path(path).expanduser()
    cols = _columns(src)
    f = _File(src)
    result = CarveResult(source=str(src), pages=f.pages)
    by_seq: dict[int, dict] = {}
    for _page, _rowid, values in f.leaf_cells():
        result.cells_read += 1
        if len(values) > len(cols) or len(values) < cols.index("seq") + 1:
            result.rejected += 1
            continue
        row = dict(zip(cols, values))
        # REAL affinity: SQLite writes an integral float as an integer, and
        # reads it back as a float. The hash was taken over the float.
        if isinstance(row.get("timestamp"), int):
            row["timestamp"] = float(row["timestamp"])
        if not isinstance(row.get("seq"), int) or not _record_hash_holds(row):
            result.rejected += 1
            continue
        prior = by_seq.get(row["seq"])
        if prior is None:
            by_seq[row["seq"]] = row
        elif prior["record_hash"] != row["record_hash"]:
            # Two verified records under one seq is a fork in the file. Both
            # are the record as written; neither is chosen here.
            if row["seq"] not in result.conflicts:
                result.conflicts.append(row["seq"])
    result.records = [by_seq[s] for s in sorted(by_seq)]
    for rec in result.records:
        before = by_seq.get(rec["seq"] - 1)
        result.linked[rec["seq"]] = (
            before is not None and rec["previous_hash"] == before["record_hash"]
        )
    result.conflicts.sort()
    return result


def write_jsonl(result: CarveResult, out: Path, want: Optional[range] = None) -> int:
    """Write the recovered records (optionally only seqs in ``want``) as JSONL."""
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in result.records:
            if want is not None and rec["seq"] not in want:
                continue
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            n += 1
    return n
