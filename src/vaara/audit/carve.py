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

import hashlib
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
    from vaara.audit.trail import AuditRecord, stored_event_type
    try:
        rec = AuditRecord(
            record_id=row["record_id"], action_id=row["action_id"],
            event_type=stored_event_type(row["event_type"]), timestamp=row["timestamp"],
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


@dataclass
class RecoveryCheck:
    """Whether a carved bundle may be declared recovered on a live trail."""

    bundle: str
    bundle_sha256: str
    seqs: list[int] = field(default_factory=list)
    links_verified: int = 0
    anchored_to_seq: Optional[int] = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.seqs)

    def to_dict(self) -> dict:
        return {
            "bundle": self.bundle, "bundle_sha256": self.bundle_sha256,
            "recovered_seq_count": len(self.seqs),
            "first_seq": self.seqs[0] if self.seqs else None,
            "last_seq": self.seqs[-1] if self.seqs else None,
            "links_verified": self.links_verified,
            "anchored_to_seq": self.anchored_to_seq,
            "ok": self.ok, "problems": list(self.problems),
        }


def _declared_lost(con: sqlite3.Connection) -> set[int]:
    lost: set[int] = set()
    for (data,) in con.execute(
        "SELECT data FROM audit_records NOT INDEXED WHERE event_type = 'repair_gap'"
    ):
        try:
            lost.update(int(s) for s in json.loads(data).get("lost_seqs", []))
        except (ValueError, TypeError, AttributeError):
            continue
    return lost


def check_recovery(bundle: str | Path, db: str | Path) -> RecoveryCheck:
    """Check a carved JSONL bundle against the live trail it would fill.

    Every record must recompute its own hash, each must name the one before it,
    every seq must be one a REPAIR_GAP on the live trail declared lost, and no
    seq may already be on the trail. The first record is anchored when its
    previous_hash is the live record at the seq before it.
    """
    path = Path(bundle).expanduser()
    raw = path.read_bytes()
    check = RecoveryCheck(bundle=str(path), bundle_sha256=hashlib.sha256(raw).hexdigest())
    rows: list[dict] = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            check.problems.append(f"line {n} is not JSON")
            continue
        if not _record_hash_holds(row):
            check.problems.append(f"seq {row.get('seq')} does not recompute its record_hash")
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["seq"])
    check.seqs = [r["seq"] for r in rows]
    if len(set(check.seqs)) != len(check.seqs):
        check.problems.append("the bundle holds a seq twice")
    for before, after in zip(rows, rows[1:]):
        if after["seq"] == before["seq"] + 1:
            if after["previous_hash"] == before["record_hash"]:
                check.links_verified += 1
            else:
                check.problems.append(f"seq {after['seq']} does not name seq {before['seq']}")
    con = sqlite3.connect(f"{Path(db).expanduser().resolve().as_uri()}?mode=ro", uri=True)
    try:
        lost = _declared_lost(con)
        present = {s for (s,) in con.execute("SELECT seq FROM audit_records NOT INDEXED")}
        undeclared = [s for s in check.seqs if s not in lost]
        if undeclared:
            check.problems.append(
                f"{len(undeclared)} seq(s) were never declared lost on this trail, "
                f"first {undeclared[0]}")
        clash = [s for s in check.seqs if s in present]
        if clash:
            check.problems.append(f"{len(clash)} seq(s) are already on the trail, first {clash[0]}")
        if rows:
            prior = con.execute(
                "SELECT record_hash FROM audit_records NOT INDEXED WHERE seq = ?",
                (rows[0]["seq"] - 1,),
            ).fetchone()
            if prior is not None:
                if prior[0] == rows[0]["previous_hash"]:
                    check.anchored_to_seq = rows[0]["seq"] - 1
                else:
                    check.problems.append(
                        f"seq {rows[0]['seq']} does not name the trail's seq {rows[0]['seq'] - 1}")
        done = [
            json.loads(d).get("bundle_sha256") for (d,) in con.execute(
                "SELECT data FROM audit_records NOT INDEXED WHERE event_type = 'repair_recovered'")
        ]
        if check.bundle_sha256 in done:
            check.problems.append("this bundle is already declared recovered on the trail")
    finally:
        con.close()
    return check


def declare_recovery(check: RecoveryCheck, db: str | Path, source: str = "") -> Path:
    """Store the bundle beside the trail and append its REPAIR_RECOVERED record."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.audit.trail import AuditTrail

    if not check.ok:
        raise ValueError("recovery check did not pass: " + "; ".join(check.problems))
    db_path = Path(db).expanduser().resolve()
    keep_dir = db_path.parent / "recovered"
    keep_dir.mkdir(parents=True, exist_ok=True)
    kept = keep_dir / f"{check.bundle_sha256}.jsonl"
    if not kept.exists():
        data = Path(check.bundle).read_bytes()
        kept.write_bytes(data)
    backend = SQLiteAuditBackend(db_path)
    try:
        AuditTrail(on_record=backend.write_record).record_repair_recovered({
            "method": "page_carve",
            "bundle_sha256": check.bundle_sha256,
            "bundle_path": str(kept.relative_to(db_path.parent)),
            "recovered_seqs": check.seqs,
            "links_verified": check.links_verified,
            "anchored_to_seq": check.anchored_to_seq,
            "source": source,
        })
    finally:
        backend.close()
    return kept
