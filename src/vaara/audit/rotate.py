# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retention rotation: export a signed archive, verify it, then purge.

``purge_older_than`` leaves a documented hash-chain seam at the retention
boundary, and the documented workflow around it (export a signed zip first,
archive it externally, then purge) was manual. ``rotate`` is that workflow
as one fail-closed operation: the purge never runs unless the archive was
written AND re-verified from its own bytes. A rotation that cannot prove
its archive is a rotation that did not happen.

Requires the [export] extra (cryptography) via ``export_signed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from vaara.audit.sqlite_backend import SQLiteAuditBackend

logger = logging.getLogger("vaara.audit.rotate")


@dataclass
class RotateResult:
    """Return value of :func:`rotate`.

    ``ok``: archive written, verified, and (unless dry_run) purge applied.
    ``exported_records``: records in the signed archive (whole trail).
    ``purged_records``: records purged, or that would be purged in dry_run.
    ``archive_path``: the written zip, when export succeeded.
    ``errors``: human-readable failure reasons (empty if ``ok``).
    """
    ok: bool
    exported_records: int = 0
    purged_records: int = 0
    archive_path: Union[Path, None] = None
    errors: list[str] = field(default_factory=list)


def rotate(
    db_path: Union[str, Path],
    out_path: Union[str, Path],
    signer_key: Union[str, Path, bytes],
    retention_days: int,
    *,
    tenant_id: str = "",
    dry_run: bool = False,
) -> RotateResult:
    """Export the whole trail as a signed zip, verify it, then purge.

    The archive always covers the full trail (export is whole-trail), so the
    external archive chain stays self-consistent forever while the live DB
    keeps only the retention window. Purging is skipped entirely when the
    export or its verification fails, and in ``dry_run`` mode the purge
    count is reported without deleting anything.
    """
    from vaara.audit.export import export_signed
    from vaara.audit.verify import verify_signed

    if retention_days <= 0:
        return RotateResult(ok=False, errors=[f"retention_days must be > 0, got {retention_days}"])

    out = Path(out_path)
    backend = SQLiteAuditBackend(db_path, tenant_id=tenant_id)
    try:
        exported = backend.count()

        try:
            export_result = export_signed(backend.load_trail(), out, signer_key=signer_key)
        except Exception as exc:
            return RotateResult(
                ok=False,
                exported_records=exported,
                errors=[f"export failed, purge skipped: {exc}"],
            )
        if not export_result.chain_intact:
            return RotateResult(
                ok=False,
                exported_records=exported,
                archive_path=out,
                errors=["exported chain did not verify at export time, purge skipped"],
            )

        verdict = verify_signed(out)
        if not verdict.ok:
            return RotateResult(
                ok=False,
                exported_records=exported,
                archive_path=out,
                errors=["archive failed re-verification, purge skipped"] + verdict.errors,
            )

        purged = backend.purge_older_than(retention_days * 86400, dry_run=dry_run)
        logger.info(
            "rotate: archived %d record(s) to %s, %s %d record(s) older than %d day(s)",
            exported, out, "would purge" if dry_run else "purged", purged, retention_days,
        )
        return RotateResult(
            ok=True,
            exported_records=exported,
            purged_records=purged,
            archive_path=out,
        )
    finally:
        backend.close()
