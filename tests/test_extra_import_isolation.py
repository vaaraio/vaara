# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An extra must not drag in a dependency it does not declare.

``vaara[scitt]`` declares cbor2 and cryptography. It does not declare
asn1crypto, which belongs to ``vaara[timeanchor]``. But
``vaara.audit.scitt_anchor`` imported ``_signed_payload_digest`` from
``vaara.audit.receipt_anchor``, and that module imports asn1crypto at module
level for its RFC 3161 TSA. So a clean ``pip install 'vaara[scitt]'`` produced
a SCITT module that raised ModuleNotFoundError on import.

Nothing caught it because tests/test_scitt_anchor.py guards on rfc8785 only,
so in an environment without either dependency the module skipped and the
breakage stayed invisible. These tests run the import in a subprocess with the
undeclared dependency blocked, which is the only way to assert the boundary
from inside an environment that happens to have everything installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("cbor2")
pytest.importorskip("rfc8785")


def _import_with_blocked(module: str, blocked: str) -> subprocess.CompletedProcess:
    """Import ``module`` in a fresh interpreter with ``blocked`` unimportable."""
    program = textwrap.dedent(
        f"""
        import sys

        class _Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name == {blocked!r} or name.startswith({blocked!r} + "."):
                    raise ImportError("blocked by test: " + name)
                return None

        sys.meta_path.insert(0, _Blocker())
        import {module}
        print("IMPORT_OK")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
    )


def test_scitt_anchor_imports_without_the_timeanchor_extra() -> None:
    """vaara[scitt] must import with asn1crypto absent."""
    result = _import_with_blocked("vaara.audit.scitt_anchor", "asn1crypto")
    assert "IMPORT_OK" in result.stdout, (
        "vaara.audit.scitt_anchor needs asn1crypto, which the 'scitt' extra "
        "does not declare.\n" + result.stderr
    )


def test_receipt_page_imports_without_the_timeanchor_extra() -> None:
    """receipt_page only needs the digest, not the RFC 3161 producer."""
    result = _import_with_blocked("vaara.audit.receipt_page", "asn1crypto")
    assert "IMPORT_OK" in result.stdout, (
        "vaara.audit.receipt_page needs asn1crypto at import time.\n" + result.stderr
    )


def test_signed_payload_digest_is_reachable_without_the_timeanchor_extra() -> None:
    """The digest helper is pure hashlib + rfc8785 and must not need asn1crypto."""
    result = _import_with_blocked("vaara.audit.timeanchor", "asn1crypto")
    assert "IMPORT_OK" in result.stdout, result.stderr


def test_receipt_anchor_still_re_exports_the_digest_helpers() -> None:
    """Moving them must not break the documented import path."""
    pytest.importorskip("asn1crypto")
    from vaara.audit.receipt_anchor import _signed_payload_digest, anchored_digest
    from vaara.audit.timeanchor import (
        _signed_payload_digest as moved_digest,
    )
    from vaara.audit.timeanchor import anchored_digest as moved_anchored

    assert _signed_payload_digest is moved_digest
    assert anchored_digest is moved_anchored
