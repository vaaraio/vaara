# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Point the whole suite at a throwaway home directory.

``InterceptionPipeline()`` with no trail resolves ``Path.home() / ".vaara" /
"trail" / "audit.db"``, and roughly a dozen other modules bind ``Path.home()``
into a module-level constant for approvals, policy, config, gate bundles and
the proxy trails. There was no conftest, so running ``pytest`` on a machine
that also runs Vaara appended test records to the operator's real audit trail.

For a product whose claim is an evidence chain nobody has edited, a test run
that writes into it is the wrong kind of surprise. It also made the suite
flaky: a live hook writing to the same SQLite file while tests opened it
produced ``disk I/O error`` and 58 failures that had nothing to do with the
code under test.

The redirect happens at import time rather than in a fixture. pytest imports
conftest before it imports any test module, and constants like
``approvals.APPROVALS_DIR`` are bound at *their* import, so a fixture would run
too late to move them.

Set ``VAARA_TEST_USE_REAL_HOME=1`` to opt out, for the rare case of debugging
against a real trail on purpose.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

if not os.environ.get("VAARA_TEST_USE_REAL_HOME"):
    _sandbox = Path(tempfile.mkdtemp(prefix="vaara-test-home-"))
    (_sandbox / ".vaara").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(_sandbox)
    # Path.home() reads HOME first but falls back to the password database,
    # and pathlib caches nothing, so USERPROFILE matters on Windows runners.
    os.environ["USERPROFILE"] = str(_sandbox)
    # Anything reading the trail path from the environment follows the same
    # sandbox rather than the operator's file.
    os.environ.setdefault("VAARA_DB", str(_sandbox / ".vaara" / "test-audit.db"))
