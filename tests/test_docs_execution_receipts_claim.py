# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/execution-receipts.md says what the receipt is, and what became of SEP-2787.

The page once said the execution receipt "proves what happened". A receipt is
signed by whoever observed the outcome, so it records what that issuer saw: an
upstream that never answered is not a failed call, and the receipt does not
know the difference (see the ``outcome_recorded`` states on the MCP proxy). The
page also called the receipt the post-execution sibling of SEP-2787 without
saying that MCP pull request 2787 was closed unmerged on 2026-09-22.
"""
from __future__ import annotations

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "execution-receipts.md"


def _text() -> str:
    # The page wraps lines; compare on one line of single spaces.
    return re.sub(r"\s+", " ", DOC.read_text(encoding="utf-8"))


def test_the_receipt_is_not_said_to_prove_what_happened():
    text = _text()
    assert "proves what happened" not in text
    assert "the receipt proves" not in text


def test_the_receipt_is_said_to_record_what_the_issuer_observed():
    assert "records what the issuer observed" in _text()


def test_the_closed_sep_is_named_as_closed():
    text = _text()
    assert "SEP-2787" in text
    assert "pull request 2787" in text
    assert "closed without being merged on 2026-09-22" in text
