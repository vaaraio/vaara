# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Caller-supplied names cannot forge log lines (CodeQL 131-134, 478)."""

import logging

from vaara._sanitize import log_safe
from vaara.pipeline import InterceptionPipeline


def test_line_breaks_and_controls_become_question_marks():
    assert log_safe("agent\r\nINFO forged") == "agent??INFO forged"
    assert log_safe("a\x1b[31mb") == "a?[31mb"


def test_length_is_capped_and_non_strings_render():
    assert len(log_safe("x" * 1000)) == 200
    assert log_safe(42) == "42"


def test_pipeline_log_keeps_one_line(caplog):
    pipeline = InterceptionPipeline()
    with caplog.at_level(logging.INFO, logger="vaara.pipeline"):
        pipeline.intercept(agent_id="bot\nINFO vaara.pipeline: forged", tool_name="read_file\r\nx",
                           parameters={})
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Intercepted")]
    assert lines and all("\n" not in m and "\r" not in m for m in lines)
