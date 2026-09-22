# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An adapter must not discard the provider's own aggregate verdict.

Each cloud and OSS guardrail returns both per-category detail and a single
top-level "did I intervene" field. The adapters parsed the detail and dropped
the top-level field, so a policy or filter type the parser does not model
produced no category at all, ``aggregate_verdict`` saw an empty action set,
and the finding came back ``verdict="allow"``. That lands in the hash-chained
trail as "upstream guardrail: allow" while the provider actually blocked.

Nothing catches this by shape. The contract tests assert the response shapes
still agree, and they do; what was incomplete is the dispatch. So these tests
feed each parser a response carrying a policy key it does not model, with the
provider's top-level verdict set to blocked, and assert the finding does not
say allow.

The rebuff adapter already did this (its ``injectionDetected`` cross-check)
and is the shape the other three now follow.
"""

from __future__ import annotations

from vaara.integrations._content_safety_base import aggregate_verdict
from vaara.integrations.bedrock_guardrails import parse_apply_guardrail_response
from vaara.integrations.gcp_model_armor import parse_sanitize_response
from vaara.integrations.guardrails_ai import parse_validation_outcome


# --------------------------------------------------------------------------
# Bedrock: top-level "action"
# --------------------------------------------------------------------------

def test_bedrock_unmodelled_policy_with_intervention_is_not_allow() -> None:
    """AWS blocked on a policy type the parser does not know."""
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{"someFuturePolicy": {"matches": [{"type": "WHATEVER"}]}}],
    }
    finding = parse_apply_guardrail_response(response, scanned_role="prompt")
    assert finding.verdict != "allow"
    assert finding.verdict == "block"
    assert any(c.provider_category == "guardrail_intervened" for c in finding.categories)


def test_bedrock_clean_response_stays_allow() -> None:
    """No intervention must not be turned into a block."""
    response = {"action": "NONE", "assessments": []}
    finding = parse_apply_guardrail_response(response, scanned_role="prompt")
    assert finding.verdict == "allow"
    assert finding.categories == ()


def test_bedrock_does_not_double_report_a_modelled_block() -> None:
    """When a real assessment already blocked, do not add a synthetic one."""
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{
            "wordPolicy": {
                "customWords": [{"match": "forbidden", "action": "BLOCKED"}],
            },
        }],
    }
    finding = parse_apply_guardrail_response(response, scanned_role="prompt")
    assert finding.verdict == "block"
    synthetic = [c for c in finding.categories
                 if c.provider_category == "guardrail_intervened"]
    assert synthetic == [], "a modelled block already covered this"


def test_bedrock_missing_action_field_is_not_treated_as_intervention() -> None:
    """An absent field asserts nothing: not a block, and not a pass either."""
    finding = parse_apply_guardrail_response({"assessments": []}, scanned_role="prompt")
    assert finding.verdict == "unparsed"
    assert finding.categories == ()


# --------------------------------------------------------------------------
# GCP Model Armor: top-level filterMatchState
# --------------------------------------------------------------------------

def test_gcp_unmodelled_filter_with_match_is_not_allow() -> None:
    """Google matched on a filter the parser does not know."""
    response = {
        "sanitizationResult": {
            "filterMatchState": "MATCH_FOUND",
            "filterResults": {"someFutureFilter": {"matchState": "MATCH_FOUND"}},
        },
    }
    finding = parse_sanitize_response(response, scanned_role="prompt")
    assert finding.verdict != "allow"
    assert any(c.provider_category == "filter_match_state" for c in finding.categories)


def test_gcp_no_match_stays_allow() -> None:
    response = {
        "sanitizationResult": {
            "filterMatchState": "NO_MATCH_FOUND",
            "filterResults": {},
        },
    }
    finding = parse_sanitize_response(response, scanned_role="prompt")
    assert finding.verdict == "allow"


def test_gcp_snake_case_encoding_is_read_too() -> None:
    """The SDK encoding must behave like the REST one."""
    response = {
        "sanitization_result": {
            "filter_match_state": "MATCH_FOUND",
            "filter_results": {"some_future_filter": {"match_state": "MATCH_FOUND"}},
        },
    }
    finding = parse_sanitize_response(response, scanned_role="prompt")
    assert finding.verdict != "allow"


# --------------------------------------------------------------------------
# Guardrails AI: top-level validation_passed
# --------------------------------------------------------------------------

def test_guardrails_failed_validation_with_passing_summaries_is_not_allow() -> None:
    """A Guard fails for reasons no validator owns (reask exhaustion, parsing)."""
    outcome = {
        "validation_passed": False,
        "validation_summaries": [
            {"validator_name": "DetectPII", "validator_status": "pass"},
        ],
        "error": "reask limit reached",
    }
    finding = parse_validation_outcome(outcome)
    assert finding.verdict != "allow"
    assert any(c.provider_category == "validation_failed" for c in finding.categories)


def test_guardrails_absent_flag_with_passing_summaries_stays_allow() -> None:
    """Absent is not false. A missing field must not manufacture a failure."""
    outcome = {
        "validation_summaries": [
            {"validator_name": "DetectPII", "validator_status": "pass"},
        ],
    }
    finding = parse_validation_outcome(outcome)
    assert finding.verdict == "allow"


def test_guardrails_passing_validation_stays_allow() -> None:
    outcome = {
        "validation_passed": True,
        "validation_summaries": [
            {"validator_name": "DetectPII", "validator_status": "pass"},
        ],
    }
    finding = parse_validation_outcome(outcome)
    assert finding.verdict == "allow"


def test_guardrails_does_not_double_report_a_failing_validator() -> None:
    outcome = {
        "validation_passed": False,
        "validation_summaries": [
            {"validator_name": "DetectPII", "validator_status": "fail"},
        ],
    }
    finding = parse_validation_outcome(outcome)
    assert finding.verdict == "flag"
    synthetic = [c for c in finding.categories
                 if c.provider_category == "validation_failed"]
    assert synthetic == []


# --------------------------------------------------------------------------
# The shared aggregator, which had no tests at all
# --------------------------------------------------------------------------

def test_aggregate_verdict_resolves_an_empty_category_list_to_allow() -> None:
    """Documents A1-1 rather than asserting it is correct.

    Every adapter routes through this, so it is the single place ambiguity is
    resolved, and it resolves it open. An empty list means two different
    things and the type cannot tell them apart: the provider found nothing, or
    the adapter understood nothing. Both come out "allow".

    The cross-checks above close the cases where a provider states a verdict
    the parser can compare against. A reply the parser could not read is
    handled one level up: each parser tells ``build_finding`` whether it
    understood the response, and an unread reply comes back ``"unparsed"``
    (tests below). Making empty mean block here would turn every clean scan
    into a block, so the aggregator itself stays as it is.
    """
    assert aggregate_verdict([]) == "allow"


# --------------------------------------------------------------------------
# The fourth verdict: a response the adapter could not read
# --------------------------------------------------------------------------

def test_unread_responses_come_back_unparsed_on_every_adapter() -> None:
    """Each adapter, fed a reply missing the field it reads, says unparsed."""
    from vaara.integrations.azure_content_safety import parse_responses
    from vaara.integrations.llm_guard import parse_scan_result
    from vaara.integrations.nemo_guardrails import parse_generation_response
    from vaara.integrations.rebuff import parse_detect_response

    assert parse_apply_guardrail_response({}).verdict == "unparsed"
    assert parse_sanitize_response({"sanitizationResult": {}}).verdict == "unparsed"
    assert parse_sanitize_response({"sanitizationResult": {
        "filterMatchState": "FILTER_MATCH_STATE_UNSPECIFIED"}}).verdict == "unparsed"
    assert parse_responses().verdict == "unparsed"
    assert parse_responses(shield={"unexpected": True}).verdict == "unparsed"
    assert parse_scan_result({}, {}, scanned_role="prompt").verdict == "unparsed"
    assert parse_generation_response({}).verdict == "unparsed"
    assert parse_detect_response({}).verdict == "unparsed"


def test_read_clean_responses_still_allow() -> None:
    """The same adapters, fed a clean reply they can read, still say allow."""
    from vaara.integrations.azure_content_safety import parse_responses
    from vaara.integrations.llm_guard import parse_scan_result
    from vaara.integrations.nemo_guardrails import parse_generation_response
    from vaara.integrations.rebuff import parse_detect_response

    assert parse_apply_guardrail_response(
        {"action": "NONE", "assessments": []}).verdict == "allow"
    assert parse_sanitize_response({"sanitizationResult": {
        "filterMatchState": "NO_MATCH_FOUND"}}).verdict == "allow"
    assert parse_responses(
        analyze_text={"categoriesAnalysis": []}).verdict == "allow"
    assert parse_scan_result({"Toxicity": True}, {"Toxicity": 0.0},
                             scanned_role="prompt").verdict == "allow"
    assert parse_generation_response(
        {"log": {"activated_rails": []}}).verdict == "allow"
    assert parse_detect_response({"injectionDetected": False}).verdict == "allow"


def test_a_read_block_is_not_downgraded_to_unparsed() -> None:
    """A category the adapter did read wins over a missing aggregate field."""
    response = {"assessments": [{"wordPolicy": {
        "customWords": [{"match": "forbidden", "action": "BLOCKED"}]}}]}
    assert parse_apply_guardrail_response(response).verdict == "block"
