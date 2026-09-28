"""A deployer can override a guardrail-to-article mapping without adapter code.

docs/COMPLIANCE.md says the mapping table is a published artefact a
deployer can read, dispute and override. The override goes through
``lookup``, which every adapter reads, so a changed row reaches the
findings the adapters produce.
"""

import json

import pytest

from vaara.integrations import _content_safety_articles as cs
from vaara.integrations._content_safety_articles import CategoryMapping


@pytest.fixture(autouse=True)
def _clean():
    cs.clear_overrides()
    yield
    cs.clear_overrides()


def test_override_replaces_the_published_row():
    published = cs.lookup("llm-guard", "Sentiment")
    assert published.ai_act_articles == ()

    cs.override_mapping(CategoryMapping(
        "llm-guard", "Sentiment", "sentiment", ("Art. 13",), (), "deployer: tone is disclosed",
    ))

    assert cs.lookup("llm-guard", "Sentiment").ai_act_articles == ("Art. 13",)
    assert cs.lookup("llm-guard", "sentiment").ai_act_articles == ("Art. 13",)


def test_override_can_map_a_category_the_table_does_not_list():
    assert cs.lookup("llm-guard", "Gibberish") is None
    cs.override_mapping(CategoryMapping("llm-guard", "Gibberish", "output_validation", ("Art. 15",), ()))
    assert cs.lookup("llm-guard", "Gibberish").vaara_category == "output_validation"


def test_removing_an_override_restores_the_published_row():
    cs.override_mapping(CategoryMapping("rebuff", "canary_leak", "pii", ("Art. 10",), ()))
    cs.remove_override("rebuff", "canary_leak")
    assert cs.lookup("rebuff", "canary_leak").vaara_category == "secrets_leak"


def test_the_published_table_is_unchanged_by_an_override():
    cs.override_mapping(CategoryMapping("rebuff", "canary_leak", "pii", ("Art. 10",), ()))
    [row] = [m for m in cs.all_mappings_for("rebuff") if m.provider_category == "canary_leak"]
    assert row.vaara_category == "secrets_leak"


def test_an_adapter_finding_carries_the_override():
    from vaara.integrations.llm_guard import parse_scan_result

    before = parse_scan_result(
        {"PromptInjection": False}, {"PromptInjection": 0.9}, scanned_role="user",
    )
    assert "Art. 9" not in before.ai_act_articles()

    cs.override_mapping(CategoryMapping(
        "llm-guard", "PromptInjection", "adversarial", ("Art. 15", "Art. 9"), ("LLM01",),
    ))
    after = parse_scan_result(
        {"PromptInjection": False}, {"PromptInjection": 0.9}, scanned_role="user",
    )
    assert "Art. 9" in after.ai_act_articles()
    assert "Art. 9" in after.to_audit_context()["upstream_guardrail"]["ai_act_articles"]


def test_load_overrides_from_a_file(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps([{
        "provider": "gcp-model-armor", "provider_category": "virus_scan",
        "vaara_category": "malicious_file", "ai_act_articles": ["Art. 15", "Art. 9"],
        "owasp_llm": ["LLM05"], "notes": "deployer",
    }]))
    assert cs.load_overrides(str(path)) == 1
    assert cs.lookup("gcp-model-armor", "virus_scan").ai_act_articles == ("Art. 15", "Art. 9")


def test_a_bad_row_applies_nothing(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps([
        {"provider": "rebuff", "provider_category": "canary_leak", "vaara_category": "pii"},
        {"provider": "rebuff", "provider_category": "model_injection", "ai_act_articles": "Art. 15"},
    ]))
    with pytest.raises(ValueError):
        cs.load_overrides(str(path))
    assert cs.lookup("rebuff", "canary_leak").vaara_category == "secrets_leak"
