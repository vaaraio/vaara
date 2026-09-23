"""Contract test for the NeMo Guardrails adapter against the real library.

``test_integrations_nemo_guardrails.py`` builds responses by hand in the
shape the adapter assumes. This file runs a real ``LLMRails`` with an input
rail, over langchain-core's fake chat model, so the adapter meets the
``GenerationResponse`` NeMo actually returns. Offline: no model, no key.

It found one mismatch. Called with ``messages``, ``generate`` returns the
reply as a list of message dicts, and the adapter handed that list back as
the reply text.

Skips when nemoguardrails is absent. The ``guardrail-contracts`` CI job
installs it and fails on a skip.
"""

from __future__ import annotations

import pytest

nemo = pytest.importorskip("nemoguardrails", reason="pip install nemoguardrails")
fake_models = pytest.importorskip(
    "langchain_core.language_models.fake_chat_models",
    reason="pip install langchain-core",
)

from vaara.integrations.nemo_guardrails import (  # noqa: E402
    NemoGuardrailsAdapter,
    _with_rails_log,
    parse_generation_response,
)

_COLANG = '''
define bot refuse to respond
  "I can't help with that."

define flow no secrets
  $ok = execute check_no_secret
  if not $ok
    bot refuse to respond
    stop
'''

_YAML = '''
models: []
rails:
  input:
    flows:
      - no secrets
'''


@pytest.fixture
def adapter():
    config = nemo.RailsConfig.from_content(colang_content=_COLANG, yaml_content=_YAML)
    rails = nemo.LLMRails(
        config, llm=fake_models.FakeListChatModel(responses=["general answer"] * 5),
    )

    async def check_no_secret(context=None):
        return "sk-" not in (context or {}).get("user_message", "")

    rails.register_action(check_no_secret, "check_no_secret")
    return NemoGuardrailsAdapter(rails)


def test_a_rail_that_stops_the_turn_blocks(adapter):
    text, finding = adapter.generate([{"role": "user", "content": "key sk-123"}])
    assert text == "I can't help with that."
    assert finding.verdict == "block"
    stopped = [c for c in finding.categories if c.action == "BLOCKED"]
    assert [c.evidence["rail_name"] for c in stopped] == ["no secrets"]


def test_a_clean_turn_allows_and_returns_the_reply_as_text(adapter):
    text, finding = adapter.generate([{"role": "user", "content": "hello"}])
    assert text == "general answer"
    assert finding.verdict == "allow"
    assert finding.categories, "the rails log was not requested or not read"


def test_forced_log_option_is_a_valid_generation_options():
    from nemoguardrails.rails.llm.options import GenerationOptions

    assert GenerationOptions(**_with_rails_log(None)).log.activated_rails is True
    merged = _with_rails_log({"log": {"llm_calls": True}})
    options = GenerationOptions(**merged)
    assert options.log.activated_rails is True
    assert options.log.llm_calls is True
    typed = _with_rails_log(GenerationOptions())
    assert typed.log.activated_rails is True


def test_real_activated_rail_fields_parse():
    from nemoguardrails.rails.llm.options import (
        ActivatedRail,
        GenerationLog,
        GenerationResponse,
    )

    rail = ActivatedRail(
        type="output", name="self check output",
        decisions=["execute self_check_output", "refuse to respond", "stop"],
        executed_actions=[], stop=True,
    )
    response = GenerationResponse(
        response=[{"role": "assistant", "content": "no"}],
        log=GenerationLog(activated_rails=[rail]),
    )
    finding = parse_generation_response(response)
    assert finding.verdict == "block"
    assert finding.categories[0].provider_category == "output_rails.self_check"
