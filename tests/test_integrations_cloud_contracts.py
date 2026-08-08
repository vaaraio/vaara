"""Contract tests for the cloud-guardrail adapters.

Every other adapter test in this repo builds its fixture by hand, in the
shape Vaara *assumes* the provider returns. Those tests pass whether or
not the assumption is true, which is how three shipped adapters came to
disagree with their upstreams while CI stayed green:

* ``gcp_model_armor`` read camelCase keys off a proto-plus message that
  emits snake_case, checked ``hasattr(instance, "to_dict")`` when
  proto-plus puts ``to_dict`` on the metaclass, and compared enums to
  strings when the default conversion emits integers. A HIGH-confidence
  MATCH_FOUND produced verdict "allow".
* ``azure_content_safety`` called ``analyze_text(text=...)`` when the GA
  SDK takes a required positional ``options``, and silently no-opped on
  three endpoints that do not exist on the client at all.
* ``bedrock_guardrails`` gated on ``detected``, which is optional in the
  AWS model, rather than ``action``, which is required.

So these tests build fixtures from the provider's OWN types — proto
messages, the botocore service model, the real client signature — and
assert the adapter agrees with them. No credentials and no network: the
SDKs describe their own contracts offline.

Each class skips when its SDK is absent. The `adapter-contracts` CI job
installs all three so the skips do not hide a regression.
"""

from __future__ import annotations

import inspect

import pytest


class TestGcpModelArmorContract:
    """Fixtures built from real google-cloud-modelarmor proto messages."""

    @staticmethod
    def _sdk():
        return pytest.importorskip(
            "google.cloud.modelarmor_v1",
            reason="pip install google-cloud-modelarmor",
        )

    def _matched_response(self, ma):
        return ma.SanitizeUserPromptResponse(
            sanitization_result=ma.SanitizationResult(
                filter_match_state=ma.FilterMatchState.MATCH_FOUND,
                filter_results={
                    "rai": ma.FilterResult(
                        rai_filter_result=ma.RaiFilterResult(
                            match_state=ma.FilterMatchState.MATCH_FOUND,
                            rai_filter_type_results={
                                "dangerous": ma.RaiFilterResult.RaiFilterTypeResult(
                                    match_state=ma.FilterMatchState.MATCH_FOUND,
                                    confidence_level=ma.DetectionConfidenceLevel.HIGH,
                                )
                            },
                        )
                    ),
                },
            )
        )

    def test_to_dict_lives_on_the_metaclass_not_the_instance(self):
        """The assumption that broke the adapter, pinned as a fact."""
        ma = self._sdk()
        response = self._matched_response(ma)
        assert not hasattr(response, "to_dict")
        assert callable(getattr(type(response), "to_dict", None))

    def test_a_real_match_blocks(self):
        ma = self._sdk()
        from vaara.integrations.gcp_model_armor import GcpModelArmorAdapter

        response = self._matched_response(ma)

        class _Client:
            def sanitize_user_prompt(self, request=None):
                return response

            def sanitize_model_response(self, request=None):
                return response

        adapter = GcpModelArmorAdapter(
            _Client(), template="projects/p/locations/l/templates/t"
        )
        finding = adapter.scan_prompt("dangerous text")

        assert finding.verdict == "block"
        assert finding.raw, "the upstream response must survive into the audit trail"
        assert "responsible_ai.dangerous" in {
            c.provider_category for c in finding.categories
        }

    def test_a_real_clean_response_allows(self):
        ma = self._sdk()
        from vaara.integrations.gcp_model_armor import GcpModelArmorAdapter

        response = ma.SanitizeUserPromptResponse(
            sanitization_result=ma.SanitizationResult(
                filter_match_state=ma.FilterMatchState.NO_MATCH_FOUND,
                filter_results={},
            )
        )

        class _Client:
            def sanitize_user_prompt(self, request=None):
                return response

            def sanitize_model_response(self, request=None):
                return response

        adapter = GcpModelArmorAdapter(_Client(), template="t")
        assert adapter.scan_prompt("hello").verdict == "allow"

    def test_sdk_and_rest_encodings_agree(self):
        """snake_case + string enums (SDK) vs camelCase (REST)."""
        ma = self._sdk()
        from vaara.integrations.gcp_model_armor import (
            GcpModelArmorAdapter,
            parse_sanitize_response,
        )

        sdk_dict = GcpModelArmorAdapter._to_dict(self._matched_response(ma))
        rest_dict = {
            "sanitizationResult": {
                "filterResults": {
                    "rai": {
                        "raiFilterResult": {
                            "matchState": "MATCH_FOUND",
                            "raiFilterTypeResults": {
                                "dangerous": {
                                    "matchState": "MATCH_FOUND",
                                    "confidenceLevel": "HIGH",
                                }
                            },
                        }
                    }
                }
            }
        }
        from_sdk = parse_sanitize_response(sdk_dict)
        from_rest = parse_sanitize_response(rest_dict)

        assert from_sdk.verdict == from_rest.verdict == "block"
        assert [c.provider_category for c in from_sdk.categories] == [
            c.provider_category for c in from_rest.categories
        ]

    def test_integer_enums_still_match(self):
        """Library-default conversion emits ints; matching must survive it."""
        ma = self._sdk()
        from vaara.integrations.gcp_model_armor import parse_sanitize_response

        as_ints = type(self._matched_response(ma)).to_dict(self._matched_response(ma))
        assert (
            as_ints["sanitization_result"]["filter_results"]["rai"][
                "rai_filter_result"
            ]["match_state"]
            == ma.FilterMatchState.MATCH_FOUND.value
        )
        assert parse_sanitize_response(as_ints).verdict == "block"


class TestAzureContentSafetyContract:
    """Assertions against the real ContentSafetyClient surface."""

    @staticmethod
    def _client_cls():
        module = pytest.importorskip(
            "azure.ai.contentsafety",
            reason="pip install azure-ai-contentsafety",
        )
        return module.ContentSafetyClient

    def test_analyze_text_takes_one_required_positional_options(self):
        """`analyze_text(text=...)` raised TypeError against the real client."""
        signature = inspect.signature(self._client_cls().analyze_text)
        parameters = [p for name, p in signature.parameters.items() if name != "self"]
        first = parameters[0]

        assert first.name == "options"
        assert first.default is inspect.Parameter.empty
        assert first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert "text" not in signature.parameters

    def test_rest_only_endpoints_are_absent_from_the_client(self):
        """Pins why those endpoints must raise rather than no-op."""
        client_cls = self._client_cls()
        for name in (
            "shield_prompt",
            "detect_text_protected_material",
            "detect_groundedness",
        ):
            assert not hasattr(client_cls, name), (
                f"{name} now exists on ContentSafetyClient. The adapter can call "
                "it directly instead of requiring an injected client."
            )

    def test_adapter_accepts_the_real_client_and_sends_a_mapping(self):
        from vaara.integrations.azure_content_safety import AzureContentSafetyAdapter

        sent = {}

        class _RealisticClient:
            """Mirrors the GA signature: one positional options mapping."""

            def analyze_text(self, options):
                sent["options"] = options
                return {"categoriesAnalysis": [{"category": "Hate", "severity": 6}]}

        finding = AzureContentSafetyAdapter(_RealisticClient()).scan_prompt("bad text")

        assert sent["options"] == {"text": "bad text"}
        assert finding.verdict == "block"

    def test_requesting_an_absent_endpoint_raises(self):
        from vaara.integrations.azure_content_safety import AzureContentSafetyAdapter

        class _RealisticClient:
            def analyze_text(self, options):
                return {"categoriesAnalysis": []}

        adapter = AzureContentSafetyAdapter(_RealisticClient())
        with pytest.raises(RuntimeError, match="shield_prompt"):
            adapter.scan_prompt("text", include={"shield"})


class TestRebuffContract:
    """Rebuff ships two response types that disagree on field names."""

    @staticmethod
    def _types():
        pytest.importorskip("rebuff", reason="pip install rebuff")
        from rebuff.rebuff import DetectApiSuccessResponse
        from rebuff.sdk import RebuffDetectionResponse

        return DetectApiSuccessResponse, RebuffDetectionResponse

    def test_the_two_response_shapes_really_do_disagree(self):
        """Pins the drift that made the self-hosted path fail open."""
        api, sdk = self._types()

        def fields(cls):
            return set(getattr(cls, "model_fields", None) or cls.__fields__)

        assert "modelScore" in fields(api)
        # Not model_score — the self-hosted SDK names it after the vendor.
        assert "openai_score" in fields(sdk)
        assert "model_score" not in fields(sdk)
        assert fields(api).isdisjoint(fields(sdk))

    def _hit(self, cls, camel):
        if camel:
            return cls(
                heuristicScore=0.95, modelScore=0.99, vectorScore={"topScore": 0.97},
                runHeuristicCheck=True, runVectorCheck=True, runLanguageModelCheck=True,
                maxHeuristicScore=0.75, maxModelScore=0.9, maxVectorScore=0.9,
                injectionDetected=True,
            )
        return cls(
            heuristic_score=0.95, openai_score=0.99, vector_score=0.97,
            run_heuristic_check=True, run_vector_check=True,
            run_language_model_check=True, max_heuristic_score=0.75,
            max_model_score=0.9, max_vector_score=0.9, injection_detected=True,
        )

    def test_both_response_shapes_block_the_same_injection(self):
        api, sdk = self._types()
        from vaara.integrations.rebuff import parse_detect_response

        assert parse_detect_response(self._hit(api, camel=True)).verdict == "block"
        assert parse_detect_response(self._hit(sdk, camel=False)).verdict == "block"

    def test_self_hosted_model_layer_is_read_from_openai_score(self):
        _, sdk = self._types()
        from vaara.integrations.rebuff import parse_detect_response

        response = sdk(
            heuristic_score=0.0, openai_score=0.99, vector_score=0.0,
            run_heuristic_check=True, run_vector_check=True,
            run_language_model_check=True, max_heuristic_score=0.75,
            max_model_score=0.9, max_vector_score=0.9, injection_detected=True,
        )
        triggered = parse_detect_response(response).triggered_categories()
        assert [c.provider_category for c in triggered] == ["model_injection"]

    def test_clean_responses_still_allow(self):
        api, sdk = self._types()
        from vaara.integrations.rebuff import parse_detect_response

        clean_api = api(
            heuristicScore=0.0, modelScore=0.0, vectorScore={"topScore": 0.0},
            runHeuristicCheck=True, runVectorCheck=True, runLanguageModelCheck=True,
            maxHeuristicScore=0.75, maxModelScore=0.9, maxVectorScore=0.9,
            injectionDetected=False,
        )
        clean_sdk = sdk(
            heuristic_score=0.0, openai_score=0.0, vector_score=0.0,
            run_heuristic_check=True, run_vector_check=True,
            run_language_model_check=True, max_heuristic_score=0.75,
            max_model_score=0.9, max_vector_score=0.9, injection_detected=False,
        )
        assert parse_detect_response(clean_api).verdict == "allow"
        assert parse_detect_response(clean_sdk).verdict == "allow"

    def test_detect_injection_signature_is_what_the_adapter_calls(self):
        pytest.importorskip("rebuff")
        from rebuff import Rebuff, RebuffSdk

        for cls in (Rebuff, RebuffSdk):
            parameters = inspect.signature(cls.detect_injection).parameters
            assert "user_input" in parameters
            assert hasattr(cls, "is_canary_word_leaked") or cls is RebuffSdk


class TestBedrockGuardrailsContract:
    """Assertions against the botocore service model for ApplyGuardrail."""

    @staticmethod
    def _operation():
        pytest.importorskip("botocore", reason="pip install boto3")
        import botocore.session

        service = botocore.session.get_session().get_service_model("bedrock-runtime")
        return service.operation_model("ApplyGuardrail")

    def test_request_shape_matches_what_the_adapter_sends(self):
        operation = self._operation()
        required = set(operation.input_shape.required_members)

        assert required == {
            "guardrailIdentifier",
            "guardrailVersion",
            "source",
            "content",
        }

        content_item = operation.input_shape.members["content"].member
        assert "text" in content_item.members
        assert "text" in content_item.members["text"].required_members

    def test_every_response_field_the_adapter_reads_exists(self):
        operation = self._operation()
        assessment = operation.output_shape.members["assessments"].member

        expected = {
            "topicPolicy": ("topics", {"name", "type", "action", "detected"}),
            "contentPolicy": (
                "filters",
                {"type", "confidence", "filterStrength", "action", "detected"},
            ),
            "contextualGroundingPolicy": (
                "filters",
                {"type", "threshold", "score", "action", "detected"},
            ),
        }
        for policy, (collection, fields) in expected.items():
            members = set(assessment.members[policy].members[collection].member.members)
            assert fields <= members, f"{policy}.{collection} lost {fields - members}"

        word = assessment.members["wordPolicy"]
        for collection in ("customWords", "managedWordLists"):
            assert {"match", "action"} <= set(word.members[collection].member.members)

        sensitive = assessment.members["sensitiveInformationPolicy"]
        for collection in ("piiEntities", "regexes"):
            assert "action" in sensitive.members[collection].member.members

    def test_detected_is_optional_while_action_is_required(self):
        """The reason the adapter must not gate solely on `detected`."""
        operation = self._operation()
        assessment = operation.output_shape.members["assessments"].member

        for policy, collection in (
            ("topicPolicy", "topics"),
            ("contentPolicy", "filters"),
            ("contextualGroundingPolicy", "filters"),
        ):
            shape = assessment.members[policy].members[collection].member
            assert "detected" not in shape.required_members
            assert "action" in shape.required_members

    def test_an_enforcing_action_without_detected_still_blocks(self):
        self._operation()  # skip unless botocore is present
        from vaara.integrations.bedrock_guardrails import (
            parse_apply_guardrail_response,
        )

        response = {
            "assessments": [
                {
                    "topicPolicy": {
                        "topics": [
                            {"name": "legal", "type": "DENY", "action": "BLOCKED"}
                        ]
                    }
                }
            ]
        }
        assert parse_apply_guardrail_response(response).verdict == "block"

    def test_explicit_detected_false_is_still_honoured(self):
        self._operation()
        from vaara.integrations.bedrock_guardrails import (
            parse_apply_guardrail_response,
        )

        response = {
            "assessments": [
                {
                    "topicPolicy": {
                        "topics": [
                            {
                                "name": "legal",
                                "type": "DENY",
                                "action": "NONE",
                                "detected": False,
                            }
                        ]
                    }
                }
            ]
        }
        assert parse_apply_guardrail_response(response).verdict == "allow"
