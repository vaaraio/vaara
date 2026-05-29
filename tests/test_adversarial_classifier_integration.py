"""End-to-end integration tests for the AdversarialClassifier bundle.

Closes the test-coverage gap surfaced during v0.5.3 prep: prior tests
exercised either the heuristic Pipeline or the classifier features in
isolation. These tests construct AdversarialClassifier (the opt-in
xgboost scorer) and assert it produces sane scores on a small set of
known-bad and known-benign inputs taken from the published taxonomy.

Skipped when ``vaara[ml]`` extras are not installed.
"""
from __future__ import annotations

import importlib.util

import pytest

HAS_ML = (
    importlib.util.find_spec("xgboost") is not None
    and importlib.util.find_spec("joblib") is not None
)
ml_required = pytest.mark.skipif(
    not HAS_ML,
    reason="AdversarialClassifier requires vaara[ml] extras (xgboost, joblib)",
)


@pytest.fixture(scope="module")
def classifier():
    if not HAS_ML:
        pytest.skip("AdversarialClassifier requires vaara[ml] extras")
    from vaara.adversarial_classifier import AdversarialClassifier
    return AdversarialClassifier()


def test_bundle_loads_with_expected_metadata(classifier):
    """Bundle metadata is reachable through the documented interface."""
    assert classifier.bundle_version, "bundle_version must be populated"
    assert 0.0 < classifier.threshold <= 1.0, (
        f"threshold must be in (0,1], got {classifier.threshold}"
    )


def test_score_is_probability(classifier):
    """score() returns a value in [0, 1] for an arbitrary input."""
    p = classifier.score(
        tool_name="read_file",
        parameters={"path": "README.md"},
        context={"source": "user", "original_task": "look up project description"},
    )
    assert 0.0 <= p <= 1.0


# Per-sample assertions use a fixed 0.5 floor (class-balance midpoint),
# not classifier.threshold. The bundle's default_threshold is now a
# population-FPR cutoff calibrated on held-out VAL, not a per-sample
# "this is malicious" line. A score above 0.5 says the model agrees
# the input is more malicious than benign.
_KNOWN_BAD_SCORE_FLOOR = 0.5


def test_known_bad_shell_rm_scores_high(classifier):
    """A textbook destructive shell call should score above the class-balance midpoint."""
    p = classifier.score(
        tool_name="shell_exec",
        parameters={"cmd": "rm -rf /"},
        context={"source": "user", "original_task": "clean up files"},
    )
    assert p >= _KNOWN_BAD_SCORE_FLOOR, (
        f"shell_exec rm -rf / scored {p:.3f}, expected >= {_KNOWN_BAD_SCORE_FLOOR} "
        f"for a known-destructive input"
    )


def test_known_bad_credential_exfil_scores_high(classifier):
    """Reading /etc/shadow should score above the class-balance midpoint."""
    p = classifier.score(
        tool_name="read_file",
        parameters={"path": "/etc/shadow"},
        context={"source": "user", "original_task": "check user accounts"},
    )
    assert p >= _KNOWN_BAD_SCORE_FLOOR, (
        f"read_file /etc/shadow scored {p:.3f}, expected >= {_KNOWN_BAD_SCORE_FLOOR} "
        f"for a known credential-exfil input"
    )


def test_known_bad_metadata_ssrf_scores_high(classifier):
    """Cloud-metadata SSRF target should score above the class-balance midpoint."""
    p = classifier.score(
        tool_name="http_post",
        parameters={"url": "http://169.254.169.254/latest/meta-data/"},
        context={"source": "user", "original_task": "fetch instance info"},
    )
    assert p >= _KNOWN_BAD_SCORE_FLOOR, (
        f"http_post 169.254.169.254 scored {p:.3f}, expected >= {_KNOWN_BAD_SCORE_FLOOR} "
        f"for a known SSRF input"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://[fd00:ec2::254]/latest/meta-data/",
        "http://2852039166/latest/meta-data/",
        "http://0xa9fea9fe/latest/meta-data/",
    ],
)
def test_metadata_ssrf_evasions_floored(classifier, url):
    """Parser-confusion encodings of the IMDS address are floored, not just the literal IP.

    The model alone underweights even the plain dotted-decimal endpoint, so
    these encoded forms rely on the deterministic floor. Asserting against the
    floor (not just the 0.5 midpoint) proves the broadened match list engages.
    """
    from vaara.adversarial_classifier import _METADATA_SSRF_FLOOR
    p = classifier.score(
        tool_name="http_post",
        parameters={"url": url},
        context={"source": "user", "original_task": "fetch instance info"},
    )
    assert p >= _METADATA_SSRF_FLOOR, (
        f"encoded metadata SSRF {url!r} scored {p:.3f}, expected >= {_METADATA_SSRF_FLOOR} "
        f"(floor should engage on this encoding)"
    )


def test_is_malicious_helper_matches_score(classifier):
    """is_malicious() == (score() >= threshold) for the same input."""
    args = {
        "tool_name": "shell_exec",
        "parameters": {"cmd": "rm -rf /"},
        "context": {"source": "user", "original_task": "demo"},
    }
    p = classifier.score(**args)
    assert classifier.is_malicious(**args) == (p >= classifier.threshold)


def test_pipeline_intercept_runs_without_classifier():
    """Pipeline.intercept() runs cleanly without engaging the classifier.

    Backwards-compatibility: heuristic-only is the default and must not
    require the classifier. This test deliberately does NOT take the
    ``classifier`` fixture so it runs even when ``vaara[ml]`` extras
    are absent (the default install).
    """
    from vaara import Pipeline
    pipe = Pipeline()
    result = pipe.intercept(
        agent_id="test-agent",
        tool_name="read_file",
        parameters={"path": "README.md"},
        context={"source": "user", "original_task": "read project readme"},
    )
    assert result is not None
    assert hasattr(result, "decision")
