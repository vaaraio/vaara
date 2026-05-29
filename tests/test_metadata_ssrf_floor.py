"""Pattern-level coverage for the cloud-metadata SSRF floor.

The deterministic floor in ``AdversarialClassifier.score`` lifts a bare
http_post to a cloud instance-metadata endpoint above the calibrated
threshold regardless of what the learned model returns. These tests assert
the match list covers the well-known parser-confusion encodings of the AWS
IMDS address and does not fire on benign lookalikes.

These exercise only the module-level regex tuple, so they run without the
``vaara[ml]`` extras (the floor itself still needs the model to produce the
base score, exercised in test_adversarial_classifier_integration.py).
"""
from __future__ import annotations

import json

import pytest

from vaara.adversarial_classifier import _METADATA_SSRF_PATTERNS


def _matches(text: str) -> bool:
    return any(p.search(text) for p in _METADATA_SSRF_PATTERNS)


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.170.2/v2/credentials",
        # AWS IMDS over IPv6 link-local.
        "http://[fd00:ec2::254]/latest/meta-data/",
        "http://[FD00:EC2::254]/latest/meta-data/",
        # 169.254.169.254 as a dotless 32-bit decimal / hex integer.
        "http://2852039166/latest/meta-data/",
        "http://0xa9fea9fe/latest/meta-data/",
        "http://0xA9FEA9FE/latest/meta-data/",
    ],
)
def test_metadata_evasions_match(url):
    """Each known encoding of a metadata endpoint is matched by the floor."""
    blob = json.dumps({"tool_name": "http_post", "parameters": {"url": url}})
    assert _matches(blob), f"floor failed to match metadata SSRF encoding: {url}"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1/users",
        "https://github.com/vaaraio/vaara",
        # Magic decimal as a slice of a longer digit run must not match.
        "https://shop.example.com/order/12852039166",
        "https://metrics.example.com/ts/28520391660",
        # Off-by-one from the magic decimal.
        "http://2852039167/health",
        # Longer hex token sharing the prefix must not match.
        "http://0xa9fea9feee/blob",
        # Dotless digits that are not the IMDS integer.
        "http://169254169254/legacy",
    ],
)
def test_benign_lookalikes_do_not_match(url):
    """Benign hosts and numeric lookalikes are not floored by the pattern list."""
    blob = json.dumps({"tool_name": "http_get", "parameters": {"url": url}})
    assert not _matches(blob), f"floor false-positive on benign input: {url}"
