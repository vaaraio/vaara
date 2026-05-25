"""Prompt-injection detection over Vaara's adversarial classifier.

Wraps the model that produced vaara-bench-v1's headline numbers
(``bench/vaara-bench-v1-results.json``) so the buyer-visible "prompt
injection blocking" feature is the same engine that ships the cross-
validated recall and FPR reported there.

The classifier was trained on tool-call shapes (``tool_name`` +
``parameters`` + ``context``). For free text we wrap the input as a
synthetic call so the same model surface applies; the heuristic
recipient field is ``ml.text`` so the bundle's tool-name vocabulary
does not bias the score toward any specific tool family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_DEFAULT_THRESHOLD = 0.9226  # v0.32 pick on VAL at target FPR 5%


@dataclass(frozen=True)
class InjectionResult:
    """Result of a single prompt-injection check.

    Attributes:
        detected: ``True`` iff the score meets or exceeds the threshold.
        score: Adversarial probability in [0, 1].
        threshold: Decision threshold applied.
        bundle_version: Classifier bundle the score came from.
        backend: Identifier of the scoring backend ("vaara_adversarial"
            for the shipped ML bundle, "heuristic" for the regex
            fallback when the ml extra is not installed).
    """

    detected: bool
    score: float
    threshold: float
    bundle_version: str
    backend: str

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "score": self.score,
            "threshold": self.threshold,
            "bundle_version": self.bundle_version,
            "backend": self.backend,
        }


# Conservative heuristic patterns for the no-ml fallback. Matched
# case-insensitively, OR-joined. Intentionally narrow so the heuristic
# does not become Vaara's de-facto detector — the ML path is preferred.
_HEURISTIC_INJECTION_PATTERNS = (
    r"ignore (?:all )?(?:the )?(?:previous|prior|above)(?: instructions)?",
    r"disregard (?:the )?(?:above|previous|prior)",
    r"system prompt",
    r"you are now",
    r"jailbreak",
    r"do anything now",
    r"DAN mode",
    r"developer mode",
    r"forget (?:the )?(?:above|previous|prior|all)",
    r"reveal (?:the )?(?:system )?prompt",
    r"\bsudo\s+(?:rm|cat|chmod|chown)\b",
    r"<\|.*?\|>",
)


def _heuristic_score(text: str) -> float:
    import re

    hits = 0
    for pat in _HEURISTIC_INJECTION_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits += 1
    if not hits:
        return 0.0
    # Diminishing returns: one hit lands at 0.65, two at 0.85, three+ at 0.95.
    return min(0.95, 0.40 + 0.25 * hits)


def detect_injection(
    text: str,
    *,
    threshold: Optional[float] = None,
    use_ml: bool = True,
) -> InjectionResult:
    """Score free text for prompt-injection likelihood.

    Args:
        text: The text to scan. Typical inputs are user-supplied prompt
            contents, retrieved-document chunks, or tool-output strings
            that flow back into an agent's context.
        threshold: Decision threshold in [0, 1]. Defaults to 0.9226
            (v0.32 VAL pick at target FPR 5%).
        use_ml: When True and the ml extra is installed, route through
            the AdversarialClassifier. When False or the extra is
            absent, fall back to the heuristic regex set.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    th = _DEFAULT_THRESHOLD if threshold is None else float(threshold)

    if use_ml:
        try:
            from vaara.adversarial_classifier import AdversarialClassifier

            clf = AdversarialClassifier()
            score = clf.score(
                tool_name="ml.text",
                parameters={"prompt": text},
                context={},
            )
            return InjectionResult(
                detected=score >= th,
                score=score,
                threshold=th,
                bundle_version=clf.bundle_version,
                backend="vaara_adversarial",
            )
        except (ImportError, FileNotFoundError):
            pass  # fall through to heuristic

    score = _heuristic_score(text)
    return InjectionResult(
        detected=score >= th,
        score=score,
        threshold=th,
        bundle_version="heuristic/v1",
        backend="heuristic",
    )
