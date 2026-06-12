"""TPM evidence chain: a continuous-attestation loop bound to a SEP-2828 record.

Phase 1 of the hardware-governance binding, the continuous-attestation twin of the
single :mod:`vaara.attestation._tpm_binding` quote. Where Phase 0 binds one TPM 2.0
quote to a record at a point in time, this binds an *ordered sequence* of quotes
taken over a window, the Keylime-style loop: each tick re-quotes the TPM and folds
the kernel's grown IMA log into a rolling, hash-linked chain. A regulator replays
the whole chain offline and learns something a lone quote cannot tell them, that
the measured platform state held *continuously* across the window with no gap an
operator could slip a swap through.

What a ``continuous`` chain proves that one quote does not
---------------------------------------------------------

1. **Ordering, tamper-evident.** Each link's ``extraData`` is
   ``SHA-512(jcs(record) || prev_digest || seq)`` where ``prev_digest`` is the
   SHA-256 of the previous link's signed quote (genesis: 32 zero bytes). The AK
   signs the quote and the quote covers ``extraData``, so dropping, reordering, or
   splicing a link changes the ``prev_digest`` a later link committed to and its
   binding fails. The chain cannot be re-cut after the fact.
2. **Single uninterrupted boot.** Every link carries the TPM's own
   ``TPMS_CLOCK_INFO``. A ``continuous`` verdict requires the TPM clock strictly
   increasing and ``resetCount`` / ``restartCount`` constant across the window: a
   reboot (which increments ``resetCount`` and opens an unmeasured gap) breaks
   continuity rather than passing silently.
3. **Append-only measurement.** Each link's IMA log must extend the previous
   link's, entry for entry, and still replay to that link's quoted PCR 10. An
   operator who loads unmeasured software mid-window and unloads it cannot produce
   an IMA log that both grows monotonically and replays at every tick.

What it still does NOT prove (the honest boundary)
--------------------------------------------------

The per-link boundaries from Phase 0 all carry over unchanged: the AK is trusted
as supplied (``ak_chain_basis`` stays ``caller_supplied_unverified`` until the EK
chain is validated), IMA measures files not decision semantics
(``decision_logic_basis`` ``not_established``), and IMA-policy completeness is
unchecked. What the chain *does* move is freshness: a lone Phase-0 quote carries no
verifier challenge, so ``freshness_basis`` is ``not_established`` there. A
``continuous`` chain raises that to ``chain_continuity`` -- not a live
verifier-issued nonce (this stays offline-verifiable, so there is no interactive
challenger), but a real proof that the quotes form an unbroken, ordered, single-boot
sequence bounded by the TPM's own monotonic clock. Binding the chain head and tail
to an external trusted timestamp (the transparency-log anchor) is what would pin
that window to wall-clock; that is left to the caller.

Schema ``vaara.tpm-evidence-chain/v0``.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._tpm import (
    TPMAttestationError,
    TPMSAttest,
    parse_tpms_attest,
)
from vaara.attestation._tpm_binding import (
    TPMBindingVerdict,
    bind_record_to_chain_extra_data,
    verify_tpm_binding,
)

TPM_CHAIN_SCHEMA = "vaara.tpm-evidence-chain/v0"

# The genesis link has no predecessor; its prev_digest is a fixed 32 zero bytes.
GENESIS_PREV_DIGEST = bytes(32)


@dataclass(frozen=True)
class TPMChainLink:
    """One tick of a continuous-attestation chain, before verification.

    Raw pieces a single quote produces: the binary ``TPMS_ATTEST`` (``attest``),
    its ``TPMT_SIGNATURE`` (``signature``), the PEM AK public key (``ak_pub_pem``),
    the quoted PCR values keyed by index, and the ascii IMA log at that tick. The
    position in the chain (``seq``) is the link's index in the ordered list, so it
    is not carried here; the verifier derives it.
    """

    attest: bytes
    signature: bytes
    ak_pub_pem: bytes
    pcr_values: dict[int, bytes]
    ima_log: str


def link_digest(attest_bytes: bytes) -> bytes:
    """The ``prev_digest`` a successor link commits to: ``SHA-256`` of the quote.

    Committing to the signed attest bytes is enough to chain: a swapped signature
    over the same attest is rejected by the per-link signature check, and any
    change to the attest itself changes this digest, so the successor's binding
    fails.
    """
    return hashlib.sha256(attest_bytes).digest()


def _ima_lines(ima_log: str) -> list[str]:
    """Non-empty, stripped IMA log lines, for the append-only prefix check."""
    return [ln.strip() for ln in ima_log.splitlines() if ln.strip()]


def _is_prefix(earlier: list[str], later: list[str]) -> bool:
    """True if ``earlier`` is a leading prefix of ``later`` (append-only growth)."""
    return len(later) >= len(earlier) and later[: len(earlier)] == earlier


@dataclass(frozen=True)
class TPMChainVerdict:
    """Verdict over a continuous-attestation chain bound to one SEP-2828 record.

    ``tier`` is one of ``unverified`` (a link failed its own crypto/binding check,
    which includes a broken hash-linkage), ``linked`` (every link individually
    verifies and the chain is hash-linked in order, but the window is not shown
    continuous -- a single link, a reboot, a clock that did not advance, or an IMA
    log that did not grow append-only), or ``continuous`` (``linked`` plus a strict
    monotonic single-boot window with append-only IMA across at least two links).

    The boolean sub-results make the tier reconstructable: ``links_bound`` (all
    links pass Phase-0 verification including the chain ``extraData`` binding),
    ``clock_monotonic``, ``reboot_free`` (``resetCount``/``restartCount`` constant),
    ``ima_append_only``, ``ak_stable`` (one AK across the window). The ``*_basis``
    fields carry the Phase-0 honesty record forward; ``freshness_basis`` is the one
    the chain moves, to ``chain_continuity`` on a ``continuous`` verdict. ``window``
    surfaces the first/last TPM clock and IMA entry counts without asserting
    wall-clock. ``links`` holds each per-link verdict. ``reason`` is non-normative.

    ``ok`` is the overall answer: in default mode the chain is ``continuous`` and no
    link's IMA PCR pin mismatched. In ``strict`` mode it additionally requires the
    EK-rooted ``attested`` tier, which v0 cannot reach, so strict is honestly
    unavailable until the AK chain is validated.
    """

    schema: str
    tier: str
    ok: bool
    strict: bool
    n_links: int
    links_bound: bool
    clock_monotonic: bool
    reboot_free: bool
    ima_append_only: bool
    ak_stable: bool
    reset_count: Optional[int]
    restart_count: Optional[int]
    window: dict[str, Any]
    pcr_pin_basis: str
    ak_chain_basis: str
    ima_policy_basis: str
    decision_logic_basis: str
    freshness_basis: str
    links: list[dict[str, Any]]
    record: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tier": self.tier,
            "ok": self.ok,
            "strict": self.strict,
            "n_links": self.n_links,
            "links_bound": self.links_bound,
            "clock_monotonic": self.clock_monotonic,
            "reboot_free": self.reboot_free,
            "ima_append_only": self.ima_append_only,
            "ak_stable": self.ak_stable,
            "reset_count": self.reset_count,
            "restart_count": self.restart_count,
            "window": self.window,
            "pcr_pin_basis": self.pcr_pin_basis,
            "ak_chain_basis": self.ak_chain_basis,
            "ima_policy_basis": self.ima_policy_basis,
            "decision_logic_basis": self.decision_logic_basis,
            "freshness_basis": self.freshness_basis,
            "links": self.links,
            "record": self.record,
            "reason": self.reason,
        }


def _chain_reason(
    *,
    n_links: int,
    links_bound: bool,
    clock_monotonic: bool,
    reboot_free: bool,
    ima_append_only: bool,
    ak_stable: bool,
    any_pin_mismatch: bool,
    tier: str,
    strict: bool,
    ok: bool,
) -> str:
    """A non-normative explanation that always carries the trust caveats."""
    parts: list[str] = []
    if not links_bound:
        parts.append(
            "at least one link did not verify: its signature, record binding, PCR "
            "digest, or IMA replay failed, or the hash-linkage to its predecessor "
            "is broken (a dropped, reordered, or spliced link)"
        )
    elif n_links < 2:
        parts.append(
            "every link verifies and binds, but a single link cannot demonstrate a "
            "continuous window; supply at least two ticks"
        )
    elif not clock_monotonic:
        parts.append(
            "every link verifies, but the TPM clock did not strictly advance across "
            "the chain, so the ticks are not a forward-ordered window"
        )
    elif not reboot_free:
        parts.append(
            "every link verifies, but resetCount or restartCount changed mid-chain: "
            "the platform rebooted, opening an unmeasured gap that breaks continuity"
        )
    elif not ima_append_only:
        parts.append(
            "every link verifies, but the IMA log did not grow append-only across "
            "the chain, so an earlier measured state was rewritten"
        )
    elif not ak_stable:
        parts.append(
            "every link verifies, but the attestation key changed mid-chain, so the "
            "window is not a single attester's continuous record"
        )
    else:
        parts.append(
            "every link verifies and binds to this record in order, the TPM clock "
            "strictly advances on one uninterrupted boot, and the IMA log grows "
            "append-only across the window"
        )
    if any_pin_mismatch:
        parts.append(
            "but at least one link's IMA PCR does NOT match the pinned reference "
            "(a different measured state ran at that tick)"
        )
    parts.append(
        "the AK was trusted as supplied and not validated to a TPM vendor root "
        "(EK chain deferred), so a self-generated key passes the same check"
    )
    parts.append(
        "IMA measures which files and executables loaded, not decision semantics; "
        "the decision content is what the signed record carries"
    )
    if tier == "continuous":
        parts.append(
            "freshness rests on chain continuity (ordered, single-boot, "
            "monotonic-clock window), not a live verifier challenge; anchor the "
            "chain head and tail to a trusted timestamp to bound it to wall-clock"
        )
    else:
        parts.append(
            "extraData carries the record hash and chain position, not a verifier "
            "challenge, so freshness is not established"
        )
    if strict and not ok:
        parts.append(
            "strict mode requires an AK validated to a TPM vendor root (EK chain), "
            "which v0 cannot establish"
        )
    return "; ".join(parts) + "."


def verify_tpm_chain(
    record: dict[str, Any],
    links: "list[TPMChainLink]",
    *,
    expected_ima_pcr: Optional[str] = None,
    strict: bool = False,
) -> TPMChainVerdict:
    """Verify a continuous-attestation chain binds to a SEP-2828 record. One verdict.

    ``record`` is the on-disk record dict every link binds to; ``links`` is the
    ordered sequence of :class:`TPMChainLink` ticks (position in the list is the
    link's ``seq``). ``expected_ima_pcr`` optionally pins each link's quoted PCR 10
    against a vetted reference. ``strict`` requires the EK-rooted ``attested`` tier
    (unreachable in v0).

    Each link is verified with the full Phase-0 check
    (:func:`~vaara.attestation._tpm_binding.verify_tpm_binding`) against its
    chain-extended ``extraData``, then the cross-link invariants are applied:
    hash-linkage (already enforced through each link's binding), strictly
    increasing TPM clock, constant reset/restart counts, append-only IMA growth,
    and a stable AK. A link that fails to parse or verify yields
    ``tier='unverified'`` with the failing link flagged in ``links``, never a
    traceback. Raises :class:`ValueError` if ``record`` is not a JSON object or
    ``links`` is empty.
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"record must be a JSON object, got {type(record).__name__}"
        )
    if not links:
        raise ValueError("a TPM evidence chain must have at least one link")

    n_links = len(links)
    link_verdicts: list[TPMBindingVerdict] = []
    parsed_attests: list[Optional[TPMSAttest]] = []
    prev_digest = GENESIS_PREV_DIGEST
    for seq, link in enumerate(links):
        expected_extra = bind_record_to_chain_extra_data(record, prev_digest, seq)
        verdict = verify_tpm_binding(
            record,
            link.attest,
            link.signature,
            link.ak_pub_pem,
            pcr_values=link.pcr_values,
            ima_log=link.ima_log,
            expected_ima_pcr=expected_ima_pcr,
            expected_extra_data=expected_extra,
            strict=False,
        )
        link_verdicts.append(verdict)
        try:
            parsed_attests.append(parse_tpms_attest(link.attest))
        except TPMAttestationError:
            parsed_attests.append(None)
        # The successor commits to THIS link's quote, regardless of its verdict;
        # a tampered link then fails its successor's binding too.
        prev_digest = link_digest(link.attest)

    links_bound = all(v.tier != "unverified" for v in link_verdicts)
    any_pin_mismatch = any(
        v.pcr_pin_basis == "pin_mismatch" for v in link_verdicts
    )

    # Cross-link temporal and measurement invariants. Computed over every link;
    # an unparseable attest forces the relevant invariant False.
    all_parsed = all(a is not None for a in parsed_attests)
    clock_monotonic = all_parsed and n_links >= 2
    if all_parsed:
        for prev, cur in zip(parsed_attests, parsed_attests[1:]):
            assert prev is not None and cur is not None  # narrowed by all_parsed
            if cur.clock <= prev.clock:
                clock_monotonic = False
                break

    reset_count: Optional[int] = None
    restart_count: Optional[int] = None
    reboot_free = all_parsed
    if all_parsed:
        first = parsed_attests[0]
        assert first is not None
        reset_count = first.reset_count
        restart_count = first.restart_count
        for a in parsed_attests:
            assert a is not None
            if a.reset_count != reset_count or a.restart_count != restart_count:
                reboot_free = False
                break

    ima_append_only = True
    prev_lines: Optional[list[str]] = None
    for link in links:
        cur_lines = _ima_lines(link.ima_log)
        if prev_lines is not None and not _is_prefix(prev_lines, cur_lines):
            ima_append_only = False
            break
        prev_lines = cur_lines

    first_ak = links[0].ak_pub_pem
    ak_stable = all(link.ak_pub_pem == first_ak for link in links)

    window = {
        "clock_first": parsed_attests[0].clock
        if parsed_attests[0] is not None
        else None,
        "clock_last": parsed_attests[-1].clock
        if parsed_attests[-1] is not None
        else None,
        "ima_entries_first": len(_ima_lines(links[0].ima_log)),
        "ima_entries_last": len(_ima_lines(links[-1].ima_log)),
    }

    chain_ok = bool(
        links_bound
        and n_links >= 2
        and clock_monotonic
        and reboot_free
        and ima_append_only
        and ak_stable
    )

    if not links_bound:
        tier = "unverified"
    elif chain_ok:
        tier = "continuous"
    else:
        tier = "linked"

    # v0 never validates the AK chain, IMA policy, or decision logic.
    ak_chain_basis = "caller_supplied_unverified"
    ima_policy_basis = "not_established"
    decision_logic_basis = "not_established"
    freshness_basis = (
        "chain_continuity" if tier == "continuous" else "not_established"
    )

    if expected_ima_pcr is None:
        pcr_pin_basis = "unpinned"
    elif any_pin_mismatch:
        pcr_pin_basis = "pin_mismatch"
    else:
        pcr_pin_basis = "pinned"

    if strict:
        ok = False  # EK-rooted attested tier is unreachable in v0
    else:
        ok = bool(chain_ok and not any_pin_mismatch)

    reason = _chain_reason(
        n_links=n_links,
        links_bound=links_bound,
        clock_monotonic=clock_monotonic,
        reboot_free=reboot_free,
        ima_append_only=ima_append_only,
        ak_stable=ak_stable,
        any_pin_mismatch=any_pin_mismatch,
        tier=tier,
        strict=strict,
        ok=ok,
    )

    return TPMChainVerdict(
        schema=TPM_CHAIN_SCHEMA,
        tier=tier,
        ok=ok,
        strict=strict,
        n_links=n_links,
        links_bound=links_bound,
        clock_monotonic=clock_monotonic,
        reboot_free=reboot_free,
        ima_append_only=ima_append_only,
        ak_stable=ak_stable,
        reset_count=reset_count,
        restart_count=restart_count,
        window=window,
        pcr_pin_basis=pcr_pin_basis,
        ak_chain_basis=ak_chain_basis,
        ima_policy_basis=ima_policy_basis,
        decision_logic_basis=decision_logic_basis,
        freshness_basis=freshness_basis,
        links=[v.to_dict() for v in link_verdicts],
        record=record,
        reason=reason,
    )


__all__ = [
    "GENESIS_PREV_DIGEST",
    "TPM_CHAIN_SCHEMA",
    "TPMChainLink",
    "TPMChainVerdict",
    "link_digest",
    "verify_tpm_chain",
]
