"""TPM binding: bind a TPM 2.0 quote + IMA log to a SEP-2828 record.

The vendor-neutral, commodity-hardware twin of
:mod:`vaara.attestation._enforcement`. Where enforcement binds an AMD SEV-SNP
report, this binds an ordinary TPM 2.0 quote and the kernel's IMA
runtime-measurement log to a specific signed execution record. It is the verify
side; the quote, its signature, the quoted PCR values, and the IMA log arrive
pre-captured (the capture script under ``scripts/tpm/`` requests them from the
chip at runtime).

The chain a passing check establishes
-------------------------------------

One ``bound`` verdict ties four links together, each checkable offline by someone
who trusts neither Vaara nor the operator:

1. The AK signature verifies over the exact quote bytes, so the quote was
   produced by whoever holds that attestation key.
2. ``extraData`` in the quote equals ``SHA-256(jcs(record))``, so the quote was
   taken *for this record* and not lifted from elsewhere.
3. The supplied PCR values recompute the ``pcrDigest`` the AK signed, so those
   values are the ones the TPM quoted, not a convenient substitute.
4. The supplied IMA log replays to the quoted PCR 10 value, so the list of
   measured executables and files is exactly the one folded into the quote.

What it does NOT prove (the honest boundary)
--------------------------------------------

1. That the AK belongs to a genuine TPM. The AK public key is trusted as
   supplied; its endorsement-key certificate chain to a TPM vendor root, and the
   credential-activation that binds AK to EK, are not validated. A self-generated
   key produces a byte-identical quote and passes the same check, so
   ``ak_chain_basis`` is always ``caller_supplied_unverified`` in v0. This is the
   exact analogue of the deferred AMD KDS chain on the SEV-SNP side.
2. That the measured software *decided anything*. IMA measures which files and
   executables were loaded, not decision semantics. The platform and the code
   were un-tampered; *what the agent decided and why* is what the signed record
   carries. ``decision_logic_basis`` is therefore always ``not_established``; the
   binding of record-meaning to platform-proof is the whole point.
3. That the IMA policy was complete. v0 does not check which IMA policy was
   loaded, so a narrow policy could measure little. ``ima_policy_basis`` is
   always ``not_established``.
4. Freshness against replay. ``extraData`` carries the record hash, not a
   verifier-issued challenge, so a captured quote can be re-presented against the
   same record. ``freshness_basis`` is always ``not_established`` in this
   Phase-0 flow; a continuous-attestation loop (Phase 1) is what closes it.

Stated plainly: until ``ak_chain_basis`` reaches an EK-rooted value, this verdict
has no component the submitter cannot forge, the same deliberate contrast the
enforcement module draws against the un-forgeable eIDAS handoff anchor. The word
``attested`` is reserved for the future tier that validates the AK to a TPM
vendor root.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_canonical import canonical_json
from vaara.attestation._tpm import (
    IMA_PCR,
    TPMAttestationError,
    TPMSAttest,
    hash_name,
    parse_tpms_attest,
    parse_tpmt_signature,
    pcr_digest_over,
    replay_ima_pcr,
)

TPM_BINDING_SCHEMA = "vaara.tpm-binding-attestation/v0"


def bind_record_to_extra_data(record: dict[str, Any]) -> bytes:
    """The 32-byte TPM quote ``extraData`` that binds a quote to a record.

    ``extraData = SHA-256(canonical_json(record))`` over the FULL on-disk record
    dict, *including* its top-level ``signature`` field. Same preimage discipline as
    :func:`~vaara.attestation._enforcement.bind_record_to_report_data` (hash the
    whole record, not the signed-block subset, to close the signature-malleability
    gap where a stripped or swapped signature would canonicalise the same).

    SHA-256, not SHA-512: a quote's ``qualifyingData`` is a ``TPM2B_DATA`` whose
    ceiling is ``sizeof(TPMU_HA)`` (the TPM's largest *implemented* hash), so a
    64-byte nonce is rejected ``TPM_RC_SIZE`` on any TPM without SHA-512 (most fTPMs
    cap at SHA-384 = 48 bytes; confirmed against an AMD fTPM 2026-06-12). 32 bytes
    fits every TPM 2.0, where SHA-256 is mandatory. SEV-SNP ``REPORT_DATA`` is a
    flat 64-byte field with no such constraint, so the enforcement side still uses
    SHA-512; only the TPM nonce changes.
    """
    return hashlib.sha256(canonical_json(record)).digest()


def bind_record_to_chain_extra_data(
    record: dict[str, Any], prev_digest: bytes, seq: int
) -> bytes:
    """The ``extraData`` for one link of a continuous-attestation chain.

    ``extraData = SHA-256(canonical_json(record) || prev_digest || seq_be64)``.
    Where :func:`bind_record_to_extra_data` binds a lone quote to a record, this
    additionally folds in the position in the chain (``seq``, a big-endian u64) and
    the predecessor link (``prev_digest``, the SHA-256 of the previous link's signed
    quote bytes; the genesis link uses 32 zero bytes). Because the AK signs the
    quote and the quote covers this ``extraData``, the linkage is tamper-evident:
    dropping, reordering, or splicing a link changes the ``prev_digest`` a later
    link committed to, so its binding no longer holds. 32 bytes, for the same
    ``TPM2B_DATA`` portability reason as the Phase-0 binding (a 64-byte nonce is
    rejected ``TPM_RC_SIZE`` on a TPM without SHA-512).
    """
    if seq < 0:
        raise ValueError("seq must be non-negative")
    return hashlib.sha256(
        canonical_json(record) + prev_digest + seq.to_bytes(8, "big")
    ).digest()


@dataclass(frozen=True)
class TPMBindingVerdict:
    """Verdict over a TPM 2.0 quote + IMA log bound to a SEP-2828 record.

    ``tier`` is one of ``unverified`` (a crypto or binding link failed), ``bound``
    (all four links hold against the supplied AK), or ``pcr_pinned`` (``bound`` and
    the quoted IMA PCR matches a caller-supplied vetted reference). ``attested`` is
    reserved for a future tier that validates the AK to a TPM vendor root and is
    never emitted in v0.

    The ``*_basis`` fields are the honesty record: ``ak_chain_basis`` (AK trusted
    without an EK chain), ``ima_policy_basis`` (IMA policy completeness unchecked),
    ``decision_logic_basis`` (IMA measures files, not decisions), and
    ``freshness_basis`` (no verifier challenge, so replay is not prevented). The
    boolean sub-results make the tier reconstructable. ``pcr_context`` surfaces raw
    quote fields without asserting anything about them. ``reason`` is non-normative.

    ``ok`` is the overall answer: in default mode every link holds and any supplied
    PCR pin matched. In ``strict`` mode it additionally requires the EK-rooted
    ``attested`` tier, which v0 cannot reach, so a strict pass is honestly
    unavailable until the AK chain is validated.
    """

    schema: str
    tier: str
    parsed: bool
    magic_ok: bool
    attest_type_ok: bool
    signature_algo_ok: bool
    signature_valid: bool
    bound: bool
    extra_data_expected: str
    extra_data_actual: Optional[str]
    pcr_digest_recomputed: bool
    pcr_digest_quoted: Optional[str]
    ima_pcr_index: int
    ima_pcr_quoted: Optional[str]
    ima_pcr_replayed: Optional[str]
    ima_replayed: bool
    ima_log_entries: Optional[int]
    expected_ima_pcr: Optional[str]
    pcr_pin_basis: str
    ak_chain_basis: str
    ima_policy_basis: str
    decision_logic_basis: str
    freshness_basis: str
    pcr_context: dict[str, Any]
    strict: bool
    ok: bool
    record: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tier": self.tier,
            "ok": self.ok,
            "strict": self.strict,
            "parsed": self.parsed,
            "magic_ok": self.magic_ok,
            "attest_type_ok": self.attest_type_ok,
            "signature_algo_ok": self.signature_algo_ok,
            "signature_valid": self.signature_valid,
            "bound": self.bound,
            "extra_data_expected": self.extra_data_expected,
            "extra_data_actual": self.extra_data_actual,
            "pcr_digest_recomputed": self.pcr_digest_recomputed,
            "pcr_digest_quoted": self.pcr_digest_quoted,
            "ima_pcr_index": self.ima_pcr_index,
            "ima_pcr_quoted": self.ima_pcr_quoted,
            "ima_pcr_replayed": self.ima_pcr_replayed,
            "ima_replayed": self.ima_replayed,
            "ima_log_entries": self.ima_log_entries,
            "expected_ima_pcr": self.expected_ima_pcr,
            "pcr_pin_basis": self.pcr_pin_basis,
            "ak_chain_basis": self.ak_chain_basis,
            "ima_policy_basis": self.ima_policy_basis,
            "decision_logic_basis": self.decision_logic_basis,
            "freshness_basis": self.freshness_basis,
            "pcr_context": self.pcr_context,
            "record": self.record,
            "reason": self.reason,
        }


def _normalize_hex(value: Optional[str]) -> Optional[bytes]:
    """Parse a hex string to bytes; None on absent or malformed input."""
    if not isinstance(value, str):
        return None
    try:
        return bytes.fromhex(value.strip())
    except ValueError:
        return None


def _verify_ak_signature(ak_pub_pem: bytes, attest_bytes: bytes, ts: Any) -> bool:
    """Verify an ECDSA ``TPMT_SIGNATURE`` over the quote bytes against the AK.

    Returns ``True`` / ``False`` for a genuine verify result. Raises
    :class:`TPMAttestationError` only for a bad *verifier* input (an unloadable AK
    PEM, a non-EC key, an unknown scheme hash); a submitter-controlled garbage
    signature is a ``False``, never a raise.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import (
            encode_dss_signature,
        )
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise TPMAttestationError(
            "cryptography not installed. Install with: "
            "pip install 'vaara[attestation]'"
        ) from exc

    try:
        ak = serialization.load_pem_public_key(ak_pub_pem)
    except (ValueError, TypeError) as exc:
        raise TPMAttestationError(f"failed to load AK public key: {exc}") from exc
    if not isinstance(ak, ec.EllipticCurvePublicKey):
        raise TPMAttestationError("AK public key is not an EC public key")

    hash_cls = {"sha256": hashes.SHA256, "sha384": hashes.SHA384,
                "sha512": hashes.SHA512}.get(hash_name(ts.hash_alg))
    if hash_cls is None:
        raise TPMAttestationError(
            f"unsupported ECDSA scheme hash 0x{ts.hash_alg:04x}"
        )
    der = encode_dss_signature(ts.r, ts.s)
    try:
        ak.verify(der, attest_bytes, ec.ECDSA(hash_cls()))
        return True
    except InvalidSignature:
        return False


def _ima_bank_alg(attest: TPMSAttest) -> Optional[int]:
    """The hash-algorithm id of the selection that quoted PCR 10, or None."""
    for selection in attest.pcr_selections:
        if IMA_PCR in selection.pcrs:
            return selection.hash_alg
    return None


def _pcr_context(attest: TPMSAttest) -> dict[str, Any]:
    """Raw quote fields surfaced for inspection. None are gated on in v0."""
    selected = []
    for selection in attest.pcr_selections:
        try:
            bank = hash_name(selection.hash_alg)
        except TPMAttestationError:
            bank = f"0x{selection.hash_alg:04x}"
        selected.append({"bank": bank, "pcrs": list(selection.pcrs)})
    return {
        "firmware_version": f"0x{attest.firmware_version:016x}",
        "reset_count": attest.reset_count,
        "restart_count": attest.restart_count,
        "safe": attest.safe,
        "selected_pcrs": selected,
    }


def _reason(
    *,
    parsed: bool,
    magic_ok: bool,
    attest_type_ok: bool,
    signature_algo_ok: bool,
    signature_valid: bool,
    bound: bool,
    pcr_digest_recomputed: bool,
    ima_replayed: bool,
    pcr_pin_basis: str,
    strict: bool,
    ok: bool,
) -> str:
    """A non-normative explanation that always carries the trust caveats."""
    parts: list[str] = []
    if not parsed:
        parts.append("the quote did not parse as a TPMS_ATTEST structure")
    elif not magic_ok:
        parts.append("the quote magic is not TPM_GENERATED_VALUE")
    elif not attest_type_ok:
        parts.append("the attest type is not TPM_ST_ATTEST_QUOTE")
    elif not signature_algo_ok:
        parts.append("the signature is not a parseable ECDSA TPMT_SIGNATURE")
    elif not signature_valid:
        parts.append("the quote signature did not verify against the supplied AK")
    elif not bound:
        parts.append(
            "extraData does not equal sha256(jcs(record)): the quote does not "
            "bind to this record"
        )
    elif not pcr_digest_recomputed:
        parts.append(
            "the supplied PCR values do not recompute the signed pcrDigest"
        )
    elif not ima_replayed:
        parts.append(
            "the IMA log does not replay to the quoted PCR 10 value"
        )
    else:
        parts.append(
            "the quote verifies against the supplied AK, extraData binds to this "
            "record, the PCR values recompute the signed digest, and the IMA log "
            "replays to the quoted PCR 10"
        )
        if pcr_pin_basis == "pinned":
            parts.append("the IMA PCR matches the pinned reference")
        elif pcr_pin_basis == "pin_mismatch":
            parts.append(
                "but the IMA PCR does NOT match the pinned reference "
                "(a different measured state ran)"
            )
        else:
            parts.append(
                "the IMA PCR is reported but not pinned; supply "
                "expected_ima_pcr from a vetted reference state to assert which "
                "software state ran"
            )
    parts.append(
        "the AK was trusted as supplied and not validated to a TPM vendor root "
        "(EK chain deferred), so a self-generated key passes the same check"
    )
    parts.append(
        "IMA measures which files and executables loaded, not decision "
        "semantics; the decision content is what the signed record carries"
    )
    parts.append(
        "extraData carries the record hash, not a verifier challenge, so a "
        "captured quote can be replayed against the same record"
    )
    if strict and not ok:
        parts.append(
            "strict mode requires an AK validated to a TPM vendor root (EK "
            "chain) and a pinned IMA PCR, which v0 cannot establish"
        )
    return "; ".join(parts) + "."


def verify_tpm_binding(
    record: dict[str, Any],
    attest_bytes: bytes,
    signature: bytes,
    ak_pub_pem: bytes,
    *,
    pcr_values: dict[int, bytes],
    ima_log: str,
    expected_ima_pcr: Optional[str] = None,
    expected_extra_data: Optional[bytes] = None,
    strict: bool = False,
) -> TPMBindingVerdict:
    """Verify a TPM 2.0 quote + IMA log binds to a SEP-2828 record. One verdict.

    ``record`` is the on-disk record dict; ``attest_bytes`` the binary
    ``TPMS_ATTEST`` quote; ``signature`` the binary ``TPMT_SIGNATURE``;
    ``ak_pub_pem`` the PEM AK public key (trusted as supplied, its EK chain not
    validated). ``pcr_values`` maps PCR index to the raw bank-digest bytes that
    were quoted; ``ima_log`` is the ascii IMA runtime-measurement log.
    ``expected_ima_pcr`` optionally pins the quoted PCR 10 (hex) against a vetted
    reference. ``expected_extra_data`` overrides the expected ``extraData``: by
    default the quote is expected to bind to ``SHA-256(jcs(record))``; the
    continuous-attestation chain passes the chain-extended binding from
    :func:`bind_record_to_chain_extra_data` so each link is checked against its own
    position and predecessor. ``strict`` requires the EK-rooted ``attested`` tier
    (unreachable in v0).

    A malformed quote, signature, or IMA log yields ``tier='unverified'`` with the
    failing link flagged, never a traceback. Raises :class:`ValueError` if
    ``record`` is not a JSON object or cannot be canonicalised. Propagates
    :class:`TPMAttestationError` only for a bad *verifier* input (an unloadable AK
    PEM, a wrong-type key).
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"record must be a JSON object, got {type(record).__name__}"
        )
    if expected_extra_data is not None:
        expected_extra = expected_extra_data
    else:
        try:
            expected_extra = bind_record_to_extra_data(record)
        except Exception as exc:  # noqa: BLE001 - canonical_json raises on bad shapes
            raise ValueError(f"cannot canonicalise record: {exc}") from exc
    extra_data_expected = expected_extra.hex()

    parsed = magic_ok = attest_type_ok = False
    signature_algo_ok = signature_valid = bound = False
    pcr_digest_recomputed = ima_replayed = False
    extra_data_actual: Optional[str] = None
    pcr_digest_quoted: Optional[str] = None
    ima_pcr_quoted: Optional[str] = None
    ima_pcr_replayed: Optional[str] = None
    ima_log_entries: Optional[int] = None
    pcr_context: dict[str, Any] = {}

    attest: Optional[TPMSAttest] = None
    try:
        attest = parse_tpms_attest(attest_bytes)
        parsed = True
    except TPMAttestationError:
        attest = None

    ts = None
    if signature:
        try:
            ts = parse_tpmt_signature(signature)
            signature_algo_ok = True
        except TPMAttestationError:
            ts = None

    if attest is not None:
        magic_ok = attest.magic_ok
        attest_type_ok = attest.is_quote
        extra_data_actual = attest.extra_data.hex()
        pcr_digest_quoted = attest.pcr_digest.hex()
        pcr_context = _pcr_context(attest)
        # compare_digest is False (not an error) on a length mismatch.
        bound = hmac.compare_digest(attest.extra_data, expected_extra)

        if ts is not None:
            signature_valid = _verify_ak_signature(ak_pub_pem, attest.raw, ts)
            # pcrDigest is hashed with the signing scheme's hash.
            try:
                recomputed = pcr_digest_over(
                    attest.pcr_selections, pcr_values, ts.hash_alg
                )
                pcr_digest_recomputed = hmac.compare_digest(
                    recomputed, attest.pcr_digest
                )
            except TPMAttestationError:
                pcr_digest_recomputed = False

        bank_alg = _ima_bank_alg(attest)
        ima_value = pcr_values.get(IMA_PCR)
        if ima_value is not None:
            ima_pcr_quoted = ima_value.hex()
        if bank_alg is not None and ima_value is not None:
            try:
                replayed = replay_ima_pcr(ima_log, bank_alg)
                ima_pcr_replayed = replayed.hex()
                ima_log_entries = len(
                    [ln for ln in ima_log.splitlines() if ln.strip()]
                )
                ima_replayed = hmac.compare_digest(replayed, ima_value)
            except TPMAttestationError:
                ima_replayed = False

    # IMA PCR pin (independent of the binding result).
    if expected_ima_pcr is None:
        pcr_pin_basis = "unpinned"
    else:
        expected_bytes = _normalize_hex(expected_ima_pcr)
        ima_value = pcr_values.get(IMA_PCR)
        if (
            ima_value is not None
            and expected_bytes is not None
            and hmac.compare_digest(ima_value, expected_bytes)
        ):
            pcr_pin_basis = "pinned"
        else:
            pcr_pin_basis = "pin_mismatch"

    # v0 never validates the AK chain, IMA policy, decision logic, or freshness.
    ak_chain_basis = "caller_supplied_unverified"
    ima_policy_basis = "not_established"
    decision_logic_basis = "not_established"
    freshness_basis = "not_established"

    crypto_ok = bool(
        parsed
        and magic_ok
        and attest_type_ok
        and signature_algo_ok
        and signature_valid
        and bound
        and pcr_digest_recomputed
        and ima_replayed
    )

    if not crypto_ok:
        tier = "unverified"
    elif pcr_pin_basis == "pinned":
        tier = "pcr_pinned"
    else:
        tier = "bound"

    if strict:
        ok = bool(
            crypto_ok
            and pcr_pin_basis == "pinned"
            and ak_chain_basis == "ek_chain_verified"  # never set in v0
        )
    else:
        ok = bool(crypto_ok and pcr_pin_basis != "pin_mismatch")

    reason = _reason(
        parsed=parsed,
        magic_ok=magic_ok,
        attest_type_ok=attest_type_ok,
        signature_algo_ok=signature_algo_ok,
        signature_valid=signature_valid,
        bound=bound,
        pcr_digest_recomputed=pcr_digest_recomputed,
        ima_replayed=ima_replayed,
        pcr_pin_basis=pcr_pin_basis,
        strict=strict,
        ok=ok,
    )

    return TPMBindingVerdict(
        schema=TPM_BINDING_SCHEMA,
        tier=tier,
        parsed=parsed,
        magic_ok=magic_ok,
        attest_type_ok=attest_type_ok,
        signature_algo_ok=signature_algo_ok,
        signature_valid=signature_valid,
        bound=bound,
        extra_data_expected=extra_data_expected,
        extra_data_actual=extra_data_actual,
        pcr_digest_recomputed=pcr_digest_recomputed,
        pcr_digest_quoted=pcr_digest_quoted,
        ima_pcr_index=IMA_PCR,
        ima_pcr_quoted=ima_pcr_quoted,
        ima_pcr_replayed=ima_pcr_replayed,
        ima_replayed=ima_replayed,
        ima_log_entries=ima_log_entries,
        expected_ima_pcr=expected_ima_pcr,
        pcr_pin_basis=pcr_pin_basis,
        ak_chain_basis=ak_chain_basis,
        ima_policy_basis=ima_policy_basis,
        decision_logic_basis=decision_logic_basis,
        freshness_basis=freshness_basis,
        pcr_context=pcr_context,
        strict=strict,
        ok=ok,
        record=record,
        reason=reason,
    )


__all__ = [
    "TPM_BINDING_SCHEMA",
    "TPMBindingVerdict",
    "bind_record_to_chain_extra_data",
    "bind_record_to_extra_data",
    "verify_tpm_binding",
]
