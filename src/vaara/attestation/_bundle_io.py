"""On-disk format for an evidence bundle: load a JSON document into a bundle.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt`` and ``vaara.attestation``.

:func:`verify_evidence_bundle` takes an :class:`EvidenceBundle` of in-memory
objects. To verify evidence someone handed you, those objects first have to
be reconstructed from files. This module defines the one JSON shape that
carries a whole bundle and the loader that turns it into an
:class:`EvidenceBundle`, so a regulator can be handed a single file and run
one command (``vaara verify-bundle``) to check it.

The shape is exactly the ``bundle`` object the ``evidence_bundle_v0``
conformance vectors already commit, so every committed vector is a valid
bundle document and the independent checker that reads that shape needs no
Vaara import:

    {
      "receipt":        { ... },                 # required, ExecutionReceipt JSON
      "did_document":   { ... },                 # identity lens
      "expected_keyid": "did:web:...#key",       # identity lens (optional pin)
      "verifying_jwk":  {"kty": "EC", ...},      # signature lens (EC P-256/RSA)
      "attestation":    { ... },                 # back-link lens, SEP-2787 JSON
      "inclusion":      {"log_index", "tree_size",
                         "siblings_hex", "root_hex"},   # inclusion lens
      "inclusion_leaf_hex": "..",                # optional inclusion-leaf override
      "consistency":    {"first_size", "second_size", "hashes_hex",
                         "first_root_hex", "second_root_hex"},  # consistency lens
      "registry":       {"entries": [ ... ]}     # revocation lens
    }

Only ``receipt`` is required. Every other key is optional; an absent key
leaves its lens not applicable, exactly as a ``None`` field would on the
dataclass. The loader is strict on what is present: a malformed block raises
:class:`ValueError` naming the offending field, rather than silently dropping
a lens.

Purely additive. Defines no new wire shape for the bundle itself; it only
parses the existing one off disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from vaara.attestation._bundle import EvidenceBundle
from vaara.attestation._receipt_types import receipt_from_dict
from vaara.attestation._revocation import RevocationRegistry
from vaara.attestation.transparency_log import ConsistencyProof, InclusionProof


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field!r} must be a JSON object, got {type(value).__name__}")
    return value


def _hex_to_bytes(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{field!r} must be a hex string, got {type(value).__name__}")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field!r} is not valid hex: {exc}") from exc


def _hex_tuple(value: Any, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field!r} must be a list of hex strings")
    return tuple(_hex_to_bytes(h, f"{field}[{i}]") for i, h in enumerate(value))


def _int(value: Any, field: str) -> int:
    # bool is an int subclass; reject it so a stray true/false is not read as 1/0.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field!r} must be an integer")
    return value


def _verifying_material(jwk: dict[str, Any]) -> Any:
    # Reuse the identity lens's JWK parser (EC P-256 and RSA) so the signature
    # lens and the identity lens accept exactly the same key material.
    from vaara.attestation._receipt_identity import _jwk_to_public_key

    return _jwk_to_public_key(jwk)


def evidence_bundle_from_json(doc: dict[str, Any]) -> EvidenceBundle:
    """Load an :class:`EvidenceBundle` from its on-disk JSON document.

    ``doc`` is the parsed JSON object described in the module docstring. Only
    ``receipt`` is required; each other key feeds one lens and may be omitted.
    Raises :class:`ValueError` with the offending field name when a present
    block is malformed, so a bad file fails loudly instead of quietly losing
    a lens.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"bundle must be a JSON object, got {type(doc).__name__}")
    if "receipt" not in doc:
        raise ValueError("bundle is missing the required 'receipt' field")

    receipt = receipt_from_dict(_require_dict(doc["receipt"], "receipt"))

    verifying_material = None
    if doc.get("verifying_jwk") is not None:
        try:
            verifying_material = _verifying_material(
                _require_dict(doc["verifying_jwk"], "verifying_jwk")
            )
        except ValueError:
            raise
        except Exception as exc:  # AttestationError, KeyError from the JWK parser
            raise ValueError(f"'verifying_jwk' is not a usable key: {exc}") from exc

    attestation = None
    if doc.get("attestation") is not None:
        from vaara.attestation._attest_types import attestation_from_dict

        try:
            attestation = attestation_from_dict(
                _require_dict(doc["attestation"], "attestation")
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"'attestation' is not a valid attestation: {exc}") from exc

    inclusion = None
    log_root = None
    if doc.get("inclusion") is not None:
        inc = _require_dict(doc["inclusion"], "inclusion")
        try:
            inclusion = InclusionProof(
                log_index=_int(inc["log_index"], "inclusion.log_index"),
                tree_size=_int(inc["tree_size"], "inclusion.tree_size"),
                siblings=_hex_tuple(inc["siblings_hex"], "inclusion.siblings_hex"),
            )
            log_root = _hex_to_bytes(inc["root_hex"], "inclusion.root_hex")
        except KeyError as exc:
            raise ValueError(f"'inclusion' is missing {exc}") from exc

    inclusion_leaf = None
    if doc.get("inclusion_leaf_hex") is not None:
        inclusion_leaf = _hex_to_bytes(doc["inclusion_leaf_hex"], "inclusion_leaf_hex")

    consistency = None
    consistency_first_root = None
    consistency_second_root = None
    if doc.get("consistency") is not None:
        con = _require_dict(doc["consistency"], "consistency")
        try:
            consistency = ConsistencyProof(
                first_size=_int(con["first_size"], "consistency.first_size"),
                second_size=_int(con["second_size"], "consistency.second_size"),
                hashes=_hex_tuple(con["hashes_hex"], "consistency.hashes_hex"),
            )
            consistency_first_root = _hex_to_bytes(
                con["first_root_hex"], "consistency.first_root_hex"
            )
            consistency_second_root = _hex_to_bytes(
                con["second_root_hex"], "consistency.second_root_hex"
            )
        except KeyError as exc:
            raise ValueError(f"'consistency' is missing {exc}") from exc

    registry = None
    if doc.get("registry") is not None:
        try:
            registry = RevocationRegistry.from_dict(
                _require_dict(doc["registry"], "registry")
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"'registry' is not a valid registry: {exc}") from exc

    expected_keyid = doc.get("expected_keyid")
    if expected_keyid is not None and not isinstance(expected_keyid, str):
        raise ValueError("'expected_keyid' must be a string")

    did_document = doc.get("did_document")
    if did_document is not None:
        did_document = _require_dict(did_document, "did_document")

    return EvidenceBundle(
        receipt=receipt,
        did_document=did_document,
        expected_keyid=expected_keyid,
        verifying_material=verifying_material,
        attestation=attestation,
        inclusion=inclusion,
        log_root=log_root,
        inclusion_leaf=inclusion_leaf,
        consistency=consistency,
        consistency_first_root=consistency_first_root,
        consistency_second_root=consistency_second_root,
        registry=registry,
    )


# ── Issuer side: assemble the on-disk bundle document ────────────────────────
#
# ``evidence_bundle_from_json`` is the verifier's loader. The functions below
# are its issuer-side mirror: the party producing evidence stitches its pieces
# into the one document the verifier reads. They take and return plain JSON
# values (the same shapes each piece already has on disk), so assembly defines
# no new wire format and adds nothing the loader does not already parse.

# Conventional file names for an issuer's separate artifacts, used by
# ``vaara build-bundle --from-dir`` and mirrored by the build_bundle_v0
# independent checker. Each maps a file in the issuer's directory to the
# bundle-document key it fills. The ``.json`` pieces are JSON objects; the
# ``.txt`` pieces are raw scalar strings.
BUNDLE_PIECE_FILES: dict[str, str] = {
    "receipt.json": "receipt",
    "did_document.json": "did_document",
    "verifying_jwk.json": "verifying_jwk",
    "attestation.json": "attestation",
    "inclusion.json": "inclusion",
    "consistency.json": "consistency",
    "registry.json": "registry",
}
BUNDLE_PIECE_SCALARS: dict[str, str] = {
    "expected_keyid.txt": "expected_keyid",
    "inclusion_leaf_hex.txt": "inclusion_leaf_hex",
}

# The optional document keys, in the order the loader documents them. ``receipt``
# is required and handled separately.
_OPTIONAL_DOC_KEYS: tuple[str, ...] = (
    "did_document",
    "expected_keyid",
    "verifying_jwk",
    "attestation",
    "inclusion",
    "inclusion_leaf_hex",
    "consistency",
    "registry",
)


def build_bundle_document(
    *,
    receipt: dict[str, Any],
    did_document: dict[str, Any] | None = None,
    expected_keyid: str | None = None,
    verifying_jwk: dict[str, Any] | None = None,
    attestation: dict[str, Any] | None = None,
    inclusion: dict[str, Any] | None = None,
    inclusion_leaf_hex: str | None = None,
    consistency: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the on-disk evidence-bundle document from the issuer's pieces.

    The issuer-side mirror of :func:`evidence_bundle_from_json`: where the
    loader reconstructs a bundle from one document, this stitches the issuer's
    separate pieces into that one document. Each argument is the JSON the piece
    already is on disk (the receipt, the SEP-2787 attestation it answers, the
    transparency-log inclusion and consistency proofs, the did:web identity
    material, the verifying JWK, and the revocation registry). ``receipt`` is
    required; every other piece is optional and, when omitted, leaves its lens
    not applicable, exactly as on the dataclass.

    The assembled document is validated by loading it straight back through
    :func:`evidence_bundle_from_json`, so a malformed piece raises
    :class:`ValueError` naming the offending field at assembly time rather than
    producing a file that will not load. Returns the document as a plain dict;
    the caller renders it (the CLI writes it sorted with two-space indent, the
    exact bytes the ``bundle_doc_v0`` vectors commit).

    Round-trip: the document this returns, written to disk and read by
    :func:`evidence_bundle_from_json`, reconstructs the same bundle, which is
    what ``vaara verify-bundle`` checks.
    """
    doc: dict[str, Any] = {"receipt": receipt}
    optional: dict[str, Any] = {
        "did_document": did_document,
        "expected_keyid": expected_keyid,
        "verifying_jwk": verifying_jwk,
        "attestation": attestation,
        "inclusion": inclusion,
        "inclusion_leaf_hex": inclusion_leaf_hex,
        "consistency": consistency,
        "registry": registry,
    }
    for key in _OPTIONAL_DOC_KEYS:
        value = optional[key]
        if value is not None:
            doc[key] = value

    # Validate by reconstructing: a malformed piece fails here, loudly and
    # named, instead of being written to a bundle that the verifier rejects.
    evidence_bundle_from_json(doc)
    return doc


def load_bundle_pieces_from_dir(directory: Union[str, Path]) -> dict[str, Any]:
    """Discover an issuer's bundle pieces in a directory by conventional name.

    Returns a kwargs dict for :func:`build_bundle_document`: every recognised
    file present contributes its piece. The ``.json`` files in
    :data:`BUNDLE_PIECE_FILES` are parsed as JSON; the scalar ``.txt`` files in
    :data:`BUNDLE_PIECE_SCALARS` contribute their stripped text. Files outside
    the convention are ignored. Raises :class:`ValueError` naming the file when
    a JSON piece does not parse, and :class:`NotADirectoryError` when
    ``directory`` is not a directory.
    """
    base = Path(directory).expanduser()
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {base}")

    pieces: dict[str, Any] = {}
    for filename, key in BUNDLE_PIECE_FILES.items():
        path = base / filename
        if path.is_file():
            try:
                pieces[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    for filename, key in BUNDLE_PIECE_SCALARS.items():
        path = base / filename
        if path.is_file():
            pieces[key] = path.read_text(encoding="utf-8").strip()
    return pieces
