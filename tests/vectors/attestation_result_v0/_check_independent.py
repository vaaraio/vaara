"""Independent checker for attestation_result_v0 vectors.

Imports only the standard library. It does NOT import Vaara. It re-derives, from
each input verdict in ``cases.json``, the EAR (draft-ietf-rats-ear) and its AR4SI
trustworthiness vector (draft-ietf-rats-ar4si) using a second, independent
implementation of the honesty mapping, and byte-compares the result against
``expected.json``.

The mapping it reproduces:

  * instance-identity / hardware -- trust in the attesting key and hardware root.
    Contraindicated (96) if the evidence does not hold (unparseable / bad signature,
    or -- for instance-identity -- a quote/report that does not bind this record).
    Affirming (2) only when a basis reports a validated root (``ek_chain_verified``
    / ``kds_verified``). Otherwise a warning (32): verified but provenance trusted
    as supplied.
  * executables / configuration -- trust in measured state. Pin mismatch -> 96. A
    reconciled-and-pinned measurement -> 2. Reconciled-but-unpinned -> 32. Not
    reconciled -> 96. For SEV-SNP both follow the single launch measurement.
  * overall ear_status -- no higher trust than the worst claim present.

Run: ``python tests/vectors/attestation_result_v0/_check_independent.py``.
Exit code 0 on full agreement, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EAR_PROFILE = "tag:ietf.org,2026:rats/ear#04"
VAARA_PROFILE = "tag:vaara.io,2026:attestation-result#v0"
VERIFIER_CLAIMS_KEY = "vaara.io/verifier-claims"
VERIFIER_DEVELOPER = "https://vaara.io"
SCHEMA = "vaara.attestation-result/v0"
HONEST_LIMIT = (
    "The platform is attested, not the decision semantics; the decision "
    "content is carried by the signed SEP-2828 record this result references "
    "by digest. affirming requires a validated hardware root and is "
    "unreachable while the root is trusted as supplied."
)

AFF, WARN, CONTRA = 2, 32, 96


def _root(evidence_ok: bool, validated: bool) -> int:
    if not evidence_ok:
        return CONTRA
    return AFF if validated else WARN


def _meas(reconciled: bool, pin: str) -> int:
    if pin == "pin_mismatch":
        return CONTRA
    if not reconciled:
        return CONTRA
    return AFF if pin == "pinned" else WARN


def _status(vector: dict) -> str:
    if not vector:
        return "none"
    worst = max(vector.values())
    if worst >= CONTRA:
        return "contraindicated"
    if worst >= WARN:
        return "warning"
    if worst >= AFF:
        return "affirming"
    return "none"


def _derive(v: dict):
    """Return (submod_label, vector, native) for a verdict, independently."""
    schema = v.get("schema")
    if schema == "vaara.tpm-evidence-chain/v0":
        ev_ok = bool(v.get("links_bound")) and v.get("tier") != "unverified"
        validated = v.get("ak_chain_basis") == "ek_chain_verified"
        pin = str(v.get("pcr_pin_basis", "unpinned"))
        vector = {
            "instance-identity": _root(ev_ok, validated),
            "hardware": _root(ev_ok, validated),
            "executables": _meas(bool(v.get("ima_append_only")) and ev_ok, pin),
            "configuration": _meas(ev_ok, pin),
        }
        native = {
            "native_tier": v.get("tier"),
            "root_trust_basis": v.get("ak_chain_basis"),
            "pcr_pin_basis": pin,
            "ima_policy_basis": v.get("ima_policy_basis"),
            "freshness_basis": v.get("freshness_basis"),
            "decision_semantics_basis": v.get("decision_logic_basis"),
            "chain_continuous": v.get("tier") == "continuous",
            "n_links": v.get("n_links"),
        }
        return "tpm", vector, native

    if schema == "vaara.tpm-binding-attestation/v0":
        ev_ok = bool(v.get("parsed")) and bool(v.get("signature_valid"))
        bound = bool(v.get("bound"))
        validated = v.get("ak_chain_basis") == "ek_chain_verified"
        pin = str(v.get("pcr_pin_basis", "unpinned"))
        vector = {
            "instance-identity": _root(ev_ok, validated) if (ev_ok and bound) else CONTRA,
            "hardware": _root(ev_ok, validated),
        }
        if v.get("ima_log_entries"):
            vector["executables"] = _meas(bool(v.get("ima_replayed")), pin)
        vector["configuration"] = _meas(bool(v.get("pcr_digest_recomputed")), pin)
        native = {
            "native_tier": v.get("tier"),
            "root_trust_basis": v.get("ak_chain_basis"),
            "pcr_pin_basis": pin,
            "ima_policy_basis": v.get("ima_policy_basis"),
            "freshness_basis": v.get("freshness_basis"),
            "decision_semantics_basis": v.get("decision_logic_basis"),
            "bound_record_digest": v.get("extra_data_expected"),
        }
        return "tpm", vector, native

    if schema == "vaara.enforcement-attestation/v0":
        ev_ok = bool(v.get("parsed")) and bool(v.get("signature_valid"))
        bound = bool(v.get("bound"))
        validated = v.get("vcek_chain_basis") == "kds_verified"
        mbasis = str(v.get("measurement_basis", "unpinned"))
        vector = {
            "instance-identity": _root(ev_ok, validated) if (ev_ok and bound) else CONTRA,
            "hardware": _root(ev_ok, validated),
        }
        if v.get("measurement"):
            measured = _meas(True, mbasis)
            vector["executables"] = measured
            vector["configuration"] = measured
        native = {
            "native_tier": v.get("tier"),
            "root_trust_basis": v.get("vcek_chain_basis"),
            "measurement_basis": mbasis,
            "tcb_appraisal": "not_established",
            "enforcement_logic_basis": v.get("enforcement_logic_basis"),
            "bound_record_digest": v.get("report_data_expected"),
        }
        return "sev-snp", vector, native

    raise ValueError(f"unrecognised verdict schema {schema!r}")


def _build(verdict: dict, iat: int, build: str) -> dict:
    label, vector, native = _derive(verdict)
    status = _status(vector)
    claims = {
        "schema": SCHEMA,
        "source_schema": verdict.get("schema"),
        "result_is_unsigned": True,
        "honest_limit": HONEST_LIMIT,
    }
    claims.update({k: val for k, val in native.items() if val is not None})
    submod = {
        "eat_profile": VAARA_PROFILE,
        "ear_status": status,
        "ear_trustworthiness_vector": vector,
        VERIFIER_CLAIMS_KEY: claims,
    }
    return {
        "eat_profile": EAR_PROFILE,
        "iat": iat,
        "ear_status": status,
        "ear_verifier_id": {"developer": VERIFIER_DEVELOPER, "build": build},
        "submods": {label: submod},
    }


def main() -> int:
    cases_doc = json.loads((HERE / "cases.json").read_text())
    expected = json.loads((HERE / "expected.json").read_text())
    iat = cases_doc["iat"]
    build = cases_doc["verifier_build"]

    failures = 0
    for case in cases_doc["cases"]:
        name = case["name"]
        got = _build(case["verdict"], iat, build)
        want = expected[name]
        if got != want:
            failures += 1
            print(f"MISMATCH {name}:")
            print(f"  want: {json.dumps(want, sort_keys=True)}")
            print(f"  got:  {json.dumps(got, sort_keys=True)}")
    total = len(cases_doc["cases"])
    if failures:
        print(f"\n{failures}/{total} cases disagree")
        return 1
    print(f"independent checker reproduced all {total} EARs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
