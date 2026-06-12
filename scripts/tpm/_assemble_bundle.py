"""Assemble a vaara.tpm-evidence-bundle/v0 from tpm2-tools capture outputs.

Called by ``capture-tpm-binding.sh`` after it has driven the TPM. Kept separate
so the canonicalisation of the record (``SHA-256(jcs(record))`` for the quote
nonce) and the bundle assembly run through the *same* vaara code the verifier
uses, with no re-implementation in shell. Run under an interpreter that has
``vaara[attestation]`` installed.

Inputs are files on disk; the only structural assumption about the tpm2-tools
output is that ``tpm2_quote -m`` writes the raw ``TPMS_ATTEST`` (the bytes from
the ``magic`` field onward) and ``-s`` writes the marshalled ``TPMT_SIGNATURE``,
which is what the verifier's parsers expect. If a future tpm2-tools changes that,
this is the one place to adjust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vaara.attestation.receipt import (
    bind_record_to_extra_data,
    build_tpm_bundle_document,
)
from vaara.attestation._tpm import IMA_PCR


def _extra_data_hex(record_path: str) -> None:
    """Print the quote nonce (hex) for a record. Used by the shell wrapper."""
    record = json.loads(Path(record_path).read_text())
    sys.stdout.write(bind_record_to_extra_data(record).hex())


def _assemble(args: argparse.Namespace) -> None:
    record = json.loads(Path(args.record).read_text())
    attest_bytes = Path(args.attest).read_bytes()
    signature = Path(args.signature).read_bytes()
    ak_pem = Path(args.ak_pem).read_bytes()
    ima_log = Path(args.ima_log).read_text()
    pcr10 = bytes.fromhex(args.pcr10.strip())

    doc = build_tpm_bundle_document(
        record,
        attest_bytes,
        signature,
        ak_pem,
        {IMA_PCR: pcr10},
        ima_log,
        bank="sha256",
        expected_ima_pcr=args.expected_ima_pcr,
    )
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    sys.stderr.write(f"wrote {args.out}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extra = sub.add_parser("extra-data", help="print SHA-256(jcs(record)) hex")
    p_extra.add_argument("record")

    p_asm = sub.add_parser("assemble", help="build the bundle JSON")
    p_asm.add_argument("--record", required=True)
    p_asm.add_argument("--attest", required=True)
    p_asm.add_argument("--signature", required=True)
    p_asm.add_argument("--ak-pem", required=True, dest="ak_pem")
    p_asm.add_argument("--ima-log", required=True, dest="ima_log")
    p_asm.add_argument("--pcr10", required=True)
    p_asm.add_argument("--expected-ima-pcr", default=None, dest="expected_ima_pcr")
    p_asm.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "extra-data":
        _extra_data_hex(args.record)
    else:
        _assemble(args)


if __name__ == "__main__":
    main()
