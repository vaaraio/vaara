# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Assemble a vaara.tpm-evidence-chain/v0 from tpm2-tools capture outputs.

Called by ``capture-tpm-chain.sh`` after it has driven the TPM once per tick. Kept
separate so the per-link nonce (the chain-extended ``extraData``) and the chain
assembly run through the *same* vaara code the verifier uses, with no
re-implementation in shell.

The chain nonce for tick ``seq`` is
``SHA-256(jcs(record) || prev_digest || seq_be64)`` where ``prev_digest`` is the
SHA-256 of the previous tick's ``TPMS_ATTEST`` bytes (the genesis tick uses 32 zero
bytes). The shell computes it per tick with the ``extra-data`` subcommand before
asking for that tick's quote, then assembles every tick at the end.

Run under an interpreter that has ``vaara[attestation]`` installed. The only
structural assumption about tpm2-tools output is the same as the single-shot
script: ``tpm2_quote -m`` writes the raw ``TPMS_ATTEST`` and ``-s`` the marshalled
``TPMT_SIGNATURE``. If a future tpm2-tools changes that, this and
``_assemble_bundle.py`` are the places to adjust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from vaara.attestation._tpm import IMA_PCR
from vaara.attestation._tpm_chain import GENESIS_PREV_DIGEST
from vaara.attestation.receipt import (
    TPMChainLink,
    bind_record_to_chain_extra_data,
    build_tpm_chain_document,
)


def _extra_data_hex(args: argparse.Namespace) -> None:
    """Print the chain nonce (hex) for one tick. Used by the shell wrapper."""
    record = json.loads(Path(args.record).read_text())
    if args.prev_attest:
        prev_digest = hashlib.sha256(Path(args.prev_attest).read_bytes()).digest()
    else:
        prev_digest = GENESIS_PREV_DIGEST
    sys.stdout.write(
        bind_record_to_chain_extra_data(record, prev_digest, args.seq).hex()
    )


def _assemble(args: argparse.Namespace) -> None:
    record = json.loads(Path(args.record).read_text())
    ak_pem = Path(args.ak_pem).read_bytes()
    links_dir = Path(args.links_dir)

    # Tick files are NNNN.attest / NNNN.sig / NNNN.ima / NNNN.pcr10, in order.
    stems = sorted({p.stem for p in links_dir.glob("*.attest")})
    if not stems:
        sys.stderr.write(f"error: no *.attest tick files in {links_dir}\n")
        raise SystemExit(4)

    links: list[TPMChainLink] = []
    for stem in stems:
        attest = (links_dir / f"{stem}.attest").read_bytes()
        signature = (links_dir / f"{stem}.sig").read_bytes()
        ima_log = (links_dir / f"{stem}.ima").read_text()
        pcr10 = bytes.fromhex((links_dir / f"{stem}.pcr10").read_text().strip())
        links.append(
            TPMChainLink(attest, signature, ak_pem, {IMA_PCR: pcr10}, ima_log)
        )

    doc = build_tpm_chain_document(
        record,
        links,
        bank="sha256",
        expected_ima_pcr=args.expected_ima_pcr,
    )
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    sys.stderr.write(f"wrote {args.out} ({len(links)} links)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extra = sub.add_parser(
        "extra-data", help="print the chain nonce hex for one tick"
    )
    p_extra.add_argument("--record", required=True)
    p_extra.add_argument("--seq", type=int, required=True)
    p_extra.add_argument(
        "--prev-attest",
        dest="prev_attest",
        default=None,
        help="previous tick's TPMS_ATTEST file; omit for the genesis tick",
    )

    p_asm = sub.add_parser("assemble", help="build the chain JSON from tick files")
    p_asm.add_argument("--record", required=True)
    p_asm.add_argument("--ak-pem", required=True, dest="ak_pem")
    p_asm.add_argument("--links-dir", required=True, dest="links_dir")
    p_asm.add_argument("--expected-ima-pcr", default=None, dest="expected_ima_pcr")
    p_asm.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "extra-data":
        _extra_data_hex(args)
    else:
        _assemble(args)


if __name__ == "__main__":
    main()
