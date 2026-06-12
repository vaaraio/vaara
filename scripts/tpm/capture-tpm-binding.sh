#!/usr/bin/env bash
# capture-tpm-binding.sh: produce a vaara.tpm-evidence-bundle/v0 from a real TPM.
#
# Phase 0 of the hardware-governance binding: take a signed SEP-2828 record, ask
# the local TPM 2.0 for a quote whose extraData carries SHA-256(jcs(record)) over
# PCR 10, grab the kernel IMA measurement log, and stitch all of it into one
# bundle that `vaara verify-tpm-binding` checks offline, trusting no operator.
#
# This is the hardware-touching half. The pure verifier needs none of this; this
# script exists only to feed it real evidence on a box that has a TPM.
#
# Requirements (none are present in CI; this runs on a TPM box):
#   - tpm2-tools (tpm2_createek, tpm2_createak, tpm2_quote, tpm2_pcrread)
#   - access to /dev/tpm0 or /dev/tpmrm0: membership in the `tss` group, or root
#   - root (or sudo) to read the IMA log, which is mode 0640 root-only
#   - a Python with `vaara[attestation]` installed (set VAARA_PYTHON, default
#     ./.venv/bin/python then python3)
#
# Usage:
#   scripts/tpm/capture-tpm-binding.sh RECORD.json OUT_BUNDLE.json [EXPECTED_IMA_PCR_HEX]
#
# The AK created here is an ephemeral ECC (NIST P-256, ECDSA-SHA256) attestation
# key, which is what the v0 verifier reads. Its endorsement-key chain is NOT
# exported or validated yet (that is the deferred `attested` tier); the bundle
# records the AK as caller-supplied.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RECORD.json OUT_BUNDLE.json [EXPECTED_IMA_PCR_HEX]" >&2
  exit 2
fi

RECORD="$1"
OUT="$2"
EXPECTED_IMA_PCR="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ASSEMBLE="${SCRIPT_DIR}/_assemble_bundle.py"

# Pick a Python that has vaara installed.
if [[ -n "${VAARA_PYTHON:-}" ]]; then
  PY="${VAARA_PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
else
  PY="python3"
fi

for tool in tpm2_createek tpm2_createak tpm2_quote tpm2_pcrread tpm2_flushcontext; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: ${tool} not found; install tpm2-tools" >&2
    exit 3
  fi
done

if [[ ! -f "${RECORD}" ]]; then
  echo "error: record not found: ${RECORD}" >&2
  exit 1
fi

IMA_LOG_SRC="/sys/kernel/security/ima/ascii_runtime_measurements_sha256"
if [[ ! -r "${IMA_LOG_SRC}" ]]; then
  echo "note: ${IMA_LOG_SRC} is not readable as this user; re-run with sudo or" >&2
  echo "      a kernel that exposes the sha256 IMA bank. The bundle needs it." >&2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "==> computing quote nonce = SHA-256(jcs(record))"
EXTRA_DATA_HEX="$("${PY}" "${ASSEMBLE}" extra-data "${RECORD}")"

# fTPMs implement only a handful of transient object slots. A previous run that
# died mid-way (or any other TPM user) can leave the EK/AK or a session resident,
# and tpm2_createak then fails 0x902 (TPM_RC_OBJECT_MEMORY, "out of memory for
# object contexts"). Flush stale transient objects and sessions first; harmless
# when nothing is loaded.
echo "==> flushing any stale transient objects + sessions"
tpm2_flushcontext -t >/dev/null 2>&1 || true
tpm2_flushcontext -l >/dev/null 2>&1 || true
tpm2_flushcontext -s >/dev/null 2>&1 || true

echo "==> creating ephemeral ECC endorsement + attestation keys"
tpm2_createek -c "${WORK}/ek.ctx" -G ecc >/dev/null
tpm2_createak \
  -C "${WORK}/ek.ctx" \
  -c "${WORK}/ak.ctx" \
  -G ecc -g sha256 -s ecdsa \
  -u "${WORK}/ak.pem" -f pem >/dev/null

echo "==> reading PCR 10 (sha256 bank)"
# tpm2_pcrread prints lines like "  10: 0x<HEX>"; pull the hex for PCR 10.
PCR10_HEX="$(tpm2_pcrread sha256:10 | awk '/10:/ { gsub(/0x/, "", $2); print tolower($2) }')"
if [[ -z "${PCR10_HEX}" ]]; then
  echo "error: could not read PCR 10 from sha256 bank" >&2
  exit 4
fi

echo "==> taking the quote over PCR 10 with the record nonce"
tpm2_quote \
  -c "${WORK}/ak.ctx" \
  -l "sha256:10" \
  -q "${EXTRA_DATA_HEX}" \
  -g sha256 \
  -m "${WORK}/quote.msg" \
  -s "${WORK}/quote.sig" \
  -o "${WORK}/quote.pcrs" >/dev/null

echo "==> capturing the IMA measurement log"
if [[ -r "${IMA_LOG_SRC}" ]]; then
  cp "${IMA_LOG_SRC}" "${WORK}/ima.log"
else
  # Last resort: needs privilege. Fails loudly rather than shipping an empty log.
  cat "${IMA_LOG_SRC}" > "${WORK}/ima.log"
fi

echo "==> assembling the bundle"
EXTRA_ARGS=()
if [[ -n "${EXPECTED_IMA_PCR}" ]]; then
  EXTRA_ARGS+=(--expected-ima-pcr "${EXPECTED_IMA_PCR}")
fi
"${PY}" "${ASSEMBLE}" assemble \
  --record "${RECORD}" \
  --attest "${WORK}/quote.msg" \
  --signature "${WORK}/quote.sig" \
  --ak-pem "${WORK}/ak.pem" \
  --ima-log "${WORK}/ima.log" \
  --pcr10 "${PCR10_HEX}" \
  --out "${OUT}" \
  "${EXTRA_ARGS[@]}"

echo "==> done. verify with:"
echo "    vaara verify-tpm-binding ${OUT}"
