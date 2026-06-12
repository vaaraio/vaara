#!/usr/bin/env bash
# capture-tpm-chain.sh: produce a vaara.tpm-evidence-chain/v0 from a real TPM.
#
# Phase 1 of the hardware-governance binding, the continuous-attestation loop. It
# takes one signed SEP-2828 record and, on a fixed interval, asks the local TPM 2.0
# for a sequence of quotes over PCR 10. Each tick's extraData carries the
# chain-extended nonce SHA-256(jcs(record) || prev_digest || seq), so the ticks are
# hash-linked: a regulator running `vaara verify-tpm-chain` offline learns the
# measured platform held continuously across the window, with no reboot and an
# append-only IMA log, trusting no operator.
#
# One ephemeral ECC AK is created once and reused for every tick, so the chain is a
# single attester's record (`ak_stable`). Its endorsement-key chain is NOT exported
# or validated yet (the deferred `attested` tier).
#
# Requirements (none in CI; this runs on a TPM box):
#   - tpm2-tools (tpm2_createek, tpm2_createak, tpm2_quote, tpm2_pcrread)
#   - access to /dev/tpm0 or /dev/tpmrm0: membership in the `tss` group, or root
#   - root (or sudo) to read the IMA log, which is mode 0640 root-only
#   - a Python with `vaara[attestation]` (set VAARA_PYTHON, default ./.venv/bin/python)
#
# Usage:
#   scripts/tpm/capture-tpm-chain.sh RECORD.json OUT_CHAIN.json [TICKS] [INTERVAL_SEC]
# Defaults: TICKS=3, INTERVAL_SEC=2.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 RECORD.json OUT_CHAIN.json [TICKS] [INTERVAL_SEC]" >&2
  exit 2
fi

RECORD="$1"
OUT="$2"
TICKS="${3:-3}"
INTERVAL="${4:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ASSEMBLE="${SCRIPT_DIR}/_assemble_chain.py"

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
  echo "note: ${IMA_LOG_SRC} is not readable as this user; re-run with sudo." >&2
fi

WORK="$(mktemp -d)"
LINKS="${WORK}/links"
mkdir -p "${LINKS}"
trap 'rm -rf "${WORK}"' EXIT

# fTPMs implement only a handful of transient object slots. A previous run that
# died mid-way (or any other TPM user) can leave the EK/AK or a session resident,
# and tpm2_createak then fails 0x902 (TPM_RC_OBJECT_MEMORY, "out of memory for
# object contexts"). Flush stale transient objects and sessions first; harmless
# when nothing is loaded.
echo "==> flushing any stale transient objects + sessions"
tpm2_flushcontext -t >/dev/null 2>&1 || true
tpm2_flushcontext -l >/dev/null 2>&1 || true
tpm2_flushcontext -s >/dev/null 2>&1 || true

echo "==> creating one ephemeral ECC EK + AK, reused for every tick"
tpm2_createek -c "${WORK}/ek.ctx" -G ecc >/dev/null
tpm2_createak \
  -C "${WORK}/ek.ctx" \
  -c "${WORK}/ak.ctx" \
  -G ecc -g sha256 -s ecdsa \
  -u "${WORK}/ak.pem" -f pem >/dev/null

PREV_ATTEST=""
for (( seq=0; seq<TICKS; seq++ )); do
  STEM="$(printf '%04d' "${seq}")"
  echo "==> tick ${seq}: computing chain nonce"
  if [[ -n "${PREV_ATTEST}" ]]; then
    EXTRA_DATA_HEX="$("${PY}" "${ASSEMBLE}" extra-data \
      --record "${RECORD}" --seq "${seq}" --prev-attest "${PREV_ATTEST}")"
  else
    EXTRA_DATA_HEX="$("${PY}" "${ASSEMBLE}" extra-data \
      --record "${RECORD}" --seq "${seq}")"
  fi

  PCR10_HEX="$(tpm2_pcrread sha256:10 | awk '/10:/ { gsub(/0x/, "", $2); print tolower($2) }')"
  if [[ -z "${PCR10_HEX}" ]]; then
    echo "error: could not read PCR 10 from sha256 bank" >&2
    exit 4
  fi

  # Each tpm2_quote ContextLoads the AK into a transient object slot. fTPMs have
  # only a few and the loaded objects accumulate across ticks until ContextLoad
  # fails 0x902 (TPM_RC_OBJECT_MEMORY). Flush transient objects before each quote:
  # the AK is reloaded from ak.ctx on disk, so every tick runs on a clean slate.
  tpm2_flushcontext -t >/dev/null 2>&1 || true

  tpm2_quote \
    -c "${WORK}/ak.ctx" \
    -l "sha256:10" \
    -q "${EXTRA_DATA_HEX}" \
    -g sha256 \
    -m "${LINKS}/${STEM}.attest" \
    -s "${LINKS}/${STEM}.sig" \
    -o "${WORK}/${STEM}.pcrs" >/dev/null

  printf '%s' "${PCR10_HEX}" > "${LINKS}/${STEM}.pcr10"
  if [[ -r "${IMA_LOG_SRC}" ]]; then
    cp "${IMA_LOG_SRC}" "${LINKS}/${STEM}.ima"
  else
    cat "${IMA_LOG_SRC}" > "${LINKS}/${STEM}.ima"
  fi

  PREV_ATTEST="${LINKS}/${STEM}.attest"
  if (( seq < TICKS - 1 )); then
    sleep "${INTERVAL}"
  fi
done

echo "==> assembling the chain (${TICKS} ticks)"
"${PY}" "${ASSEMBLE}" assemble \
  --record "${RECORD}" \
  --ak-pem "${WORK}/ak.pem" \
  --links-dir "${LINKS}" \
  --out "${OUT}"

echo "==> done. verify with:"
echo "    vaara verify-tpm-chain ${OUT}"
