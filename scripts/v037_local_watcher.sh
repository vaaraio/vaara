#!/usr/bin/env bash
# v0.37 local-side watcher. Polls the droplet for generator progress + pulls
# new entries via rsync. Reports state transitions to .v037_watch/progress.log
# on a ~60s cadence. Exits when all three droplet generators have written
# their target count of entries (or the user kills it).
#
# Per feedback_monitor_droplet_jobs_continuously.md: continuous monitoring,
# not fire-and-forget. The harness layer (Claude session) tails this log
# in turn and surfaces state transitions to the user.
#
# Auto-shutdown opt-in: set DROPLET_NAME=<doctl name> and the watcher will
# `doctl compute droplet delete --force` after a clean three-cat finish.
# Without DROPLET_NAME the watcher exits clean and leaves the droplet up.
#
# Usage:
#   DROPLET=root@REMOTE_IP DROPLET_NAME=vaara-v037 N_PER_CAT=300 \
#     ./scripts/v037_local_watcher.sh
set -euo pipefail

DROPLET="${DROPLET:?set DROPLET=root@DROPLET_IP before launching}"
DROPLET_NAME="${DROPLET_NAME:-}"
N_PER_CAT="${N_PER_CAT:-300}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"
MODEL_TAG="${MODEL_TAG:-llama33}"
EXPECTED_MODEL="${EXPECTED_MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"

WATCH_DIR=".v037_watch"
GEN_LOCAL="tests/adversarial/generated"
GEN_REMOTE="/root/v037-out/"
LOG_REMOTE="/root/vllm-logs"

mkdir -p "${WATCH_DIR}" "${GEN_LOCAL}"

ts() { date -u +%FT%TZ; }
note() { echo "[$(ts)] $*" | tee -a "${WATCH_DIR}/progress.log"; }

note "watcher start; droplet=${DROPLET} target=${N_PER_CAT}/category interval=${INTERVAL_SEC}s shutdown=${DROPLET_NAME:-off}"

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${DROPLET}" 'echo ok' >/dev/null 2>&1; then
  note "FATAL: ssh ${DROPLET} 'echo ok' failed. Check droplet IP + key auth."
  exit 2
fi

declare -A LAST_COUNT=( [TM]=0 [PE]=0 [DE]=0 )

while true; do
  rsync -avz --partial \
    "${DROPLET}:${GEN_REMOTE}" "${GEN_LOCAL}/" \
    >>"${WATCH_DIR}/rsync.log" 2>&1 || note "rsync cycle failed (non-fatal)"

  done_count=0
  for pfx in TM PE DE; do
    f="${GEN_LOCAL}/${pfx}-v037-${MODEL_TAG}.jsonl"
    cur=0
    if [[ -f "$f" ]]; then
      cur=$(wc -l < "$f" | tr -d ' ')
    fi
    prev=${LAST_COUNT[$pfx]}
    if [[ "$cur" -gt "$prev" ]]; then
      note "${pfx}: ${prev} -> ${cur} (+$((cur - prev)))"
      LAST_COUNT[$pfx]=$cur
    fi
    if [[ "$cur" -ge "$N_PER_CAT" ]]; then
      done_count=$((done_count + 1))
    fi
  done

  vllm_model=$(ssh -o BatchMode=yes "${DROPLET}" \
    "curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -oE '\"id\"[[:space:]]*:[[:space:]]*\"[^\"]*\"' | head -1 | grep -oE '\"[^\"]*\"\$' | tr -d '\"'" \
    2>/dev/null || echo "SSH_FAIL")
  if [[ "$vllm_model" != "${EXPECTED_MODEL}" ]]; then
    note "WARNING vllm_model='${vllm_model}' (expected ${EXPECTED_MODEL})"
    ssh -o BatchMode=yes "${DROPLET}" "tail -20 ${LOG_REMOTE}/vllm_llama33.log" 2>&1 \
      | tee -a "${WATCH_DIR}/progress.log" || true
  fi

  if [[ "$done_count" -ge 3 ]]; then
    note "all three generators reached ${N_PER_CAT} entries; watcher exiting clean"
    note "next steps: dedupe pass + schema validation + MANIFEST regen + bench eval"
    if [[ -n "${DROPLET_NAME}" ]]; then
      note "auto-shutdown: doctl compute droplet delete ${DROPLET_NAME} --force"
      if command -v doctl >/dev/null 2>&1; then
        doctl compute droplet delete "${DROPLET_NAME}" --force 2>&1 \
          | tee -a "${WATCH_DIR}/progress.log" || note "doctl delete failed (manual shutdown required)"
      else
        note "doctl not on PATH; shut down ${DROPLET_NAME} manually"
      fi
    fi
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
