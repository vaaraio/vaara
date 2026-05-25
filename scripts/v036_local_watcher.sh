#!/usr/bin/env bash
# v0.36 local-side watcher. Polls the droplet for generator progress + pulls
# new entries via rsync. Reports state transitions to .v036_watch/progress.log
# on a ~60s cadence. Exits when all three droplet generators have written
# their target count of entries (or the user kills it).
#
# Per feedback_monitor_droplet_jobs_continuously.md: continuous monitoring,
# not fire-and-forget. The harness layer (Claude session) tails this log
# in turn and surfaces state transitions to the user.
#
# Usage:
#   DROPLET=root@1.2.3.4 N_PER_CAT=700 ./scripts/v036_local_watcher.sh
#
# Side effects:
#   - .v036_watch/progress.log: append-only state log
#   - .v036_watch/rsync.log: rsync output per cycle
#   - tests/adversarial/generated/{TM,PE,DE}-v036-mixtral.jsonl: pulled artefacts
set -euo pipefail

DROPLET="${DROPLET:?set DROPLET=root@DROPLET_IP before launching}"
N_PER_CAT="${N_PER_CAT:-700}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

WATCH_DIR=".v036_watch"
GEN_LOCAL="tests/adversarial/generated"
GEN_REMOTE="/root/v036-out/"
LOG_REMOTE="/root/vllm-logs"

mkdir -p "${WATCH_DIR}" "${GEN_LOCAL}"

ts() { date -u +%FT%TZ; }
note() { echo "[$(ts)] $*" | tee -a "${WATCH_DIR}/progress.log"; }

note "watcher start; droplet=${DROPLET} target=${N_PER_CAT}/category interval=${INTERVAL_SEC}s"

# Sanity: confirm ssh works before entering loop
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${DROPLET}" 'echo ok' >/dev/null 2>&1; then
  note "FATAL: ssh ${DROPLET} 'echo ok' failed. Check droplet IP + key auth."
  exit 2
fi

declare -A LAST_COUNT=( [TM]=0 [PE]=0 [DE]=0 )

while true; do
  # 1. rsync any new jsonl back to local repo
  rsync -avz --partial \
    "${DROPLET}:${GEN_REMOTE}" "${GEN_LOCAL}/" \
    >>"${WATCH_DIR}/rsync.log" 2>&1 || note "rsync cycle failed (non-fatal)"

  # 2. count entries per category locally and report deltas
  done_count=0
  for pfx in TM PE DE; do
    f="${GEN_LOCAL}/${pfx}-v036-mixtral.jsonl"
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

  # 3. vLLM liveness + model-identity check + recent log tail (early warning)
  vllm_model=$(ssh -o BatchMode=yes "${DROPLET}" \
    "curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -oE '\"id\"[[:space:]]*:[[:space:]]*\"[^\"]*\"' | head -1 | grep -oE '\"[^\"]*\"\$' | tr -d '\"'" \
    2>/dev/null || echo "SSH_FAIL")
  if [[ "$vllm_model" != "mistralai/Mixtral-8x7B-Instruct-v0.1" ]]; then
    note "WARNING vllm_model='${vllm_model}' (expected mistralai/Mixtral-8x7B-Instruct-v0.1)"
    ssh -o BatchMode=yes "${DROPLET}" "tail -20 ${LOG_REMOTE}/vllm_mixtral.log" 2>&1 \
      | tee -a "${WATCH_DIR}/progress.log" || true
  fi

  # 4. exit when all three legs hit the target
  if [[ "$done_count" -ge 3 ]]; then
    note "all three generators reached ${N_PER_CAT} entries; watcher exiting clean"
    note "next steps: dedupe pass + Claude leg join + schema validation + MANIFEST regen"
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
