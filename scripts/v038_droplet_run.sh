#!/usr/bin/env bash
# v0.38 droplet-side driver. Mirrors v037_droplet_run.sh.
# Phase 1: re-runs the three existing attacker families (TM/PE/DE) against
# Llama-3.3-70B-Instruct-FP8-dynamic at a fresh seed for PAIR scale-up
# (combined with v037 outputs, halves the Wilson CI on each sub-cell).
# 4th attacker family (IPI / BIPIA-aligned) ships once its generator is
# authored — this driver is forward-compatible (just add a 4th category).
#
# NO destructive EXIT trap. Local-side watcher controls cleanup + shutdown.
# Idempotent: re-running detects existing vllm health + generator pidfiles
# and skips. Safe to re-run on partial-progress recovery.
set -euo pipefail

LOG_DIR=/root/vllm-logs
WORK_DIR=/root/v038
HF_CACHE=/root/hf-cache
GEN="${WORK_DIR}/scripts/generate_targeted_v037.py"
OUT_DIR=/root/v038-out
MODEL="${MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
MODEL_TAG="${MODEL_TAG:-llama33}"
PORT=8000
N_PER_CAT="${N_PER_CAT:-300}"
SEED="${SEED:-43}"
OUT_SUFFIX="${OUT_SUFFIX:-v038-${MODEL_TAG}-s${SEED}}"

mkdir -p "$LOG_DIR" "$WORK_DIR" "$OUT_DIR"

# 1. vllm container (skip if already healthy AND serving the right model)
served_model=""
if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null \
    | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | head -1 \
    | grep -oE '"[^"]*"$' \
    | tr -d '"' > /tmp/v038_served_model 2>/dev/null; then
  served_model=$(cat /tmp/v038_served_model)
fi
if [[ "${served_model}" == "${MODEL}" ]]; then
  echo "[v038] vllm already healthy on :${PORT} serving ${served_model}"
else
  if [[ -n "${served_model}" ]]; then
    echo "[v038] WARNING: :${PORT} is serving '${served_model}', not '${MODEL}'. Relaunching."
  fi
  echo "[v038] removing any stale vllm containers"
  docker rm -f vllm-llama33 vllm-mixtral vllm-qwen vllm 2>/dev/null || true

  echo "[v038] launching vllm container for ${MODEL}"
  docker run -d --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size 32G --ipc=host --network host \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -v "${HF_CACHE}":/root/.cache/huggingface \
    --name vllm-llama33 \
    rocm/vllm:latest \
    vllm serve "${MODEL}" \
      --host 0.0.0.0 --port "${PORT}" \
      --max-model-len 8192 \
      --enforce-eager \
      --gpu-memory-utilization 0.92 \
      >"${LOG_DIR}/vllm_llama33.log" 2>&1

  echo "[v038] waiting for /v1/models (max 30 min)"
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[v038] vllm healthy after ${i} x 10s"
      break
    fi
    sleep 10
  done
  if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[v038] vllm did not become healthy; tail of log:" >&2
    tail -40 "${LOG_DIR}/vllm_llama33.log" >&2
    exit 2
  fi
fi

# 2. fire the three existing generators in parallel under nohup
cd "${WORK_DIR}"
declare -A PREFIX=( [tool_misuse]=TM [privilege_escalation]=PE [data_exfil]=DE )
for cat in tool_misuse privilege_escalation data_exfil; do
  pfx=${PREFIX[$cat]}
  log="${LOG_DIR}/gen_${cat}.log"
  pidf="${LOG_DIR}/gen_${cat}.pid"
  if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "[v038] ${cat} generator already running pid=$(cat "${pidf}")"
    continue
  fi
  nohup python3 "${GEN}" \
    --category "${cat}" \
    --n "${N_PER_CAT}" \
    --base-url "http://localhost:${PORT}/v1" \
    --model "${MODEL}" \
    --model-tag "${MODEL_TAG}" \
    --random-seed "${SEED}" \
    --out "${OUT_DIR}/${pfx}-${OUT_SUFFIX}.jsonl" \
    >"${log}" 2>&1 &
  echo $! >"${pidf}"
  echo "[v038] launched ${cat} generator pid=$(cat "${pidf}") -> ${pfx}-${OUT_SUFFIX}.jsonl"
done

echo "[v038] all three generators running; outputs in ${OUT_DIR}"
echo "[v038] monitor: tail -f ${LOG_DIR}/gen_*.log"
