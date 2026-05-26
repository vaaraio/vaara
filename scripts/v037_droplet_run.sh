#!/usr/bin/env bash
# v0.37 droplet-side driver. Launches Llama-3.3-70B-Instruct FP8-dynamic
# under rocm/vllm:latest in a detached docker container, waits for the
# /v1/models endpoint, then fires three parallel category generators
# under nohup. Exits with vllm and the three generators still running.
# Monitor from the local side via scripts/v037_local_watcher.sh.
#
# NO destructive EXIT trap. Local-side watcher controls cleanup + shutdown.
# Lesson baked in from the v0.31 incident: trap final_rsync EXIT included
# `ssh DROPLET shutdown` and any SIGTERM (pkill) destroyed the model cache.
# This script just brings services up and exits clean.
#
# Idempotent: re-running detects existing vllm health + generator pidfiles
# and skips. Safe to re-run on partial-progress recovery.
#
# HF_TOKEN must be set in the environment if the model repo is gated.
# The RedHatAI FP8-dynamic variant is currently public; gating may change.
set -euo pipefail

LOG_DIR=/root/vllm-logs
WORK_DIR=/root/v037
HF_CACHE=/root/hf-cache
GEN="${WORK_DIR}/scripts/generate_targeted_v037.py"
OUT_DIR=/root/v037-out
MODEL="${MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
MODEL_TAG="${MODEL_TAG:-llama33}"
PORT=8000
N_PER_CAT="${N_PER_CAT:-300}"

mkdir -p "$LOG_DIR" "$WORK_DIR" "$OUT_DIR"

# 1. vllm container (skip if already healthy AND serving the right model)
served_model=""
if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null \
    | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | head -1 \
    | grep -oE '"[^"]*"$' \
    | tr -d '"' > /tmp/v037_served_model 2>/dev/null; then
  served_model=$(cat /tmp/v037_served_model)
fi
if [[ "${served_model}" == "${MODEL}" ]]; then
  echo "[v037] vllm already healthy on :${PORT} serving ${served_model}"
else
  if [[ -n "${served_model}" ]]; then
    echo "[v037] WARNING: :${PORT} is serving '${served_model}', not '${MODEL}'. Relaunching."
  fi
  echo "[v037] removing any stale vllm containers"
  docker rm -f vllm-llama33 vllm-mixtral vllm-qwen vllm 2>/dev/null || true

  echo "[v037] launching vllm container for ${MODEL}"
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
      --quantization fp8 \
      >"${LOG_DIR}/vllm_llama33.log" 2>&1

  echo "[v037] waiting for /v1/models (max 30 min)"
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[v037] vllm healthy after ${i} x 10s"
      break
    fi
    sleep 10
  done
  if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[v037] vllm did not become healthy; tail of log:" >&2
    tail -40 "${LOG_DIR}/vllm_llama33.log" >&2
    exit 2
  fi
fi

# 2. fire the three generators in parallel under nohup
cd "${WORK_DIR}"
declare -A PREFIX=( [tool_misuse]=TM [privilege_escalation]=PE [data_exfil]=DE )
for cat in tool_misuse privilege_escalation data_exfil; do
  pfx=${PREFIX[$cat]}
  log="${LOG_DIR}/gen_${cat}.log"
  pidf="${LOG_DIR}/gen_${cat}.pid"
  if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "[v037] ${cat} generator already running pid=$(cat "${pidf}")"
    continue
  fi
  nohup python3 "${GEN}" \
    --category "${cat}" \
    --n "${N_PER_CAT}" \
    --base-url "http://localhost:${PORT}/v1" \
    --model "${MODEL}" \
    --model-tag "${MODEL_TAG}" \
    --random-seed 42 \
    --out "${OUT_DIR}/${pfx}-v037-${MODEL_TAG}.jsonl" \
    >"${log}" 2>&1 &
  echo $! >"${pidf}"
  echo "[v037] launched ${cat} generator pid=$(cat "${pidf}") -> ${pfx}-v037-${MODEL_TAG}.jsonl"
done

echo "[v037] all three generators running; outputs in ${OUT_DIR}"
echo "[v037] monitor: tail -f ${LOG_DIR}/gen_*.log"
