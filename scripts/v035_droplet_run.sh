#!/usr/bin/env bash
# v0.35 droplet-side driver. Launches Qwen-72B under rocm/vllm:latest in a
# detached docker container, waits for the /v1/models endpoint, then fires
# three parallel matched-benign generators under nohup. Exits with vllm and
# the three generators still running; monitor from local side via the
# tail-friendly logs in ~/vllm-logs/.
set -euo pipefail

LOG_DIR=/root/vllm-logs
WORK_DIR=/root/v035
HF_CACHE=/root/hf-cache
GEN="${WORK_DIR}/scripts/generate_matched_benign_v035.py"
OUT_DIR=/root/v035-out
MODEL="Qwen/Qwen2.5-72B-Instruct"
PORT=8000

mkdir -p "$LOG_DIR" "$WORK_DIR" "$OUT_DIR"

# 1. vllm container (skip if already healthy)
if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[v035] vllm already healthy on :${PORT}"
else
  echo "[v035] removing any stale vllm-qwen container"
  docker rm -f vllm-qwen 2>/dev/null || true

  echo "[v035] launching vllm container"
  docker run -d --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size 32G --ipc=host --network host \
    -v "${HF_CACHE}":/root/.cache/huggingface \
    --name vllm-qwen \
    rocm/vllm:latest \
    vllm serve "${MODEL}" \
      --host 0.0.0.0 --port "${PORT}" \
      --max-model-len 8192 \
      --enforce-eager \
      --gpu-memory-utilization 0.92 \
      >"${LOG_DIR}/vllm_qwen.log" 2>&1

  echo "[v035] waiting for /v1/models (max 30 min)"
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[v035] vllm healthy after ${i} x 10s"
      break
    fi
    sleep 10
  done
  if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[v035] vllm did not become healthy; tail of log:" >&2
    tail -40 "${LOG_DIR}/vllm_qwen.log" >&2
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
    echo "[v035] ${cat} generator already running pid=$(cat "${pidf}")"
    continue
  fi
  nohup python3 "${GEN}" \
    --category "${cat}" \
    --n 700 \
    --base-url "http://localhost:${PORT}/v1" \
    --model "${MODEL}" \
    --random-seed 42 \
    --out "${OUT_DIR}/BT-v035-${pfx}.jsonl" \
    >"${log}" 2>&1 &
  echo $! >"${pidf}"
  echo "[v035] launched ${cat} generator pid=$(cat "${pidf}") -> BT-v035-${pfx}.jsonl"
done

echo "[v035] all three generators running; outputs in ${OUT_DIR}"
echo "[v035] monitor: tail -f ${LOG_DIR}/gen_*.log"
