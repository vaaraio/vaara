#!/usr/bin/env bash
# Droplet-side driver, parameterised. Serves one model under vLLM and fires one
# generator per category in parallel.
#
# The v037 and v038 runners hardcoded their three categories in the for-loop, so
# every new category set meant editing a copy of the script on a paid box. This
# takes the list from the environment instead, which is what makes a second
# attacker model a one-liner rather than a fourth copy of this file.
#
#   MODE=attack  CATS="prompt_injection ssrf_via_tools" SEED=77 \
#   OUT_SUFFIX=v041-llama33-s77 bash v041_droplet_run.sh
#
# NO destructive EXIT trap, and nothing here touches git. The generators write
# INTO /root/vaara/tests/adversarial, so a git operation on that working tree
# can take finished output with it. That is not a theory: it destroyed 282
# finished SSRF entries on 2026-09-10.
set -euo pipefail

LOG_DIR=/root/vllm-logs
WORK_DIR=/root/vaara
GEN_ATTACK="${WORK_DIR}/scripts/generate_targeted_v037.py"
GEN_BENIGN="${WORK_DIR}/scripts/generate_matched_benign_v035.py"

MODE="${MODE:-attack}"
MODEL="${MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
MODEL_TAG="${MODEL_TAG:-llama33}"
SEED="${SEED:-42}"
N_PER_CAT="${N_PER_CAT:-700}"
PORT="${PORT:-8000}"
CATS="${CATS:-prompt_injection ssrf_via_tools destructive_actions credential_exfil}"
OUT_SUFFIX="${OUT_SUFFIX:-v041-${MODEL_TAG}-s${SEED}}"

declare -A PREFIX=(
  [tool_misuse]=TM [privilege_escalation]=PE [data_exfil]=DE
  [prompt_injection]=PI [ssrf_via_tools]=SR
  [destructive_actions]=DA [credential_exfil]=CE
)

mkdir -p "$LOG_DIR"

# 1. vLLM, reused if it is already serving the model we want.
served=$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "")
if [[ "${served}" == "${MODEL}" ]]; then
  echo "[v041] vllm already serving ${MODEL}"
else
  if [[ -n "${served}" ]]; then
    echo "[v041] :${PORT} serves '${served}', not '${MODEL}'. Replacing."
    docker rm -f vllm-server 2>/dev/null || true
    sleep 10
  fi
  echo "[v041] launching vllm for ${MODEL}"
  docker run -d --rm --name vllm-server \
    --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --network host \
    --shm-size 16g \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    vllm/vllm-openai-rocm:latest \
    vllm serve "${MODEL}" \
      --host 0.0.0.0 --port "${PORT}" \
      --max-model-len 8192 \
      --enforce-eager \
      --gpu-memory-utilization 0.92 \
      --quantization fp8 \
      >"${LOG_DIR}/vllm_${MODEL_TAG}.log" 2>&1

  echo "[v041] waiting for /v1/models (max 60 min; a cold model pull is slow)"
  for i in $(seq 1 360); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[v041] vllm healthy after ${i} x 10s"
      break
    fi
    sleep 10
  done
  if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[v041] vllm did not become healthy; tail of log:" >&2
    tail -40 "${LOG_DIR}/vllm_${MODEL_TAG}.log" >&2
    exit 2
  fi
fi

# 2. one generator per category, in parallel, detached.
cd "${WORK_DIR}"
for cat in ${CATS}; do
  pfx="${PREFIX[$cat]:-}"
  if [[ -z "${pfx}" ]]; then
    echo "[v041] unknown category '${cat}', skipping" >&2
    continue
  fi
  log="${LOG_DIR}/gen_${MODE}_${cat}_${OUT_SUFFIX}.log"
  pidf="${LOG_DIR}/gen_${MODE}_${cat}_${OUT_SUFFIX}.pid"
  if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "[v041] ${cat} already running pid=$(cat "${pidf}")"
    continue
  fi

  if [[ "${MODE}" == "benign" ]]; then
    out="${WORK_DIR}/tests/adversarial/benign_generated/BT-${OUT_SUFFIX}-${pfx}.jsonl"
    nohup python3 "${GEN_BENIGN}" \
      --category "${cat}" --n "${N_PER_CAT}" \
      --base-url "http://localhost:${PORT}/v1" \
      --model "${MODEL}" --random-seed "${SEED}" \
      --out "${out}" >"${log}" 2>&1 &
  else
    out="${WORK_DIR}/tests/adversarial/generated/${pfx}-${OUT_SUFFIX}.jsonl"
    nohup python3 "${GEN_ATTACK}" \
      --category "${cat}" --n "${N_PER_CAT}" \
      --base-url "http://localhost:${PORT}/v1" \
      --model "${MODEL}" --model-tag "${MODEL_TAG}" \
      --random-seed "${SEED}" --dedupe-prior \
      --out "${out}" >"${log}" 2>&1 &
  fi
  echo $! >"${pidf}"
  echo "[v041] ${MODE} ${cat} pid=$(cat "${pidf}") -> $(basename "${out}")"
done

echo "[v041] all launched. monitor: tail -f ${LOG_DIR}/gen_${MODE}_*_${OUT_SUFFIX}.log"
