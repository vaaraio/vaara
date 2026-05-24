#!/usr/bin/env bash
# v031_bench_orchestrate.sh — vaara-bench-v2 driver.
#
# Runs from Henri's home server. Wraps itself in tmux. Drives an MI300x
# droplet that serves vLLM on port 8000, one attacker model at a time.
# Local box runs the Vaara stack + PAIR + corpus extension orchestration
# against the droplet endpoint. Results land in tests/adversarial/v031/.
#
# Architecture:
#   droplet -> vLLM in tmux session "vllm", one model at a time
#   local   -> this script, in tmux session "vaara-bench-v2"
#   network -> local curls droplet:8000 OpenAI-compatible API
#
# Survivability:
#   - self-wraps in tmux locally
#   - vLLM on droplet runs inside its own tmux session
#   - rsync vllm.log after every model swap (never lose more than one run)
#   - trap EXIT runs a final rsync + leaves droplet up so we can re-pull
#
# Usage:
#   ./scripts/v031_bench_orchestrate.sh
# Optional env overrides:
#   DROPLET=root@1.2.3.4
#   ATTACKERS="Qwen/Qwen2.5-32B-Instruct meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct"
#   GENERATOR="Qwen/Qwen2.5-72B-Instruct"
#   PAIR_MAX_ITERS=5
#   EXTEND_PER_STYLE=500

set -euo pipefail

# ---------- config ----------
DROPLET="${DROPLET:-root@134.199.203.161}"
DROPLET_IP="${DROPLET##*@}"
SESSION="${SESSION:-vaara-bench-v2}"
ATTACKERS="${ATTACKERS:-Qwen/Qwen2.5-32B-Instruct meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B-Instruct}"
GENERATOR="${GENERATOR:-Qwen/Qwen2.5-72B-Instruct}"
PAIR_MAX_ITERS="${PAIR_MAX_ITERS:-5}"
EXTEND_PER_STYLE="${EXTEND_PER_STYLE:-500}"

OUT_ROOT="tests/adversarial/v031"
ENDPOINT="http://${DROPLET_IP}:8000"

# ---------- tmux self-wrap ----------
if [[ -z "${TMUX:-}" ]]; then
    echo "[wrap] re-execing inside tmux session '${SESSION}'"
    exec tmux new-session -A -s "$SESSION" "$0" "$@"
fi

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/orchestrate.log"
exec > >(tee -a "$LOG") 2>&1

ts() { date -u +%FT%TZ; }
note() { echo "[$(ts)] $*"; }

# ---------- exit handler ----------
final_rsync() {
    note "trap: final rsync of droplet state"
    local rsync_ok=1
    rsync -avz "${DROPLET}:/root/vllm.log" "$OUT_ROOT/vllm-final.log" || rsync_ok=0
    rsync -avz "${DROPLET}:/root/" "$OUT_ROOT/droplet-home/" \
        --include='*.log' --include='*.json' --include='*.jsonl' \
        --include='*/' --exclude='*' || rsync_ok=0
    if [[ $rsync_ok -eq 1 ]]; then
        note "rsync ok; scheduling droplet shutdown in 5 min"
        ssh "$DROPLET" "shutdown -h +5" || note "shutdown command failed; run manually"
    else
        note "rsync failed; droplet left running for manual recovery"
        note "  re-run rsync or shutdown manually:"
        note "    rsync -avz '${DROPLET}:/root/' '$OUT_ROOT/droplet-home/' --include='*.log' --include='*.json' --include='*.jsonl' --include='*/' --exclude='*'"
        note "    ssh ${DROPLET} 'shutdown -h +1'"
    fi
}
trap final_rsync EXIT

# ---------- vLLM helpers ----------
vllm_up() {
    local model="$1"
    note "vllm: swap to $model"
    ssh -o StrictHostKeyChecking=accept-new "$DROPLET" "
        tmux kill-session -t vllm 2>/dev/null || true
        docker rm -f vllm 2>/dev/null || true
        tmux new-session -d -s vllm \"docker run --rm --name vllm \
            --device=/dev/kfd --device=/dev/dri --group-add video \
            --ipc=host --shm-size=16g \
            --network=host \
            -v /root/.cache/huggingface:/root/.cache/huggingface \
            rocm/vllm:latest python3 -m vllm.entrypoints.openai.api_server \
            --model $model --host 0.0.0.0 --port 8000 \
            --gpu-memory-utilization 0.92 --max-model-len 8192 \
            2>&1 | tee /root/vllm.log\"
    "
    note "vllm: waiting for $model to become ready"
    local deadline=$(( $(date +%s) + 3600 ))
    until curl -sf "${ENDPOINT}/v1/models" >/dev/null 2>&1; do
        if (( $(date +%s) > deadline )); then
            note "vllm: did not become ready within an hour, aborting"
            return 1
        fi
        sleep 10
    done
    note "vllm: ready: $model"
}

vllm_cache_purge() {
    note "vllm: purging HF cache on droplet to free disk"
    ssh "$DROPLET" "rm -rf /root/.cache/huggingface/hub/* 2>/dev/null || true"
}

# ---------- 1. Corpus extension via the frontier generator ----------
note "step 1/5: corpus extension with $GENERATOR"
vllm_up "$GENERATOR"

for style in roleplay hypothetical fakemode; do
    note "extend: jailbreak style=$style n=$EXTEND_PER_STYLE"
    python research/droplet_sync/research/e1_generate.py \
        --base-url "${ENDPOINT}/v1" \
        --model "$GENERATOR" \
        --style "$style" \
        --n "$EXTEND_PER_STYLE" \
        --out "tests/adversarial/generated/JB-${style}-v031.jsonl"
done

note "extend: benign read_file canonical-paths"
python research/droplet_sync/research/e2_generate.py \
    --base-url "${ENDPOINT}/v1" \
    --model "$GENERATOR" \
    --n "$EXTEND_PER_STYLE" \
    --out "tests/adversarial/benign_generated/BT-canonical-v031.jsonl"

rsync -avz "${DROPLET}:/root/vllm.log" "$OUT_ROOT/vllm-generator.log"
vllm_cache_purge

# ---------- 2. Multi-attacker PAIR ----------
note "step 2/5: multi-attacker PAIR"
for model in $ATTACKERS; do
    vllm_up "$model"
    safe="${model//\//_}"
    note "pair: attacker=$model"
    python scripts/eval_pair_attack.py \
        --endpoint "$ENDPOINT" \
        --model "$model" \
        --max-iters "$PAIR_MAX_ITERS" \
        --out "$OUT_ROOT/pair_${safe}.json"
    rsync -avz "${DROPLET}:/root/vllm.log" "$OUT_ROOT/vllm-pair-${safe}.log"
    vllm_cache_purge
done

# ---------- 3. Classifier retrain (local, CPU) ----------
note "step 3/5: classifier retrain on extended corpus"
python scripts/train_adversarial_classifier.py \
    --json-out "$OUT_ROOT/classifier_v031.json"

# ---------- 4. Distribution-shift eval (local) ----------
note "step 4/5: distribution-shift eval"
python scripts/eval_distribution_shift.py \
    --corpus-root tests/adversarial \
    --out "$OUT_ROOT/distribution_shift_v031.json"

# ---------- 5. Adversarial corpus end-to-end re-eval (local) ----------
note "step 5/5: adversarial corpus end-to-end eval (with classifier)"
python scripts/eval_adversarial.py \
    --corpus-dir tests/adversarial \
    --with-classifier \
    --out "$OUT_ROOT/adversarial_v031.json"

note "done. artefacts:"
ls -la "$OUT_ROOT"

# Trap handles final rsync. Droplet stays up for verification.
