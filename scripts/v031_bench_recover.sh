#!/usr/bin/env bash
# v031_bench_recover.sh — recover the v0.31 bench after the gated-Llama failure.
#
# State: step 1 (corpus extension) shipped 2000 entries to local disk.
# Step 2 first attacker (Qwen2.5-32B-Instruct) PAIR shipped pair_Qwen_Qwen2.5-32B-Instruct.json.
# Step 2 second attacker (meta-llama/Llama-3.3-70B-Instruct) FAILED at vLLM load — gated HF repo,
# no auth on droplet. Skipping it; substituting Qwen2.5-72B-Instruct as the second (and final)
# attacker since 72B weights are already cached on the droplet from step 1.
#
# Net: bench ships with two attackers (Qwen2.5-32B + Qwen2.5-72B). Llama is a follow-up once
# HF_TOKEN is on the droplet.
#
# Survivability: tmux session "vaara-bench-v2", trap on EXIT does final rsync + auto-shutdown
# when rsync succeeded (matching the canonical orchestrator pattern).

set -euo pipefail

DROPLET="root@134.199.203.161"
DROPLET_IP="${DROPLET##*@}"
SESSION="vaara-bench-v2"
ENDPOINT="http://${DROPLET_IP}:8000"
ATTACKER="Qwen/Qwen2.5-72B-Instruct"
OUT_ROOT="tests/adversarial/v031"

if [[ -z "${TMUX:-}" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[wrap] session '${SESSION}' already exists. attach with: tmux attach -t ${SESSION}"
        exit 1
    fi
    echo "[wrap] launching detached in tmux session '${SESSION}'"
    tmux new-session -d -s "$SESSION" "bash $0 $*"
    echo "[wrap] attach with: tmux attach -t ${SESSION}"
    exit 0
fi

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/recover.log"
exec > >(tee -a "$LOG") 2>&1

ts() { date -u +%FT%TZ; }
note() { echo "[$(ts)] $*"; }

final_rsync() {
    note "trap: final rsync of droplet state"
    local rsync_ok=1
    rsync -avz "${DROPLET}:/root/vllm.log" "$OUT_ROOT/vllm-recover-final.log" || rsync_ok=0
    rsync -avz "${DROPLET}:/root/" "$OUT_ROOT/droplet-home/" \
        --include='*.log' --include='*.json' --include='*.jsonl' \
        --include='*/' --exclude='*' || rsync_ok=0
    if [[ $rsync_ok -eq 1 ]]; then
        note "rsync ok; scheduling droplet shutdown in 5 min (pre-authorized)"
        ssh "$DROPLET" "shutdown -h +5" || note "shutdown command failed; run manually"
    else
        note "rsync failed; droplet left running for manual recovery"
        note "  re-run: rsync -avz '${DROPLET}:/root/' '$OUT_ROOT/droplet-home/' --include='*.log' --include='*.json' --include='*.jsonl' --include='*/' --exclude='*'"
        note "  then:   ssh ${DROPLET} 'shutdown -h +1'"
    fi
}
trap final_rsync EXIT

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

# ---------- step 2 second attacker (substitute) ----------
note "step 2 recovery: PAIR with $ATTACKER (substitute for gated Llama-3.3-70B-Instruct)"
vllm_up "$ATTACKER"
safe="${ATTACKER//\//_}"
.venv/bin/python scripts/eval_pair_attack.py \
    --endpoint "$ENDPOINT" \
    --model "$ATTACKER" \
    --max-iters 5 \
    --out "$OUT_ROOT/pair_${safe}.json"
rsync -avz "${DROPLET}:/root/vllm.log" "$OUT_ROOT/vllm-pair-${safe}.log"

# ---------- step 3 classifier retrain (local CPU) ----------
note "step 3: classifier retrain on extended corpus"
.venv/bin/python scripts/train_adversarial_classifier.py \
    --json-out "$OUT_ROOT/classifier_v031.json"

# ---------- step 4 distribution-shift eval (local) ----------
note "step 4: distribution-shift eval"
.venv/bin/python scripts/eval_distribution_shift.py \
    --corpus-root tests/adversarial \
    --out "$OUT_ROOT/distribution_shift_v031.json"

# ---------- step 5 adversarial corpus end-to-end re-eval (local) ----------
note "step 5: adversarial corpus end-to-end eval (with classifier)"
.venv/bin/python scripts/eval_adversarial.py \
    --corpus-dir tests/adversarial \
    --with-classifier \
    --out "$OUT_ROOT/adversarial_v031.json"

note "done. artefacts:"
ls -la "$OUT_ROOT"

# Trap handles final rsync + shutdown.
