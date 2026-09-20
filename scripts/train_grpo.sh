#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:?usage: bash scripts/train_grpo.sh /path/to/data}"
MODEL="${MODEL:?set MODEL=/path/to/base/model}"
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
REWARD_MODE="${REWARD_MODE:-multi}"
MAX_STEPS="${MAX_STEPS:-}"
ARGS=(--data-dir "$DATA_DIR" --model "$MODEL" --reward-mode "$REWARD_MODE" --n-gpus "$N_GPUS" --nnodes "$NNODES")
if [[ -n "$MAX_STEPS" ]]; then ARGS+=(--max-steps "$MAX_STEPS"); fi
echo "[Search-R1] 即将启动 veRL 多卡训练：GPUs=$N_GPUS, nodes=$NNODES, reward=$REWARD_MODE"
python scripts/train_grpo.py "${ARGS[@]}" --execute
