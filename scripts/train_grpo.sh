#!/usr/bin/env bash
# 4x RTX 4090 上的 GRPO 训练入口（默认从 SFT checkpoint 启动）
set -euo pipefail
DATA_DIR="${1:?usage: bash scripts/train_grpo.sh /path/to/data}"
MODEL="${MODEL:-}"
N_GPUS="${N_GPUS:-4}"
NNODES="${NNODES:-1}"
REWARD_MODE="${REWARD_MODE:-multi}"
MAX_STEPS="${MAX_STEPS:-}"
PYTHON="${PYTHON:-python}"
ARGS=(--data-dir "$DATA_DIR" --reward-mode "$REWARD_MODE" --n-gpus "$N_GPUS" --nnodes "$NNODES")
if [[ -n "$MODEL" ]]; then ARGS+=(--model "$MODEL"); fi
if [[ -n "$MAX_STEPS" ]]; then ARGS+=(--max-steps "$MAX_STEPS"); fi
echo "[Search-R1] 即将启动 veRL 多卡训练：GPUs=$N_GPUS, nodes=$NNODES, reward=$REWARD_MODE"
"$PYTHON" scripts/train_grpo.py "${ARGS[@]}" --execute
