#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:?usage: run_verl_train.sh DATA_DIR MODEL [N_GPUS]}"
MODEL="${2:?usage: run_verl_train.sh DATA_DIR MODEL [N_GPUS]}"
N_GPUS="${3:-1}"
export VERL_ATTN_IMPL="${VERL_ATTN_IMPL:-sdpa}"
export VERL_VLLM_PREFIX_CACHING="${VERL_VLLM_PREFIX_CACHING:-0}"
export SEARCH_REWARD_MODE="${SEARCH_REWARD_MODE:-multi}"
python -m verl.trainer.main_ppo \
  "data.train_files=${DATA_DIR}/processed/train.parquet" \
  "data.val_files=${DATA_DIR}/processed/eval.parquet" \
  algorithm.adv_estimator=grpo \
  "actor_rollout_ref.model.path=${MODEL}" \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.enable_prefix_caching=false \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  "custom_reward_function.path=$(pwd)/scripts/verl_custom_reward.py" \
  custom_reward_function.name=compute_score \
  "trainer.n_gpus_per_node=${N_GPUS}" \
  trainer.nnodes=1
