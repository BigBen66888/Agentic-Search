# veRL 接入说明（4×4090）

本仓库不复制 veRL/Search-R1 训练框架。训练入口同时使用 Search-R1 的 `retriever.url`、`rollout.n_agent` 与 veRL 的 `custom_reward_function`。仓库没有提供已固定版本的同时支持这三个接口的 fork；相邻旧版 Search-R1 副本没有 `custom_reward_function`。因此安装普通 `verl` 或未经适配的 Search-R1 均不能直接证明二值/多级奖励生效。完整 GPU 训练须先验证接口兼容性和奖励输出，并记录实际 fork commit。

## 一、接入点

### 1. Reward Manager

使用 upstream 的 `verl.trainer.main_ppo` + `custom_reward_function` 接口：

    custom_reward_function.path=<repo>/scripts/verl_custom_reward.py
    custom_reward_function.name=compute_score

`compute_score` 内部委托给 `search_r1_refine.rl.verl_adapter.SearchRewardAdapter`。
奖励模式由环境变量 `SEARCH_REWARD_MODE` 决定，该变量由 `scripts/train_grpo.py`
按 `--reward-mode` 注入：

- `binary`：B1，答案 EM 的 0/1 奖励
- `multi`：B2，五级加权奖励

`ground_truth` 需包含：

    {"target": ["gold answer"], "supporting_facts": [...]}

### 2. GRPO 优势估计

以下增强优势估计器是本仓库提供的适配器。当前 `train_grpo.py` 没有把它接入外部 veRL 的 `compute_advantage`；若使用它，须在训练框架中显式调用并记录补丁：

```python
from search_r1_refine.rl.verl_advantage_adapter import compute_search_grpo_advantage

advantages = compute_search_grpo_advantage(
    rewards=sequence_rewards,
    group_ids=prompt_group_ids,
    config={"success_weight": 1.5, "failure_weight": 0.6,
            "zscore_epsilon": 1e-6, "clip_sigma": 5.0},
)
```

未接入此适配器时，训练框架使用自身的 GRPO 优势估计；不得将轨迹加权与 ±5σ 裁剪描述为该次训练已经生效。

## 二、4×4090 参数

`scripts/train_grpo.py` 生成的命令已包含：

    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    actor_rollout_ref.actor.use_kl_loss=true
    actor_rollout_ref.actor.state_masking=true
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

采样数默认用 Search-R1 fork 的 `actor_rollout_ref.rollout.n_agent`；
若使用 upstream veRL，加 `--upstream-verl` 改用 `actor_rollout_ref.rollout.n`。

## 三、执行顺序

```bash
python scripts/train_grpo.py --data-dir /data/search_r1 --n-gpus 4 --reward-mode multi --dry-run
python scripts/train_grpo.py --data-dir /data/search_r1 --n-gpus 4 --reward-mode multi --execute
```

B1 与 B2 各跑一次，不要混在同一个 run 里。首次先 `--max-steps 50` 短跑。

## 四、最低验收

1. `torch.cuda.device_count()` 为 4，型号为 4090
2. 检索机 `/health` 返回 `ok`，`ways` 包含 `dense` 与 `bm25`
3. `processed/train_pool.parquet` 与 `processed/eval.parquet` 存在
4. B1 短跑：奖励能落到 response 末 token
5. B2 短跑：无 NaN，记录 reward breakdown、KL、advantage 均值/标准差/最大值
6. 最终四个变体用同一 RRF、同一 test、同一评测脚本
