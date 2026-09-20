# veRL + 多卡训练接入说明

本仓库的训练主框架是 veRL，不是自写 PPO/GRPO Trainer。search_r1_refine 只保存数据、检索、奖励和优势估计的改造模块；参考项目中的 veRL、Ray、vLLM 训练框架需要在 Kaggle 或公司服务器单独安装。

本项目的正式训练路径是：

- B1：veRL + GRPO + 原始 0/1 EM reward；
- B2：veRL + GRPO + 多级 reward + group 归一化/离群裁剪；
- PPO 不作为正式训练方案，不启动 critic 模型。

## 一、veRL 中需要接入的两个位置

### 1. Reward Manager

在参考 Search-R1 的 verl/trainer/main_ppo_format.py 中，原来选择 qa_em_format.compute_score_em 的位置，改为：

    from search_r1_refine.rl.verl_adapter import SearchRewardAdapter

    adapter = SearchRewardAdapter(
        tokenizer=tokenizer,
        mode=config.reward_model.mode,       # binary 或 multi
        max_turns=config.max_turns,
    )

在 RewardManager.__call__ 解码出 sequences_str 和 ground_truth 后：

    score = adapter(sequences_str, ground_truth)
    reward_tensor[i, valid_response_length - 1] = score

B1 使用 reward_model.mode=binary；B2 使用 reward_model.mode=multi。

ground_truth 需要包含：

    {
      "target": ["gold answer"],
      "supporting_facts": [...]
    }

### 2. GRPO Advantage

veRL 原有的 GRPO group 归一化可以先保留，完成 smoke 后再接入增强版本：

    from search_r1_refine.rl.verl_advantage_adapter import compute_search_grpo_advantage

    advantages = compute_search_grpo_advantage(
        rewards=sequence_rewards,
        group_ids=prompt_group_ids,
        config={
            "success_weight": 1.5,
            "failure_weight": 0.6,
            "zscore_epsilon": 1e-6,
            "clip_sigma": 5.0,
        },
    )

然后按 veRL 原有逻辑将 outcome advantage 扩展到 response token mask。请保留配置开关，异常时回退到原始 veRL GRPO。

## 二、多卡运行原则

训练由 Ray/veRL 负责多卡调度，不要用单卡 Python 进程假装多卡。启动前确认：

    nvidia-smi
    python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
    ray status

单机 8 卡推荐参数：

    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    actor_rollout_ref.rollout.n_agent=5
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.grad_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

多机时：

    trainer.n_gpus_per_node=<每台机器 GPU 数>
    trainer.nnodes=<机器数>

多机运行前必须让所有机器看到相同的数据目录、模型目录、代码 commit 和 Python 环境；Ray head/worker、NCCL、SSH 和防火墙属于服务器部署问题，先用单机多卡跑通。

## 三、训练入口

生成命令但不执行：

    python scripts/train_grpo.py \
      --data-dir /data/search_r1 \
      --model Qwen/Qwen2.5-3B \
      --reward-mode multi \
      --n-gpus 8 \
      --nnodes 1 \
      --dry-run

确认命令后执行：

    python scripts/train_grpo.py \
      --data-dir /data/search_r1 \
      --model Qwen/Qwen2.5-3B \
      --reward-mode multi \
      --n-gpus 8 \
      --nnodes 1 \
      --execute

脚本会在每个阶段打印 GPU、torch、Ray 环境检查，数据/索引文件检查，veRL 启动参数，日志目录和训练进程退出码。

B1 与 B2 分别执行一次，不要把两个 reward mode 混在同一个 run 中。

## 四、为什么不把 veRL 整个复制到本仓库

veRL、Ray、vLLM 和 CUDA 版本强绑定，直接复制会让 GitHub 仓库变得巨大且难以复现。本仓库只保留以下改造接口：

- search_r1_refine/rl/verl_adapter.py
- search_r1_refine/rl/verl_advantage_adapter.py
- 本文档中的 upstream 接入点

上传 GitHub 后，在服务器上 clone 本仓库和参考 Search-R1，再应用这两个适配器即可。

## 五、最小验收

在正式训练前必须完成：

1. 单机多卡 torch.cuda.device_count() 正确；
2. 检索服务 /health 返回 ok；
3. train.parquet 和 eval.parquet 存在；
4. B1 用 5%–10% 数据短跑，reward 能正常落在 response 末 token；
5. B2 短跑无 NaN，记录 reward breakdown、KL、advantage mean/std/max；
6. 最终 B0/B1/B2 使用同一 RRF、同一 test/dev 和同一评测脚本。

