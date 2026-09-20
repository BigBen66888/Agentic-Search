# Search-R1 Refine

面向 NQ、TriviaQA、PopQA、HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle 的多轮搜索 Agent 改造工程。

本仓库只保存代码、配置、测试和文档，不保存数据集、Wikipedia 语料、FAISS/BM25 索引、模型权重或密钥。设计目标是：上传 GitHub 后，在 Kaggle/公司服务器上按命令下载数据、处理数据、建立索引、启动检索服务、执行 GRPO 训练和 test 在线评测。

## 目录

    configs/                 数据、检索、RL、评测配置
    search_r1_refine/        可测试的 Python 包
      data/                  下载、过滤、改写、难度分层
      retrieval/             FAISS/BM25/RRF 和服务
      rl/                    多级奖励与 GRPO 优势估计
      evaluation/            在线评测和指标
    scripts/                 Kaggle/服务器入口脚本
    tests/                   单元测试和 smoke test
    docs/                    运行说明与实验记录模板

## 从零安装与运行

完整中文步骤见 docs/从零安装与运行.md；veRL 多卡接入见 docs/verl_integration.md。不要先执行下载或训练，先确认 NVIDIA 驱动和磁盘空间，再按 00-base、01-torch、02-retrieval、03-verl-training 顺序安装。

本项目的正式训练必须使用 veRL + Ray 多卡调度，PPO 不作为正式方案。训练入口会检查 torch、GPU 数量、veRL、Parquet 和索引，并持续打印阶段进度。当前默认 GPU 依赖按 CUDA 12.1 配置。

## 快速开始（Linux/Kaggle）

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements/00-base.txt
    pip install -r requirements/01-torch-cu121.txt
    pip install -r requirements/02-retrieval.txt
    pip install -r requirements/03-verl-training.txt
    pip install -e . --no-deps

    python scripts/download_data.py --data-dir /kaggle/working/search_r1_data --qa-only
    python scripts/prepare_data.py --data-dir /kaggle/working/search_r1_data

    # Wikipedia 语料和索引体积很大
    python scripts/download_data.py --data-dir /kaggle/working/search_r1_data --corpus
    python scripts/build_indices.py --data-dir /kaggle/working/search_r1_data --corpus /kaggle/working/search_r1_data/corpus/wiki-18/wiki-18.jsonl

    pytest -q
    python scripts/smoke_test.py

    # upstream veRL 环境准备好后，单机 8 卡
    N_GPUS=8 MODEL=Qwen/Qwen2.5-3B bash scripts/train_grpo.sh /kaggle/working/search_r1_data

## 数据和训练

训练处理的 7 个 source：nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle。
官方 test/dev 不改写、不扩增、不参与阈值调参。所有样本保存 source、原始 id、split、rewrite_type、difficulty_score、difficulty_reasons。

v0.3 的低成本正式对照只有三组：

1. B0：base model + RRF，无 RL。
2. B1：RRF + 原始 0/1 EM reward 的 GRPO。
3. B2：RRF + 多级 reward 的 GRPO。

B0 不训练；BM25/Dense/RRF 的检索比较不需要重复训练。search_r1_refine/rl 的奖励和优势估计是独立模块；docs/verl_integration.md 说明如何接入 upstream veRL，scripts/train_grpo.py 负责生成训练启动命令。数据处理同时输出 JSONL（审计/评测）和 Parquet（兼容参考 veRL 数据集读取器）。

所有下载脚本都要求显式 data-dir；仓库根目录不会生成大文件。图片中的 5% 过滤、2.5 倍扩展、99.8% Recall 等都是目标/待验证结果。
