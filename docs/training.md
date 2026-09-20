# Training notes

本仓库不复制整个 upstream veRL。训练前需要：

1. 安装与参考 Search-R1 兼容的 veRL/Ray/vLLM 环境。
2. 准备 data/processed/train.jsonl 和 eval.jsonl。
3. 启动 scripts/serve_retriever.py。
4. 用 scripts/train_grpo.py --reward-mode binary 跑 B1，再用 --reward-mode multi 跑 B2。
5. 将 search_r1_refine.rl 的 reward/advantage adapter 接入 upstream veRL 的实际 reward path。
6. 先用 5%–10% 数据短跑，检查 NaN、KL、reward breakdown、EM/F1 和格式率。

B0 不需要训练：使用同一检索服务和 base model 直接在线评测。

