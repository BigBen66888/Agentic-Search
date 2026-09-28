# 实验记录与复现凭据

README 的结果表为**实测**值，评测集为 NQ `test` 与 HotpotQA `dev`。为让后续复跑可核对，运行结束后请将以下产物与该次 Git commit 对应保存：

1. 环境：日期、Git commit、Python / PyTorch / CUDA / veRL / vLLM 版本、训练机和检索机硬件。
2. 数据：`download_manifest.json`、`processed/data_stats.json`、QA split 与样本数、语料来源与 revision。
3. 检索：`indices/index_stats.json`、BM25 backend、FAISS nlist / PQ / nprobe、`/health` 返回值。
4. 训练：基模与教师模型 revision、SFT 采样和保留条数、GRPO 步数、随机种子、训练日志。
5. 评测：每组 `artifacts/eval/<variant>/eval_traces.jsonl` 与 `eval_summary.json`、`artifacts/report/comparison_report.md`，以及失败案例。
6. 绘图：由实际训练日志生成曲线，标明 step、指标定义、滑动平均窗口与随机种子。

复跑时不要合并不同 commit、不同评测集或不同索引的 `eval_summary.json`。`compare_models.py` 可合并已有汇总，换实验配置时应使用新的数据目录或加 `--no-merge`。
