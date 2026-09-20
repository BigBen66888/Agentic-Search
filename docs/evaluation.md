# Online evaluation

Agent 服务需要提供：

    POST /generate
    {"prompt": "..."}
    -> {"text": "<think>...</think><search>...</search><answer>...</answer>"}

运行：

    python scripts/evaluate_online.py --data-dir /path/to/search_r1_data \
      --agent-url http://127.0.0.1:9000/generate \
      --retriever-url http://127.0.0.1:8000/retrieve

评测保存每条 action trace，并按 source 计算答案、证据、格式、效率、token 和 latency。

