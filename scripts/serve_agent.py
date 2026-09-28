#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Serve a policy as a search agent for evaluation.

    python scripts/serve_agent.py --model Qwen/Qwen2.5-3B \
        --retriever-url http://<retrieval-box>:8000/retrieve --port 9000

Exposes ``POST /generate`` with ``{"prompt": ...}`` and returns the full
multi-turn trajectory text, so every variant (base / SFT / SFT+GRPO / reference
model) is evaluated through exactly the same interface.
"""
import argparse
import time

from fastapi import FastAPI
from pydantic import BaseModel

from search_r1_refine.evaluation.agent_loop import run_local_episode
from search_r1_refine.sft.teacher import RetrieverClient

app = FastAPI(title="Search-R1 Refine Agent")
STATE = {}


class GenerateRequest(BaseModel):
    prompt: str
    max_turns: int = None
    temperature: float = None


@app.get("/health")
def health():
    return {"status": "ok" if STATE.get("llm") else "not_ready", "model": STATE.get("model")}


@app.post("/generate")
def generate(req: GenerateRequest):
    started = time.perf_counter()
    result = run_local_episode(
        STATE["llm"], STATE["tokenizer"], STATE["retriever"], req.prompt,
        max_turns=req.max_turns or STATE["max_turns"],
        max_tokens=STATE["max_tokens"],
        temperature=STATE["temperature"] if req.temperature is None else req.temperature,
    )
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    STATE["model"] = args.model
    STATE["max_turns"] = args.max_turns
    STATE["max_tokens"] = args.max_tokens
    STATE["temperature"] = args.temperature
    STATE["tokenizer"] = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    STATE["retriever"] = RetrieverClient(url=args.retriever_url, topk=args.topk, max_chars=args.max_chars)
    STATE["llm"] = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
                       tensor_parallel_size=args.tensor_parallel_size,
                       max_model_len=args.max_model_len, trust_remote_code=True)
    print(f"[Search-R1] Agent 服务就绪 model={args.model} port={args.port}", flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
