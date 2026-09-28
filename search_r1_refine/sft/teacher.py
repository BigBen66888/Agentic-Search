"""Strong-teacher trajectory sampling for SFT cold start.

The teacher is any OpenAI-compatible chat endpoint (DeepSeek / Qwen / GPT /
a local vLLM server). It is driven through the same multi-turn search loop the
policy will later run, so the resulting trajectories are on-policy in shape:
``<think>`` -> ``<search>`` -> ``<information>`` -> ... -> ``<answer>``
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import requests

SYSTEM_PROMPT = (
    "You are a search agent. Reason step by step inside <think>...</think>. "
    "When you need external evidence, emit exactly one line <search>query</search> and stop. "
    "The system will return <information>...</information>. "
    "You may search at most a few times. "
    "Finish with <｜hy_place▁holder▁no▁14｜>short answer<｜hy_place▁holder▁no▁14｜> containing only the answer string."
)

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S | re.I)
PARTIAL_SEARCH_RE = re.compile(r"<search>(.*)$", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


class ChatClient:
    """Minimal OpenAI-compatible chat completion client."""

    def __init__(self, model: str, base_url: str = None, api_key: str = None,
                 temperature: float = 0.3, max_tokens: int = 512, timeout: int = 180):
        self.model = model
        self.base_url = (base_url or os.environ.get("LLM_API_BASE")
                         or "https://api.deepseek.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None,
             temperature: float = None, max_tokens: int = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.post(f"{self.base_url}/chat/completions",
                                     json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"empty completion: {data}")
        message = choices[0].get("message", {})
        return str(message.get("content", "") or "")


class RetrieverClient:
    """HTTP client for the two-way RRF retrieval service."""

    def __init__(self, url: str = "http://127.0.0.1:8000/retrieve", topk: int = 5,
                 max_chars: int = 1200, timeout: int = 60):
        self.url = url
        self.topk = topk
        self.max_chars = max_chars
        self.timeout = timeout
        self.session = requests.Session()

    def search_with_titles(self, query: str):
        payload = {"queries": [query], "topk": self.topk, "return_text": True, "max_chars": self.max_chars}
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        docs = (response.json() or {}).get("result", [[]])[0]
        if not docs:
            return "<information>No relevant passages found.</information>", []
        parts, titles = [], []
        for i, doc in enumerate(docs, 1):
            title = str(doc.get("title", ""))
            text = (doc.get("text") or "").strip().replace("\n", " ")
            if title:
                titles.append(title)
            parts.append(f"[{i}] {title}: {text}")
        return "<information>" + "\n".join(parts) + "</information>", titles

    def search(self, query: str) -> str:
        return self.search_with_titles(query)[0]


def _normalise_completion(text: str) -> str:
    """Cut generation at the first complete action tag."""
    match = SEARCH_RE.search(text)
    if match:
        return text[:match.end()]
    if "</answer>" in text.lower():
        idx = text.lower().rindex("</answer>")
        return text[:idx + len("</answer>")]
    return text


def run_teacher_episode(client: ChatClient, retriever: RetrieverClient, question: str,
                        max_turns: int = 4, temperature: float = 0.3) -> Dict:
    """Drive one multi-turn search episode and return the trajectory."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    turns, searches = [], []
    for turn in range(max_turns + 1):
        remaining = max_turns - len(searches)
        if remaining <= 0:
            messages.append({"role": "user", "content": "No more searches allowed. Answer now with <answer>...</answer>."})
        completion = _normalise_completion(
            client.chat(messages, stop=["</search>", "</answer>"], temperature=temperature)
        )
        match = SEARCH_RE.search(completion)
        if match and len(searches) < max_turns:
            query = match.group(1).strip()[:200]
            messages.append({"role": "assistant", "content": completion})
            information = retriever.search(query)
            messages.append({"role": "user", "content": information})
            searches.append(query)
            turns.append({"type": "search", "content": completion, "query": query, "information": information})
            continue
        partial = PARTIAL_SEARCH_RE.search(completion)
        if partial and not ANSWER_RE.search(completion) and len(searches) < max_turns:
            query = partial.group(1).strip()[:200]
            completion = completion[:partial.start()] + f"<search>{query}</search>"
            messages.append({"role": "assistant", "content": completion})
            information = retriever.search(query)
            messages.append({"role": "user", "content": information})
            searches.append(query)
            turns.append({"type": "search", "content": completion, "query": query, "information": information})
            continue
        if not ANSWER_RE.search(completion):
            completion = completion + "<answer>" + completion.strip().split("\n")[-1][:80] + "</answer>"
        messages.append({"role": "assistant", "content": completion})
        turns.append({"type": "answer", "content": completion})
        break

    # includes the <information> observations, which the evidence reward needs
    full_text = "\n".join(m["content"] for m in messages if m["role"] != "system")
    return {
        "messages": messages,
        "searches": searches,
        "search_count": len(searches),
        "turns": len(turns),
        "text": full_text,
    }


def trajectory_to_training_row(sample: Dict, trajectory: Dict, parts: Dict) -> Dict:
    """Keep only user/assistant turns; the system prompt is re-added at train time."""
    messages = [m for m in trajectory["messages"] if m["role"] in ("user", "assistant")]
    return {
        "sample_id": sample.get("sample_id"),
        "source": sample.get("source"),
        "question": sample.get("question"),
        "answers": sample.get("answers"),
        "messages": messages,
        "search_count": trajectory["search_count"],
        "searches": trajectory["searches"],
        "reward": parts,
        "reward_total": float(parts.get("total", 0.0)),
        "text": trajectory["text"],
        "system": SYSTEM_PROMPT,
    }
