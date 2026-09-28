"""Multi-turn search loop driven by a local vLLM engine.

Kept separate from the teacher loop because the policy is served locally
(offline vLLM) rather than over an OpenAI-compatible HTTP API. The observable
behaviour is identical: think -> search -> information -> ... -> answer.
"""
from __future__ import annotations

import re
from typing import Dict, List

from search_r1_refine.sft.teacher import SYSTEM_PROMPT, RetrieverClient

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S | re.I)
PARTIAL_SEARCH_RE = re.compile(r"<search>(.*)$", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def run_local_episode(llm, tokenizer, retriever: RetrieverClient, question: str,
                      max_turns: int = 4, max_tokens: int = 512,
                      temperature: float = 0.0) -> Dict:
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    stop = ["</search>", "</answer>"]
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=stop)
    searches: List[str] = []
    pieces: List[str] = []
    retrieved_titles: List[str] = []

    for _ in range(max_turns + 1):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion = llm.generate([prompt], params)[0].outputs[0].text
        match = SEARCH_RE.search(completion)
        partial = PARTIAL_SEARCH_RE.search(completion)
        if match and len(searches) < max_turns:
            query = match.group(1).strip()[:200]
            completion = completion[:match.end()]
        elif partial and not ANSWER_RE.search(completion) and len(searches) < max_turns:
            query = partial.group(1).strip()[:200]
            completion = completion[:partial.start()] + f"<search>{query}</search>"
        else:
            if not ANSWER_RE.search(completion):
                completion = completion + "<answer>" + completion.strip().split("\n")[-1][:80] + "</answer>"
            messages.append({"role": "assistant", "content": completion})
            pieces.append(completion)
            break
        messages.append({"role": "assistant", "content": completion})
        pieces.append(completion)
        information, titles = retriever.search_with_titles(query)
        messages.append({"role": "user", "content": information})
        pieces.append(information)
        retrieved_titles.extend(titles)
        searches.append(query)

    return {"text": "\n".join(pieces), "searches": searches,
            "search_count": len(searches), "retrieved_titles": retrieved_titles}
