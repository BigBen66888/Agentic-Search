"""LLM-as-a-judge scoring.

Rule metrics (EM / F1) punish paraphrases that are actually correct. A judge
model scores semantic equivalence against the gold answer on a 1-5 scale which
is normalised to 0-1, so it can be averaged alongside the rule metrics.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from search_r1_refine.sft.teacher import ChatClient

JUDGE_PROMPT = """You are grading a short answer to a factual question.

Question: {question}
Gold answer(s): {gold}
Predicted answer: {prediction}

Rules:
- Ignore capitalization, punctuation, articles, and extra words that do not change the meaning.
- A prediction is correct if it names the same entity/value as any gold answer.
- Partial credit for a correct but incomplete answer.

Reply with JSON only: {{"score": <integer 1-5>, "reason": "<short>"}}
1 = wrong, 3 = partially correct, 5 = fully correct.
"""

SCORE_RE = re.compile(r'"score"\s*:\s*([1-5](?:\.\d+)?)')


class Judge:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None,
                 enabled: bool = True, temperature: float = 0.0, max_tokens: int = 128):
        self.enabled = enabled
        self.model = model or os.environ.get("JUDGE_MODEL") or os.environ.get("TEACHER_MODEL") or "deepseek-chat"
        self.client = ChatClient(model=self.model, base_url=base_url, api_key=api_key,
                                 temperature=temperature, max_tokens=max_tokens)

    def score(self, question: str, prediction: str, gold: List[str]) -> Optional[float]:
        if not self.enabled:
            return None
        prompt = JUDGE_PROMPT.format(question=question, gold=gold[:5], prediction=prediction)
        try:
            reply = self.client.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            print(f"[Search-R1] judge 调用失败：{exc}", flush=True)
            return None
        match = SCORE_RE.search(reply)
        if not match:
            return None
        value = float(match.group(1))
        return max(0.0, min(1.0, (value - 1) / 4.0))

    def score_batch(self, items: List[Dict]) -> List[Optional[float]]:
        return [self.score(i.get("question", ""), i.get("prediction", ""), i.get("answers", []))
                for i in items]
