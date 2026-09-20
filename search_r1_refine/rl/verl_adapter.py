"""Small adapter used by the upstream veRL reward manager.

The upstream trainer owns tokenization and DataProto batching. This module only
owns the deterministic trajectory scoring, so it can be imported from the
reward manager without copying the whole trainer.
"""
from typing import Any, Dict, Iterable, List
from .reward import score_trajectory

class SearchRewardAdapter:
    def __init__(self, tokenizer=None, mode="multi", max_turns=4, weights=None):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_turns = max_turns
        self.weights = weights

    def score_text(self, text: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        return score_trajectory(
            text,
            ground_truth.get("target", ground_truth.get("answers", [])),
            ground_truth.get("supporting_facts"),
            mode=self.mode,
            max_turns=self.max_turns,
            weights=self.weights,
        )

    def __call__(self, solution_str: str, ground_truth: Dict[str, Any]) -> float:
        return float(self.score_text(solution_str, ground_truth)["total"])

