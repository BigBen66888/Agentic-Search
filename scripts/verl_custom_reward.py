"""Official veRL custom-reward hook for B1/B2 Search-R1 training.

``SEARCH_REWARD_MODE`` is set by scripts/train_grpo.py from ``--reward-mode``,
so binary (B1) and multi-level (B2) runs can no longer collapse into one.
"""
import os

from search_r1_refine.rl.verl_adapter import SearchRewardAdapter

_ADAPTER = None


def _adapter():
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = SearchRewardAdapter(
            mode=os.environ.get("SEARCH_REWARD_MODE", "multi"),
            max_turns=int(os.environ.get("SEARCH_REWARD_MAX_TURNS", "4")),
        )
    return _ADAPTER


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    if isinstance(ground_truth, dict):
        target = ground_truth
    elif isinstance(ground_truth, (list, tuple)):
        target = {"target": list(ground_truth)}
    else:
        target = {"target": [str(ground_truth)] if ground_truth else []}
    return _adapter().score_text(solution_str, target)["total"]
