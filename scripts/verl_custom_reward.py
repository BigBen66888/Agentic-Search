"""Official veRL custom-reward hook for B1/B2 Search-R1 training."""
import os
from search_r1_refine.rl.verl_adapter import SearchRewardAdapter

def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    mode = os.environ.get("SEARCH_REWARD_MODE", "multi")
    if isinstance(ground_truth, dict):
        target = ground_truth
    elif isinstance(ground_truth, (list, tuple)):
        target = {"target": list(ground_truth)}
    else:
        target = {"target": [str(ground_truth)] if ground_truth else []}
    return SearchRewardAdapter(mode=mode).score_text(solution_str, target)["total"]
