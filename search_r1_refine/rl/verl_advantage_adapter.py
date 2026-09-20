"""Driver-side adapter for GRPO outcome advantages."""
from typing import Sequence
import torch
from .advantage import grouped_advantages

def compute_search_grpo_advantage(rewards: torch.Tensor, group_ids: Sequence[str], config=None):
    config = config or {}
    return grouped_advantages(
        rewards,
        group_ids,
        success_weight=float(config.get("success_weight", 1.5)),
        failure_weight=float(config.get("failure_weight", 0.6)),
        epsilon=float(config.get("zscore_epsilon", 1e-6)),
        clip_sigma=float(config.get("clip_sigma", 5.0)),
    )

