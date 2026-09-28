"""Driver-side adapter for GRPO outcome advantages."""
from typing import Dict, List, Optional, Sequence
import torch
from .advantage import grouped_advantages, trajectory_weight


def compute_search_grpo_advantage(rewards: torch.Tensor, group_ids: Sequence[str], config=None,
                                  rows: Optional[Sequence[Dict]] = None) -> torch.Tensor:
    """Drop-in replacement for veRL's GRPO advantage.

    ``rows`` optionally carries the originating sample dicts so per-trajectory
    difficulty weights can be applied on top of the success/failure weighting.
    """
    config = config or {}
    weights: Optional[List[float]] = None
    if rows is not None and len(rows) == len(group_ids):
        weights = [trajectory_weight(row) for row in rows]
    return grouped_advantages(
        rewards,
        group_ids,
        success_weight=float(config.get("success_weight", 1.5)),
        failure_weight=float(config.get("failure_weight", 0.6)),
        epsilon=float(config.get("zscore_epsilon", 1e-6)),
        clip_sigma=float(config.get("clip_sigma", 5.0)),
        weights=weights,
    )
