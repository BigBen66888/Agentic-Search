"""GRPO advantage estimation: trajectory weighting, group z-score, outlier clipping."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

# torch is only needed for the tensor maths; the weighting helpers stay usable
# on a CPU-only box where the training stack is not installed.
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    torch = None

DIFFICULTY_WEIGHTS = {"easy": 0.8, "medium": 1.0, "hard": 1.2}


def trajectory_weight(row: Optional[Dict] = None, difficulty_weights: Dict[str, float] = None) -> float:
    """Weight per trajectory: harder prompts contribute a stronger learning signal."""
    if not row:
        return 1.0
    table = difficulty_weights or DIFFICULTY_WEIGHTS
    return float(table.get(str(row.get("difficulty", "medium")), 1.0))


def grouped_advantages(rewards: torch.Tensor, group_ids: Sequence[str], *,
                       success_weight: float = 1.5, failure_weight: float = 0.6,
                       epsilon: float = 1e-6, clip_sigma: float = 5.0,
                       weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Group-normalised advantages.

    Steps per prompt group:
    1. trajectory weighting  - successes x1.5, failures x0.6 (times per-trajectory weight)
    2. z-score normalisation - subtract group mean, divide by group std
    3. outlier clipping       - clamp to +/- clip_sigma
    """
    if torch is None:
        raise ModuleNotFoundError("grouped_advantages requires torch")
    rewards = rewards.reshape(-1).float()
    groups = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[str(gid)].append(i)
    output = torch.zeros_like(rewards)
    for _, indices in groups.items():
        values = rewards[indices].clone()
        values = values * torch.where(values > 0, success_weight, failure_weight)
        if weights is not None:
            scale = torch.tensor([float(weights[i]) for i in indices], dtype=values.dtype, device=values.device)
            values = values * scale
        std = values.std(unbiased=False)
        z = (values - values.mean()) / (std + epsilon) if std > epsilon else torch.zeros_like(values)
        output[indices] = torch.clamp(z, -clip_sigma, clip_sigma)
    return output
