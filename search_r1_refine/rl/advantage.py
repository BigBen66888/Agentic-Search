from collections import defaultdict, deque
from typing import Dict, List, Sequence
import torch

def grouped_advantages(rewards: torch.Tensor, group_ids: Sequence[str], *, success_weight=1.5, failure_weight=.6, epsilon=1e-6, clip_sigma=5.0):
    rewards = rewards.reshape(-1).float()
    groups = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[str(gid)].append(i)
    output = torch.zeros_like(rewards)
    for _, indices in groups.items():
        values = rewards[indices]
        values = values * torch.where(values > 0, success_weight, failure_weight)
        std = values.std(unbiased=False)
        z = (values - values.mean()) / (std + epsilon) if std > epsilon else torch.zeros_like(values)
        output[indices] = torch.clamp(z, -clip_sigma, clip_sigma)
    return output

class SuccessFIFO:
    def __init__(self, max_size=1000):
        self.items = deque(maxlen=max_size)
    def add(self, trajectory: Dict):
        if trajectory.get("success", False):
            self.items.append(trajectory)
    def sample(self, n: int) -> List[Dict]:
        return list(self.items)[-min(n, len(self.items)):] if self.items else []
    def __len__(self):
        return len(self.items)

