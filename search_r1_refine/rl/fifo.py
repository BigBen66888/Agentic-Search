"""FIFO replay buffer of successful trajectories.

During GRPO a fraction of every training batch (default 10%) is drawn from
prompts that the policy has already solved. Without this, a question that was
answered once correctly may never reappear, and the behaviour silently decays.
"""
from __future__ import annotations

import json
import os
import random
import threading
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

DEFAULT_MAX_SIZE = 20000


class SuccessFIFO:
    """Thread-safe bounded queue; keeps the most recent successful episodes."""

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        self.max_size = int(max_size)
        self._items: Deque[Dict[str, Any]] = deque(maxlen=self.max_size)
        self._lock = threading.Lock()

    def add(self, trajectory: Dict[str, Any]) -> bool:
        if not trajectory:
            return False
        success = trajectory.get("success")
        if success is None:
            success = float(trajectory.get("reward_total", trajectory.get("reward", 0.0))) > 0
        if not success:
            return False
        with self._lock:
            self._items.append(dict(trajectory))
        return True

    def extend(self, trajectories: Iterable[Dict[str, Any]]) -> int:
        return sum(1 for t in trajectories if self.add(t))

    def sample(self, n: int) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._items)
        if not items or n <= 0:
            return []
        return items[-min(n, len(items)):]

    def sample_prompts(self, n: int, seed: int = 0) -> List[Dict[str, Any]]:
        """Uniform sample of stored prompts (not just the newest ones)."""
        with self._lock:
            items = list(self._items)
        if not items or n <= 0:
            return []
        rng = random.Random(seed)
        rng.shuffle(items)
        return items[:min(n, len(items))]

    def __len__(self) -> int:
        return len(self._items)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            items = list(self._items)
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str, max_size: int = DEFAULT_MAX_SIZE) -> "SuccessFIFO":
        buffer = cls(max_size)
        if not path or not os.path.exists(path):
            return buffer
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    buffer.add(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return buffer


def trajectory_to_prompt_row(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a stored episode back into a training prompt row."""
    messages = trajectory.get("messages") or []
    user_turns = [m for m in messages if m.get("role") == "user"]
    question = trajectory.get("question") or (user_turns[0].get("content", "") if user_turns else "")
    return {
        "sample_id": f"{trajectory.get('sample_id', 'fifo')}:replay",
        "source": trajectory.get("source", "fifo"),
        "split": "train",
        "question": question,
        "answers": trajectory.get("answers", []),
        "prompt_text": trajectory.get("prompt_text") or question,
        "prompt": [{"role": "user", "content": trajectory.get("prompt_text") or question}],
        "supporting_facts": trajectory.get("supporting_facts"),
        "reward_model": {"style": "rule",
                         "ground_truth": {"target": trajectory.get("answers", []),
                                          "supporting_facts": trajectory.get("supporting_facts")}},
        "extra_info": {"split": "train", "replay": True},
        "metadata": {"replay": True, "reward": trajectory.get("reward_total")},
    }


def build_mixed_pool(pool: Sequence[Dict[str, Any]], fifo: SuccessFIFO, ratio: float = 0.10,
                     seed: int = 0) -> Dict[str, Any]:
    """Replace ``ratio`` of the pool with prompts drawn from successful history.

    Returns ``{"rows": [...], "replay_rows": n, "pool_rows": n, "ratio": ratio}``.
    The result size equals the original pool size, so the RL step count is
    unchanged.
    """
    pool_rows = list(pool)
    if not pool_rows or ratio <= 0 or len(fifo) == 0:
        return {"rows": pool_rows, "replay_rows": 0, "pool_rows": len(pool_rows), "ratio": ratio}
    n_replay = min(int(round(len(pool_rows) * ratio)), len(fifo))
    if n_replay <= 0:
        return {"rows": pool_rows, "replay_rows": 0, "pool_rows": len(pool_rows), "ratio": ratio}
    rng = random.Random(seed)
    replay = [trajectory_to_prompt_row(t) for t in fifo.sample_prompts(n_replay, seed)]
    indices = list(range(len(pool_rows)))
    rng.shuffle(indices)
    for position, index in enumerate(indices[:n_replay]):
        pool_rows[index] = replay[position]
    return {"rows": pool_rows, "replay_rows": n_replay, "pool_rows": len(pool_rows), "ratio": ratio}
