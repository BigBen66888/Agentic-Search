import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.rl.advantage import DIFFICULTY_WEIGHTS, trajectory_weight


def test_trajectory_weight_by_difficulty():
    assert trajectory_weight({"difficulty": "hard"}) == DIFFICULTY_WEIGHTS["hard"]
    assert trajectory_weight({"difficulty": "easy"}) == DIFFICULTY_WEIGHTS["easy"]
    assert trajectory_weight({}) == 1.0
    assert trajectory_weight(None) == 1.0


def test_grouped_advantage_math():
    import pytest

    torch = pytest.importorskip("torch")
    from search_r1_refine.rl.advantage import grouped_advantages

    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = grouped_advantages(rewards, ["a", "a", "b", "b"])
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) <= 5.0
    assert float(out[0]) > float(out[1])  # success beats failure inside a group


def test_grouped_advantage_clipping():
    import pytest

    torch = pytest.importorskip("torch")
    from search_r1_refine.rl.advantage import grouped_advantages

    out = grouped_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ["g"] * 6, clip_sigma=1.0)
    assert float(out.abs().max()) <= 1.0 + 1e-6


def test_grouped_advantage_with_trajectory_weights():
    import pytest

    torch = pytest.importorskip("torch")
    from search_r1_refine.rl.advantage import grouped_advantages

    rewards = torch.tensor([1.0, 1.0])
    plain = grouped_advantages(rewards, ["g", "g"])
    weighted = grouped_advantages(rewards, ["g", "g"], weights=[1.0, 2.0])
    assert not torch.allclose(plain, weighted)


def test_zero_variance_group_is_neutral():
    import pytest

    torch = pytest.importorskip("torch")
    from search_r1_refine.rl.advantage import grouped_advantages

    out = grouped_advantages(torch.tensor([0.5, 0.5]), ["g", "g"])
    assert float(out.abs().max()) == 0.0
