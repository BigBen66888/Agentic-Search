import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.rl.fifo import (SuccessFIFO, build_mixed_pool,
                                      trajectory_to_prompt_row)


def _trajectory(i, reward=0.9, success=True):
    return {
        "sample_id": f"s{i}",
        "source": "nq",
        "question": f"question {i}?",
        "answers": [f"answer {i}"],
        "prompt_text": f"question {i}?",
        "reward_total": reward,
        "success": success,
        "messages": [{"role": "user", "content": f"question {i}?"},
                     {"role": "assistant", "content": "<answer>answer</answer>"}],
    }


def test_fifo_only_stores_successes():
    fifo = SuccessFIFO(max_size=10)
    assert fifo.add(_trajectory(1, success=True)) is True
    assert fifo.add(_trajectory(2, success=False)) is False
    assert len(fifo) == 1


def test_fifo_infers_success_from_reward():
    fifo = SuccessFIFO()
    fifo.add({"reward_total": 0.0})
    fifo.add({"reward_total": 0.5})
    assert len(fifo) == 1


def test_fifo_is_bounded():
    fifo = SuccessFIFO(max_size=3)
    for i in range(10):
        fifo.add(_trajectory(i))
    assert len(fifo) == 3
    assert [t["sample_id"] for t in fifo.sample(10)] == ["s7", "s8", "s9"]


def test_fifo_round_trip():
    root = tempfile.mkdtemp(prefix="sr1_fifo_")
    try:
        path = os.path.join(root, "fifo.jsonl")
        fifo = SuccessFIFO()
        for i in range(5):
            fifo.add(_trajectory(i))
        fifo.save(path)
        reloaded = SuccessFIFO.load(path)
        assert len(reloaded) == 5
        assert reloaded.sample(1)[0]["sample_id"] == "s4"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_build_mixed_pool_replaces_ratio():
    pool = [{"sample_id": f"p{i}", "question": f"q{i}?", "answers": ["a"]} for i in range(100)]
    fifo = SuccessFIFO()
    for i in range(50):
        fifo.add(_trajectory(i))
    result = build_mixed_pool(pool, fifo, ratio=0.10, seed=0)
    assert result["pool_rows"] == 100
    assert result["replay_rows"] == 10
    assert len(result["rows"]) == 100
    replays = [r for r in result["rows"] if (r.get("metadata") or {}).get("replay")]
    assert len(replays) == 10
    assert all(r["sample_id"].endswith(":replay") for r in replays)


def test_build_mixed_pool_noop_without_fifo():
    pool = [{"sample_id": "p0", "question": "q?", "answers": ["a"]}]
    result = build_mixed_pool(pool, SuccessFIFO(), ratio=0.10)
    assert result["replay_rows"] == 0 and result["rows"] == pool


def test_trajectory_to_prompt_row_keeps_ground_truth():
    row = trajectory_to_prompt_row(_trajectory(7))
    assert row["answers"] == ["answer 7"]
    assert row["question"] == "question 7?"
    assert row["reward_model"]["ground_truth"]["target"] == ["answer 7"]
    assert row["metadata"]["replay"] is True
