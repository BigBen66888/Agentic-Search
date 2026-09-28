import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.data.prepare import answer_rate, prepare, split_is_usable


class Args:
    def __init__(self, root, sources, output_dir):
        self.dataset_name = "mock"
        self.sources = sources
        self.data_dir = root
        self.output_dir = output_dir
        self.eval_split = "test"
        self.eval_splits = {}
        self.min_answer_rate = 0.9
        self.min_rows = 10
        self.min_chars = 5
        self.max_chars = 2000
        self.require_intent = True
        self.allow_dev_fallback = False
        self.augment = False
        self.easy_max = 1
        self.medium_max = 3


def _write(root, source, split, rows):
    directory = os.path.join(root, "qa", source)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_split_is_usable():
    good = [{"question": "q", "golden_answers": ["a"]}] * 12
    assert split_is_usable(good)[0] is True
    assert split_is_usable([{"question": "q", "golden_answers": []}] * 12)[1].startswith("answers_missing")
    assert split_is_usable(good[:3])[1].startswith("too_few_rows")
    assert split_is_usable(None)[1] == "split_missing"


def test_answer_rate():
    assert answer_rate([{"golden_answers": ["a"]}] * 10) == 1.0
    assert answer_rate([{"golden_answers": []}] * 10) == 0.0


def test_prepare_keeps_only_official_test_with_answers():
    root = tempfile.mkdtemp(prefix="sr1_prepare_")
    try:
        _write(root, "nq", "test", [{"question": "Who wrote Hamlet?", "golden_answers": ["Shakespeare"]}] * 12)
        _write(root, "nq", "train", [{"question": "train q?", "golden_answers": ["a"]}] * 5)
        # hotpotqa resolves to dev by default; an answer-free dev must be dropped
        _write(root, "hotpotqa", "dev", [{"question": "Who is X?", "golden_answers": []}] * 12)
        _write(root, "popqa", "dev", [{"question": "Who is Y?", "golden_answers": ["a"]}] * 12)

        out = os.path.join(root, "processed")
        stats = prepare(Args(root, ["nq", "hotpotqa", "popqa"], out))

        assert stats["eval_sources"] == ["nq"]
        assert stats["dropped_sources"]["hotpotqa"].startswith("answers_missing")
        assert stats["dropped_sources"]["popqa"].startswith("split_missing")
        assert stats["train_pool_rows"] == 5
        rows = [json.loads(x) for x in open(os.path.join(out, "eval.jsonl"), encoding="utf-8")]
        assert all(r["split"] == "test" for r in rows)
        assert rows[0]["prompt"][0]["role"] == "user"
        assert "search" in rows[0]["prompt_text"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prepare_dev_fallback_is_opt_in():
    root = tempfile.mkdtemp(prefix="sr1_fallback_")
    try:
        _write(root, "popqa", "dev", [{"question": "Who is Y?", "golden_answers": ["a"]}] * 12)
        args = Args(root, ["popqa"], os.path.join(root, "processed"))
        assert prepare(args)["eval_sources"] == []
        args.allow_dev_fallback = True
        assert prepare(args)["eval_sources"] == ["popqa"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_per_source_eval_split_override():
    root = tempfile.mkdtemp(prefix="sr1_splits_")
    try:
        _write(root, "hotpotqa", "test", [{"question": "Who is X?", "golden_answers": ["x"]}] * 12)
        args = Args(root, ["hotpotqa"], os.path.join(root, "processed"))
        args.eval_splits = {"hotpotqa": "test"}
        stats = prepare(args)
        assert stats["eval_sources"] == ["hotpotqa"]
        assert stats["by_source"]["hotpotqa"]["split"] == "test"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_difficulty_layers_recorded():
    root = tempfile.mkdtemp(prefix="sr1_layers_")
    try:
        rows = [{"question": "Who wrote Hamlet?", "golden_answers": ["Shakespeare"]}] * 12
        _write(root, "nq", "test", rows)
        _write(root, "nq", "train", rows[:6])
        stats = prepare(Args(root, ["nq"], os.path.join(root, "processed")))
        assert sum(stats["difficulty"].values()) == stats["train_pool_rows"]
        assert set(stats["difficulty"]) == {"easy", "medium", "hard"}
    finally:
        shutil.rmtree(root, ignore_errors=True)
