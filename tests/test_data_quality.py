import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.data.augment import (AugmentConfig, balance_targets, layer_counts,
                                           parse_variants, plan_deficits, validate_variant)
from search_r1_refine.data.difficulty import bucket, length_score, multihop_score, score_difficulty
from search_r1_refine.data.quality import answer_is_complete, filter_rows, has_question_intent, quality_filter


def test_question_length_gate():
    assert quality_filter("Who wrote Hamlet?", ["Shakespeare"]).keep
    assert "question_too_short" in quality_filter("Hi?", ["x"]).reasons
    assert "question_too_long" in quality_filter("x" * 3000 + "?", ["x"]).reasons


def test_answer_completeness_gate():
    assert answer_is_complete(["Paris"])[0]
    assert "empty_answer" in answer_is_complete([])[1]
    assert "placeholder_answer" in answer_is_complete(["unknown"])[1]
    assert "answer_too_long" in answer_is_complete(["x" * 300])[1]


def test_special_character_gate():
    reasons = quality_filter("Who\x00 wrote Hamlet?", ["Shakespeare"]).reasons
    assert "control_character" in reasons
    assert "no_alphanumeric" in quality_filter("???!!!", ["x"]).reasons


def test_intent_detection():
    assert has_question_intent("Who wrote Hamlet?")
    assert has_question_intent("Name the capital of France")
    assert has_question_intent("法国的首都是哪里")
    assert not has_question_intent("The capital of France is Paris")
    assert "no_question_intent" in quality_filter("Paris is the capital of France.",
                                                  ["Paris"]).reasons


def test_filter_rows_reports_reasons():
    rows = [
        {"question": "Who wrote Hamlet?", "answers": ["Shakespeare"]},
        {"question": "This is a statement.", "answers": ["x"]},
    ]
    kept, rejected = filter_rows(rows)
    assert len(kept) == 1 and len(rejected) == 1
    assert "no_question_intent" in rejected[0]["filter_reasons"]


def test_difficulty_dimensions():
    assert length_score("x" * 100)[0] == 2
    assert length_score("x" * 50)[0] == 1
    assert length_score("short")[0] == 0
    assert multihop_score("Who directed the film whose lead actor was born in Paris?")[0] >= 1
    score, reasons = score_difficulty({"question": "In 1999, who wrote the book that was not adapted?"})
    assert score >= 2
    assert "time_constraint" in reasons and "negation" in reasons
    assert bucket(0) == "easy" and bucket(2) == "medium" and bucket(9) == "hard"


def test_difficulty_without_retrieval_measurement():
    score, reasons = score_difficulty({"question": "Who wrote Hamlet?"}, retrieval_hit=None)
    assert "retrieval_miss" not in reasons
    score_miss, reasons_miss = score_difficulty({"question": "Who wrote Hamlet?"}, retrieval_hit=0)
    assert score_miss == score + 2 and "retrieval_miss" in reasons_miss


def _rows(source, layers):
    rows = []
    for layer, count in layers.items():
        for i in range(count):
            rows.append({"source": source, "difficulty": layer, "question": f"q{i}?"})
    return rows


def test_layer_planning_only_fills_deficits():
    rows = _rows("nq", {"easy": 10, "medium": 4, "hard": 6})
    counts = layer_counts(rows)
    assert counts[("nq", "easy")] == 10
    targets = balance_targets(rows, "mean")
    assert targets["nq"] == round((10 + 4 + 6) / 3)
    deficits = plan_deficits(rows, "mean")
    assert ("nq", "easy") not in deficits          # already above target
    assert deficits[("nq", "medium")] == 7 - 4
    assert deficits[("nq", "hard")] == 7 - 6


def test_no_deficit_when_layers_balanced():
    rows = _rows("hotpotqa", {"easy": 5, "medium": 5, "hard": 5})
    assert plan_deficits(rows, "mean") == {}


def test_variant_parsing_and_validation():
    assert parse_variants('{"variants": ["a", "b"]}', 2) == ["a", "b"]
    assert parse_variants("garbage", 2) == []
    assert parse_variants('prefix {"variants": ["a"]} suffix', 1) == ["a"]
    original = {"question": "Who wrote Hamlet in 1601?"}
    assert validate_variant(original, "In 1601, who wrote Hamlet, if you know?")
    assert not validate_variant(original, "Who wrote Hamlet?")          # drops the year
    assert not validate_variant(original, "Who wrote Hamlet in 1601?")  # identical


def test_augment_config_defaults():
    config = AugmentConfig()
    assert config.model == "deepseek-4.1-flash"
    assert config.balance_target == "mean"
