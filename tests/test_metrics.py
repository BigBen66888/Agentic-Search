import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.evaluation.metrics import exact_match, last_answer, score_row, summarize
from search_r1_refine.evaluation.rank import gold_titles, merge_rankings, mrr_at_k, ndcg_at_k, recall_at_k


def test_exact_match_and_f1():
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("Paris.", ["paris"]) == 1.0
    assert exact_match("London", ["paris"]) == 0.0
    assert exact_match("", ["paris"]) == 0.0


def test_last_answer_takes_final_block():
    text = "<answer>first</answer><think>x</think><answer>second</answer>"
    assert last_answer(text) == "second"


def test_rank_metrics():
    gold = {"paris"}
    assert recall_at_k(["paris", "rome"], gold, 10) == 1.0
    assert recall_at_k(["rome"], gold, 10) == 0.0
    assert mrr_at_k(["rome", "paris"], gold, 10) == 0.5
    assert mrr_at_k(["rome"], gold, 10) == 0.0
    assert ndcg_at_k(["paris"], gold, 10) == 1.0
    assert 0 < ndcg_at_k(["rome", "paris"], gold, 10) < 1.0
    assert recall_at_k(["x"], set(), 10) != recall_at_k(["x"], set(), 10)  # nan when no gold


def test_gold_titles_from_supporting_facts():
    facts = [{"title": "Paris"}, {"title": "France"}]
    assert gold_titles(facts) == {"paris", "france"}


def test_merge_rankings_keeps_best_rank():
    merged = merge_rankings([["a", "b"], ["b", "c"]])
    assert merged[0] == "a"


def test_score_row_and_summarize():
    row = {
        "source": "nq",
        "question": "Who wrote Hamlet?",
        "answers": ["Shakespeare"],
        "text": "<think>t</think><search>hamlet author</search>"
                "<information>Paris is capital</information><answer>Shakespeare</answer>",
        "retrieved_titles": ["Hamlet", "Paris"],
        "supporting_facts": [{"title": "Hamlet"}],
        "search_count": 1,
        "latency_ms": 10,
        "tokens": 20,
    }
    scored = score_row(row)
    assert scored["em"] == 1.0 and scored["answer_f1"] == 1.0
    assert scored["recall@10"] == 1.0 and scored["mrr@10"] == 1.0
    summary = summarize([row])
    assert summary["count"] == 1
    assert summary["overall"]["em"] == 1.0
    assert "nq" in summary["by_source"]
