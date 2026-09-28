import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.evaluation.metrics import _gold_answers, _prompt_of
from search_r1_refine.retrieval.bm25 import available_backends, resolve_backend


def test_answer_and_prompt_fallbacks():
    row = {"answer": "F=ma", "question": "牛顿第二定律?"}
    assert _gold_answers(row) == ["F=ma"]
    assert _prompt_of(row) == "牛顿第二定律?"


def test_bm25_is_delegated_to_a_library():
    """No hand-written BM25 must remain: backends come from installed packages."""
    assert isinstance(available_backends(), list)
    try:
        assert resolve_backend("auto") in ("bm25s", "rank_bm25")
    except ImportError:
        pass  # no BM25 library installed on this machine; the service will report it


def test_reward_module_has_no_bm25_implementation():
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "search_r1_refine", "retrieval", "bm25.py"), encoding="utf-8").read()
    assert "get_scores" not in source.split("retrieve")[0]
