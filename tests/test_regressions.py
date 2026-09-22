from search_r1_refine.evaluation.metrics import _gold_answers, _prompt_of
from search_r1_refine.retrieval.build_index import bm25_tokenize

def test_answer_and_prompt_fallbacks():
    row = {"answer": "F=ma", "question": "牛顿第二定律?"}
    assert _gold_answers(row) == ["F=ma"]
    assert _prompt_of(row) == "牛顿第二定律?"

def test_cjk_bm25_tokenizer_has_unigrams_and_bigrams():
    tokens = bm25_tokenize("牛顿第二定律是什么")
    assert "牛" in tokens and "牛顿" in tokens and "定律" in tokens
