from search_r1_refine.retrieval.rrf import reciprocal_rank_fusion

def test_rrf_deduplicates():
    out = reciprocal_rank_fusion([[{"doc_id":"a"},{"doc_id":"b"}],[{"doc_id":"b"},{"doc_id":"c"}]], topk=3)
    assert [x["doc_id"] for x in out] == ["b","a","c"]

def test_rrf_tie_is_deterministic():
    out = reciprocal_rank_fusion([[{"doc_id":"a"}],[{"doc_id":"b"}]], topk=2)
    assert [x["doc_id"] for x in out] == ["a","b"]

