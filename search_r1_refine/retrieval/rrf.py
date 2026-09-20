from collections import defaultdict
from typing import Dict, List, Sequence

def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[Dict]], *, k: float = 60.0, topk: int = 10, id_key: str = "doc_id") -> List[Dict]:
    scores = defaultdict(float)
    docs = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = str(doc[id_key])
            scores[doc_id] += 1.0 / (k + rank)
            docs.setdefault(doc_id, dict(doc))
    order = sorted(scores, key=lambda x: (-scores[x], x))
    output = []
    for doc_id in order[:topk]:
        item = dict(docs[doc_id])
        item["doc_id"] = doc_id
        item["rrf_score"] = scores[doc_id]
        output.append(item)
    return output

