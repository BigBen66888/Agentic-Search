import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_r1_refine.evaluation.report import build_comparison, to_markdown, write_report


def _summary(em, f1, mrr):
    block = {"count": 10, "em": em, "answer_f1": f1, "judge": None,
             "recall@10": 0.4, "mrr@10": mrr, "ndcg@10": 0.4,
             "evidence": 0.5, "format": 1.0, "efficiency": 0.8,
             "avg_searches": 2.0, "avg_latency_ms": 100.0, "avg_tokens": 120.0}
    return {"count": 10, "overall": dict(block), "by_source": {"nq": dict(block)}}


def test_build_comparison_has_deltas():
    summaries = {"base": _summary(0.2, 0.3, 0.25), "sft": _summary(0.3, 0.42, 0.35)}
    comparison = build_comparison(summaries, baseline="base")
    assert comparison["variants"] == ["base", "sft"]
    assert comparison["overall"]["base"]["ΔEM"] == 0.0
    assert abs(comparison["overall"]["sft"]["ΔEM"] - 0.1) < 1e-6
    assert "nq" in comparison["by_source"]


def test_markdown_and_files():
    root = tempfile.mkdtemp(prefix="sr1_report_")
    try:
        summaries = {"base": _summary(0.2, 0.3, 0.25), "grpo": _summary(0.35, 0.45, 0.4)}
        comparison = build_comparison(summaries, baseline="base")
        markdown = to_markdown(comparison)
        assert "| variant |" in markdown and "grpo" in markdown
        paths = write_report(comparison, root)
        for path in paths.values():
            assert os.path.exists(path)
    finally:
        shutil.rmtree(root, ignore_errors=True)
