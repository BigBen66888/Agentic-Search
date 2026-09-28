"""Comparison report across model variants."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence

METRICS = ["em", "answer_f1", "judge", "recall@10", "mrr@10", "ndcg@10",
           "evidence", "format", "efficiency", "avg_searches", "avg_latency_ms"]

PRETTY = {
    "em": "EM",
    "answer_f1": "F1",
    "judge": "Judge",
    "recall@10": "Recall@10",
    "mrr@10": "MRR@10",
    "ndcg@10": "NDCG@10",
    "evidence": "Evidence",
    "format": "Format",
    "efficiency": "Efficiency",
    "avg_searches": "Searches",
    "avg_latency_ms": "Latency(ms)",
}


def _value(block: Dict, key: str):
    value = block.get(key)
    return None if value is None else round(float(value), 4)


def build_comparison(summaries: Dict[str, Dict], baseline: str = "base") -> Dict:
    variants = list(summaries)
    sources = sorted({s for v in summaries.values() for s in v.get("by_source", {})})
    base_block = summaries.get(baseline, {}).get("overall", {})

    def row_for(block: Dict, base: Dict) -> Dict:
        row = {PRETTY.get(m, m): _value(block, m) for m in METRICS}
        row["n"] = block.get("count")
        for metric in ("em", "answer_f1", "mrr@10", "ndcg@10"):
            current, reference = block.get(metric), base.get(metric)
            if current is not None and reference is not None:
                row[f"Δ{PRETTY.get(metric, metric)}"] = round(float(current) - float(reference), 4)
        return row

    overall = {v: row_for(summaries[v].get("overall", {}), base_block) for v in variants}
    per_source = {}
    for source in sources:
        per_source[source] = {
            v: row_for(summaries[v].get("by_source", {}).get(source, {}),
                       summaries.get(baseline, {}).get("by_source", {}).get(source, {}))
            for v in variants
        }
    return {"variants": variants, "baseline": baseline, "overall": overall, "by_source": per_source}


def _table(rows: Dict[str, Dict]) -> List[str]:
    if not rows:
        return []
    columns = ["variant"] + [c for c in list(next(iter(rows.values())).keys()) if c != "variant"]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for name, values in rows.items():
        cells = [name]
        for column in columns[1:]:
            value = values.get(column)
            cells.append("" if value is None else f"{value}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_markdown(comparison: Dict) -> str:
    lines = ["# Search-R1 Refine 对比报告", ""]
    lines.append(f"基线变体：`{comparison['baseline']}`；Δ 列表示相对基线的绝对提升。")
    lines.append("")
    lines.append("## 整体（合并所有保留数据集）")
    lines.append("")
    overall = {name: dict(values, variant=name) for name, values in comparison["overall"].items()}
    lines.extend(_table(overall))
    lines.append("")
    for source, rows in comparison["by_source"].items():
        lines.append(f"## 数据集：{source}")
        lines.append("")
        lines.extend(_table({name: dict(values, variant=name) for name, values in rows.items()}))
        lines.append("")
    return "\n".join(lines)


def write_report(comparison: Dict, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    markdown = os.path.join(output_dir, "comparison_report.md")
    payload = os.path.join(output_dir, "comparison.json")
    csv_path = os.path.join(output_dir, "comparison_overall.csv")
    with open(markdown, "w", encoding="utf-8") as f:
        f.write(to_markdown(comparison))
    with open(payload, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + METRICS + ["count"])
        for name, values in comparison["overall"].items():
            writer.writerow([name] + [_value(values, PRETTY.get(m, m)) for m in METRICS]
                            + [values.get("n")])
    return {"markdown": markdown, "json": payload, "csv": csv_path}
