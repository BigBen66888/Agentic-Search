import re
from typing import Iterable, List

from .text import f1_score

def blocks(text: str, tag: str) -> List[str]:
    return [x.strip() for x in re.findall(fr"<{tag}>(.*?)</{tag}>", text or "", re.S | re.I)]

def format_score(text):
    answers, searches = blocks(text, "answer"), blocks(text, "search")
    if not answers or any(not x for x in searches):
        return 0.0
    return 1.0 if (not searches or re.search(r"<think>.*?</think>", text or "", re.S | re.I)) else 0.0

def answer_score(text, answers: Iterable[str]):
    found = blocks(text, "answer")
    return max((f1_score(found[-1], x) for x in answers), default=0.0) if found else 0.0

def evidence_score(text, answers, supporting_facts=None):
    info = " ".join(blocks(text, "information")).lower()
    if not info:
        return 0.0
    answer_hit = any(str(a).lower() in info for a in answers if a)
    if not supporting_facts:
        return float(answer_hit)
    titles = {str(x.get("title", x.get("doc_title", ""))).lower() for x in supporting_facts}
    title_hit = sum(1 for title in titles if title and title in info)
    return .5 * float(answer_hit) + .5 * min(title_hit / max(len(titles), 1), 1.0)

def efficiency_score(text, max_turns=4):
    searches = blocks(text, "search")
    invalid = len(re.findall(r"invalid action|previous action is invalid", text or "", re.I))
    repeated = len(searches) - len({x.lower().strip() for x in searches})
    if not searches:
        return .5 if blocks(text, "answer") else 0.0
    penalty = .12 * max(0, len(searches)-1) + .15 * invalid + .15 * repeated
    penalty += .25 if len(searches) > max_turns else 0
    return max(0.0, min(1.0, 1.0 - penalty))

def strategy_score(text, answers, supporting_facts=None):
    searches, infos = blocks(text, "search"), blocks(text, "information")
    unique = len({x.lower().strip() for x in searches}) / max(len(searches), 1)
    novelty = len({x[:160].lower() for x in infos}) / max(len(infos), 1)
    multi = evidence_score(text, answers, supporting_facts) if supporting_facts else min(len(infos) / 2, 1.0)
    stop = 1.0 if blocks(text, "answer") and len(searches) <= 4 else 0.0
    return .30 * unique + .30 * novelty + .25 * multi + .15 * stop

def score_trajectory(text, answers, supporting_facts=None, mode="multi", max_turns=4, weights=None):
    answers = list(answers)
    if mode in {"binary", "em"}:
        value = float(answer_score(text, answers) >= .999)
        return {"total": value, "answer": value, "evidence": 0., "format": 0., "efficiency": 0., "strategy": 0.}
    weights = weights or {"answer": .55, "evidence": .15, "format": .10, "efficiency": .10, "strategy": .10}
    parts = {"answer": answer_score(text, answers), "evidence": evidence_score(text, answers, supporting_facts),
             "format": format_score(text), "efficiency": efficiency_score(text, max_turns),
             "strategy": strategy_score(text, answers, supporting_facts)}
    parts["total"] = sum(weights[key] * parts[key] for key in weights)
    return parts

