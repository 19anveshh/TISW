"""Deterministic metrics for evaluation results."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Optional


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())).strip()


def fact_coverage(answer: str, reference_facts: Iterable[str]) -> dict:
    """Conservative lexical coverage; semantic quality is left to the judge."""
    facts = [str(f).strip() for f in reference_facts if str(f).strip()]
    normalized = normalize_text(answer)
    matched = [fact for fact in facts if normalize_text(fact) in normalized]
    return {
        "matched": len(matched),
        "required": len(facts),
        "score": (len(matched) / len(facts)) if facts else None,
        "matched_facts": matched,
    }


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> Optional[float]:
    if k <= 0:
        return None
    selected = retrieved[:k]
    return sum(item in relevant for item in selected) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> Optional[float]:
    if not relevant:
        return None
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> Optional[float]:
    if not relevant:
        return None
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], graded_relevance: dict[str, float], k: int) -> Optional[float]:
    if not graded_relevance or k <= 0:
        return None

    def dcg(items: list[str]) -> float:
        return sum((2 ** float(graded_relevance.get(item, 0)) - 1) / math.log2(index + 2) for index, item in enumerate(items))

    actual = dcg(retrieved[:k])
    ideal = dcg(sorted(graded_relevance, key=graded_relevance.get, reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def summarize_records(records: list[dict]) -> dict:
    """Aggregate only measured numeric values, preserving not-run states."""
    values: dict[str, list[float]] = defaultdict(list)
    statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        for metric_name, metric in (record.get("metrics") or {}).items():
            if isinstance(metric, dict):
                status = metric.get("status", "measured" if metric.get("value") is not None else "not_run")
                statuses[metric_name][status] += 1
                value = metric.get("value")
            else:
                status = "measured"
                statuses[metric_name][status] += 1
                value = metric
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[metric_name].append(float(value))
    summary = {}
    for name in sorted(set(values) | set(statuses)):
        observed = values.get(name, [])
        summary[name] = {
            "mean": (sum(observed) / len(observed)) if observed else None,
            "count": len(observed),
            "statuses": dict(statuses[name]),
        }
    return summary


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
