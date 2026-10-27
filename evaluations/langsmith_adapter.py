"""Local-only schemas for a future LangSmith evaluation integration.

No LangSmith package or network client is imported here. The returned payloads
can later be passed to an explicitly configured SDK client in an online job.
"""

from __future__ import annotations

from .rubrics import rubric_for_case
from . import DATASET_VERSION


def case_to_langsmith_example(case: dict) -> dict:
    """Convert one fixed case into a local LangSmith-example-shaped payload."""
    rubric_groups = sorted(rubric_for_case(case))
    return {
        "inputs": {
            "question": case.get("input", ""),
            "history": case.get("history", []),
            "expected_mode": case.get("expected_mode", "normal"),
        },
        "outputs": {
            "reference_facts": case.get("reference_facts", []),
            "evaluation_criteria": case.get("evaluation_criteria", []),
        },
        "metadata": {
            "case_id": case.get("id"),
            "category": case.get("category"),
            "subject": case.get("subject"),
            "dataset_version": case.get("dataset_version") or DATASET_VERSION,
            "rubric_groups": rubric_groups,
            "paired_group": (case.get("metadata") or {}).get("paired_group"),
            "paired_mode": (case.get("metadata") or {}).get("paired_mode"),
        },
    }


def records_to_langsmith_results(records: list[dict]) -> list[dict]:
    """Convert local records into experiment-result-shaped rows."""
    results = []
    for record in records:
        result = record.get("result") or {}
        results.append({
            "example_id": record.get("case_id"),
            "outputs": {"response": record.get("output", "")},
            "reference_outputs": {
                "status": record.get("status"),
                "metrics": record.get("metrics", {}),
            },
            "metadata": {
                "category": record.get("category"),
                "mode": record.get("mode"),
                "response_source": record.get("response_source"),
                "latency_ms": result.get("latency_ms"),
                "usage": result.get("usage"),
            },
        })
    return results


def build_pair_rows(records: list[dict]) -> list[dict]:
    """Build local Normal/TISW rows without running an experiment."""
    groups: dict[str, dict[str, dict]] = {}
    for record in records:
        group = (record.get("metadata") or {}).get("paired_group")
        mode = (record.get("metadata") or {}).get("paired_mode") or record.get("mode")
        if group and mode:
            groups.setdefault(group, {})[mode] = record
    rows = []
    for group, modes in sorted(groups.items()):
        rows.append({
            "paired_group": group,
            "normal": modes.get("normal"),
            "tisw": modes.get("tisw"),
            "comparison_status": "MEASURED" if all(modes.get(mode, {}).get("status") in {"PASS", "FAIL"} for mode in ("normal", "tisw")) else "NOT_RUN",
        })
    return rows
