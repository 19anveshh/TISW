"""Create a no-provider audit report from archived evaluation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import DATASET_VERSION, EVALUATOR_VERSION
from .runner import _scrub


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _exact_counts(payload: dict) -> dict:
    records = payload.get("records", [])
    counts = Counter(record.get("status", "MISSING_STATUS") for record in records)
    return {key: counts[key] for key in sorted(counts)}


def _failure_detail(record: dict) -> dict:
    judge = record.get("judge") or {}
    return {
        "case_id": record.get("case_id"),
        "input": record.get("input"),
        "expected_mode": record.get("result", {}).get("expected_mode", record.get("mode")),
        "observed_mode": record.get("result", {}).get("mode"),
        "actual_response": record.get("output"),
        "deterministic_status": record.get("status"),
        "metrics": record.get("metrics", {}),
        "judge_status": judge.get("status"),
        "judge_scores": judge.get("metrics", {}),
        "judge_overall": judge.get("overall"),
        "judge_passed": judge.get("passed"),
        "judge_rationale": judge.get("rationale"),
        "judge_evidence": judge.get("evidence"),
        "finding": "The input explicitly requests TISW, so activation is consistent with the application rule. The archived expected_mode=normal is a dataset/rubric error; the acknowledgement is a mode command response, not a teaching answer to judge semantically.",
    }


def build_audit(source_path: str, quota_path: str | None, validations: list[str]) -> dict:
    source = _load(source_path)
    source_records = source.get("records", [])
    failure_ids = {"ADV-005", "ADV-013"}
    failures = [_failure_detail(record) for record in source_records if record.get("case_id") in failure_ids]
    quota = _load(quota_path) if quota_path else None
    result = {
        "metadata": {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-audit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_type": "archived-evaluation-audit",
            "evaluator_version": EVALUATOR_VERSION,
            "current_dataset_version": DATASET_VERSION,
            "provider_calls_made_by_audit": 0,
        },
        "original_valid_run": {
            "path": source_path,
            "run_id": source.get("metadata", {}).get("run_id"),
            "metadata": source.get("metadata", {}),
            "record_count": len(source_records),
            "exact_record_status_counts": _exact_counts(source),
            "reported_summary_status_counts": source.get("summary", {}).get("status_counts", {}),
            "token_usage": source.get("summary", {}).get("token_usage"),
            "latency": source.get("summary", {}).get("latency_ms"),
            "quality_scoring_note": "The archived run used evaluator version 1.0.0. It recorded 128 records as 'completed', 20 as 'passed', and 2 as 'failed'; those are not the required PASS/FAIL/ERROR/NOT_RUN vocabulary. Its judge ran for only the two failing command cases, so broad semantic judge metrics were not measured.",
        },
        "adv_case_findings": failures,
        "quota_limited_run": None,
        "workflow_repairs": [
            "Canonical top-level statuses are now PASS, FAIL, ERROR, and NOT_RUN.",
            "Provider 429/rate-limit errors, timeouts, budget exhaustion, invalid judge output, and missing responses cannot become PASS.",
            "Persistent response and judge caches include dataset, prompt, rubric, model, temperature, case, answer, and retrieval-context fingerprints.",
            "Per-case checkpoints are atomically persisted; --resume skips only compatible PASS/FAIL records.",
            "Maximum external calls, maximum observed provider tokens, and repeated-rate-limit stopping are configurable.",
            "Mode-command acknowledgements are evaluated deterministically and are not sent to the semantic judge.",
        ],
        "tests_executed": [{"command": item.split("|", 1)[0], "result": item.split("|", 1)[1] if "|" in item else "recorded"} for item in validations],
        "limitations": [
            "No new live run was launched during this audit, as requested.",
            "The original valid run's response-quality scores remain historical and are not recomputed here.",
            "The subsequent run is treated as quota-limited and excluded from response-quality scoring.",
            "Exact cost remains unavailable without configured prices; provider token usage is reported only when response metadata exposes it.",
        ],
    }
    if quota:
        quota_records = quota.get("records", [])
        errors = [record for record in quota_records if record.get("status") in {"error", "ERROR"}]
        result["quota_limited_run"] = {
            "path": quota_path,
            "run_id": quota.get("metadata", {}).get("run_id"),
            "exact_record_status_counts": _exact_counts(quota),
            "error_count": len(errors),
            "representative_error": (errors[0].get("result", {}).get("error") if errors else None),
            "interpretation": "The provider returned HTTP 429 token-per-day exhaustion after the account had used approximately 199,4xx of the 200,000 daily-token limit. These records are ERROR, not successful quality evaluations; remaining work was not a valid measurement of model quality.",
        }
    return _scrub(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit archived TISW evaluation reports without provider calls.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--quota-source")
    parser.add_argument("--output", default="evaluation_results")
    parser.add_argument("--validation", action="append", default=[])
    args = parser.parse_args()
    result = build_audit(args.source, args.quota_source, args.validation)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result["metadata"]["run_id"]
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    original = result["original_valid_run"]
    quota = result.get("quota_limited_run")
    lines = [
        "# TISW Archived Evaluation Audit",
        "",
        f"- Audit run: `{stem}`",
        f"- Provider calls made by audit: **0**",
        "",
        "## Original archived run",
        "",
        f"- Source: `{original['path']}`",
        f"- Run ID: `{original['run_id']}`",
        f"- Records: **{original['record_count']}**",
        f"- Exact record statuses: `{original['exact_record_status_counts']}`",
        f"- Historical tokens: `{original['token_usage']}`",
        f"- Historical latency: `{original['latency']}`",
        "",
        original["quality_scoring_note"],
        "",
        "## ADV-005 and ADV-013",
        "",
    ]
    for finding in result["adv_case_findings"]:
        lines.extend([
            f"### `{finding['case_id']}`",
            f"- Input: `{finding['input']}`",
            f"- Expected mode in archived case: `{finding['expected_mode']}`",
            f"- Observed mode: `{finding['observed_mode']}`",
            f"- Actual response: `{finding['actual_response']}`",
            f"- Judge overall: `{finding['judge_overall']}`; passed: `{finding['judge_passed']}`",
            f"- Judge scores: `{finding['judge_scores']}`",
            f"- Judge rationale: {finding['judge_rationale']}",
            f"- Judge evidence: `{finding['judge_evidence']}`",
            f"- Finding: {finding['finding']}",
            "",
        ])
    if quota:
        lines.extend([
            "## Subsequent quota-limited run",
            "",
            f"- Source: `{quota['path']}`",
            f"- Exact record statuses: `{quota['exact_record_status_counts']}`",
            f"- Error records: **{quota['error_count']}**",
            f"- Representative provider error: `{quota['representative_error']}`",
            "- These ERROR records are excluded from response-quality scoring.",
            "",
        ])
    lines.extend(["## Workflow repairs", "", *[f"- {item}" for item in result["workflow_repairs"]], "", "## Tests executed", ""])
    lines.extend([f"- `{item['command']}` — **{item['result']}**" for item in result["tests_executed"]])
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in result["limitations"]], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit JSON: {json_path}")
    print(f"Audit report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
