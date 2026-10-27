"""Command-line evaluation runner for the existing TISW workflow.

Examples:

    uv run python -m evaluations.runner --offline
    uv run python -m evaluations.runner --live --judge
    uv run python -m evaluations.runner --live --baseline results/<run>.json

The offline mode runs deterministic state checks, fixture code checks, and
dataset/schema validation.  Response-quality and latency claims require
``--live`` and a configured Groq key.  No result is silently converted into a
score when a provider, judge, or retrieval dependency was unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from runtime_config import configure_runtime

from . import DATASET_VERSION, EVALUATOR_VERSION
from .coding import run_python_snippet
from .dataset import load_cases
from .metrics import (
    fact_coverage,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    summarize_records,
)
from .rubrics import RUBRIC_VERSION, rubric_prompt


PROMPT_VERSION = "application-prompts-from-main.py@current"
DEFAULT_OUTPUT_DIR = Path("evaluation_results")
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / ".cache"
SECRET_PATTERN = re.compile(r"gsk_[A-Za-z0-9_-]+")
PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"
NOT_RUN = "NOT_RUN"


_APPLICATION_MODULE: Any = None
_RUNTIME_CONFIG: Any = None


def _application() -> Any:
    global _APPLICATION_MODULE, _RUNTIME_CONFIG
    if _RUNTIME_CONFIG is None:
        # Direct unit-test helpers are offline unless a CLI run has explicitly
        # selected a runtime mode first.
        _RUNTIME_CONFIG = configure_runtime(offline=True)
    if _APPLICATION_MODULE is None:
        import main
        _APPLICATION_MODULE = main
    return _APPLICATION_MODULE


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before an external call when the configured budget is exhausted."""


class EvaluationCallBudget:
    def __init__(self, max_calls: int, max_tokens: int):
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.external_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.unavailable_usage_calls = 0

    def reserve(self) -> None:
        if self.max_calls and self.external_calls >= self.max_calls:
            raise EvaluationBudgetExceeded(f"maximum external calls reached ({self.max_calls})")
        if self.max_tokens and self.total_tokens >= self.max_tokens:
            raise EvaluationBudgetExceeded(f"maximum provider tokens reached ({self.max_tokens})")
        self.external_calls += 1

    def record(self, usage: dict) -> None:
        values = [usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
        if not any(isinstance(value, int) and value >= 0 for value in values):
            self.unavailable_usage_calls += 1
            return
        self.input_tokens += usage.get("input_tokens") or 0
        self.output_tokens += usage.get("output_tokens") or 0
        self.total_tokens += usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)

    def snapshot(self) -> dict:
        return {
            "max_external_calls": self.max_calls,
            "max_total_tokens": self.max_tokens,
            "external_calls": self.external_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage_unavailable_calls": self.unavailable_usage_calls,
        }


class BudgetedLLM:
    """Wrap a provider client so every generation/judge call is accounted for."""

    def __init__(self, llm: Any, budget: EvaluationCallBudget, usage_sink: list[dict]):
        self.llm = llm
        self.budget = budget
        self.usage_sink = usage_sink

    def invoke(self, messages: list[Any]) -> Any:
        self.budget.reserve()
        try:
            response = self.llm.invoke(messages)
        except Exception:
            raise
        usage = _usage_from_message(response)
        self.budget.record(usage)
        self.usage_sink.append(usage)
        return response


def _scrub(value: Any) -> Any:
    """Prevent obvious provider secrets from entering results or traces."""
    if isinstance(value, str):
        return SECRET_PATTERN.sub("[REDACTED_API_KEY]", value)
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _scrub(item) for key, item in value.items()}
    return value


def _hash_file_or_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _json_hash(value: Any) -> str:
    return _hash_file_or_value(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _cache_key(stage: str, payload: Any) -> str:
    return f"{stage}-{_json_hash(payload)}"


def _response_cache_key(case: dict, retrieved_context: Optional[str], config_fingerprint: dict) -> str:
    return _cache_key("response", {
        "cache_schema": "response-v1",
        "case": case,
        "retrieved_context": retrieved_context,
        "config": config_fingerprint,
    })


def _config_fingerprint() -> dict:
    application = _application()
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "temperature": 0.3,
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": {
            "normal": _hash_file_or_value(application.NORMAL_MODE_PROMPT),
            "tisw": _hash_file_or_value(application.TISW_MODE_PROMPT),
        },
        "dataset_version": DATASET_VERSION,
    }


def _cache_read(cache_dir: Path, key: str) -> Optional[dict]:
    path = cache_dir / f"{key}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _cache_write(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{key}.json"
    temporary = cache_dir / f".{key}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(_scrub(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(target)


def _checkpoint_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(_scrub(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _checkpoint_read(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class EvaluationLLM:
    """Minimal fake model used for deterministic graph/state checks."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content="[evaluation fixture response]")


def _metric(value: Any = None, *, status: str = "measured", explanation: str = "", evidence: Any = None) -> dict:
    if value is None and status == "measured":
        status = "not_run"
    result = {"value": value, "status": status}
    if explanation:
        result["explanation"] = explanation
    if evidence is not None:
        result["evidence"] = _scrub(evidence)
    return result


def _run_with_timeout(function, timeout_seconds: float):
    """Run a call with a best-effort timeout on all supported platforms."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(function)
    try:
        return future.result(timeout=max(0.1, timeout_seconds)), None
    except FutureTimeout:
        future.cancel()
        return None, f"exceeded {timeout_seconds:.2f}s (worker cancellation is best-effort on this platform)"
    finally:
        # Do not block a timed-out evaluation on a provider worker.  The
        # provider client itself remains responsible for its own connection
        # cleanup; a timed-out result is never reused.
        executor.shutdown(wait=False, cancel_futures=True)


def _usage_from_message(message: Any) -> dict:
    """Extract provider usage without depending on one LangChain version."""
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    provider_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    def first(*keys):
        for source in (usage, provider_usage, response_metadata):
            for key in keys:
                value = source.get(key) if isinstance(source, dict) else None
                if isinstance(value, (int, float)):
                    return int(value)
        return None
    return {
        "input_tokens": first("input_tokens", "prompt_tokens"),
        "output_tokens": first("output_tokens", "completion_tokens"),
        "total_tokens": first("total_tokens"),
    }


def _merge_usage(target: dict, usage: dict) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if isinstance(usage.get(key), int):
            target[key] = (target.get(key) or 0) + usage[key]


def _sum_usage(events: list[dict]) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in events:
        _merge_usage(total, event)
    return total


def _build_rag_fixture(case: dict, cache: dict) -> tuple[Optional[str], list[str], dict]:
    """Retrieve a case's fixture documents with the real hybrid RAG engine."""
    if not case.get("rag"):
        return None, [], {"status": "not_applicable"}
    key = _json_hash(case.get("rag_documents", []))
    try:
        if key not in cache:
            from rag_engine import get_session_document_index, prune_session_index

            session_id = f"evaluation-{uuid.uuid4().hex}"
            index = get_session_document_index(session_id)
            for document in case.get("rag_documents", []):
                index.add_document(document["filename"], document["text"].encode("utf-8"))
            cache[key] = (session_id, index)
        _session_id, index = cache[key]
        chunks = index.hybrid_search(case["input"], final_top_k=6)
        filenames = [chunk.filename for chunk in chunks]
        context = "\n\n".join(
            f"--- EXCERPT {index + 1} ---\n[Source: {chunk.filename}]\n{chunk.text}"
            for index, chunk in enumerate(chunks)
        )
        return context or "[No relevant excerpts were found in the uploaded documents for this question.]", filenames, {"status": "measured", "count": len(chunks)}
    except Exception as exc:
        return None, [], {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}


def _cleanup_rag_cache(cache: dict) -> None:
    try:
        from rag_engine import prune_session_index
        for session_id, _index in cache.values():
            prune_session_index(session_id)
    except Exception:
        pass


def _run_mode_case(case: dict) -> dict:
    """Exercise mode selection and checkpoint state with the real graph."""
    scenario = case.get("mode_scenario") or {}
    steps = scenario.get("steps") or [case["input"]]
    expected = scenario.get("expected_modes") or [case.get("expected_mode", "normal")]
    fake = EvaluationLLM()
    graph = _application().create_tutor_graph(fake, InMemorySaver())
    thread_id = f"mode-eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    observed = []
    errors = []
    rag_context = scenario.get("rag_context")
    for index, text in enumerate(steps):
        payload = {"messages": [HumanMessage(content=text)]}
        if index == len(steps) - 1 and rag_context:
            payload["rag_context"] = rag_context
        try:
            state = graph.invoke(payload, config={"configurable": {"thread_id": thread_id}})
            observed.append(state.get("active_mode", "normal"))
        except Exception as exc:
            observed.append(None)
            errors.append(f"{type(exc).__name__}: {exc}")

    isolation_ok = True
    if scenario.get("isolation"):
        try:
            isolated_state = graph.invoke(
                {"messages": [HumanMessage(content="What is Python?")]},
                config={"configurable": {"thread_id": f"isolated-{uuid.uuid4().hex}"}},
            )
            isolation_ok = isolated_state.get("active_mode", "normal") == "normal"
        except Exception as exc:
            isolation_ok = False
            errors.append(f"isolation_{type(exc).__name__}: {exc}")

    pass_steps = observed == expected
    passed = pass_steps and isolation_ok and not errors
    return {
        "status": PASS if passed else FAIL,
        "response_source": "mock_provider",
        "observed_modes": observed,
        "expected_modes": expected,
        "isolation_passed": isolation_ok,
        "llm_calls": len(fake.calls),
        "errors": errors,
        "metrics": {
            "mode_state_accuracy": _metric(1.0 if pass_steps else 0.0, explanation="Exact per-step mode state comparison.", evidence={"expected": expected, "observed": observed}),
            "conversation_isolation": _metric(1.0 if isolation_ok else 0.0, explanation="Fresh thread must begin in normal mode."),
        },
    }


def _run_code_fixture(case: dict) -> dict:
    task = case.get("code_task")
    if not task:
        return {"status": "not_applicable"}
    result = run_python_snippet(task["code"], task.get("expected_stdout"), timeout_seconds=3.0)
    status_map = {"passed": PASS, "failed": FAIL, "timeout": ERROR, "error": ERROR, "rejected": ERROR}
    return {"status": status_map.get(result.status, ERROR), "execution": result.to_dict()}


def _rag_metrics(retrieved: list[str], relevant_sources: dict[str, float]) -> dict:
    relevant = set(relevant_sources)
    return {
        "rag_precision_at_3": _metric(precision_at_k(retrieved, relevant, 3), explanation="Relevant source filenames among the first three retrieved chunks."),
        "rag_recall_at_6": _metric(recall_at_k(retrieved, relevant, 6), explanation="Relevant source filenames recovered in the first six chunks."),
        "rag_mrr": _metric(mean_reciprocal_rank(retrieved, relevant), explanation="Reciprocal rank of the first relevant source."),
        "rag_ndcg_at_6": _metric(ndcg_at_k(retrieved, relevant_sources, 6), explanation="Graded source ranking using the case relevance grades."),
    }


def _judge_json(text: Any) -> Optional[dict]:
    """Parse a judge object, accepting only JSON or a JSON code fence."""
    if isinstance(text, dict):
        return text
    cleaned = str(text or "").strip()
    candidates = [cleaned]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _case_requests_json(case: dict) -> bool:
    """Detect an explicit user/test requirement for JSON output."""
    text = " ".join(
        str(case.get(key, ""))
        for key in ("input", "evaluation_criteria")
    )
    return bool(re.search(
        r"(?:return|respond|output|provide|format|write|give)[^.!?\n]{0,100}\bjson\b"
        r"|\bjson\b[^.!?\n]{0,100}(?:only|object|format)",
        text,
        flags=re.IGNORECASE,
    ))


def _judge_error(reason: str, response: Any = None, *, raw_response: Any = None) -> dict:
    result = {"status": ERROR.lower(), "reason": reason}
    if response is not None:
        result["usage"] = _usage_from_message(response)
    if raw_response is not None:
        result["response_excerpt"] = _scrub(str(raw_response)[:500])
    return result


def _judge_answer(judge_llm: Any, case: dict, answer: str, retrieved_context: Optional[str]) -> dict:
    explicit_json = _case_requests_json(case)
    prompt = f"""You are a blinded evaluator for a tutoring response. Do not reward length, headings, or an analogy by themselves. Score only observable evidence.

CASE:
{json.dumps({k: case.get(k) for k in ('id', 'category', 'subject', 'input', 'history', 'expected_mode', 'reference_facts', 'evaluation_criteria')}, ensure_ascii=True, indent=2)}

APPLICABLE RUBRIC:
{rubric_prompt(case)}

RETRIEVED CONTEXT (may be empty):
{retrieved_context or '[none]'}

ANSWER TO SCORE:
{answer[:16000]}

OUTPUT-CONTRACT INTERPRETATION:
- The JSON-only format below applies to your judge response, not to the assistant answer.
- Do not penalize the assistant for using ordinary prose, Markdown, or code unless the user explicitly requested a different output format or the rubric makes that format necessary.
- An explicit JSON-output requirement was detected in this case: {str(explicit_json).lower()}.

Return JSON only with this shape:
{{"scores": {{"group.metric": 1}}, "overall": 1, "passed": true, "rationale": "...", "evidence": ["..."]}}
Each score must be an integer from 1 to 5. Include every applicable rubric metric using keys such as general.factual_correctness and tisw.beginner_clarity. Set passed true only when the answer meets the user's case criteria without a material error. Only enforce JSON on the assistant answer when the explicit requirement above is true.
"""
    try:
        response = judge_llm.invoke([HumanMessage(content=prompt)])
        raw_response = getattr(response, "content", response)
        parsed = _judge_json(raw_response)
        if not parsed:
            return _judge_error("judge_returned_non_json", response, raw_response=raw_response)
        scores = parsed.get("scores")
        if not isinstance(scores, dict):
            return _judge_error("judge_json_missing_scores", response)
        if not scores:
            return _judge_error("judge_json_empty_scores", response)
        metrics = {}
        for name, value in scores.items():
            if not isinstance(name, str):
                return _judge_error("judge_score_name_not_string", response)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                return _judge_error(f"judge_score_out_of_range_or_non_integer:{name}", response)
            metrics[f"judge.{name}"] = _metric(float(value), explanation="LLM judge score; inspect rationale and evidence.")
        overall = parsed.get("overall")
        if isinstance(overall, bool) or not isinstance(overall, int) or not 1 <= overall <= 5:
            return _judge_error("judge_overall_out_of_range_or_non_integer", response)
        if not isinstance(parsed.get("passed"), bool):
            return _judge_error("judge_passed_missing_or_not_boolean", response)
        if not isinstance(parsed.get("rationale"), str) or not parsed["rationale"].strip():
            return _judge_error("judge_rationale_missing_or_empty", response)
        if not isinstance(parsed.get("evidence"), list):
            return _judge_error("judge_evidence_missing_or_not_list", response)
        return {
            "status": "measured",
            "metrics": metrics,
            "overall": overall,
            "passed": parsed["passed"],
            "rationale": _scrub(str(parsed.get("rationale", ""))),
            "evidence": _scrub(parsed.get("evidence", [])),
            "usage": _usage_from_message(response),
        }
    except Exception as exc:
        return _judge_error(f"{type(exc).__name__}: {exc}")


def _execute_live_case(
    case: dict,
    llm: Any,
    *,
    timeout_seconds: float,
    rag_cache: dict,
    budget: EvaluationCallBudget,
    cache_dir: Path,
    config_fingerprint: dict,
) -> dict:
    retrieved_context = None
    retrieved_sources: list[str] = []
    rag_status = {"status": "not_applicable"}
    if case.get("rag"):
        retrieved_context, retrieved_sources, rag_status = _build_rag_fixture(case, rag_cache)

    response_key = _response_cache_key(case, retrieved_context, config_fingerprint)
    cached = _cache_read(cache_dir, response_key)
    if cached:
        cached["cache_hit"] = True
        cached["response_source"] = "cache"
        return cached

    usage_events: list[dict] = []
    graph = _application().create_tutor_graph(BudgetedLLM(llm, budget, usage_events), InMemorySaver())
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"

    history = case.get("history", [])
    if case.get("expected_mode") == "tisw" and not any(_application().detect_mode_command(str(item.get("content", ""))) == "activate" for item in history if item.get("role") == "user"):
        history = [{"role": "user", "content": "Use TISW."}] + history

    calls = []
    for item in history:
        if item.get("role") != "user":
            continue
        calls.append((str(item.get("content", "")), None))
    calls.append((case["input"], retrieved_context))

    final_state = None
    final_message = ""
    latencies = []
    timed_out = None
    started_case = time.perf_counter()
    try:
        for index, (text, context) in enumerate(calls):
            payload = {"messages": [HumanMessage(content=text)], "rag_context": context}
            started = time.perf_counter()
            final_state, timeout_reason = _run_with_timeout(
                lambda payload=payload: graph.invoke(payload, config={"configurable": {"thread_id": thread_id}}),
                timeout_seconds,
            )
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            if timeout_reason:
                timed_out = timeout_reason
                break
            messages = (final_state or {}).get("messages", [])
            if messages:
                final_message = str(messages[-1].content)
        if timed_out:
            result = {
                "status": ERROR,
                "response_source": "live_provider",
                "output": "",
                "latency_ms": (time.perf_counter() - started_case) * 1000,
                "call_latencies_ms": latencies,
                "timeout_reason": timed_out,
                "usage": _sum_usage(usage_events),
                "retrieved_sources": retrieved_sources,
                "rag_status": rag_status,
            }
            return result
        if final_state is None:
            raise RuntimeError("graph returned no state")
        if not final_message.strip():
            return {
                "status": ERROR,
                "response_source": "live_provider",
                "error": "missing model response",
                "output": "",
                "latency_ms": (time.perf_counter() - started_case) * 1000,
                "call_latencies_ms": latencies,
                "usage": _sum_usage(usage_events),
                "retrieved_sources": retrieved_sources,
                "rag_status": rag_status,
            }
        expected_mode = case.get("expected_mode", "normal")
        actual_mode = final_state.get("active_mode", "normal")
        facts = fact_coverage(final_message, case.get("reference_facts", []))
        metrics = {
            "reference_fact_coverage": _metric(facts["score"], status="measured" if facts["score"] is not None else "not_run", explanation="Conservative lexical reference-fact coverage; not a substitute for semantic judging.", evidence=facts),
            "mode_expectation": _metric(1.0 if actual_mode == expected_mode else 0.0, explanation="Mode returned by the actual graph for this case.", evidence={"expected": expected_mode, "observed": actual_mode}),
        }
        if case.get("rag"):
            metrics.update(_rag_metrics(retrieved_sources, case.get("relevant_sources", {})))
        result = {
            "status": PASS if actual_mode == expected_mode else FAIL,
            "response_source": "live_provider",
            "output": _scrub(final_message),
            "mode": actual_mode,
            "expected_mode": expected_mode,
            "latency_ms": (time.perf_counter() - started_case) * 1000,
            "call_latencies_ms": latencies,
            "usage": _sum_usage(usage_events),
            "retrieved_sources": retrieved_sources,
            "retrieved_context": _scrub(retrieved_context),
            "rag_status": rag_status,
            "metrics": metrics,
        }
        _cache_write(cache_dir, response_key, result)
        return result
    except EvaluationBudgetExceeded as exc:
        return {
            "status": NOT_RUN,
            "response_source": "live_provider_budget_blocked",
            "reason": str(exc),
            "output": "",
            "latency_ms": (time.perf_counter() - started_case) * 1000,
            "call_latencies_ms": latencies,
            "usage": _sum_usage(usage_events),
            "retrieved_sources": retrieved_sources,
            "rag_status": rag_status,
        }
    except Exception as exc:
        return {
            "status": ERROR,
            "response_source": "live_provider",
            "error": f"{type(exc).__name__}: {exc}",
            "output": _scrub(final_message),
            "latency_ms": (time.perf_counter() - started_case) * 1000,
            "call_latencies_ms": latencies,
            "usage": _sum_usage(usage_events),
            "retrieved_sources": retrieved_sources,
            "rag_status": rag_status,
        }


def _case_record(
    case: dict,
    result: dict,
    *,
    run_mode: str,
    judge_llm: Any = None,
    retry_limit: int = 0,
    cache_dir: Optional[Path] = None,
    judge_fingerprint: Optional[dict] = None,
    cached_judge: Optional[dict] = None,
) -> dict:
    record = {
        "case_id": case["id"],
        "category": case["category"],
        "subject": case["subject"],
        "mode": case.get("expected_mode", "normal"),
        "input": _scrub(case["input"]),
        "status": result.get("status", "error"),
        "response_source": result.get("response_source", "unknown"),
        "metadata": _scrub(case.get("metadata", {})),
        "output": _scrub(result.get("output", "")),
        "result": _scrub({k: v for k, v in result.items() if k not in {"output", "metrics"}}),
        "metrics": result.get("metrics", {}),
        "evaluator": {"version": EVALUATOR_VERSION, "rubric_version": RUBRIC_VERSION, "run_mode": run_mode},
        "retries": [],
    }
    code_result = result.get("fixture") or _run_code_fixture(case)
    if code_result.get("status") != "not_applicable":
        record["code_execution"] = code_result
        record["metrics"]["fixture_code_execution"] = _metric(1.0 if code_result["status"] == PASS else 0.0, explanation="Execution of the fixed fixture snippet, not generated model code.")

    command_only = bool(_application().detect_mode_command(case.get("input", "")))
    if command_only:
        record["judge"] = {"status": NOT_RUN, "reason": "mode command is evaluated deterministically; acknowledgement is not a teaching answer"}
    elif cached_judge is not None or (judge_llm is not None and result.get("status") in {PASS, FAIL} and result.get("output")):
        judge_key = _cache_key("judge", {
            "cache_schema": "judge-v1",
            "case_id": case["id"],
            "answer": result["output"],
            "retrieved_context": result.get("retrieved_context"),
            "case": case,
            "config": judge_fingerprint or {},
        })
        judge = cached_judge or (_cache_read(cache_dir, judge_key) if cache_dir else None)
        if judge:
            judge["cache_hit"] = True
        else:
            judge = _judge_answer(judge_llm, case, result["output"], result.get("retrieved_context"))
            if cache_dir and judge.get("status") == "measured":
                _cache_write(cache_dir, judge_key, judge)
        record["judge"] = judge
        if judge.get("status") == "measured":
            record["metrics"].update(judge.get("metrics", {}))
            if record["status"] == PASS:
                record["status"] = PASS if judge.get("passed") else FAIL
        elif record["status"] == PASS:
            # A generated answer without a valid judge result is not a
            # successful quality evaluation.
            record["status"] = ERROR
        elif judge.get("status") == ERROR.lower() and record["status"] in {PASS, FAIL}:
            # Keep the deterministic result for diagnosis, but do not expose
            # a quality FAIL when the judge itself failed to produce a score.
            record["deterministic_status"] = record["status"]
            record["status"] = ERROR
        if retry_limit and judge.get("status") == "measured" and not judge.get("passed"):
            record["retry_policy"] = "one additional attempt for a measured quality failure; production output is never changed"
    return record


def _cached_judge_for_result(case: dict, result: dict, cache_dir: Path) -> Optional[dict]:
    if not result.get("output"):
        return None
    judge_fingerprint = {
        "evaluator_version": EVALUATOR_VERSION,
        "model": os.getenv("EVAL_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")),
        "temperature": 0.0,
        "rubric_version": RUBRIC_VERSION,
    }
    judge_key = _cache_key("judge", {
        "cache_schema": "judge-v1",
        "case_id": case["id"],
        "answer": result["output"],
        "retrieved_context": result.get("retrieved_context"),
        "case": case,
        "config": judge_fingerprint,
    })
    judge = _cache_read(cache_dir, judge_key)
    if judge:
        judge["cache_hit"] = True
    return judge


def _offline_records(cases: list[dict], include_rag: bool, cache_dir: Path) -> list[dict]:
    records = []
    rag_cache = {}
    config_fingerprint = _config_fingerprint()
    try:
        for case in cases:
            started = time.perf_counter()
            if case.get("mode_scenario"):
                result = _run_mode_case(case)
                cached_judge = None
            elif case.get("code_task"):
                result = {"status": NOT_RUN, "response_source": "missing_cache", "reason": "offline mode has no compatible cached response", "metrics": {}}
                fixture = _run_code_fixture(case)
                result["fixture"] = fixture
                result["metrics"]["fixture_code_execution"] = _metric(1.0 if fixture.get("status") == PASS else 0.0, explanation="Execution of the fixed fixture snippet, not generated model code.")
                cached_judge = None
            elif not case.get("rag"):
                response_key = _response_cache_key(case, None, config_fingerprint)
                result = _cache_read(cache_dir, response_key)
                if result:
                    result["cache_hit"] = True
                    result["response_source"] = "cache"
                    cached_judge = _cached_judge_for_result(case, result, cache_dir)
                    if cached_judge is None:
                        result["status"] = NOT_RUN
                        result["reason"] = "compatible cached response found but no compatible cached judge result"
                else:
                    result = {"status": NOT_RUN, "response_source": "missing_cache", "reason": "offline mode has no compatible cached response", "metrics": {}}
                    cached_judge = None
            elif case.get("rag") and include_rag:
                context, sources, rag_status = _build_rag_fixture(case, rag_cache)
                result = {"status": PASS if rag_status.get("status") == "measured" else ERROR, "response_source": "deterministic_fixture", "retrieved_sources": sources, "rag_status": rag_status, "metrics": _rag_metrics(sources, case.get("relevant_sources", {})) if rag_status.get("status") == "measured" else {}}
                cached_judge = None
            else:
                result = {"status": NOT_RUN, "response_source": "missing_cache", "reason": "offline mode has no compatible cached response", "metrics": {}}
                cached_judge = None
            result["latency_ms"] = (time.perf_counter() - started) * 1000
            records.append(_case_record(case, result, run_mode="offline", cached_judge=cached_judge))
    finally:
        _cleanup_rag_cache(rag_cache)
    return records


def _is_rate_limit_record(record: dict) -> bool:
    text = json.dumps(record, ensure_ascii=True).lower()
    return any(marker in text for marker in ("ratelimit", "rate_limit", "rate limit", "429", "tokens per day"))


def _not_run_record(case: dict, reason: str) -> dict:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "subject": case["subject"],
        "mode": case.get("expected_mode", "normal"),
        "input": _scrub(case["input"]),
        "status": NOT_RUN,
        "response_source": "provider_stop",
        "output": "",
        "result": {"status": NOT_RUN, "reason": reason},
        "metrics": {},
        "evaluator": {"version": EVALUATOR_VERSION, "rubric_version": RUBRIC_VERSION, "run_mode": "live"},
        "retries": [],
    }


def _provider_error_record(case: dict, reason: str) -> dict:
    record = _not_run_record(case, reason)
    record["status"] = ERROR
    record["result"] = {
        "status": ERROR,
        "error": reason,
        "provider_failure": True,
    }
    return record


def _live_records(
    cases: list[dict],
    *,
    use_judge: bool,
    timeout_seconds: float,
    retry_limit: int,
    cache_dir: Path,
    checkpoint_path: Path,
    checkpoint_fingerprint: dict,
    max_external_calls: int,
    max_total_tokens: int,
    max_rate_limit_errors: int,
) -> tuple[list[dict], dict]:
    application = _application()
    llm = application.load_and_validate_groq_llm()
    budget = EvaluationCallBudget(max_external_calls, max_total_tokens)
    judge_llm = None
    if use_judge:
        from langchain_groq import ChatGroq
        model = os.getenv("EVAL_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
        judge_llm = BudgetedLLM(
            ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.0),
            budget,
            [],
        )
    records = []
    checkpoint = _checkpoint_read(checkpoint_path)
    if checkpoint.get("fingerprint") == checkpoint_fingerprint:
        existing = {record.get("case_id"): record for record in checkpoint.get("records", []) if record.get("status") in {PASS, FAIL}}
        records.extend(existing.values())
    existing_ids = {record["case_id"] for record in records}
    rag_cache = {}
    stop_reason = checkpoint.get("stop_reason")
    rate_limit_errors = 0
    config_fingerprint = {
        "evaluator_version": EVALUATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "temperature": 0.3,
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": {"normal": _hash_file_or_value(application.NORMAL_MODE_PROMPT), "tisw": _hash_file_or_value(application.TISW_MODE_PROMPT)},
        "dataset_version": DATASET_VERSION,
    }
    judge_fingerprint = {
        "evaluator_version": EVALUATOR_VERSION,
        "model": os.getenv("EVAL_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")),
        "temperature": 0.0,
        "rubric_version": RUBRIC_VERSION,
    }

    def save_checkpoint() -> None:
        _checkpoint_write(checkpoint_path, {
            "schema": "tisw-evaluation-checkpoint-v1",
            "fingerprint": checkpoint_fingerprint,
            "records": records,
            "budget": budget.snapshot(),
            "stop_reason": stop_reason,
        })

    try:
        for case in cases:
            if case["id"] in existing_ids:
                continue
            if stop_reason:
                records.append(_not_run_record(case, stop_reason))
                continue
            # State-control cases are deterministic graph checks even during a
            # live run. They must not be diluted by semantic answer scores.
            result = _run_mode_case(case) if case.get("mode_scenario") else _execute_live_case(
                case,
                llm,
                timeout_seconds=timeout_seconds,
                rag_cache=rag_cache,
                budget=budget,
                cache_dir=cache_dir,
                config_fingerprint=config_fingerprint,
            )
            record = _case_record(
                case,
                result,
                run_mode="live",
                judge_llm=judge_llm,
                retry_limit=retry_limit,
                cache_dir=cache_dir,
                judge_fingerprint=judge_fingerprint,
            )
            # Retries compare an independent fresh invocation. They never alter
            # the first answer stored in the record or production behavior.
            if retry_limit and record.get("judge", {}).get("status") == "measured" and not record["judge"].get("passed"):
                retry_result = _execute_live_case(case, llm, timeout_seconds=timeout_seconds, rag_cache=rag_cache, budget=budget, cache_dir=cache_dir, config_fingerprint={**config_fingerprint, "retry": 2})
                retry_judge = _judge_answer(judge_llm, case, retry_result.get("output", ""), retry_result.get("retrieved_context")) if judge_llm and retry_result.get("output") else {"status": "not_run", "reason": "no retry output"}
                record["retries"].append({"attempt": 2, "result": retry_result, "judge": retry_judge})
            records.append(record)
            existing_ids.add(case["id"])
            if _is_rate_limit_record(record):
                rate_limit_errors += 1
                if rate_limit_errors >= max_rate_limit_errors:
                    stop_reason = f"stopped after {rate_limit_errors} consecutive provider rate-limit errors"
            else:
                rate_limit_errors = 0
            if budget.max_calls and budget.external_calls >= budget.max_calls:
                stop_reason = f"stopped at maximum external-call budget ({budget.max_calls})"
            if budget.max_tokens and budget.total_tokens >= budget.max_tokens:
                stop_reason = f"stopped at maximum provider-token budget ({budget.max_tokens})"
            save_checkpoint()
    except KeyboardInterrupt:
        stop_reason = "interrupted by user; completed records preserved in checkpoint"
        seen = {record["case_id"] for record in records}
        for case in cases:
            if case["id"] not in seen:
                records.append(_not_run_record(case, stop_reason))
        save_checkpoint()
    finally:
        _cleanup_rag_cache(rag_cache)
    # A provider can fail before the case result reaches the normal record
    # construction path (for example, a rate-limit exception raised by a
    # worker). Preserve that attempted case as ERROR; only later cases are
    # NOT_RUN.
    if stop_reason and "rate-limit" in stop_reason.lower():
        seen = {record["case_id"] for record in records}
        if not any(record.get("status") == ERROR and _is_rate_limit_record(record) for record in records):
            attempted = next((case for case in cases if case["id"] not in seen), None)
            if attempted is not None:
                records.append(_provider_error_record(attempted, stop_reason))
    if stop_reason:
        for case in cases:
            if case["id"] not in {record["case_id"] for record in records}:
                records.append(_not_run_record(case, stop_reason))
    save_checkpoint()
    return records, {
        "model": getattr(llm, "model_name", None) or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "judge_enabled": use_judge,
        "budget": budget.snapshot(),
        "stop_reason": stop_reason,
    }


def _baseline_comparison(current_summary: dict, baseline_path: Optional[str]) -> dict:
    if not baseline_path:
        return {"status": "not_requested"}
    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        baseline_summary = baseline.get("summary", {}).get("metrics", {})
        regressions = []
        for name, current in current_summary.items():
            before = baseline_summary.get(name, {}).get("mean")
            after = current.get("mean")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and after < before - 0.05:
                regressions.append({"metric": name, "baseline": before, "current": after, "drop": before - after})
        return {"status": "measured", "baseline": str(baseline_path), "regressions": regressions, "passed": not regressions}
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _run_calibration(path: str, output_dir: Path) -> int:
    """Compare judge scores with manually supplied labels.

    The checked-in template intentionally has no labels. This command refuses
    to turn placeholders into an apparent agreement score.
    """
    calibration_path = Path(path)
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read calibration file: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    cases = payload.get("cases", [])
    if not cases or any(not isinstance(case.get("human_scores"), dict) or not case.get("human_scores") for case in cases):
        print("Calibration is blocked: add real blinded human_scores to the template before running it.", file=sys.stderr)
        return 2
    if any("<paste" in str(case.get("answer", "")) for case in cases):
        print("Calibration is blocked: replace placeholder answers with anonymized reviewed answers.", file=sys.stderr)
        return 2
    try:
        from langchain_groq import ChatGroq
        model = os.getenv("EVAL_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
        judge = ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.0)
    except Exception as exc:
        print(f"Calibration provider unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    rows = []
    absolute_errors = []
    exact = []
    for case in cases:
        judge_result = _judge_answer(judge, {"id": case["id"], "category": "calibration", "subject": "calibration", "input": case["question"], "history": [], "expected_mode": "normal", "reference_facts": [], "evaluation_criteria": [case.get("notes", "")]}, case["answer"], None)
        predicted = judge_result.get("metrics", {})
        human = case["human_scores"]
        for metric_name, human_score in human.items():
            predicted_score = (predicted.get(f"judge.{metric_name}") or {}).get("value")
            if isinstance(human_score, (int, float)) and isinstance(predicted_score, (int, float)):
                absolute_errors.append(abs(float(predicted_score) - float(human_score)))
                exact.append(float(predicted_score) == float(human_score))
        rows.append({"id": case["id"], "judge": judge_result, "human_scores": human})
    result = {
        "version": payload.get("version"),
        "evaluator_version": EVALUATOR_VERSION,
        "status": "measured",
        "case_count": len(cases),
        "exact_agreement": sum(exact) / len(exact) if exact else None,
        "mean_absolute_error": sum(absolute_errors) / len(absolute_errors) if absolute_errors else None,
        "rows": _scrub(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"calibration-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Calibration result: {target}")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows"}}, indent=2))
    return 0


def _build_summary(records: list[dict], metadata: dict, baseline_path: Optional[str]) -> dict:
    statuses = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    categories = {}
    modes = {}
    for grouping, key in ((categories, "category"), (modes, "mode")):
        for record in records:
            grouping.setdefault(record[key], []).append(record)
    return {
        "total_cases": len(records),
        "status_counts": statuses,
        "PASS": statuses.get(PASS, 0),
        "FAIL": statuses.get(FAIL, 0),
        "ERROR": statuses.get(ERROR, 0),
        "NOT_RUN": statuses.get(NOT_RUN, 0),
        "metrics": summarize_records(records),
        "by_category": {key: {"total": len(value), "statuses": _status_counts(value), "metrics": summarize_records(value)} for key, value in categories.items()},
        "by_mode": {key: {"total": len(value), "statuses": _status_counts(value), "metrics": summarize_records(value)} for key, value in modes.items()},
        "latency_ms": _latency_summary(records),
        "token_usage": _token_summary(records, metadata),
        "retry_effectiveness": _retry_summary(records),
        "baseline": _baseline_comparison(summarize_records(records), baseline_path),
        "weakest_metrics": _weakest_metrics(summarize_records(records)),
        "critical_failures": [record["case_id"] for record in records if record["category"] == "mode_state" and record["status"] != PASS],
    }


def _status_counts(records: list[dict]) -> dict:
    result = {}
    for record in records:
        result[record["status"]] = result.get(record["status"], 0) + 1
    return result


def _latency_summary(records: list[dict]) -> dict:
    values = [float(record.get("result", {}).get("latency_ms", record.get("latency_ms"))) for record in records if isinstance(record.get("result", {}).get("latency_ms", record.get("latency_ms")), (int, float))]
    return {"count": len(values), "median_ms": percentile(values, 0.5), "p95_ms": percentile(values, 0.95)}


def _token_summary(records: list[dict], metadata: dict) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    observed_case_ids = set()
    for record in records:
        usage = record.get("result", {}).get("usage") or {}
        if any(isinstance(usage.get(key), int) and usage.get(key) > 0 for key in totals):
            observed_case_ids.add(record["case_id"])
            _merge_usage(totals, usage)
        judge_usage = (record.get("judge") or {}).get("usage") or {}
        if any(isinstance(judge_usage.get(key), int) and judge_usage.get(key) > 0 for key in totals):
            observed_case_ids.add(record["case_id"])
            _merge_usage(totals, judge_usage)
    observed = len(observed_case_ids)
    input_price = metadata.get("cost", {}).get("input_usd_per_million", 0.0)
    output_price = metadata.get("cost", {}).get("output_usd_per_million", 0.0)
    estimated_cost = (totals["input_tokens"] * input_price + totals["output_tokens"] * output_price) / 1_000_000
    return {
        "status": "measured" if observed else "not_run",
        "cases_with_provider_usage": observed,
        "totals": totals if observed else None,
        "estimated_cost_usd": estimated_cost if observed and (input_price or output_price) else None,
        "cost_status": "estimated_from_provider_usage_and_configured_prices" if observed and (input_price or output_price) else "unmeasured",
    }


def _retry_summary(records: list[dict]) -> dict:
    attempts = [
        (record, retry)
        for record in records
        for retry in record.get("retries", [])
    ]
    improved = 0
    comparable = 0
    for record, retry in attempts:
        before = (record.get("judge") or {}).get("overall")
        after = (retry.get("judge") or {}).get("overall")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            comparable += 1
            improved += int(after > before)
    return {
        "attempted": len(attempts),
        "comparable_judge_pairs": comparable,
        "improved_pairs": improved,
        "improvement_rate": (improved / comparable) if comparable else None,
        "policy": "evaluation-only, maximum one additional attempt, never auto-updates production prompts",
    }


def _weakest_metrics(metrics: dict) -> list[dict]:
    measured = [(name, data.get("mean")) for name, data in metrics.items() if isinstance(data.get("mean"), (int, float))]
    return [{"metric": name, "mean": value} for name, value in sorted(measured, key=lambda item: item[1])[:10]]


def _markdown_report(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# TISW Evaluation Report",
        "",
        f"- Run: `{result['metadata']['run_id']}`",
        f"- Timestamp (UTC): `{result['metadata']['timestamp']}`",
        f"- Dataset: `{result['metadata']['dataset_version']}` ({result['metadata']['dataset_hash']})",
        f"- Evaluator: `{result['metadata']['evaluator_version']}`; rubric `{result['metadata']['rubric_version']}`",
        f"- Mode: `{result['metadata']['run_mode']}`; model: `{result['metadata'].get('model') or 'not run'}`",
        "",
        "## Execution",
        "",
        f"Total cases: **{summary['total_cases']}**; PASS: **{summary['PASS']}**; FAIL: **{summary['FAIL']}**; ERROR: **{summary['ERROR']}**; NOT_RUN: **{summary['NOT_RUN']}**.",
        "",
        "Values below are measured only where the metric status is `measured`; `not_run` and `blocked` are not treated as zero.",
        "",
        "## Metrics",
        "",
        "| Metric | Mean | Observed | Statuses |",
        "|---|---:|---:|---|",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(f"| `{name}` | {metric.get('mean') if metric.get('mean') is not None else '—'} | {metric.get('count', 0)} | {metric.get('statuses', {})} |")
    lines.extend(["", "## By category", "", "| Category | Total | Statuses |", "|---|---:|---|"])
    for category, data in sorted(summary["by_category"].items()):
        lines.append(f"| `{category}` | {data['total']} | {data['statuses']} |")
    lines.extend(["", "## Normal versus TISW", "", "| Mode | Total | Statuses |", "|---|---:|---|"])
    for mode, data in sorted(summary["by_mode"].items()):
        lines.append(f"| `{mode}` | {data['total']} | {data['statuses']} |")
    lines.extend(["", "## Reliability and performance", "", f"Latency: `{summary['latency_ms']}`", f"Token usage: `{summary['token_usage']}`", f"Retry effectiveness: `{summary['retry_effectiveness']}`", "", "Token usage and cost are only reported when provider usage metadata was returned. Estimated cost configuration is recorded in the JSON result."])
    lines.extend(["", "## Weakest measured metrics", ""])
    for item in summary["weakest_metrics"]:
        lines.append(f"- `{item['metric']}`: {item['mean']}")
    if summary["critical_failures"]:
        lines.extend(["", "## Critical failures", "", *[f"- `{case_id}`" for case_id in summary["critical_failures"]]])
    failed = [record for record in result["records"] if record["status"] in {FAIL, ERROR, NOT_RUN}]
    if failed:
        lines.extend(["", "## Representative failures", ""])
        for record in failed[:10]:
            lines.append(f"- `{record['case_id']}` ({record['category']}): {record.get('result', {}).get('error') or record.get('result', {}).get('reason') or record['status']}")
    lines.extend(["", "## Limitations", "", "- LLM-judge values require a live run with `--judge`; they are not inferred from answer length.", "- Learning gain, delayed retention, and transfer outcomes are not measured by this response-quality suite; see `evaluations/learning_effectiveness_plan.md`.", "- The code runner is process-isolated with a deny-list and timeout; strong OS/container isolation remains recommended for untrusted generated code.", "- No LangSmith integration was found in the application, so local JSON traces are used."])
    return "\n".join(lines) + "\n"


def run_evaluation(args: argparse.Namespace) -> tuple[dict, int]:
    global _RUNTIME_CONFIG
    load_dotenv()
    runtime = configure_runtime(offline=True if not args.live else None)
    _RUNTIME_CONFIG = runtime
    application = _application()
    cases = load_cases()
    selected = cases
    if args.category:
        allowed = set(args.category)
        selected = [case for case in cases if case["category"] in allowed]
    if args.case_id:
        requested = set(args.case_id)
        known = {case["id"] for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"Unknown case ID(s): {', '.join(missing)}")
        selected = [case for case in cases if case["id"] in requested]
    if args.limit:
        selected = selected[:args.limit]
    mode = "live" if args.live else "offline"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir = Path(args.output or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    checkpoint_path = Path(args.resume) if args.resume else output_dir / f"{run_id}.checkpoint.json"
    checkpoint_fingerprint = {
        "schema": "tisw-evaluation-checkpoint-v1",
        "evaluator_version": EVALUATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_hash": _json_hash(cases),
        "selected_case_ids": [case["id"] for case in selected],
        "run_mode": mode,
        "judge_enabled": bool(args.judge),
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": {"normal": _hash_file_or_value(application.NORMAL_MODE_PROMPT), "tisw": _hash_file_or_value(application.TISW_MODE_PROMPT)},
    }
    if args.live:
        try:
            records, live_metadata = _live_records(
                selected,
                use_judge=args.judge,
                timeout_seconds=args.timeout,
                retry_limit=args.retry_quality,
                cache_dir=cache_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_fingerprint=checkpoint_fingerprint,
                max_external_calls=args.max_external_calls,
                max_total_tokens=args.max_total_tokens,
                max_rate_limit_errors=args.max_rate_limit_errors,
            )
        except Exception as exc:
            failure = {
                "status": ERROR,
                "reason": f"Live evaluation could not start: {type(exc).__name__}: {exc}",
                "hint": "Configure a valid GROQ_API_KEY or run --offline for deterministic checks.",
            }
            (output_dir / f"{run_id}.blocked.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            print(json.dumps(failure, indent=2), file=sys.stderr)
            return failure, 2
    else:
        records = _offline_records(selected, include_rag=args.include_rag, cache_dir=cache_dir)
        live_metadata = {"model": None, "judge_enabled": False}

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "dataset_hash": _json_hash(cases),
        "selected_case_count": len(selected),
        "evaluator_version": EVALUATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": {
            "normal_mode": _hash_file_or_value(application.NORMAL_MODE_PROMPT),
            "tisw_mode": _hash_file_or_value(application.TISW_MODE_PROMPT),
        },
        "run_mode": mode,
        "model": live_metadata.get("model"),
        "judge_enabled": live_metadata.get("judge_enabled", False),
        "temperature": 0.3 if args.live else None,
        "timeout_seconds": args.timeout,
        "retry_limit": args.retry_quality,
        "cache_dir": str(cache_dir),
        "checkpoint_path": str(checkpoint_path),
        "resume_requested": bool(args.resume),
        "max_external_calls": args.max_external_calls,
        "max_total_tokens": args.max_total_tokens,
        "max_rate_limit_errors": args.max_rate_limit_errors,
        "run_control": live_metadata.get("stop_reason"),
        "budget_snapshot": live_metadata.get("budget"),
        "cost": {
            "status": "estimated_only",
            "input_usd_per_million": float(os.getenv("EVAL_INPUT_USD_PER_MILLION", "0")),
            "output_usd_per_million": float(os.getenv("EVAL_OUTPUT_USD_PER_MILLION", "0")),
            "note": "Zero values mean no price was configured; exact cost is not claimed.",
        },
        "offline": runtime.offline,
        "langsmith_tracing_enabled": runtime.tracing_enabled,
    }
    result = {"metadata": metadata, "summary": _build_summary(records, metadata, args.baseline), "records": _scrub(records)}
    json_path = output_dir / f"{run_id}.json"
    markdown_path = output_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(f"Evaluation JSON: {json_path}")
    print(f"Evaluation report: {markdown_path}")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))

    critical_failed = bool(result["summary"]["critical_failures"])
    baseline_failed = result["summary"]["baseline"].get("status") == "measured" and not result["summary"]["baseline"].get("passed", True)
    if (critical_failed or (baseline_failed and args.fail_on_regression)) and not args.no_fail_on_critical:
        return result, 1
    return result, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed TISW evaluation dataset.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true", help="Run deterministic checks without a Groq response model (default).")
    group.add_argument("--live", action="store_true", help="Run cases through the actual Groq-backed LangGraph workflow.")
    parser.add_argument("--judge", action="store_true", help="Use a configured Groq LLM judge during --live.")
    parser.add_argument("--retry-quality", type=int, choices=(0, 1), default=0, help="At most one independent additional attempt after a measured judge failure.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Best-effort per graph invocation timeout in seconds.")
    parser.add_argument("--include-rag", action="store_true", help="Run actual hybrid retrieval for RAG fixtures in offline mode.")
    parser.add_argument("--category", action="append", help="Restrict to a category; may be repeated.")
    parser.add_argument("--case-id", action="append", help="Restrict to an explicit stable case ID; may be repeated.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    parser.add_argument("--output", help="Output directory for JSON and Markdown reports.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Persistent response/judge cache directory.")
    parser.add_argument("--resume", help="Resume from a compatible checkpoint JSON without repeating PASS/FAIL cases.")
    parser.add_argument("--max-external-calls", type=int, default=300, help="Maximum provider generation plus judge calls for this run.")
    parser.add_argument("--max-total-tokens", type=int, default=180000, help="Maximum observed provider tokens; zero disables this limit.")
    parser.add_argument("--max-rate-limit-errors", type=int, default=2, help="Stop and mark remaining cases NOT_RUN after this many rate-limit errors.")
    parser.add_argument("--baseline", help="Previous JSON report to compare against.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Return non-zero when baseline metrics regress by more than 0.05.")
    parser.add_argument("--no-fail-on-critical", action="store_true", help="Do not fail the process for critical mode-state failures.")
    parser.add_argument("--calibrate", help="Run judge calibration from a manually labelled JSON file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.live:
        load_dotenv()
        runtime = configure_runtime()
        if runtime.offline:
            print("--live is blocked while TISW_OFFLINE=true", file=sys.stderr)
            return 2
    if args.calibrate:
        return _run_calibration(args.calibrate, Path(args.output or DEFAULT_OUTPUT_DIR))
    if not args.live and not args.offline:
        args.offline = True
    if args.judge and not args.live:
        print("--judge requires --live", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be positive", file=sys.stderr)
        return 2
    _result, exit_code = run_evaluation(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
