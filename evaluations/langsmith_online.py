"""Publish the fixed evaluation set and run a bounded LangSmith experiment.

This command is deliberately opt-in. It requires ``LANGSMITH_API_KEY`` and
``GROQ_API_KEY`` only when ``--run`` is requested; neither secret is written to
the dataset, experiment metadata, reports, or logs.

Examples::

    uv run python -m evaluations.langsmith_online --publish
    uv run python -m evaluations.langsmith_online --publish --run --limit 10
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate

from main import create_tutor_graph, load_and_validate_groq_llm

from . import DATASET_VERSION, EVALUATOR_VERSION
from .dataset import load_cases
from .langsmith_adapter import case_to_langsmith_example
from .metrics import fact_coverage


DEFAULT_DATASET_NAME = f"TISW {DATASET_VERSION}"


def _require_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required for this command and was not found in the environment.")
    return value


def _select_cases(args: argparse.Namespace) -> list[dict]:
    cases = load_cases()
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
        found = {case["id"] for case in cases}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Unknown case ID(s): {', '.join(missing)}")
    if args.category:
        allowed = set(args.category)
        cases = [case for case in cases if case["category"] in allowed]
    if args.limit:
        cases = cases[:args.limit]
    return cases


def publish_dataset(client: Client, dataset_name: str, cases: list[dict]) -> tuple[Any, int]:
    """Create the versioned dataset and insert only missing stable case IDs."""
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name,
            description=(
                f"TISW fixed evaluation dataset {DATASET_VERSION}; "
                f"published by evaluator {EVALUATOR_VERSION}."
            ),
            metadata={
                "application": "TISW",
                "dataset_version": DATASET_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
            },
        )

    existing_ids = {
        (example.metadata or {}).get("case_id")
        for example in client.list_examples(dataset_id=dataset.id, limit=1000)
        if getattr(example, "metadata", None)
    }
    examples = []
    for case in cases:
        if case.get("id") in existing_ids:
            continue
        example = case_to_langsmith_example(case)
        examples.append({
            "inputs": example["inputs"],
            "reference_outputs": example["outputs"],
            "metadata": example["metadata"],
        })
    if examples:
        client.create_examples(dataset_id=dataset.id, examples=examples, max_concurrency=1)
    return dataset, len(examples)


def _history_messages(history: list[dict]) -> list[Any]:
    messages = []
    for item in history:
        role = item.get("role")
        content = str(item.get("content", ""))
        if role == "user":
            messages.append(HumanMessage(content=content))
    return messages


def _target_factory():
    llm = load_and_validate_groq_llm()

    def target(inputs: dict[str, Any]) -> dict[str, str]:
        question = str(inputs.get("question", "")).strip()
        graph = create_tutor_graph(llm, InMemorySaver())
        messages = _history_messages(inputs.get("history") or [])
        messages.append(HumanMessage(content=question))
        result = graph.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": f"langsmith-{uuid.uuid4().hex}"}},
        )
        output = result.get("messages") or []
        return {"response": str(output[-1].content) if output else ""}

    return target


def _fact_evaluator(run: Any, example: Any) -> dict[str, Any]:
    output = run.outputs or {}
    reference = example.outputs or {}
    coverage = fact_coverage(str(output.get("response", "")), reference.get("reference_facts", []))
    return {
        "key": "fact_coverage",
        "score": coverage["score"] if coverage["score"] is not None else 0.0,
        "comment": f"Matched {coverage['matched']} of {coverage['required']} reference facts.",
    }


def run_experiment(client: Client, dataset_name: str, *, max_concurrency: int) -> Any:
    return evaluate(
        _target_factory(),
        data=dataset_name,
        evaluators=[_fact_evaluator],
        client=client,
        experiment_prefix="tisw-production-review",
        metadata={
            "application": "TISW",
            "dataset_version": DATASET_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
        },
        max_concurrency=max_concurrency,
        upload_results=True,
        error_handling="log",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish and evaluate TISW cases in LangSmith.")
    parser.add_argument("--publish", action="store_true", help="Create the hosted dataset and add missing cases.")
    parser.add_argument("--run", action="store_true", help="Run the bounded hosted experiment after publishing.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--category", action="append")
    parser.add_argument("--max-concurrency", type=int, choices=(1, 2, 3), default=1)
    return parser


def _dataset_name_for_selection(args: argparse.Namespace) -> str:
    if args.dataset_name != DEFAULT_DATASET_NAME or not (args.limit or args.case_id or args.category):
        return args.dataset_name
    selection = "|".join([
        str(args.limit or "all"),
        ",".join(sorted(args.case_id or [])),
        ",".join(sorted(args.category or [])),
    ])
    suffix = hashlib.sha256(selection.encode("utf-8")).hexdigest()[:10]
    return f"{args.dataset_name} sample-{suffix}"


def main() -> int:
    args = build_parser().parse_args()
    if not args.publish and not args.run:
        raise SystemExit("Choose --publish, --run, or both.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive.")
    client = Client(api_key=_require_key("LANGSMITH_API_KEY"))
    cases = _select_cases(args)
    dataset_name = _dataset_name_for_selection(args)
    dataset, added = publish_dataset(client, dataset_name, cases)
    print(f"LangSmith dataset ready: {dataset.name} ({added} new examples)")
    if args.run:
        _require_key("GROQ_API_KEY")
        result = run_experiment(client, dataset_name, max_concurrency=args.max_concurrency)
        print(f"LangSmith experiment complete: {result.experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
