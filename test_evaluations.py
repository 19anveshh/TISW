"""Offline regression checks for the evaluation framework itself."""

from evaluations.dataset import load_cases
from evaluations.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from evaluations.runner import EvaluationCallBudget, ERROR, PASS, _case_record, _execute_live_case, _judge_answer, _provider_error_record, _run_code_fixture, _run_mode_case
from langchain_core.messages import AIMessage
from pathlib import Path
from tempfile import TemporaryDirectory


class FakeJudge:
    def invoke(self, _messages):
        return AIMessage(content='{"scores":{"general.relevance":5,"general.factual_correctness":4},"overall":4,"passed":true,"rationale":"focused","evidence":["answers the question"]}')


class BadJudge:
    def invoke(self, _messages):
        return AIMessage(content="not-json")


class CapturingJudge:
    def __init__(self, content=None):
        self.content = content or '{"scores":{"general.relevance":5},"overall":5,"passed":true,"rationale":"meets the request","evidence":["direct answer"]}'
        self.prompt = ""

    def invoke(self, messages):
        self.prompt = str(messages[0].content)
        return AIMessage(content=self.content)


class MissingScoresJudge:
    def invoke(self, _messages):
        return AIMessage(content='{"overall":4,"passed":true,"rationale":"ok","evidence":[]}')


class OutOfRangeJudge:
    def invoke(self, _messages):
        return AIMessage(content='{"scores":{"general.relevance":6},"overall":6,"passed":true,"rationale":"ok","evidence":[]}')


class FailingFormatJudge:
    def invoke(self, _messages):
        return AIMessage(content='{"scores":{"general.instruction_following":1},"overall":1,"passed":false,"rationale":"required JSON was not provided","evidence":["plain text answer"]}')


class FakeResponseLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        message = AIMessage(content="cached response")
        message.usage_metadata = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        return message


class RateLimitedLLM:
    def invoke(self, _messages):
        raise RuntimeError("RateLimitError: 429 tokens per day")


class EmptyResponseLLM:
    def invoke(self, _messages):
        return AIMessage(content="")


def run_evaluation_tests():
    cases = load_cases()
    assert len(cases) == 150
    assert len({case["id"] for case in cases}) == 150
    required = {"id", "input", "history", "expected_mode", "reference_facts", "evaluation_criteria", "metadata"}
    assert all(required.issubset(case) for case in cases)

    mode_cases = [case for case in cases if case["category"] == "mode_state"]
    mode_results = [_run_mode_case(case) for case in mode_cases]
    assert all(result["status"] == PASS for result in mode_results)
    provider_error = _provider_error_record(cases[0], "stopped after 1 consecutive provider rate-limit errors")
    assert provider_error["status"] == ERROR
    assert provider_error["result"]["provider_failure"] is True

    assert precision_at_k(["a", "b", "c"], {"a"}, 3) == 1 / 3
    assert recall_at_k(["a", "b"], {"a", "c"}, 2) == 0.5
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert ndcg_at_k(["a", "b"], {"a": 3.0, "b": 1.0}, 2) == 1.0

    code_case = next(case for case in cases if case.get("code_task"))
    code_result = _run_code_fixture(code_case)
    assert code_result["status"] == PASS, code_result
    judge_result = _judge_answer(
        FakeJudge(),
        {"id": "JUDGE-001", "category": "factual", "subject": "test", "input": "What is 2 + 2?", "history": [], "expected_mode": "normal", "reference_facts": ["4"], "evaluation_criteria": ["Answer correctly."]},
        "4",
        None,
    )
    assert judge_result["status"] == "measured"
    assert judge_result["metrics"]["judge.general.relevance"]["value"] == 5.0

    natural_case = {"id": "NATURAL-001", "category": "factual", "subject": "test", "input": "What is 2 + 2?", "history": [], "expected_mode": "normal", "reference_facts": ["4"], "evaluation_criteria": ["Answer correctly."]}
    natural_judge = CapturingJudge()
    natural_result = _judge_answer(natural_judge, natural_case, "The answer is 4.", None)
    assert natural_result["status"] == "measured"
    assert "explicit JSON-output requirement was detected in this case: false" in natural_judge.prompt

    json_case = {"id": "JSON-001", "category": "factual", "subject": "test", "input": "Return a JSON object with the answer.", "history": [], "expected_mode": "normal", "reference_facts": ["4"], "evaluation_criteria": ["Return JSON."]}
    json_judge = CapturingJudge()
    json_result = _judge_answer(json_judge, json_case, '{"answer": "4"}', None)
    assert json_result["status"] == "measured"
    assert "explicit JSON-output requirement was detected in this case: true" in json_judge.prompt

    code_case = {"id": "CODE-TEST", "category": "coding", "subject": "test", "input": "Write Python that adds two numbers.", "history": [], "expected_mode": "normal", "reference_facts": [], "evaluation_criteria": ["Provide runnable code."]}
    assert _judge_answer(CapturingJudge(), code_case, "def add(a, b): return a + b", None)["status"] == "measured"
    assert _judge_answer(MissingScoresJudge(), natural_case, "4", None)["reason"] == "judge_json_missing_scores"
    assert _judge_answer(OutOfRangeJudge(), natural_case, "4", None)["reason"] == "judge_score_out_of_range_or_non_integer:general.relevance"
    assert "RateLimitError" in _judge_answer(RateLimitedLLM(), natural_case, "4", None)["reason"]

    format_record = _case_record(json_case, {"status": PASS, "output": "The answer is 4.", "metrics": {}}, run_mode="live", judge_llm=FailingFormatJudge())
    assert format_record["status"] == "FAIL"
    assert format_record["judge"]["passed"] is False
    assert cases_by_id(cases)["ADV-005"]["expected_mode"] == "tisw"
    assert cases_by_id(cases)["ADV-013"]["expected_mode"] == "tisw"

    with TemporaryDirectory() as temp_dir:
        fake_llm = FakeResponseLLM()
        case = cases_by_id(cases)["FACT-001"]
        config = {"model": "fixture-model", "temperature": 0.3, "prompt_version": "test"}
        first = _execute_live_case(case, fake_llm, timeout_seconds=3, rag_cache={}, budget=EvaluationCallBudget(10, 100), cache_dir=Path(temp_dir), config_fingerprint=config)
        second = _execute_live_case(case, fake_llm, timeout_seconds=3, rag_cache={}, budget=EvaluationCallBudget(10, 100), cache_dir=Path(temp_dir), config_fingerprint=config)
        assert first["status"] == PASS
        assert second.get("cache_hit") is True
        assert fake_llm.calls == 1

        bad_record = _case_record(case, first, run_mode="live", judge_llm=BadJudge(), cache_dir=Path(temp_dir), judge_fingerprint={"model": "bad"})
        assert bad_record["status"] == ERROR

        rate_limited = _execute_live_case(case, RateLimitedLLM(), timeout_seconds=3, rag_cache={}, budget=EvaluationCallBudget(10, 100), cache_dir=Path(temp_dir) / "rate", config_fingerprint={"model": "rate-model"})
        assert rate_limited["status"] == "ERROR"
        assert "RateLimitError" in rate_limited["error"]

        empty_response = _execute_live_case(case, EmptyResponseLLM(), timeout_seconds=3, rag_cache={}, budget=EvaluationCallBudget(10, 100), cache_dir=Path(temp_dir) / "empty", config_fingerprint={"model": "empty-model"})
        assert empty_response["status"] == "ERROR"
        assert empty_response["error"] == "missing model response"
    print("OK: 150-case dataset, deterministic mode checks, retrieval math, and isolated code fixture passed")


def cases_by_id(cases):
    return {case["id"]: case for case in cases}


if __name__ == "__main__":
    run_evaluation_tests()
