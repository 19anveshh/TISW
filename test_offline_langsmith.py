"""Offline-only runtime and LangSmith configuration checks."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_groq import ChatGroq


def run_offline_langsmith_tests() -> None:
    import runtime_config

    with patch.dict(
        os.environ,
        {
            "TISW_OFFLINE": "true",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_TRACING_V2": "true",
            "LANGCHAIN_TRACING_V2": "true",
        },
        clear=False,
    ):
        config = runtime_config.configure_runtime()
        assert config.offline is True
        assert config.tracing_enabled is False
        assert os.environ["LANGSMITH_TRACING_V2"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

        def blocked_connect(*_args, **_kwargs):
            raise AssertionError("offline test attempted a network connection")

        sys.modules.pop("main", None)
        with patch.object(ChatGroq, "__init__", return_value=None) as constructor, patch.object(socket.socket, "connect", blocked_connect):
            import main
            assert main.tutor_app is None
            assert constructor.call_count == 0
            try:
                main.load_and_validate_groq_llm()
            except RuntimeError as exc:
                assert "TISW_OFFLINE" in str(exc)
            else:
                raise AssertionError("offline Groq initialization did not fail closed")
            assert constructor.call_count == 0

    with patch.dict(
        os.environ,
        {
            "TISW_OFFLINE": "false",
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_TRACING_V2": "false",
            "LANGCHAIN_TRACING_V2": "false",
        },
        clear=False,
    ):
        config = runtime_config.configure_runtime()
        assert config.tracing_enabled is False

    with patch.dict(
        os.environ,
        {
            "TISW_OFFLINE": "false",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_TRACING_V2": "false",
            "LANGCHAIN_TRACING_V2": "false",
        },
        clear=False,
    ):
        config = runtime_config.configure_runtime()
        assert config.tracing_requested is True
        assert config.tracing_enabled is True
        assert os.environ["LANGSMITH_TRACING_V2"] == "true"

    # Re-enter the safe default before importing the evaluator.
    runtime_config.configure_runtime(offline=True)
    import evaluations.runner as runner
    import evaluations.langsmith_adapter as adapter
    from evaluations.dataset import load_cases

    cases = load_cases()
    example = adapter.case_to_langsmith_example(cases[0])
    assert example["metadata"]["case_id"] == cases[0]["id"]
    assert "question" in example["inputs"]

    with TemporaryDirectory() as temp_dir:
        fact = next(case for case in cases if case["id"] == "FACT-001")
        records = runner._offline_records([fact], include_rag=False, cache_dir=Path(temp_dir))
        assert records[0]["status"] == runner.NOT_RUN
        assert records[0]["response_source"] == "missing_cache"

        current_config = runner._config_fingerprint()
        stale_config = {**current_config, "evaluator_version": "1.1.2"}
        assert runner._response_cache_key(fact, None, stale_config) != runner._response_cache_key(fact, None, current_config)
        current_key = runner._response_cache_key(fact, None, current_config)
        runner._cache_write(Path(temp_dir), current_key, {"status": runner.PASS, "output": "cached"})
        stale_records = runner._offline_records([fact], include_rag=False, cache_dir=Path(temp_dir))
        assert stale_records[0]["status"] == runner.NOT_RUN
        assert stale_records[0]["response_source"] == "cache"
        assert "compatible cached judge" in stale_records[0]["result"]["reason"]

        checkpoint = Path(temp_dir) / "checkpoint.json"
        runner._checkpoint_write(checkpoint, {"fingerprint": {"evaluator_version": "1.1.3"}, "records": []})
        assert runner._checkpoint_read(checkpoint)["fingerprint"]["evaluator_version"] == "1.1.3"

    print("OK: offline provider/tracing guards, cache-miss status, and local LangSmith mapping passed")


if __name__ == "__main__":
    run_offline_langsmith_tests()
