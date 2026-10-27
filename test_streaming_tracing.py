"""Offline streaming and request-scoped tracing tests using only fakes."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import server
from main import create_tutor_streaming_graph


class FakeStreamingGraph:
    def __init__(self, answer: str = "Hello \u4e16\u754c\n```python\nprint(1)\n```"):
        self.answer = answer
        self.calls = 0
        self.final_event_emitted = False
        self.inputs = []

    def get_state(self, _config):
        return type("State", (), {"values": {"active_mode": "normal", "messages": []}})()

    async def astream_events(self, payload, config=None, version="v2"):
        self.calls += 1
        self.inputs.append((payload, config, version))
        yield {"event": "on_chain_start", "name": "LangGraph", "data": {}}
        chunks = ["Hello", " ", "\u4e16\u754c\n", "```python\nprint(1)\n```"]
        for text in chunks:
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatGroq",
                "metadata": {"langgraph_node": "tutor"},
                "data": {"chunk": AIMessageChunk(content=text)},
            }
        self.final_event_emitted = True
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"messages": [AIMessage(content=self.answer)]}},
        }


class FakeSyncGraph:
    def __init__(self):
        self.calls = 0

    def get_state(self, _config):
        return type("State", (), {"values": {"active_mode": "normal", "messages": []}})()

    def invoke(self, payload, config=None):
        self.calls += 1
        return {"messages": [AIMessage(content="sync response")], "active_mode": "normal"}


@contextmanager
def _fake_trace_context(*_args, **_kwargs):
    yield


def _events(text: str) -> list[dict]:
    frames = [frame for frame in text.split("\n\n") if frame.strip()]
    parsed = []
    for frame in frames:
        event = next(line[6:].strip() for line in frame.splitlines() if line.startswith("event:"))
        data = next(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
        parsed.append({"event": event, "data": json.loads(data)})
    return parsed


async def _collect(generator) -> list[dict]:
    collected = []
    async for frame in generator:
        collected.append(frame)
    return _events("".join(collected))


def run_streaming_tracing_tests() -> None:
    assert server._extract_tutor_delta({
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": "tutor"},
        "data": {"chunk": AIMessageChunk(content="visible")},
    }) == "visible"
    assert server._extract_tutor_delta({
        "event": "on_chat_model_stream",
        "metadata": {},
        "data": {"chunk": AIMessageChunk(content="hidden")},
    }) == ""
    assert server._extract_tutor_delta({
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": "other"},
        "data": {"chunk": AIMessageChunk(content="other")},
    }) == ""
    assert server._extract_tutor_delta({
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": "tutor"},
        "data": {"chunk": AIMessageChunk(content=[{"type": "tool_call_chunk", "args": "secret"}])},
    }) == ""
    assert server._extract_tutor_delta({
        "event": "on_chain_stream",
        "metadata": {"langgraph_node": "tutor"},
        "data": {"chunk": AIMessageChunk(content="snapshot")},
    }) == ""
    assert server._final_response_from_output({
        "messages": [AIMessage(content="final")],
    }) == "final"
    assert server._final_response_from_output({
        "messages": [HumanMessage(content="user text")],
    }) == ""

    client = TestClient(server.app)
    runtime = client.get("/api/runtime")
    assert runtime.status_code == 200
    runtime_data = runtime.json()
    assert runtime_data["offline"] is True
    assert runtime_data["tracing_available"] is False
    assert "api_key" not in runtime.text.lower()
    streaming_graph = FakeStreamingGraph()
    sync_graph = FakeSyncGraph()

    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "tutor_stream_app", streaming_graph), \
         patch.object(server, "tutor_app", sync_graph), \
         patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context):
        response = client.post(
            "/api/chat/stream",
            json={"message": "Explain this", "thread_id": "stream-test", "tracing_enabled": True},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        assert [item["event"] for item in events] == ["start", "delta", "delta", "delta", "delta", "done"]
        deltas = "".join(item["data"]["text"] for item in events if item["event"] == "delta")
        assert deltas == streaming_graph.answer
        assert events[-1]["data"]["response"] == streaming_graph.answer
        assert streaming_graph.calls == 1
        assert streaming_graph.inputs[0][2] == "v2"

        non_stream = client.post("/api/chat", json={"message": "Keep compatibility", "thread_id": "sync-test"})
        assert non_stream.status_code == 200, non_stream.text
        assert non_stream.json()["response"] == "sync response"
        assert sync_graph.calls == 1

    # The first delta is observable before the graph's final state event.
    async def collect_prefix():
        graph = FakeStreamingGraph()
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "tutor_stream_app", graph), \
             patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context):
            stream = server._stream_chat_events(
                server.ChatRequest(message="prefix", thread_id="prefix-test"),
                None,
            )
            first = await anext(stream)
            second = await anext(stream)
            assert 'event: start' in first
            assert 'event: delta' in second
            assert graph.final_event_emitted is False
            await stream.aclose()

    asyncio.run(collect_prefix())

    # Request correlation regression: the exact browser text must reach the
    # graph, and sequential turns must not replay or cross-associate answers.
    async def collect_request_correlation():
        class EchoGraph:
            def __init__(self):
                self.calls = []

            def get_state(self, _config):
                return type("State", (), {"values": {"active_mode": "normal", "messages": []}})()

            async def astream_events(self, payload, config=None, version="v2"):
                question = payload["messages"][-1].content
                self.calls.append((question, config, version))
                answer = f"answer-for:{question}"
                yield {
                    "event": "on_chat_model_stream",
                    "name": "ChatGroq",
                    "metadata": {"langgraph_node": "tutor"},
                    "data": {"chunk": AIMessageChunk(content=answer)},
                }
                yield {
                    "event": "on_chain_end",
                    "name": config["run_name"],
                    "data": {"output": {"messages": [AIMessage(content=answer)]}},
                }

        graph = EchoGraph()
        server._thread_locks.clear()
        server._turn_records.clear()
        first = server.ChatRequest(
            message="What is Python?",
            thread_id="correlation-thread",
            turn_id="python-turn",
        )
        second = server.ChatRequest(
            message="Explain binary search.",
            thread_id="correlation-thread",
            turn_id="binary-turn",
        )
        diagnostics = []
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "tutor_stream_app", graph), \
             patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context), \
             patch.object(
                 server,
                 "_stream_diagnostic",
                 side_effect=lambda stage, **details: diagnostics.append((stage, details)),
             ):
            first_events = await _collect(server._stream_chat_events(first, None))
            second_events = await _collect(server._stream_chat_events(second, None))
            replay_events = await _collect(server._stream_chat_events(first, None))

        assert [call[0] for call in graph.calls] == [
            "What is Python?",
            "Explain binary search.",
        ]
        assert first_events[-1]["data"] == {
            "response": "answer-for:What is Python?",
            "thread_id": "correlation-thread",
            "turn_id": "python-turn",
            "mode": "normal",
            "rag_used": False,
            "sources": [],
        }
        assert second_events[-1]["data"]["response"] == "answer-for:Explain binary search."
        assert replay_events[-1]["data"]["response"] == "answer-for:What is Python?"
        assert replay_events[-1]["data"]["replay"] is True
        assert len(graph.calls) == 2
        assert not server._thread_locks
        python_diagnostics = [
            (stage, details)
            for stage, details in diagnostics
            if details.get("turn_id") == "python-turn"
        ]
        assert [stage for stage, _details in python_diagnostics[:4]] == [
            "request_received", "graph_input", "first_delta", "completed",
        ]
        assert len({details["request_id"] for _stage, details in python_diagnostics[:4]}) == 1
        assert python_diagnostics[1][1]["message"] == "What is Python?"

    asyncio.run(collect_request_correlation())

    # Offline mode wins over a requested UI toggle and never touches the graph.
    offline_graph = FakeStreamingGraph()
    with patch.dict(os.environ, {"TISW_MOCK_STREAMING": "false"}, clear=False), \
         patch.object(server, "is_offline_mode", return_value=True), patch.object(server, "tutor_stream_app", offline_graph):
        response = client.post("/api/chat/stream", json={"message": "blocked", "tracing_enabled": True})
        events = _events(response.text)
        assert events[-1]["event"] == "error"
        assert "offline" in events[-1]["data"]["message"].lower()
        assert offline_graph.calls == 0

    # Explicit local fixture mode supports visual browser checks without a
    # provider; it is never enabled by ordinary offline mode.
    with patch.dict(os.environ, {"TISW_MOCK_STREAMING": "true"}, clear=False), \
         patch.object(server, "is_offline_mode", return_value=True), \
         patch.object(server, "tutor_stream_app", offline_graph):
        response = client.post("/api/chat/stream", json={"message": "mock"})
        mock_events = _events(response.text)
        assert mock_events[0]["data"]["mock"] is True
        assert mock_events[-1]["event"] == "done"
        assert mock_events[-1]["data"]["mock"] is True
        assert offline_graph.calls == 0

    # Runtime tracing context is request-scoped and does not mutate env flags.
    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "_langsmith_key_configured", return_value=True), \
         patch("langsmith.tracing_context", side_effect=_fake_trace_context) as tracing_factory:
        with server._request_tracing_context(
            True,
            thread_id="trace-thread",
            mode="tisw",
            rag_used=True,
            streaming=True,
            request_id="request-1",
        ):
            pass
        assert tracing_factory.call_count == 1
        kwargs = tracing_factory.call_args.kwargs
        assert kwargs["enabled"] is True
        assert kwargs["project_name"] == "tisw"
        assert kwargs["metadata"]["thread_id_hash"] != "trace-thread"
        assert "trace-thread" not in json.dumps(kwargs["metadata"])

    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "_langsmith_key_configured", return_value=True), \
         patch("langsmith.tracing_context", side_effect=_fake_trace_context) as tracing_factory:
        with server._request_tracing_context(
            False,
            thread_id="off-thread",
            mode="normal",
            rag_used=False,
            streaming=True,
            request_id="request-off",
        ):
            pass
        assert tracing_factory.call_args.kwargs["enabled"] is False

    # A tracing exporter cleanup failure must not convert a completed model
    # response into the generic stream failure shown by the browser.
    class FailingTraceContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise RuntimeError("simulated trace exporter cleanup failure")

    trace_failure_graph = FakeStreamingGraph(answer="trace-safe response")
    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "tutor_stream_app", trace_failure_graph), \
         patch.object(server, "_langsmith_key_configured", return_value=True), \
         patch("langsmith.tracing_context", return_value=FailingTraceContext()):
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "trace cleanup probe",
                "thread_id": "trace-cleanup-thread",
                "turn_id": "trace-cleanup-turn",
                "tracing_enabled": True,
            },
        )
        trace_events = _events(response.text)
        assert trace_events[-1]["event"] == "done"
        assert trace_events[-1]["data"]["response"] == "trace-safe response"
        assert trace_failure_graph.calls == 1

    with patch.object(server, "is_offline_mode", return_value=True), \
         patch.object(server, "_langsmith_key_configured", return_value=True), \
         patch("langsmith.tracing_context", side_effect=AssertionError("offline tracing was constructed")):
        with server._request_tracing_context(
            True,
            thread_id="offline-thread",
            mode="normal",
            rag_used=False,
            streaming=False,
            request_id="request-2",
        ):
            pass

    # The native streaming graph calls the model once and checkpoints one final answer.
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def astream(self, _messages, config=None):
            self.calls += 1
            for text in ("one", " ", "two"):
                yield AIMessageChunk(content=text)

    async def run_graph():
        llm = FakeLLM()
        graph = create_tutor_streaming_graph(llm, InMemorySaver())
        config = {"configurable": {"thread_id": "checkpoint-stream"}}
        result = await graph.ainvoke({"messages": [HumanMessage(content="question")]}, config=config)
        assert llm.calls == 1
        assert result["messages"][-1].content == "one two"
        state = graph.get_state(config).values
        assert state["messages"][-1].content == "one two"

    asyncio.run(run_graph())

    async def run_concurrency_and_idempotency():
        server._thread_locks.clear()
        server._turn_records.clear()
        first_started = asyncio.Event()
        both_started = asyncio.Event()
        release = asyncio.Event()

        class ControlledLLM:
            def __init__(self):
                self.calls = 0

            async def astream(self, _messages, config=None):
                self.calls += 1
                call = self.calls
                if call == 1:
                    first_started.set()
                    await release.wait()
                elif call == 2:
                    both_started.set()
                yield AIMessageChunk(content=f"answer-{call}")

        llm = ControlledLLM()
        checkpointer = InMemorySaver()
        graph = create_tutor_streaming_graph(llm, checkpointer)
        payload_one = server.ChatRequest(message="one", thread_id="same-thread", turn_id="turn-one")
        payload_two = server.ChatRequest(message="two", thread_id="same-thread", turn_id="turn-two")

        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "tutor_stream_app", graph), \
             patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context):
            first_task = asyncio.create_task(_collect(server._stream_chat_events(payload_one, None)))
            await first_started.wait()
            second_task = asyncio.create_task(_collect(server._stream_chat_events(payload_two, None)))
            release.set()
            first_events, second_events = await asyncio.gather(first_task, second_task)

        assert both_started.is_set()
        assert first_events[-1]["data"]["response"] == "answer-1"
        assert second_events[-1]["data"]["response"] == "answer-2"
        state_messages = graph.get_state({"configurable": {"thread_id": "same-thread"}}).values["messages"]
        assert [message.content for message in state_messages] == ["one", "answer-1", "two", "answer-2"]
        assert not server._thread_locks

        # Different threads are allowed to run concurrently.
        server._thread_locks.clear()
        server._turn_records.clear()
        thread_starts = {"a": asyncio.Event(), "b": asyncio.Event()}
        thread_release = asyncio.Event()

        class ParallelLLM:
            def __init__(self): self.calls = 0

            async def astream(self, messages, config=None):
                self.calls += 1
                text = str(messages[-1].content)
                thread_starts["a" if text == "thread-a" else "b"].set()
                await thread_release.wait()
                yield AIMessageChunk(content=text)

        parallel_graph = create_tutor_streaming_graph(ParallelLLM(), InMemorySaver())
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "tutor_stream_app", parallel_graph), \
             patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context):
            task_a = asyncio.create_task(_collect(server._stream_chat_events(server.ChatRequest(message="thread-a", thread_id="a", turn_id="a1"), None)))
            task_b = asyncio.create_task(_collect(server._stream_chat_events(server.ChatRequest(message="thread-b", thread_id="b", turn_id="b1"), None)))
            await thread_starts["a"].wait()
            await thread_starts["b"].wait()
            thread_release.set()
            await asyncio.gather(task_a, task_b)
        assert not server._thread_locks

        # Cancellation restores the pre-turn checkpoint and permits a retry.
        server._thread_locks.clear()
        server._turn_records.clear()
        cancel_started = asyncio.Event()
        cancel_release = asyncio.Event()

        class RetryLLM:
            def __init__(self): self.calls = 0

            async def astream(self, _messages, config=None):
                self.calls += 1
                if self.calls == 1:
                    cancel_started.set()
                    yield AIMessageChunk(content="partial")
                    await cancel_release.wait()
                else:
                    yield AIMessageChunk(content="recovered")

        retry_llm = RetryLLM()
        retry_graph = create_tutor_streaming_graph(retry_llm, InMemorySaver())
        retry_payload = server.ChatRequest(message="retry me", thread_id="retry-thread", turn_id="same-turn")
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "tutor_stream_app", retry_graph), \
             patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context):
            cancel_task = asyncio.create_task(_collect(server._stream_chat_events(retry_payload, None)))
            await cancel_started.wait()
            cancel_task.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass
            assert not retry_graph.get_state({"configurable": {"thread_id": "retry-thread"}}).values
            recovered = await _collect(server._stream_chat_events(retry_payload, None))
            replay = await _collect(server._stream_chat_events(retry_payload, None))

        recovered_state = retry_graph.get_state({"configurable": {"thread_id": "retry-thread"}}).values["messages"]
        assert [message.content for message in recovered_state] == ["retry me", "recovered"]
        assert recovered[-1]["data"]["response"] == "recovered"
        assert replay[-1]["data"]["response"] == "recovered"
        assert retry_llm.calls == 2
        assert not server._thread_locks

        # Exceptions release the lock and mark the turn failed.
        server._thread_locks.clear()
        server._turn_records.clear()
        class ErrorThenSuccess:
            def __init__(self): self.calls = 0
            async def astream(self, _messages, config=None):
                self.calls += 1
                if self.calls == 1: raise RuntimeError("mock failure")
                yield AIMessageChunk(content="after failure")

        error_graph = create_tutor_streaming_graph(ErrorThenSuccess(), InMemorySaver())
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "tutor_stream_app", error_graph), \
             patch.object(server, "_request_tracing_context", side_effect=_fake_trace_context):
            failed = await _collect(server._stream_chat_events(server.ChatRequest(message="fails", thread_id="error-thread", turn_id="error-turn"), None))
            succeeded = await _collect(server._stream_chat_events(server.ChatRequest(message="fails", thread_id="error-thread", turn_id="error-turn"), None))
        assert failed[-1]["event"] == "error"
        assert succeeded[-1]["data"]["response"] == "after failure"
        assert not server._thread_locks

    asyncio.run(run_concurrency_and_idempotency())
    print("OK: native SSE streaming, cancellation boundary, offline tracing override, request tracing, and persistence passed")


if __name__ == "__main__":
    run_streaming_tracing_tests()
