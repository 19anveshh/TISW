"""Span-tree tests: one trace per unit of work, and nothing at all when untraced.

LangSmith is replaced with a recorder that resolves parents the same way the SDK
does — from a context variable — so these assertions prove the nesting the trace
viewer will show, including across the `asyncio.to_thread` and worker-thread
boundaries.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import asyncio
import contextvars
import json
import os
import tempfile
import time
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

os.environ["TISW_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="tisw-tracing-"), "tracing.db")
os.environ["GROQ_API_KEY"] = "gsk_span_test_key"

import rag_engine
import server
import tracing

DOC_BODY = (
    "Paging split sections. The quantum section explains superposition with the "
    "marker ZEBRA-CORPUS-9137 kept inside the document body.\n\n"
) * 3

_SECRET_MARKER = "lsv2_span_marker_secret"


class RecordedRun:
    def __init__(self, span: dict):
        self._span = span

    def add_outputs(self, outputs: dict) -> None:
        self._span["outputs"].append(dict(outputs))

    def add_metadata(self, metadata: dict) -> None:
        self._span["added_metadata"].append(dict(metadata))

    def add_tags(self, tags: list) -> None:
        self._span["added_tags"].append(list(tags))

    def post(self) -> None:
        raise AssertionError("span tests must never post to LangSmith")

    def patch(self, *_args, **_kwargs) -> None:
        raise AssertionError("span tests must never post to LangSmith")


class SpanRecorder:
    """A langsmith.trace replacement that records the tree our code asks for."""

    def __init__(self):
        self.spans: list[dict] = []
        self.contexts: list[dict] = []
        self._stack = contextvars.ContextVar("test_span_stack", default=())

    @contextmanager
    def trace(self, name: str, run_type: str = "chain", **kwargs):
        stack = self._stack.get()
        span = {
            "name": name,
            "parent": stack[-1] if stack else None,
            "run_type": run_type,
            "project_name": kwargs.get("project_name"),
            "inputs": dict(kwargs.get("inputs") or {}),
            "tags": list(kwargs.get("tags") or []),
            "metadata": dict(kwargs.get("metadata") or {}),
            "outputs": [],
            "added_metadata": [],
            "added_tags": [],
        }
        self.spans.append(span)
        token = self._stack.set(stack + (name,))
        try:
            yield RecordedRun(span)
        finally:
            self._stack.reset(token)

    @contextmanager
    def tracing_context(self, *args, **kwargs):
        self.contexts.append(kwargs)
        yield

    def pairs(self) -> list[tuple]:
        return [(span["parent"], span["name"]) for span in self.spans]

    def roots(self) -> list[str]:
        return [span["name"] for span in self.spans if span["parent"] is None]

    def assert_nested(self, *roots: str) -> None:
        """Every span must reach a named root; an orphan becomes its own trace."""
        by_name = {span["name"]: span["parent"] for span in self.spans}
        for span in self.spans:
            seen = []
            current = span["name"]
            while current is not None and current in by_name:
                if current in seen:
                    raise AssertionError(f"span cycle at {current}")
                seen.append(current)
                current = by_name[current]
            assert current is None, f"{span['name']} has an unknown parent"
            assert seen[-1] in roots, f"{span['name']} is rooted at {seen[-1]}, not {roots}"

    def payload_dump(self) -> str:
        return json.dumps(
            [
                {
                    "inputs": span["inputs"],
                    "outputs": span["outputs"],
                    "metadata": span["metadata"],
                    "added_metadata": span["added_metadata"],
                    "tags": span["tags"] + span["added_tags"],
                }
                for span in self.spans
            ],
            default=str,
        )


class FakeEmbedding:
    """Deterministic embedding substitute; no model download."""

    def embed(self, texts, batch_size=None):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "quantum" in lowered or "superposition" in lowered:
                vectors.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            elif "paging" in lowered or "section" in lowered:
                vectors.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            else:
                vectors.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        return vectors


class TutorStreamGraph:
    def __init__(self, answer: str = "grounded answer"):
        self.answer = answer

    def get_state(self, _config):
        return type("State", (), {"values": {"active_mode": "normal", "messages": []}})()

    def invoke(self, _payload, config=None):
        return {"messages": [AIMessage(content=self.answer)], "active_mode": "normal"}

    async def astream_events(self, payload, config=None, version="v2"):
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatGroq",
            "metadata": {"langgraph_node": "tutor"},
            "data": {"chunk": AIMessageChunk(content=self.answer)},
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"messages": [AIMessage(content=self.answer)]}},
        }


@contextmanager
def traced_langsmith(recorder: SpanRecorder):
    with patch("langsmith.trace", side_effect=recorder.trace), \
         patch("langsmith.tracing_context", side_effect=recorder.tracing_context), \
         patch.object(tracing, "can_trace", return_value=True), \
         patch.object(rag_engine, "get_embedding_model", return_value=FakeEmbedding()):
        yield


def _upload(client: TestClient, token: str, filename: str = "notes.txt", tracing_enabled: bool = True):
    headers = {"X-Session-Token": token}
    if tracing_enabled:
        headers["X-Tracing-Enabled"] = "true"
    return client.post(
        "http://localhost/api/upload".replace("http://localhost", ""),
        files={"file": (filename, DOC_BODY.encode("utf-8"), "text/plain")},
        headers=headers,
    )


def _wait_until_indexed(client: TestClient, token: str, doc_id: str, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        documents = client.get(
            "/api/documents", headers={"X-Session-Token": token}
        ).json()["documents"]
        status = next((d["status"] for d in documents if d["doc_id"] == doc_id), "missing")
        if status in {"ready", "failed", "missing"}:
            return status
        time.sleep(0.05)
    raise AssertionError(f"document {doc_id} never finished indexing")


def run_upload_ingest_tree() -> None:
    """A traced upload produces one upload trace and one ingestion trace."""
    client = TestClient(server.app)
    token = server._issue_session_token("gsk_span_test_key", "test-model", client_id="span-client")
    recorder = SpanRecorder()

    with patch.object(server, "is_offline_mode", return_value=False), traced_langsmith(recorder):
        response = _upload(client, token)
        assert response.status_code == 202, response.text
        doc_id = response.json()["doc_id"]
        assert _wait_until_indexed(client, token, doc_id) == "ready"

    names = [span["name"] for span in recorder.spans]
    assert names[0] == "tisw.upload", names
    assert recorder.roots() == ["tisw.upload", "tisw.ingest"], recorder.roots()
    pairs = set(recorder.pairs())
    assert {
        ("tisw.upload", "rag.stage_document"),
        ("rag.stage_document", "rag.extract_text"),
        ("rag.stage_document", "rag.chunk"),
        ("tisw.ingest", "rag.index_document"),
        ("rag.index_document", "rag.embed_document"),
        ("rag.index_document", "rag.index_merge"),
        ("tisw.ingest", "rag.persist_document"),
    } <= pairs, pairs
    assert all(span["project_name"] == "tisw" for span in recorder.spans if span["parent"] is None)

    stage = next(span for span in recorder.spans if span["name"] == "rag.stage_document")
    assert stage["outputs"][-1]["doc_id"] == doc_id
    extract = next(span for span in recorder.spans if span["name"] == "rag.extract_text")
    assert extract["outputs"][-1]["characters"] > 0
    chunk_span = next(span for span in recorder.spans if span["name"] == "rag.chunk")
    assert chunk_span["outputs"][-1]["chunks"] == stage["outputs"][-1]["chunk_count"]
    embed = next(span for span in recorder.spans if span["name"] == "rag.embed_document")
    assert embed["outputs"][-1]["vectors"] == chunk_span["outputs"][-1]["chunks"]
    assert embed["outputs"][-1]["dimensions"] == 3
    merge = next(span for span in recorder.spans if span["name"] == "rag.index_merge")
    assert merge["outputs"][-1]["session_documents"] == 1
    persist = next(span for span in recorder.spans if span["name"] == "rag.persist_document")
    assert persist["outputs"][-1]["persisted"] is True
    ingest = next(span for span in recorder.spans if span["name"] == "tisw.ingest")
    assert ingest["inputs"]["doc_id"] == doc_id
    assert ingest["metadata"]["session_id_hash"] != token
    recorder.assert_nested("tisw.upload", "tisw.ingest")
    print("OK: traced upload nests stage, extract, chunk, embed, merge and persist")


def run_turn_tree() -> None:
    """A traced RAG turn nests retrieval, citations and persistence in one trace."""
    client = TestClient(server.app)
    token = server._issue_session_token("gsk_span_test_key", "test-model", client_id="span-client")

    with patch.object(server, "is_offline_mode", return_value=False), traced_langsmith(recorder := SpanRecorder()):
        response = _upload(client, token, filename="seed.txt")
        doc_id = response.json()["doc_id"]
        assert _wait_until_indexed(client, token, doc_id) == "ready"
    assert recorder.roots() == ["tisw.upload", "tisw.ingest"]

    recorder = SpanRecorder()
    graph = TutorStreamGraph(answer="grounded answer")
    payload = server.ChatRequest(
        message="Explain the quantum section.",
        thread_id="span-turn-thread",
        turn_id="span-turn-id",
        rag_enabled=True,
        tracing_enabled=True,
    )

    async def stream():
        frames = []
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "create_tutor_streaming_graph", return_value=graph), \
             traced_langsmith(recorder):
            async for frame in server._stream_chat_events(payload, token, "span-client"):
                frames.append(frame)
        return frames

    frames = asyncio.run(stream())
    body = "".join(frames)
    assert "event: done" in body, body

    assert recorder.roots() == ["tisw.turn"], recorder.roots()
    pairs = set(recorder.pairs())
    assert {
        ("tisw.turn", "rag.restore_documents"),
        ("tisw.turn", "rag.query_build"),
        ("tisw.turn", "rag.retrieve"),
        ("rag.retrieve", "rag.embed_query"),
        ("rag.retrieve", "rag.vector_search"),
        ("rag.retrieve", "rag.bm25_search"),
        ("rag.retrieve", "rag.rrf_fusion"),
        ("tisw.turn", "rag.context_build"),
        ("tisw.turn", "tisw.persist"),
    } <= pairs, pairs
    recorder.assert_nested("tisw.turn")

    turn = recorder.spans[0]
    assert turn["inputs"]["message"] == "Explain the quantum section."
    assert turn["metadata"]["rag_enabled"] is True
    assert turn["metadata"]["thread_id_hash"] != "span-turn-thread"
    assert ["normal"] in turn["added_tags"]
    assert turn["added_metadata"][-1]["rag_used"] is True
    assert turn["outputs"][-1]["sources"], turn["outputs"]

    retrieve = next(span for span in recorder.spans if span["name"] == "rag.retrieve")
    assert retrieve["outputs"][-1]["chunks"] <= rag_engine.FINAL_TOP_K
    assert "text" not in json.dumps(retrieve["outputs"]), retrieve["outputs"]
    context_build = next(span for span in recorder.spans if span["name"] == "rag.context_build")
    assert context_build["outputs"][-1]["citations"] == retrieve["outputs"][-1]["chunks"]
    persist = next(span for span in recorder.spans if span["name"] == "tisw.persist")
    assert persist["outputs"][-1]["persisted"] is True

    # The non-streaming endpoint follows the same switch and the same nesting.
    recorder = SpanRecorder()
    sync_graph = TutorStreamGraph(answer="sync grounded answer")
    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "create_tutor_graph", return_value=sync_graph), \
         traced_langsmith(recorder):
        sync = client.post(
            "/api/chat",
            json={
                "message": "Explain the quantum section.",
                "thread_id": "span-sync-thread",
                "turn_id": "span-sync-turn",
                "rag_enabled": True,
                "tracing_enabled": True,
            },
            headers={"X-Session-Token": token, "X-Client-Id": "span-client"},
        )
        assert sync.status_code == 200, sync.text
        assert sync.json()["rag_used"] is True
    assert recorder.roots() == ["tisw.turn"], recorder.roots()
    assert {"non-streaming"} <= set(recorder.spans[0]["tags"]), recorder.spans[0]["tags"]
    assert ("tisw.turn", "rag.retrieve") in set(recorder.pairs())
    print("OK: traced turns nest retrieval, fusion, citations and persistence")


def run_untraced_emits_nothing() -> None:
    """The switch and offline mode are absolute: an untraced request emits zero spans."""
    client = TestClient(server.app)
    token = server._issue_session_token("gsk_span_test_key", "test-model", client_id="span-client")
    recorder = SpanRecorder()
    graph = TutorStreamGraph(answer="untraceable answer")

    # A server that could trace, asked not to.
    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "create_tutor_streaming_graph", return_value=graph), \
         traced_langsmith(recorder):
        response = _upload(client, token, filename="quiet.txt", tracing_enabled=False)
        assert response.status_code == 202, response.text
        assert _wait_until_indexed(client, token, response.json()["doc_id"]) == "ready"
        payload = server.ChatRequest(
            message="Explain the quantum section.",
            thread_id="quiet-thread",
            turn_id="quiet-turn",
            rag_enabled=True,
            tracing_enabled=False,
        )

        async def stream():
            async for _ in server._stream_chat_events(payload, token, "span-client"):
                pass

        asyncio.run(stream())

    assert recorder.spans == []
    assert recorder.contexts, "an untraced turn must still disable ambient tracing"
    assert all(kwargs.get("enabled") is False for kwargs in recorder.contexts), recorder.contexts

    # A server that must not trace at all, however the browser is switched.
    recorder = SpanRecorder()
    with patch.object(tracing, "can_trace", return_value=False), \
         patch("langsmith.trace", side_effect=recorder.trace), \
         patch("langsmith.tracing_context", side_effect=recorder.tracing_context), \
         patch.object(rag_engine, "get_embedding_model", return_value=FakeEmbedding()), \
         patch.object(server, "is_offline_mode", return_value=True):
        blocked = client.post(
            "/api/chat/stream",
            json={"message": "Explain the quantum section.", "rag_enabled": True, "tracing_enabled": True},
            headers={"X-Session-Token": token},
        )
        assert "event: error" in blocked.text
    assert recorder.spans == []
    print("OK: an untraced turn or upload produces no spans and no LangSmith parent context")


def run_payloads_stay_safe() -> None:
    """Span payloads carry counts, ids and previews — never bodies or credentials."""
    assert tracing.preview("explain gsk_leaky_key here") == "explain [REDACTED] here"
    assert tracing.redact("project lsv2_leaky_token") == "project [REDACTED]"
    client = TestClient(server.app)
    token = server._issue_session_token(_SECRET_MARKER, "test-model", client_id="span-client")
    recorder = SpanRecorder()
    graph = TutorStreamGraph(answer="safe answer")

    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "create_tutor_streaming_graph", return_value=graph), \
         traced_langsmith(recorder):
        response = _upload(client, token, filename="safe.txt")
        assert response.status_code == 202, response.text
        assert _wait_until_indexed(client, token, response.json()["doc_id"]) == "ready"
        payload = server.ChatRequest(
            message=f"Explain the section. gsk_supersecret_key_here {_SECRET_MARKER}",
            thread_id="safe-thread",
            turn_id="safe-turn",
            rag_enabled=True,
            tracing_enabled=True,
        )

        async def stream():
            async for _ in server._stream_chat_events(payload, token, "span-client"):
                pass

        asyncio.run(stream())

    assert recorder.spans, "the safety check needs spans to inspect"
    dump = recorder.payload_dump()
    assert "gsk_supersecret_key_here" not in dump
    assert _SECRET_MARKER not in dump
    assert token not in dump
    assert "ZEBRA-CORPUS-9137" not in dump, "document bodies must stay out of traces"
    assert "span-client" not in dump
    for span in recorder.spans:
        assert len(json.dumps(span["inputs"], default=str)) < 1200, span["name"]
        assert len(json.dumps(span["outputs"], default=str)) < 4000, span["name"]
    turn = next(span for span in recorder.spans if span["name"] == "tisw.turn")
    assert "gsk_supersecret_key_here" not in turn["inputs"]["message"]
    assert "[REDACTED]" in turn["inputs"]["message"]
    print("OK: span payloads are secret-free, body-free and length-bounded")


def run_span_tree_tests() -> None:
    run_upload_ingest_tree()
    run_turn_tree()
    run_untraced_emits_nothing()
    run_payloads_stay_safe()
    print("OK: the whole backend flow nests into one trace per unit of work")


if __name__ == "__main__":
    run_span_tree_tests()
