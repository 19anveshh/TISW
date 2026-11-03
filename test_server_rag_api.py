"""
test_server_rag_api.py — End-to-end validation of FastAPI server RAG endpoints.
Tests:
1. Health check
2. File upload validation (PDF, DOCX, TXT allowed; others rejected)
3. Document listing and deletion
4. RAG enabled with 0 docs -> asks user to upload
5. RAG enabled with docs -> sources returned with accurate citations
6. Session isolation: Session A documents not accessible to Session B
7. RAG OFF -> standard chatbot behavior
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import io
import os
import time
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from server import app
import server
import rag_engine
from rag_engine import get_session_document_index

client = TestClient(app)


class FakeTutorGraph:
    """Captures graph input so API tests never call Groq."""

    def __init__(self):
        self.last_payload = None

    def get_state(self, _config):
        return SimpleNamespace(values={"messages": [HumanMessage(content="Explain virtual memory") ]})

    def invoke(self, payload, config):
        self.last_payload = payload
        return {
            "messages": [AIMessage(content="Mocked tutor response")],
            "active_mode": payload.get("active_mode", "normal"),
        }


class LargeContextThenSuccessGraph(FakeTutorGraph):
    """Simulates Groq rejecting full context once, then accepting excerpts."""

    def __init__(self):
        super().__init__()
        self.attempts = 0

    def invoke(self, payload, config):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("413 request too large: tokens per minute")
        return super().invoke(payload, config)

    async def astream_events(self, payload, config=None, version="v2"):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("413 request too large: tokens per minute")
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatGroq",
            "metadata": {"langgraph_node": "tutor"},
            "data": {"chunk": AIMessageChunk(content="Mocked streamed response")},
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"messages": [AIMessage(content="Mocked streamed response")]}},
        }

@patch.object(server, "is_offline_mode", return_value=False)
def run_api_tests(_offline_guard):
    # Session issuance only stores the key; no live Groq request is made here.
    # Assign rather than setdefault: a blank GROQ_API_KEY= line in a local .env
    # otherwise counts as "already set" and the suite fails before it starts.
    os.environ["GROQ_API_KEY"] = "test-key-for-session-tests"
    print("=== 1. Testing Health Endpoint ===")
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json().get("status") == "healthy"
    print("✓ Health check OK")

    print("\n=== 2. Testing Server-Issued Session Boundary ===")
    session_a = client.post("/api/session")
    assert session_a.status_code == 200, session_a.text
    token_a = session_a.json()["session_token"]
    session_b = client.post("/api/session")
    assert session_b.status_code == 200, session_b.text
    token_b = session_b.json()["session_token"]
    assert token_a != token_b
    unauthenticated = client.get("/api/documents")
    assert unauthenticated.status_code == 401
    arbitrary_id = client.get("/api/documents", headers={"X-Client-Session-Id": "attacker-chosen-id"})
    assert arbitrary_id.status_code == 401
    print("✓ Document access requires distinct server-issued opaque tokens")

    print("\n=== 3. Testing Unsupported File Type Rejection ===")
    res_bad = client.post(
        "/api/upload",
        files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
        headers={"X-Session-Token": token_a},
    )
    assert res_bad.status_code == 400
    assert "Unsupported file type" in res_bad.json()["detail"]
    print("✓ Non-supported file extension rejected (.exe -> 400)")

    print("\n=== 4. Testing Configured Upload Size Limit ===")
    # Exercise the 413 path with a tiny patched limit instead of allocating 30 MB.
    with patch.object(server, "MAX_FILE_SIZE_BYTES", 64):
        oversized = client.post(
            "/api/upload",
            files={"file": ("oversized.txt", b"x" * 65, "text/plain")},
            headers={"X-Session-Token": token_a},
        )
    assert oversized.status_code == 413, oversized.text
    assert "upload limit" in oversized.json()["detail"].lower()
    print("✓ Oversized request receives a clear 413 response")

    print("\n=== 5. Testing Uploading TXT Document ===")
    txt_content = b"""
    Operating Systems: Virtual Memory and Paging
    Virtual memory is a storage allocation scheme in which secondary memory can be addressed as though it were part of main memory.
    The address referenced by the CPU is called a virtual address, which is mapped to a physical address by the Memory Management Unit (MMU).
    Pages are fixed-length contiguous blocks of virtual memory.
    Page tables store the mapping between virtual addresses and physical addresses.
    Page faults occur when a requested page is not present in physical RAM.
    """
    res_upload = client.post(
        "/api/upload",
        files={"file": ("operating_systems.txt", txt_content, "text/plain")},
        headers={"X-Session-Token": token_a},
    )
    assert res_upload.status_code == 202, res_upload.text
    data = res_upload.json()
    assert data["status"] == "success"
    doc_id_a = data["doc_id"]
    assert data["chunk_count"] > 0
    assert data["index_status"] == "indexing"
    print(f"✓ Uploaded TXT doc '{data['filename']}' ({data['chunk_count']} chunks, doc_id={doc_id_a}) queued for indexing")

    # Indexing runs in a worker thread; wait for the document to become searchable.
    deadline = time.time() + 30
    while time.time() < deadline:
        listed = client.get("/api/documents", headers={"X-Session-Token": token_a}).json()["documents"]
        if any(d["doc_id"] == doc_id_a and d["status"] == "ready" for d in listed):
            break
        time.sleep(0.2)
    else:
        raise AssertionError(f"Document {doc_id_a} never reached 'ready' status")
    print("✓ Background indexing completed")

    print("\n=== 6. Testing Document Listing ===")
    res_list = client.get(
        "/api/documents",
        headers={"X-Session-Token": token_a},
    )
    assert res_list.status_code == 200
    docs = res_list.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == doc_id_a
    assert docs[0]["filename"] == "operating_systems.txt"
    print(f"✓ Listed {len(docs)} document(s) for session A")

    print("\n=== 7. Testing Session Isolation ===")
    # Session B should have 0 documents
    res_list_b = client.get(
        "/api/documents",
        headers={"X-Session-Token": token_b},
    )
    assert res_list_b.status_code == 200
    assert len(res_list_b.json()["documents"]) == 0
    print("✓ Session B cannot see Session A's documents (strict isolation verified)")

    print("\n=== 8. Testing RAG Chat with 0 Documents in Session B ===")
    res_chat_empty = client.post(
        "/api/chat",
        json={"message": "What is virtual memory?", "rag_enabled": True},
        headers={"X-Session-Token": token_b},
    )
    assert res_chat_empty.status_code == 200
    reply_empty = res_chat_empty.json()
    assert reply_empty["rag_used"] is False
    assert "no indexed documents are available yet" in reply_empty["response"].lower()
    print("✓ RAG ON with 0 documents instructs user to upload a document")

    print("\n=== 9. Testing Hybrid Retrieval in Session A ===")
    index_a = get_session_document_index(token_a)
    chunks = index_a.hybrid_search("What causes a page fault and what is the MMU?", final_top_k=6)
    assert len(chunks) > 0
    assert chunks[0].filename == "operating_systems.txt"
    assert "page fault" in chunks[0].text.lower() or "mmu" in chunks[0].text.lower()
    print(f"✓ Hybrid search retrieved {len(chunks)} chunks from session A's documents")

    print("\n=== 10. Testing Mocked RAG Chat, Citations, and RAG-Off Behavior ===")
    fake_graph = FakeTutorGraph()
    retrieval_query = server._build_retrieval_query(
        fake_graph,
        {"configurable": {"thread_id": "follow-up-test"}},
        "What about paging?",
    )
    assert "Explain virtual memory" in retrieval_query
    assert "What about paging?" in retrieval_query
    with patch.object(server, "create_tutor_graph", return_value=fake_graph):
        res_chat = client.post(
            "/api/chat",
            json={"message": "What causes a page fault?", "rag_enabled": True, "mode": "normal"},
            headers={"X-Session-Token": token_a},
        )
        assert res_chat.status_code == 200, res_chat.text
        chat_data = res_chat.json()
        assert chat_data["rag_used"] is True
        assert len(chat_data["sources"]) == 1
        assert all(source["filename"] == "operating_systems.txt" for source in chat_data["sources"])
        assert chat_data["sources"][0]["chunk_id"].endswith("_full")
        assert "operating_systems.txt" in fake_graph.last_payload["rag_context"]
        assert "COMPLETE document" in fake_graph.last_payload["rag_context"]
        assert txt_content.decode().strip() in fake_graph.last_payload["rag_context"]

    fallback_graph = FakeTutorGraph()
    with patch.object(rag_engine, "FULL_DOC_CHAR_BUDGET", 1):
        with patch.object(server, "create_tutor_graph", return_value=fallback_graph):
            res_fallback = client.post(
                "/api/chat",
                json={"message": "What causes a page fault?", "rag_enabled": True, "mode": "normal"},
                headers={"X-Session-Token": token_a},
            )
    assert res_fallback.status_code == 200, res_fallback.text
    fallback_data = res_fallback.json()
    assert fallback_data["rag_used"] is True
    assert 0 < len(fallback_data["sources"]) <= 6
    assert all(not source["chunk_id"].endswith("_full") for source in fallback_data["sources"])
    assert "COMPLETE document" not in fallback_graph.last_payload["rag_context"]
    assert "EXCERPT" in fallback_graph.last_payload["rag_context"]
    print("✓ Full-document injection is used when eligible and falls back to top-k when over budget")

    retry_graph = LargeContextThenSuccessGraph()
    with patch.object(server, "create_tutor_graph", return_value=retry_graph):
        res_retry = client.post(
            "/api/chat",
            json={
                "message": "What causes a page fault?",
                "rag_enabled": True,
                "thread_id": "large-context-retry",
                "turn_id": "large-context-retry",
            },
            headers={"X-Session-Token": token_a},
        )
    assert res_retry.status_code == 200, res_retry.text
    assert retry_graph.attempts == 2
    assert "COMPLETE document" not in retry_graph.last_payload["rag_context"]
    assert all(not source["chunk_id"].endswith("_full") for source in res_retry.json()["sources"])

    analysis_graph = LargeContextThenSuccessGraph()
    with patch.object(server, "create_tutor_graph", return_value=analysis_graph):
        res_analysis = client.post(
            "/api/chat",
            json={
                "message": "How many modules are in the document?",
                "rag_enabled": True,
                "thread_id": "large-context-analysis",
                "turn_id": "large-context-analysis",
            },
            headers={"X-Session-Token": token_a},
        )
    assert res_analysis.status_code == 200, res_analysis.text
    assert analysis_graph.attempts == 2
    assert "DOCUMENT-WIDE STRUCTURE" in analysis_graph.last_payload["rag_context"]
    assert "EXCERPT" not in analysis_graph.last_payload["rag_context"]
    assert res_analysis.json()["sources"][0]["chunk_id"].endswith("_analysis")

    analysis_meta = type("Meta", (), {"filename": "modules.txt", "doc_id": "analysis", "chunk_count": 4})()
    analysis_context, analysis_sources = server._build_document_analysis_context([
        (analysis_meta, "MODULE 1: Probability\nQuestion 1: Define probability.\nMODULE 2: Statistics")
    ])
    assert "MODULE 1: Probability" in analysis_context
    assert "MODULE 2: Statistics" in analysis_context
    assert analysis_sources[0].chunk_id == "analysis_analysis"

    stream_retry_graph = LargeContextThenSuccessGraph()
    with patch.object(server, "create_tutor_streaming_graph", return_value=stream_retry_graph):
        res_stream_retry = client.post(
            "/api/chat/stream",
            json={
                "message": "What causes a page fault?",
                "rag_enabled": True,
                "thread_id": "large-context-stream-retry",
                "turn_id": "large-context-stream-retry",
            },
            headers={"X-Session-Token": token_a},
        )
    assert res_stream_retry.status_code == 200, res_stream_retry.text
    assert stream_retry_graph.attempts == 2
    assert 'event: done' in res_stream_retry.text
    assert '"chunk_id":"' in res_stream_retry.text
    assert '_full' not in res_stream_retry.text
    print("✓ Oversized full-document requests retry through hybrid retrieval for sync and streaming chat")

    fake_off_graph = FakeTutorGraph()
    with patch.object(server, "tutor_app", fake_off_graph):
        res_chat_off = client.post(
            "/api/chat",
            json={"message": "Answer naturally", "rag_enabled": False},
        )
        assert res_chat_off.status_code == 200, res_chat_off.text
        assert res_chat_off.json()["rag_used"] is False
        assert fake_off_graph.last_payload["rag_context"] is None
    print("✓ RAG ON sends grounded context and real metadata; RAG OFF sends no document context")

    print("\n=== 11. Testing Document Deletion ===")
    res_del = client.delete(
        f"/api/documents/{doc_id_a}",
        headers={"X-Session-Token": token_a},
    )
    assert res_del.status_code == 200
    res_list_after = client.get(
        "/api/documents",
        headers={"X-Session-Token": token_a},
    )
    assert len(res_list_after.json()["documents"]) == 0
    print("✓ Deleted document; session A now has 0 documents")

    print("\n✅ ALL SERVER RAG API INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    run_api_tests()
