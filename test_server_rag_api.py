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
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from server import app
import server
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

@patch.object(server, "is_offline_mode", return_value=False)
def run_api_tests(_offline_guard):
    # Session issuance only stores the key; no live Groq request is made here.
    os.environ.setdefault("GROQ_API_KEY", "test-key-for-session-tests")
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

    print("\n=== 4. Testing Uploading TXT Document ===")
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
    assert res_upload.status_code == 200, res_upload.text
    data = res_upload.json()
    assert data["status"] == "success"
    doc_id_a = data["doc_id"]
    assert data["chunk_count"] > 0
    print(f"✓ Uploaded TXT doc '{data['filename']}' ({data['chunk_count']} chunks, doc_id={doc_id_a})")

    print("\n=== 5. Testing Document Listing ===")
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

    print("\n=== 6. Testing Session Isolation ===")
    # Session B should have 0 documents
    res_list_b = client.get(
        "/api/documents",
        headers={"X-Session-Token": token_b},
    )
    assert res_list_b.status_code == 200
    assert len(res_list_b.json()["documents"]) == 0
    print("✓ Session B cannot see Session A's documents (strict isolation verified)")

    print("\n=== 7. Testing RAG Chat with 0 Documents in Session B ===")
    res_chat_empty = client.post(
        "/api/chat",
        json={"message": "What is virtual memory?", "rag_enabled": True},
        headers={"X-Session-Token": token_b},
    )
    assert res_chat_empty.status_code == 200
    reply_empty = res_chat_empty.json()
    assert reply_empty["rag_used"] is False
    assert "no documents have been uploaded yet" in reply_empty["response"].lower()
    print("✓ RAG ON with 0 documents instructs user to upload a document")

    print("\n=== 8. Testing Hybrid Retrieval in Session A ===")
    index_a = get_session_document_index(token_a)
    chunks = index_a.hybrid_search("What causes a page fault and what is the MMU?", final_top_k=6)
    assert len(chunks) > 0
    assert chunks[0].filename == "operating_systems.txt"
    assert "page fault" in chunks[0].text.lower() or "mmu" in chunks[0].text.lower()
    print(f"✓ Hybrid search retrieved {len(chunks)} chunks from session A's documents")

    print("\n=== 9. Testing Mocked RAG Chat, Citations, and RAG-Off Behavior ===")
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
        assert 0 < len(chat_data["sources"]) <= 6
        assert all(source["filename"] == "operating_systems.txt" for source in chat_data["sources"])
        assert "operating_systems.txt" in fake_graph.last_payload["rag_context"]

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

    print("\n=== 10. Testing Document Deletion ===")
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
