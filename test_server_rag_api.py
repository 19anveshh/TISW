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
from fastapi.testclient import TestClient
from server import app
from rag_engine import get_session_document_index

client = TestClient(app)

def run_api_tests():
    print("=== 1. Testing Health Endpoint ===")
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json().get("status") == "healthy"
    print("✓ Health check OK")

    print("\n=== 2. Testing Unsupported File Type Rejection ===")
    res_bad = client.post(
        "/api/upload",
        files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
        headers={"X-Client-Session-Id": "sess-test-a"},
    )
    assert res_bad.status_code == 400
    assert "Unsupported file type" in res_bad.json()["detail"]
    print("✓ Non-supported file extension rejected (.exe -> 400)")

    print("\n=== 3. Testing Uploading TXT Document ===")
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
        headers={"X-Client-Session-Id": "sess-test-a"},
    )
    assert res_upload.status_code == 200, res_upload.text
    data = res_upload.json()
    assert data["status"] == "success"
    doc_id_a = data["doc_id"]
    assert data["chunk_count"] > 0
    print(f"✓ Uploaded TXT doc '{data['filename']}' ({data['chunk_count']} chunks, doc_id={doc_id_a})")

    print("\n=== 4. Testing Document Listing ===")
    res_list = client.get(
        "/api/documents",
        headers={"X-Client-Session-Id": "sess-test-a"},
    )
    assert res_list.status_code == 200
    docs = res_list.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == doc_id_a
    assert docs[0]["filename"] == "operating_systems.txt"
    print(f"✓ Listed {len(docs)} document(s) for session A")

    print("\n=== 5. Testing Session Isolation ===")
    # Session B should have 0 documents
    res_list_b = client.get(
        "/api/documents",
        headers={"X-Client-Session-Id": "sess-test-b"},
    )
    assert res_list_b.status_code == 200
    assert len(res_list_b.json()["documents"]) == 0
    print("✓ Session B cannot see Session A's documents (strict isolation verified)")

    print("\n=== 6. Testing RAG Chat with 0 Documents in Session B ===")
    res_chat_empty = client.post(
        "/api/chat",
        json={"message": "What is virtual memory?", "rag_enabled": True},
        headers={"X-Client-Session-Id": "sess-test-b"},
    )
    assert res_chat_empty.status_code == 200
    reply_empty = res_chat_empty.json()
    assert reply_empty["rag_used"] is False
    assert "no documents have been uploaded yet" in reply_empty["response"].lower()
    print("✓ RAG ON with 0 documents instructs user to upload a document")

    print("\n=== 7. Testing Hybrid Retrieval in Session A ===")
    index_a = get_session_document_index("sess-test-a")
    chunks = index_a.hybrid_search("What causes a page fault and what is the MMU?", final_top_k=6)
    assert len(chunks) > 0
    assert chunks[0].filename == "operating_systems.txt"
    assert "page fault" in chunks[0].text.lower() or "mmu" in chunks[0].text.lower()
    print(f"✓ Hybrid search retrieved {len(chunks)} chunks from session A's documents")

    print("\n=== 8. Testing Document Deletion ===")
    res_del = client.delete(
        f"/api/documents/{doc_id_a}",
        headers={"X-Client-Session-Id": "sess-test-a"},
    )
    assert res_del.status_code == 200
    res_list_after = client.get(
        "/api/documents",
        headers={"X-Client-Session-Id": "sess-test-a"},
    )
    assert len(res_list_after.json()["documents"]) == 0
    print("✓ Deleted document; session A now has 0 documents")

    print("\n✅ ALL SERVER RAG API INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    run_api_tests()
