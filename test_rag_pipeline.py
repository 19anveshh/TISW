"""
test_rag_pipeline.py — Comprehensive validation of the RAG engine and API integration.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import io
import docx
from rag_engine import (
    SessionDocumentIndex,
    get_session_document_index,
    extract_text_from_file,
    chunk_document,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    VECTOR_CANDIDATES,
    BM25_CANDIDATES,
    FINAL_TOP_K,
)

def run_tests():
    print("=== 1. Testing Configuration Parameters ===")
    assert VECTOR_WEIGHT == 0.5, f"Expected VECTOR_WEIGHT 0.5, got {VECTOR_WEIGHT}"
    assert BM25_WEIGHT == 0.5, f"Expected BM25_WEIGHT 0.5, got {BM25_WEIGHT}"
    assert VECTOR_CANDIDATES == 10, f"Expected VECTOR_CANDIDATES 10, got {VECTOR_CANDIDATES}"
    assert BM25_CANDIDATES == 10, f"Expected BM25_CANDIDATES 10, got {BM25_CANDIDATES}"
    assert FINAL_TOP_K == 6, f"Expected FINAL_TOP_K 6, got {FINAL_TOP_K}"
    print("✓ Configuration weights (0.5/0.5), candidates (10/10), and top-k (6) verified.")

    print("\n=== 2. Testing Text Extraction & Indexing ===")
    # 2a. Plain text document
    txt_content = b"""
    Quantum Computing Fundamentals
    Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.
    Superposition refers to a quantum system being in multiple states at the same time until it is observed.
    Entanglement describes a phenomenon where quantum particles become inextricably linked regardless of distance.
    Qubits are the fundamental units of quantum information, analogous to classical binary bits.
    """
    
    # 2b. DOCX document
    doc = docx.Document()
    doc.add_heading("Machine Learning Systems", level=1)
    doc.add_paragraph("Machine learning algorithms build mathematical models based on training data to make predictions or decisions.")
    doc.add_paragraph("Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.")
    doc.add_paragraph("Deep learning uses neural networks with three or more layers to model complex non-linear representations.")
    docx_io = io.BytesIO()
    doc.save(docx_io)
    docx_bytes = docx_io.getvalue()

    session_id = "test-session-1"
    index = get_session_document_index(session_id)
    
    txt_meta = index.add_document("quantum_physics.txt", txt_content)
    assert txt_meta.doc_id in index.documents
    assert txt_meta.chunk_count > 0
    print(f"✓ Added TXT document: {txt_meta.filename} with {txt_meta.chunk_count} chunk(s).")

    docx_meta = index.add_document("machine_learning.docx", docx_bytes)
    assert docx_meta.doc_id in index.documents
    assert docx_meta.chunk_count > 0
    print(f"✓ Added DOCX document: {docx_meta.filename} with {docx_meta.chunk_count} chunk(s).")

    print("\n=== 3. Testing Hybrid Search (Dense Vector + BM25 + 50/50 RRF) ===")
    # Query matching quantum text
    query_1 = "What is superposition and entanglement in quantum mechanics?"
    results_1 = index.hybrid_search(query_1, final_top_k=6)
    assert len(results_1) > 0, "Expected at least 1 result for quantum query"
    top_result = results_1[0]
    print(f"✓ Query: '{query_1}' -> Top hit from: {top_result.filename}")
    assert "quantum" in top_result.text.lower() or "superposition" in top_result.text.lower()

    # Query matching ML text
    query_2 = "Explain gradient descent optimization"
    results_2 = index.hybrid_search(query_2, final_top_k=6)
    assert len(results_2) > 0, "Expected at least 1 result for ML query"
    print(f"✓ Query: '{query_2}' -> Top hit from: {results_2[0].filename}")
    assert "gradient descent" in results_2[0].text.lower()

    print("\n=== 4. Testing User / Session Isolation ===")
    session_2_id = "test-session-2"
    index_2 = get_session_document_index(session_2_id)
    # Session 2 should have 0 documents
    assert len(index_2.documents) == 0, "Session 2 must NOT have documents from Session 1"
    assert len(index_2.hybrid_search("quantum")) == 0, "Session 2 search must return empty"
    print("✓ Session isolation verified: Session 2 cannot access Session 1's documents or vectors.")

    print("\n=== 5. Testing Document Removal ===")
    removed = index.remove_document(txt_meta.doc_id)
    assert removed is True
    assert txt_meta.doc_id not in index.documents
    # Should now only find ML results
    res_after = index.hybrid_search("superposition", final_top_k=6)
    assert not any(r.filename == "quantum_physics.txt" for r in res_after)
    print("✓ Document removal and index resynchronization verified.")

    print("\n✅ ALL RAG PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
