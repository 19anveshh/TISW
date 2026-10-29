"""
rag_engine.py — Advanced Hybrid Document RAG Engine for TISW.

Features:
- Supported Formats: PDF (.pdf), DOCX (.docx), TXT (.txt)
- Chunking: Structure-aware chunking preserving paragraphs, headings, page numbers
- Hybrid Retrieval:
    * Dense Vector Search via FastEmbed (BAAI/bge-small-en-v1.5)
    * Sparse Keyword Search via BM25Okapi (rank-bm25)
- Fusion: Weighted Reciprocal Rank Fusion (RRF, 50/50 weights, k=60)
- Configurable top-k candidate selection (10 vector, 10 BM25 -> top 6 final)
- Strict Session Isolation: Per-session document index and memory stores
"""

import io
import os
import re
import uuid
import time
import threading
import zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding


# =====================================================================
# CONFIGURATION CONSTANTS (Easily Tunable)
# =====================================================================

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer setting without allowing invalid configuration."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


CHUNK_SIZE_CHARS = _env_int("RAG_CHUNK_SIZE_CHARS", 1600)       # Approx 400-500 tokens
CHUNK_OVERLAP_CHARS = min(
    _env_int("RAG_CHUNK_OVERLAP_CHARS", 250),
    max(1, CHUNK_SIZE_CHARS - 1),
)                                                               # Approx 60-75 tokens
VECTOR_CANDIDATES = _env_int("RAG_VECTOR_CANDIDATES", 10)      # Candidates from vector similarity
BM25_CANDIDATES = _env_int("RAG_BM25_CANDIDATES", 10)          # Candidates from BM25 keyword search
VECTOR_WEIGHT = 0.5                                             # RRF vector weight
BM25_WEIGHT = 0.5                                               # RRF BM25 weight
RRF_K = _env_int("RAG_RRF_K", 60)                               # RRF smoothing constant
FINAL_TOP_K = _env_int("RAG_FINAL_TOP_K", 6)                    # Final chunks passed to the LLM
MAX_FILE_SIZE_BYTES = _env_int("RAG_MAX_FILE_SIZE_BYTES", 15 * 1024 * 1024)  # 15 MB
MAX_FILES_PER_SESSION = _env_int("RAG_MAX_FILES_PER_SESSION", 10)             # Max documents
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: Optional[int]  # 1-indexed for PDF; None for TXT/DOCX
    text: str
    chunk_index: int
    tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
            "chunk_index": self.chunk_index,
        }


@dataclass
class DocumentMetadata:
    doc_id: str
    filename: str
    file_type: str
    size_kb: float
    chunk_count: int
    uploaded_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "size_kb": self.size_kb,
            "chunk_count": self.chunk_count,
            "uploaded_at": self.uploaded_at,
        }


# =====================================================================
# EMBEDDING MODEL SINGLETON
# =====================================================================

_embedding_lock = threading.Lock()
_embedding_model: Optional[TextEmbedding] = None


def get_embedding_model() -> TextEmbedding:
    """Returns singleton fastembed TextEmbedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                # BAAI/bge-small-en-v1.5 is fast (384-dim), accurate, and runs efficiently on CPU
                _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model


# =====================================================================
# TEXT EXTRACTION & CHUNKING
# =====================================================================

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text for BM25 keyword search, preserving code terms and identifiers."""
    return re.findall(r"\w+", text.lower())


def validate_file_content(filename: str, file_bytes: bytes) -> None:
    """Validate basic file signatures before invoking format-specific parsers."""
    ext = os.path.splitext(filename)[1].lower()
    if not file_bytes or not file_bytes.strip():
        raise ValueError(f"File '{filename}' is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = round(len(file_bytes) / (1024 * 1024), 1)
        raise ValueError(
            f"File size ({size_mb} MB) exceeds maximum allowed size "
            f"({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."
        )

    if ext == ".pdf" and not file_bytes.lstrip().startswith(b"%PDF"):
        raise ValueError(f"File '{filename}' is not a valid PDF.")
    if ext == ".docx" and not zipfile.is_zipfile(io.BytesIO(file_bytes)):
        raise ValueError(f"File '{filename}' is not a valid DOCX document.")
    if ext == ".txt" and b"\x00" in file_bytes:
        raise ValueError(f"File '{filename}' contains binary data and is not a valid text file.")


def extract_text_from_file(filename: str, file_bytes: bytes) -> List[Tuple[str, Optional[int]]]:
    """
    Extracts text from supported document formats.
    Returns a list of (page_or_section_text, page_number_or_none) tuples.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    validate_file_content(filename, file_bytes)

    extracted_sections: List[Tuple[str, Optional[int]]] = []

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                raise ValueError("PDF has no pages.")
            for page_idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    extracted_sections.append((text, page_idx))
        except Exception as exc:
            raise ValueError(f"Could not extract text from PDF '{filename}': {exc}")

    elif ext == ".docx":
        try:
            doc = DocxDocument(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            full_text = "\n\n".join(paragraphs).strip()
            if full_text:
                extracted_sections.append((full_text, None))
        except Exception as exc:
            raise ValueError(f"Could not extract text from DOCX '{filename}': {exc}")

    elif ext == ".txt":
        decoded_text = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                decoded_text = file_bytes.decode(encoding).strip()
                break
            except UnicodeDecodeError:
                continue
        if not decoded_text:
            raise ValueError(f"Could not decode text file '{filename}'.")
        extracted_sections.append((decoded_text, None))

    if not extracted_sections:
        raise ValueError(f"No readable text could be extracted from '{filename}'.")

    return extracted_sections


def chunk_document(
    doc_id: str,
    filename: str,
    sections: List[Tuple[str, Optional[int]]],
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[DocumentChunk]:
    """
    Performs structure-aware overlapping text chunking over extracted sections.
    Preserves page numbers (for PDF) and document identifiers.
    """
    chunks: List[DocumentChunk] = []
    chunk_global_idx = 0

    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")

    for section_text, page_num in sections:
        # Split section into logical paragraphs
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [section_text]

        current_chunk_parts: List[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds chunk size, split by lines or sentences
            if para_len > chunk_size:
                sub_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
                if not sub_parts:
                    sub_parts = [para[i:i + chunk_size] for i in range(0, para_len, chunk_size - overlap)]

                # A single sentence can still exceed the target size. Split only
                # that sentence by words (and finally by characters for very long
                # identifiers) so chunks never grow without bound.
                normalized_parts: List[str] = []
                for sub in sub_parts:
                    if len(sub) <= chunk_size:
                        normalized_parts.append(sub)
                        continue
                    words = sub.split()
                    current_words: List[str] = []
                    current_words_len = 0
                    for word in words:
                        if len(word) > chunk_size:
                            if current_words:
                                normalized_parts.append(" ".join(current_words))
                                current_words = []
                                current_words_len = 0
                            normalized_parts.extend(
                                word[i:i + chunk_size]
                                for i in range(0, len(word), chunk_size)
                            )
                            continue
                        proposed_len = current_words_len + len(word) + (1 if current_words else 0)
                        if proposed_len > chunk_size and current_words:
                            normalized_parts.append(" ".join(current_words))
                            current_words = [word]
                            current_words_len = len(word)
                        else:
                            current_words.append(word)
                            current_words_len = proposed_len
                    if current_words:
                        normalized_parts.append(" ".join(current_words))
                sub_parts = normalized_parts

                for sub in sub_parts:
                    if current_length + len(sub) > chunk_size and current_chunk_parts:
                        chunk_text = " ".join(current_chunk_parts).strip()
                        if chunk_text:
                            chunk_id = f"{doc_id}_c{chunk_global_idx}"
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    filename=filename,
                                    page_number=page_num,
                                    text=chunk_text,
                                    chunk_index=chunk_global_idx,
                                    tokens=tokenize_for_bm25(chunk_text),
                                )
                            )
                            chunk_global_idx += 1

                        # Keep overlap text from previous chunk
                        overlap_target = overlap
                        retained: List[str] = []
                        accum = 0
                        for p in reversed(current_chunk_parts):
                            retained.insert(0, p)
                            accum += len(p)
                            if accum >= overlap_target:
                                break
                        current_chunk_parts = retained
                        current_length = sum(len(p) for p in current_chunk_parts)

                    current_chunk_parts.append(sub)
                    current_length += len(sub)
            else:
                if current_length + para_len > chunk_size and current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts).strip()
                    if chunk_text:
                        chunk_id = f"{doc_id}_c{chunk_global_idx}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                text=chunk_text,
                                chunk_index=chunk_global_idx,
                                tokens=tokenize_for_bm25(chunk_text),
                            )
                        )
                        chunk_global_idx += 1

                    # Retain overlap
                    overlap_target = overlap
                    retained = []
                    accum = 0
                    for p in reversed(current_chunk_parts):
                        retained.insert(0, p)
                        accum += len(p)
                        if accum >= overlap_target:
                            break
                    current_chunk_parts = retained
                    current_length = sum(len(p) for p in current_chunk_parts)

                current_chunk_parts.append(para)
                current_length += para_len

        # Flush remainder of this section
        if current_chunk_parts:
            chunk_text = "\n\n".join(current_chunk_parts).strip()
            if chunk_text:
                chunk_id = f"{doc_id}_c{chunk_global_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_num,
                        text=chunk_text,
                        chunk_index=chunk_global_idx,
                        tokens=tokenize_for_bm25(chunk_text),
                    )
                )
                chunk_global_idx += 1

    return chunks


# =====================================================================
# SESSION DOCUMENT INDEX (ISOLATED PER USER/SESSION)
# =====================================================================

class SessionDocumentIndex:
    """
    Maintains an isolated hybrid index (Vectors + BM25) for a single session.
    All operations are thread-safe and isolated from other sessions.
    """

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.lock = threading.RLock()
        self.documents: Dict[str, DocumentMetadata] = {}  # doc_id -> DocumentMetadata
        self.chunks: List[DocumentChunk] = []             # all chunks across session docs
        self.embeddings: Optional[np.ndarray] = None      # 2D normalized float32 array
        self.bm25: Optional[BM25Okapi] = None

    def add_document(
        self,
        filename: str,
        file_bytes: bytes,
        doc_id: Optional[str] = None,
    ) -> DocumentMetadata:
        """Parses, chunks, embeds, and indexes a new document into this session."""
        with self.lock:
            if len(self.documents) >= MAX_FILES_PER_SESSION:
                raise ValueError(
                    f"Upload limit reached ({MAX_FILES_PER_SESSION} documents). Please remove an existing document first."
                )

            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                size_mb = round(len(file_bytes) / (1024 * 1024), 1)
                raise ValueError(
                    f"File size ({size_mb} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."
                )

            doc_id = doc_id or str(uuid.uuid4())[:8]
            sections = extract_text_from_file(filename, file_bytes)
            new_chunks = chunk_document(doc_id, filename, sections)

            if not new_chunks:
                raise ValueError(f"No chunks could be produced from '{filename}'.")

            # Generate embeddings for new chunks using FastEmbed
            embedder = get_embedding_model()
            texts = [c.text for c in new_chunks]
            new_vectors_list = list(embedder.embed(texts))
            new_vectors = np.array(new_vectors_list, dtype=np.float32)

            # L2 Normalize for cosine similarity via dot product
            norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            new_vectors = new_vectors / norms

            # Update vectors
            if self.embeddings is None or len(self.chunks) == 0:
                self.embeddings = new_vectors
            else:
                self.embeddings = np.vstack([self.embeddings, new_vectors])

            self.chunks.extend(new_chunks)

            # Rebuild BM25 index with all chunks in session
            corpus_tokens = [c.tokens for c in self.chunks]
            self.bm25 = BM25Okapi(corpus_tokens)

            meta = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                file_type=os.path.splitext(filename)[1].lower().replace(".", ""),
                size_kb=round(len(file_bytes) / 1024, 1),
                chunk_count=len(new_chunks),
                uploaded_at=time.time(),
            )
            self.documents[doc_id] = meta
            return meta

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document and its chunks, re-syncing vector and BM25 indexes."""
        with self.lock:
            if doc_id not in self.documents:
                return False

            del self.documents[doc_id]

            # Find indices of chunks to retain
            keep_indices = [idx for idx, c in enumerate(self.chunks) if c.doc_id != doc_id]

            if not keep_indices:
                self.chunks = []
                self.embeddings = None
                self.bm25 = None
            else:
                self.chunks = [self.chunks[i] for i in keep_indices]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[keep_indices]
                corpus_tokens = [c.tokens for c in self.chunks]
                self.bm25 = BM25Okapi(corpus_tokens)

            return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata for all documents in this session."""
        with self.lock:
            return [doc.to_dict() for doc in self.documents.values()]

    def hybrid_search(
        self,
        query: str,
        final_top_k: int = FINAL_TOP_K,
        vector_candidates_k: int = VECTOR_CANDIDATES,
        bm25_candidates_k: int = BM25_CANDIDATES,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        rrf_k: int = RRF_K,
    ) -> List[DocumentChunk]:
        """
        Executes hybrid retrieval over this session's documents:
        1. Dense Vector Similarity Search (Top 10)
        2. Sparse BM25 Keyword Search (Top 10)
        3. Weighted Reciprocal Rank Fusion (RRF) with deduplication
        4. Selects final top K chunks (Default 6)
        """
        with self.lock:
            if not self.chunks or self.embeddings is None or self.bm25 is None:
                return []

            query = (query or "").strip()
            if not query:
                return []

            total_chunks = len(self.chunks)
            v_k = min(vector_candidates_k, total_chunks)
            b_k = min(bm25_candidates_k, total_chunks)

            # ── 1. Vector Similarity Search ────────────────────────────────────
            embedder = get_embedding_model()
            q_emb = list(embedder.embed([query]))[0]
            q_vec = np.array(q_emb, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            sim_scores = np.dot(self.embeddings, q_vec)
            # Argsort descending
            v_ranked_indices = np.argsort(sim_scores)[::-1][:v_k]
            vector_results = [self.chunks[i] for i in v_ranked_indices]

            # ── 2. BM25 Keyword Search ─────────────────────────────────────────
            q_tokens = tokenize_for_bm25(query)
            if q_tokens:
                bm25_scores = self.bm25.get_scores(q_tokens)
                b_ranked_indices = np.argsort(bm25_scores)[::-1][:b_k]
                bm25_results = [self.chunks[i] for i in b_ranked_indices]
            else:
                bm25_results = []

            # ── 3. Weighted Reciprocal Rank Fusion (RRF) ───────────────────────
            fused = weighted_rrf(
                vector_results,
                bm25_results,
                semantic_weight=vector_weight,
                keyword_weight=bm25_weight,
                rrf_k=rrf_k,
            )

            # ── 4. Select Final Top-K ─────────────────────────────────────────
            return [chunk for chunk, _score in fused[:final_top_k]]


def weighted_rrf(
    semantic_results: List[DocumentChunk],
    keyword_results: List[DocumentChunk],
    *,
    semantic_weight: float = VECTOR_WEIGHT,
    keyword_weight: float = BM25_WEIGHT,
    rrf_k: int = RRF_K,
) -> List[Tuple[DocumentChunk, float]]:
    """Fuse two ranked chunk lists with weighted, 1-indexed RRF.

    A stable chunk ID makes duplicates from the two retrievers collapse into
    one result while preserving the first encountered chunk metadata.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive.")

    chunk_map: Dict[str, DocumentChunk] = {}
    scores: Dict[str, float] = {}
    for rank, chunk in enumerate(semantic_results, start=1):
        chunk_map.setdefault(chunk.chunk_id, chunk)
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + semantic_weight / (rrf_k + rank)
    for rank, chunk in enumerate(keyword_results, start=1):
        chunk_map.setdefault(chunk.chunk_id, chunk)
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + keyword_weight / (rrf_k + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [(chunk_map[chunk_id], scores[chunk_id]) for chunk_id in ordered_ids]


# =====================================================================
# GLOBAL SESSION REGISTRY
# =====================================================================

_registry_lock = threading.Lock()
_session_indexes: Dict[str, SessionDocumentIndex] = {}


def get_session_document_index(session_id: str) -> SessionDocumentIndex:
    """Returns or creates the isolated SessionDocumentIndex for the given session_id."""
    clean_id = (session_id or "").strip()
    if not clean_id:
        raise ValueError("A server-issued session ID is required for document indexing.")
    with _registry_lock:
        if clean_id not in _session_indexes:
            _session_indexes[clean_id] = SessionDocumentIndex(clean_id)
        return _session_indexes[clean_id]


def prune_session_index(session_id: str) -> None:
    """Removes a session's index upon session cleanup or expiration."""
    with _registry_lock:
        if session_id in _session_indexes:
            del _session_indexes[session_id]
