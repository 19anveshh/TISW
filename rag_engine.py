"""
rag_engine.py — Advanced Hybrid Document RAG Engine for TISW.

Features:
- Supported Formats: PDF (.pdf), DOCX (.docx), TXT (.txt)
- Chunking: Structure-aware chunking preserving paragraphs, headings, page numbers
- Hybrid Retrieval:
    * Dense Vector Search via FastEmbed (BAAI/bge-small-en-v1.5)
    * Sparse Keyword Search via BM25Okapi (rank-bm25)
- Fusion: Weighted Reciprocal Rank Fusion (RRF, 30/70 semantic/keyword weights, k=60)
- Configurable top-k candidate selection (10 vector, 10 BM25 -> top 6 final)
- Strict Session Isolation: Per-session document index and memory stores
- Tracing: each ingest and retrieval stage emits a LangSmith span when a trace is open
"""

import io
import os
import re
import uuid
import time
import hashlib
import threading
import zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

import tracing


# =====================================================================
# CONFIGURATION CONSTANTS (Easily Tunable)
# =====================================================================

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer setting without allowing invalid configuration."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float setting, falling back when the value is not numeric."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _arg(args: tuple, kwargs: Dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    """Positional-or-keyword access for span payload projections."""
    return tracing.arg_at(args, kwargs, index, name, default)


def _chunk_summary(chunks: List[Any]) -> Dict[str, Any]:
    texts = [getattr(chunk, "text", "") for chunk in chunks]
    return {
        "chunks": len(chunks),
        "characters": sum(len(text) for text in texts),
        "avg_chunk_characters": round(sum(len(text) for text in texts) / max(1, len(chunks))),
        "pages": sorted({getattr(chunk, "page_number", None) for chunk in chunks} - {None}),
    }


def _metadata_summary(meta: Any) -> Dict[str, Any]:
    if meta is None:
        return {"document_present": False}
    return {
        "doc_id": meta.doc_id,
        "filename": meta.filename,
        "file_type": meta.file_type,
        "size_kb": meta.size_kb,
        "chunk_count": meta.chunk_count,
        "status": meta.status,
    }


CHUNK_SIZE_CHARS = _env_int("RAG_CHUNK_SIZE_CHARS", 1600)       # Approx 400-500 tokens
CHUNK_OVERLAP_CHARS = min(
    _env_int("RAG_CHUNK_OVERLAP_CHARS", 250),
    max(1, CHUNK_SIZE_CHARS - 1),
)                                                               # Approx 60-75 tokens
VECTOR_CANDIDATES = _env_int("RAG_VECTOR_CANDIDATES", 10)      # Candidates from vector similarity
BM25_CANDIDATES = _env_int("RAG_BM25_CANDIDATES", 10)          # Candidates from BM25 keyword search
VECTOR_WEIGHT = 0.3                                             # RRF vector weight
BM25_WEIGHT = 0.7                                               # RRF BM25 weight
RRF_K = _env_int("RAG_RRF_K", 60)                               # RRF smoothing constant
FINAL_TOP_K = _env_int("RAG_FINAL_TOP_K", 6)                    # Final chunks passed to the LLM
FULL_DOC_CHAR_BUDGET = _env_int("RAG_FULL_DOC_CHAR_BUDGET", 100000)  # full-doc context cap
MAX_FILE_SIZE_BYTES = _env_int("RAG_MAX_FILE_SIZE_BYTES", 30 * 1024 * 1024)  # 30 MB
MAX_FILES_PER_SESSION = _env_int("RAG_MAX_FILES_PER_SESSION", 10)             # Max documents
EMBED_BATCH_SIZE = _env_int("RAG_EMBED_BATCH_SIZE", 8)                        # Chunks per ONNX pass
# Cosine floor for vector candidates. Without it every question "grounds" on the
# least-dissimilar chunks in the session, even when nothing is actually relevant.
MIN_VECTOR_SCORE = _env_float("RAG_MIN_VECTOR_SCORE", 0.30)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Ingestion lifecycle. 'indexing' documents are registered but not searchable yet.
INDEX_STATUS_INDEXING = "indexing"
INDEX_STATUS_READY = "ready"
INDEX_STATUS_FAILED = "failed"


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
    status: str = INDEX_STATUS_READY
    content_hash: str = ""
    full_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "size_kb": self.size_kb,
            "chunk_count": self.chunk_count,
            "uploaded_at": self.uploaded_at,
            "status": self.status,
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
                # The cold build downloads and compiles the ONNX graph, which is
                # the slowest thing a first request of a deploy can do.
                with tracing.child("rag.load_embedding_model", inputs={"model": EMBEDDING_MODEL_NAME}):
                    # BAAI/bge-small-en-v1.5 is fast (384-dim), accurate, and runs efficiently on CPU
                    _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model


def prewarm_embedding_model() -> None:
    """
    Loads the embedding model and runs one throwaway embedding.

    Render's filesystem is ephemeral, so a cold container holds no model cache:
    without this, the first upload of every deploy spends the request downloading
    ~65 MB and building the ONNX session on 0.1 CPU, which is how uploads end as
    a browser-side 'Failed to fetch'.
    """
    model = get_embedding_model()
    for _ in model.embed(["prewarm"]):
        break


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
    # UTF-16 text is full of NUL bytes; only a BOM distinguishes it from binary.
    utf16_bom = file_bytes.startswith((b"\xff\xfe", b"\xfe\xff"))
    if ext == ".txt" and b"\x00" in file_bytes and not utf16_bom:
        raise ValueError(f"File '{filename}' contains binary data and is not a valid text file.")


@tracing.traced(
    "rag.extract_text",
    inputs=lambda args, kwargs: {
        "filename": os.path.basename(str(_arg(args, kwargs, 0, "filename", ""))),
        "extension": os.path.splitext(str(_arg(args, kwargs, 0, "filename", "")))[1].lower(),
        "size_kb": round(len(_arg(args, kwargs, 1, "file_bytes", b"") or b"") / 1024, 1),
    },
    outputs=lambda sections: {
        "sections": len(sections),
        "pages_with_text": sum(1 for _text, page in sections if page is not None),
        "characters": sum(len(text) for text, _page in sections),
    },
)
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
        # utf-8-sig first so a BOM never survives into chunk 0; utf-16 only
        # decodes when a BOM is present, so binary files still fail above.
        for encoding in ("utf-8-sig", "utf-16", "utf-8", "latin-1", "cp1252"):
            try:
                decoded_text = file_bytes.decode(encoding).strip()
                break
            except UnicodeError:
                continue
        if not decoded_text:
            raise ValueError(f"Could not decode text file '{filename}'.")
        extracted_sections.append((decoded_text, None))

    if not extracted_sections:
        raise ValueError(f"No readable text could be extracted from '{filename}'.")

    return extracted_sections


@tracing.traced(
    "rag.chunk",
    inputs=lambda args, kwargs: {
        "doc_id": str(_arg(args, kwargs, 0, "doc_id", "")),
        "filename": os.path.basename(str(_arg(args, kwargs, 1, "filename", ""))),
        "sections": len(_arg(args, kwargs, 2, "sections", []) or []),
        "chunk_size": _arg(args, kwargs, 3, "chunk_size", CHUNK_SIZE_CHARS),
        "overlap": _arg(args, kwargs, 4, "overlap", CHUNK_OVERLAP_CHARS),
    },
    outputs=_chunk_summary,
)
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
        self.pending: Dict[str, List[DocumentChunk]] = {}   # doc_id -> chunks awaiting embeddings

    @tracing.traced(
        "rag.stage_document",
        inputs=lambda args, kwargs: {
            "filename": os.path.basename(str(_arg(args, kwargs, 1, "filename", ""))),
            "size_kb": round(len(_arg(args, kwargs, 2, "file_bytes", b"") or b"") / 1024, 1),
            "doc_id": _arg(args, kwargs, 3, "doc_id"),
        },
        outputs=_metadata_summary,
    )
    def stage_document(
        self,
        filename: str,
        file_bytes: bytes,
        doc_id: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Validates, parses and chunks a document, and registers it as 'indexing'.

        Embedding is deliberately not done here: it is the slow, memory-hungry
        stage, and running it inside the upload request is what killed uploads on
        small hosts. Callers finish the work with finish_document_indexing().
        """
        with self.lock:
            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                size_mb = round(len(file_bytes) / (1024 * 1024), 1)
                raise ValueError(
                    f"File size ({size_mb} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."
                )

            # The client retries uploads after a dropped response, so the same
            # bytes can arrive twice. Identical content is one document, not two.
            content_hash = hashlib.sha256(file_bytes).hexdigest()
            duplicate = next(
                (
                    d for d in self.documents.values()
                    if d.filename == filename and d.content_hash == content_hash
                ),
                None,
            )
            if duplicate is not None:
                if duplicate.status != INDEX_STATUS_FAILED:
                    raise ValueError(f"'{filename}' is already uploaded in this session.")
                self.remove_document(duplicate.doc_id)

            # A failed ingest is a dead row, not a live document: it must not
            # permanently consume one of the session's upload slots.
            active = sum(1 for d in self.documents.values() if d.status != INDEX_STATUS_FAILED)
            if active >= MAX_FILES_PER_SESSION:
                raise ValueError(
                    f"Upload limit reached ({MAX_FILES_PER_SESSION} documents). Please remove an existing document first."
                )

            doc_id = doc_id or str(uuid.uuid4())[:8]
            sections = extract_text_from_file(filename, file_bytes)
            new_chunks = chunk_document(doc_id, filename, sections)
            full_text = "\n\n".join(text for text, _page in sections if text and text.strip())
            if len(full_text) > FULL_DOC_CHAR_BUDGET:
                full_text = ""

            if not new_chunks:
                raise ValueError(f"No chunks could be produced from '{filename}'.")

            self.pending[doc_id] = new_chunks
            meta = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                file_type=os.path.splitext(filename)[1].lower().replace(".", ""),
                size_kb=round(len(file_bytes) / 1024, 1),
                chunk_count=len(new_chunks),
                uploaded_at=time.time(),
                status=INDEX_STATUS_INDEXING,
                content_hash=content_hash,
                full_text=full_text,
            )
            self.documents[doc_id] = meta
            return meta

    @tracing.traced(
        "rag.index_document",
        inputs=lambda args, kwargs: {"doc_id": str(_arg(args, kwargs, 1, "doc_id", ""))},
        outputs=_metadata_summary,
    )
    def finish_document_indexing(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Embeds a staged document outside the lock, then merges it into the live index."""
        with self.lock:
            new_chunks = self.pending.get(doc_id)
            meta = self.documents.get(doc_id)

        if not new_chunks or meta is None:
            return None

        try:
            embedder = get_embedding_model()
            texts = [c.text for c in new_chunks]
            with tracing.child(
                "rag.embed_document",
                inputs={"doc_id": doc_id, "chunks": len(new_chunks), "batch_size": EMBED_BATCH_SIZE},
            ) as embed_run:
                # Small batches keep the ONNX arena flat: the default batch of 256
                # peaks near 1.2 GB on a full-size document, which free-tier hosts
                # OOM-kill mid-request and the browser reads that as 'Failed to fetch'.
                new_vectors = np.array(
                    list(embedder.embed(texts, batch_size=EMBED_BATCH_SIZE)), dtype=np.float32
                )
                # L2 Normalize for cosine similarity via dot product
                norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-9, norms)
                new_vectors = new_vectors / norms
                tracing.add_outputs(
                    embed_run,
                    {
                        "vectors": int(new_vectors.shape[0]),
                        "dimensions": int(new_vectors.shape[1]),
                    },
                )
        except Exception:
            with self.lock:
                self.pending.pop(doc_id, None)
                if doc_id in self.documents:
                    self.documents[doc_id].status = INDEX_STATUS_FAILED
            raise

        with self.lock:
            with tracing.child(
                "rag.index_merge",
                inputs={"doc_id": doc_id, "chunks": len(new_chunks)},
            ) as merge_run:
                self.pending.pop(doc_id, None)
                if doc_id not in self.documents:
                    return None  # removed while it was still indexing

                if self.embeddings is None or len(self.chunks) == 0:
                    self.embeddings = new_vectors
                else:
                    self.embeddings = np.vstack([self.embeddings, new_vectors])

                self.chunks.extend(new_chunks)
                # Rebuild BM25 index with all chunks in session
                self.bm25 = BM25Okapi([c.tokens for c in self.chunks])
                meta.status = INDEX_STATUS_READY
                tracing.add_outputs(
                    merge_run,
                    {
                        "session_chunks": len(self.chunks),
                        "session_documents": len(self.documents),
                    },
                )
                return meta

    def add_document(
        self,
        filename: str,
        file_bytes: bytes,
        doc_id: Optional[str] = None,
    ) -> DocumentMetadata:
        """Parses, chunks, embeds, and indexes a new document into this session."""
        meta = self.stage_document(filename, file_bytes, doc_id=doc_id)
        return self.finish_document_indexing(meta.doc_id) or meta

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document and its chunks, re-syncing vector and BM25 indexes."""
        with self.lock:
            if doc_id not in self.documents:
                return False

            del self.documents[doc_id]
            self.pending.pop(doc_id, None)

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

    def ready_document_count(self) -> int:
        """Counts documents whose chunks are actually searchable right now."""
        with self.lock:
            return sum(1 for doc in self.documents.values() if doc.status == INDEX_STATUS_READY)

    def ready_full_documents(self) -> Optional[List[Tuple[DocumentMetadata, str]]]:
        """All ready documents when every one retained its full text within budget.
        Returns None when full-document injection must not be used.
        """
        with self.lock:
            ready = [d for d in self.documents.values() if d.status == INDEX_STATUS_READY]
        if not ready or any(not d.full_text for d in ready):
            return None
        if sum(len(d.full_text) for d in ready) > FULL_DOC_CHAR_BUDGET:
            return None
        return [(d, d.full_text) for d in ready]

    def status_counts(self) -> Dict[str, int]:
        """Per-lifecycle counts, so 'indexing' is never confused with 'failed'."""
        with self.lock:
            counts = {INDEX_STATUS_INDEXING: 0, INDEX_STATUS_READY: 0, INDEX_STATUS_FAILED: 0}
            for doc in self.documents.values():
                counts[doc.status] = counts.get(doc.status, 0) + 1
            return counts

    @tracing.traced(
        "rag.retrieve",
        inputs=lambda args, kwargs: {
            "session_id_hash": tracing.hash_id(getattr(_arg(args, kwargs, 0, "self"), "session_id", "")),
            "query": tracing.preview(_arg(args, kwargs, 1, "query", "")),
            "final_top_k": _arg(args, kwargs, 2, "final_top_k", FINAL_TOP_K),
            "vector_candidates": _arg(args, kwargs, 3, "vector_candidates_k", VECTOR_CANDIDATES),
            "bm25_candidates": _arg(args, kwargs, 4, "bm25_candidates_k", BM25_CANDIDATES),
            "vector_weight": _arg(args, kwargs, 5, "vector_weight", VECTOR_WEIGHT),
            "bm25_weight": _arg(args, kwargs, 6, "bm25_weight", BM25_WEIGHT),
            "rrf_k": _arg(args, kwargs, 7, "rrf_k", RRF_K),
        },
        outputs=lambda chunks: {
            **_chunk_summary(chunks),
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        },
    )
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
        query = (query or "").strip()
        if not query:
            return []

        with self.lock:
            if not self.chunks or self.embeddings is None or self.bm25 is None:
                return []
            # Consistent snapshot; the ranking below runs without the lock.
            chunks = list(self.chunks)
            embeddings = self.embeddings
            bm25 = self.bm25

        total_chunks = len(chunks)
        v_k = min(vector_candidates_k, total_chunks)
        b_k = min(bm25_candidates_k, total_chunks)

        # ── 1. Vector Similarity Search ────────────────────────────────────
        # A blocking ONNX call, deliberately outside the index lock: holding it
        # here froze uploads and deletes for the duration of every chat query.
        with tracing.child("rag.embed_query", inputs={"query": tracing.preview(query)}) as embed_run:
            embedder = get_embedding_model()
            q_emb = list(embedder.embed([query]))[0]
            tracing.add_outputs(embed_run, {"dimensions": len(q_emb)})
        q_vec = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        with tracing.child(
            "rag.vector_search",
            inputs={
                "index_chunks": total_chunks,
                "candidates_k": v_k,
                "min_score": MIN_VECTOR_SCORE,
            },
        ) as vector_run:
            sim_scores = np.dot(embeddings, q_vec)
            v_ranked_indices = [
                i for i in np.argsort(sim_scores)[::-1][:v_k]
                if sim_scores[i] >= MIN_VECTOR_SCORE
            ]
            vector_results = [chunks[i] for i in v_ranked_indices]
            tracing.add_outputs(
                vector_run,
                {
                    "candidates": len(vector_results),
                    "top_score": round(float(max([float(sim_scores[i]) for i in v_ranked_indices], default=0.0)), 4),
                    "chunk_ids": [chunks[i].chunk_id for i in v_ranked_indices],
                },
            )

        # ── 2. BM25 Keyword Search ─────────────────────────────────────────
        with tracing.child(
            "rag.bm25_search",
            inputs={"index_chunks": total_chunks, "candidates_k": b_k},
        ) as bm25_run:
            q_tokens = tokenize_for_bm25(query)
            bm25_results = []
            if q_tokens:
                bm25_scores = bm25.get_scores(q_tokens)
                # Zero scores mean no keyword matched. Ranking them anyway injected
                # arbitrary chunks into the fusion with half of the total weight.
                b_ranked_indices = [
                    i for i in np.argsort(bm25_scores)[::-1][:b_k] if bm25_scores[i] > 0
                ]
                bm25_results = [chunks[i] for i in b_ranked_indices]
            else:
                b_ranked_indices = []
            tracing.add_outputs(
                bm25_run,
                {
                    "query_tokens": len(q_tokens),
                    "candidates": len(bm25_results),
                    "top_score": round(float(max([float(bm25_scores[i]) for i in b_ranked_indices], default=0.0)), 4),
                    "chunk_ids": [chunks[i].chunk_id for i in b_ranked_indices],
                },
            )

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


@tracing.traced(
    "rag.rrf_fusion",
    inputs=lambda args, kwargs: {
        "semantic_candidates": len(_arg(args, kwargs, 0, "semantic_results", []) or []),
        "keyword_candidates": len(_arg(args, kwargs, 1, "keyword_results", []) or []),
        "semantic_weight": _arg(args, kwargs, 2, "semantic_weight", VECTOR_WEIGHT),
        "keyword_weight": _arg(args, kwargs, 3, "keyword_weight", BM25_WEIGHT),
        "rrf_k": _arg(args, kwargs, 4, "rrf_k", RRF_K),
    },
    outputs=lambda fused: {
        "fused": len(fused),
        "scores": [
            {"chunk_id": chunk.chunk_id, "score": round(score, 5)}
            for chunk, score in fused[:10]
        ],
    },
)
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
