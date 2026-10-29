"""
server.py — TISW (Teaching in a Structured Way) — FastAPI Backend

PUBLIC DEPLOYMENT MODEL
════════════════════════
Each visitor supplies their own Groq API key via the frontend.
The backend verifies it with a real Groq test request, then issues a
cryptographically random session token. The raw key is kept server-side
only, in a short-lived in-memory store keyed by that opaque token.

The visitor's browser holds ONLY the session token — never the raw key.
The token is transmitted in the X-Session-Token request header.

LOCAL DEV SESSION
═════════════════
When GROQ_API_KEY is set in .env, the frontend requests a server-issued
opaque session token from /api/session. RAG and per-session conversation state
are scoped to that token; RAG-off API calls can still use the existing global
local tutor fallback when no token is supplied.

SECURITY PROPERTIES
════════════════════
- Raw API keys are never returned in responses, logged, or stored in
  browser storage.
- Two visitors cannot accidentally use each other's keys: each token maps
  to exactly one key in the session store.
- The session store is local to this process instance. Keys vanish on
  restart (acceptable limitation — documented in README).
- GROQ_API_KEY in .env is never overwritten by visitor-supplied keys.

ENDPOINTS
══════════
  POST /api/session      — Issue a local env-key session token
  POST /api/verify-key  — Verify a visitor key, receive session token
  POST /api/chat        — Invoke LangGraph tutor (per-request LLM)
  POST /api/reset       — Issue fresh thread_id
  GET  /api/health      — Backend health
  POST /api/upload      — Document upload
  GET  /api/threads     — Persisted conversations for this browser
"""

import os
import sys
import uuid
import time
import secrets
import json
import hashlib
import asyncio
import copy
import re
import traceback
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, AsyncIterator

# ── Windows UTF-8 console fix ──────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()
from runtime_config import configure_runtime, is_offline_mode
configure_runtime()

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from main import (
    tutor_app,
    tutor_stream_app,
    global_checkpointer,
    load_and_validate_groq_llm,
    create_tutor_graph,
    create_tutor_streaming_graph,
    NORMAL_MODE_PROMPT,
    TISW_MODE_PROMPT,
    TEACHING_METHODOLOGY_PROMPT,
    build_system_message,
    detect_mode_request,
    detect_mode_command,
    detect_persistent_preference,
    _message_chunk_text,
)
from rag_engine import (
    get_session_document_index,
    prune_session_index,
    SUPPORTED_EXTENSIONS,
    FINAL_TOP_K,
    MAX_FILE_SIZE_BYTES,
)
import persistence

# ── Fallback: try to init global tutor_app if main.py left it as None ─────────
if tutor_app is None and not is_offline_mode():
    try:
        _llm = load_and_validate_groq_llm()
        tutor_app = create_tutor_graph(_llm, global_checkpointer)
        tutor_stream_app = create_tutor_streaming_graph(_llm, global_checkpointer)
        print("[INIT] ✅ LangGraph Tutor Graph initialized (env key).")
    except Exception:
        print("[INIT] ⚠️  No usable env key — visitors must supply their own Groq key.",
              file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════════════
# PER-VISITOR SESSION STORE
# ═══════════════════════════════════════════════════════════════════════════════

SESSION_TTL_SECONDS = 7200   # 2 hours; token expires after this

class _SessionEntry:
    __slots__ = ("api_key", "model", "expires_at", "checkpointer", "client_id", "documents_restored")

    def __init__(self, api_key: str, model: str, client_id: Optional[str] = None):
        self.api_key: str = api_key
        self.model: str = model
        self.expires_at: float = time.monotonic() + SESSION_TTL_SECONDS
        # Each visitor session gets its own isolated checkpointer so conversation
        # history from one user cannot bleed into another's.
        self.checkpointer: InMemorySaver = InMemorySaver()
        # Browser-stable id used to reload persisted chats and documents.
        self.client_id: Optional[str] = client_id
        self.documents_restored: bool = False

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at

    def refresh(self) -> None:
        self.expires_at = time.monotonic() + SESSION_TTL_SECONDS


# Dict[session_token -> _SessionEntry]
_session_store: Dict[str, _SessionEntry] = {}


def _prune_expired_sessions() -> None:
    """Remove expired entries and wipe their associated document indexes."""
    expired = [k for k, v in _session_store.items() if v.is_expired()]
    for k in expired:
        del _session_store[k]
        prune_session_index(k)


def _resolve_session(token: str) -> _SessionEntry:
    """
    Look up a session token. Raises 401 if missing or expired.
    Refreshes the TTL on every successful lookup.
    """
    entry = _session_store.get(token)
    if entry is None or entry.is_expired():
        if entry is not None:
            del _session_store[token]
            prune_session_index(token)
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please verify your Groq API key again.",
        )
    entry.refresh()
    return entry


def _build_llm_for_session(entry: _SessionEntry) -> ChatGroq:
    """Create a fresh ChatGroq client scoped to this visitor's key."""
    return ChatGroq(
        model=entry.model,
        api_key=entry.api_key,
        temperature=0.4,
    )


def _issue_session_token(api_key: str, model: str, client_id: Optional[str] = None) -> str:
    """Create a server-owned opaque bearer token for one isolated session."""
    _prune_expired_sessions()
    token = secrets.token_urlsafe(32)
    _session_store[token] = _SessionEntry(api_key=api_key, model=model, client_id=client_id)
    return token


def _require_session_token(token: Optional[str]) -> str:
    """Resolve only server-issued tokens; never trust arbitrary browser IDs."""
    clean_token = (token or "").strip()
    if not clean_token:
        raise HTTPException(
            status_code=401,
            detail="A server-issued session is required for document access.",
        )
    _resolve_session(clean_token)
    return clean_token


# ── Browser-stable client id (history and document persistence scope) ─────────

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _client_id_from_header(raw: Optional[str]) -> Optional[str]:
    """Validate the opaque browser id used to reload this visitor's workspace."""
    value = (raw or "").strip()
    if not value:
        return None
    if not _CLIENT_ID_RE.match(value):
        raise HTTPException(status_code=400, detail="Invalid client identifier.")
    return value


def _require_client_id(raw: Optional[str]) -> str:
    client_id = _client_id_from_header(raw)
    if not client_id:
        raise HTTPException(status_code=400, detail="A client identifier is required.")
    return client_id


def _validate_thread_id(thread_id: str) -> str:
    if not _THREAD_ID_RE.match(thread_id or ""):
        raise HTTPException(status_code=400, detail="Invalid thread identifier.")
    return thread_id


def _restore_documents_for_session(session_id: str, entry: _SessionEntry) -> None:
    """Re-index this visitor's stored documents once per issued session."""
    if not entry.client_id or entry.documents_restored:
        return
    entry.documents_restored = True
    try:
        stored = persistence.load_documents_for_restore(entry.client_id)
    except Exception as exc:
        print(f"[DOC RESTORE ERROR] {type(exc).__name__}", file=sys.stderr)
        return
    if not stored:
        return
    doc_index = get_session_document_index(session_id)
    for row in stored:
        if row["doc_id"] in doc_index.documents:
            continue
        try:
            doc_index.add_document(row["filename"], row["content"], doc_id=row["doc_id"])
        except Exception as exc:
            print(f"[DOC RESTORE ERROR] {type(exc).__name__}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="TISW AI Tutor Backend", version="3.1.0")

_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "TISW_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class VerifyKeyRequest(BaseModel):
    api_key: str
    model: Optional[str] = None      # visitor can optionally specify a model

class VerifyKeyResponse(BaseModel):
    session_token: str
    model: str
    message: str


class SessionResponse(BaseModel):
    session_token: str
    model: str
    message: str

class SourceItem(BaseModel):
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    snippet: str

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    turn_id: Optional[str] = None
    mode: Optional[str] = None
    rag_enabled: Optional[bool] = False
    tracing_enabled: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    turn_id: Optional[str] = None
    mode: str = "normal"
    rag_used: bool = False
    sources: Optional[List[SourceItem]] = None

class DocumentItem(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    size_kb: float
    chunk_count: int
    uploaded_at: float

class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]

class UploadDocumentResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    file_type: str
    size_kb: float
    chunk_count: int
    message: str

class ResetResponse(BaseModel):
    message: str
    thread_id: str


class ThreadItem(BaseModel):
    thread_id: str
    label: str
    message_count: int
    created_at: float
    updated_at: float

class ThreadListResponse(BaseModel):
    threads: List[ThreadItem]

class StoredMessage(BaseModel):
    id: int
    role: str
    content: str
    sources: List[SourceItem] = []
    rag_used: bool = False
    created_at: float

class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: List[StoredMessage]


class RuntimeStatusResponse(BaseModel):
    offline: bool
    langsmith_key_configured: bool
    tracing_available: bool
    tracing_enabled: bool
    project: str


TURN_PENDING = "pending"
TURN_COMPLETED = "completed"
TURN_CANCELLED = "cancelled"
TURN_FAILED = "failed"
TURN_TTL_SECONDS = 7200


@dataclass
class _ThreadLockEntry:
    lock: asyncio.Lock
    references: int = 0


@dataclass
class _TurnRecord:
    status: str
    fingerprint: str
    updated_at: float
    response: Optional[dict[str, Any]] = None


# These coordination structures are deliberately process-local. They protect
# the current in-memory checkpointer architecture in one worker only; a
# multi-worker deployment needs a shared lock/idempotency store.
_thread_locks: dict[tuple[str, str], _ThreadLockEntry] = {}
_turn_records: dict[tuple[str, str, str], _TurnRecord] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Minimal prompt used only for the key-verification test call
_VERIFY_PROBE = "Reply with exactly: OK"


def _classify_groq_error(error_msg: str) -> HTTPException:
    """Map Groq error strings to user-friendly HTTP errors without leaking secrets."""
    em = error_msg.lower()
    if "401" in em or "authentication" in em or "invalid api key" in em or "unauthorized" in em:
        return HTTPException(status_code=401, detail="Invalid Groq API key. Please check and try again.")
    if "429" in em or "rate_limit" in em or "rate limit" in em:
        return HTTPException(status_code=429, detail="Groq rate limit reached. Please wait a moment.")
    if "404" in em or "model_not_found" in em or "does not exist" in em:
        return HTTPException(status_code=422, detail="Model not found on Groq. Check the model name.")
    if "503" in em or "unavailable" in em:
        return HTTPException(status_code=503, detail="Groq service temporarily unavailable.")
    return HTTPException(status_code=502, detail="Groq API error. Please try again.")


def enrich_message(user_text: str, mode: Optional[str] = None) -> str:
    """User input text is passed directly. Modes are cleanly handled by the tutor graph."""
    return user_text


def _build_retrieval_query(graph: Any, config: Dict[str, Any], user_text: str) -> str:
    """Add recent context for short/ambiguous follow-ups without changing the prompt."""
    lowered = user_text.lower().strip()
    words = lowered.split()
    follow_up_markers = (
        "what about", "how about", "why is that", "why does it", "and then",
        "what does it", "how does it", "explain that", "tell me more",
    )
    needs_context = (
        len(words) <= 12
        or any(lowered.startswith(marker) for marker in follow_up_markers)
        or bool(set(words) & {"it", "this", "that", "these", "those", "they", "them"})
    )
    if not needs_context:
        return user_text

    try:
        state = graph.get_state(config)
        values = getattr(state, "values", {}) or {}
        previous_questions = [
            str(message.content).strip()
            for message in values.get("messages", [])
            if isinstance(message, HumanMessage) and str(message.content).strip()
        ]
        if previous_questions:
            return "\n".join(previous_questions[-2:] + [user_text])
    except Exception:
        # Retrieval still works with the current query if a graph has no state yet.
        pass
    return user_text


HISTORY_SEED_LIMIT = 24   # newest stored messages replayed into a fresh checkpointer


def _seed_graph_history(
    graph: Any,
    config: Dict[str, Any],
    client_id: str,
    thread_id: str,
) -> None:
    """Give a re-issued session the conversation memory of its persisted thread."""
    try:
        history = persistence.load_history_for_graph(
            client_id, thread_id, limit=HISTORY_SEED_LIMIT
        )
    except Exception as exc:
        print(f"[HISTORY SEED ERROR] {type(exc).__name__}", file=sys.stderr)
        return
    messages: List[Any] = []
    for row in history:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        if row.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif row.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    if not messages:
        return
    try:
        graph.update_state(config, {"messages": messages})
    except Exception as exc:
        print(f"[HISTORY SEED ERROR] {type(exc).__name__}", file=sys.stderr)


def _persist_completed_turn(
    client_id: Optional[str],
    thread_id: str,
    turn_id: str,
    user_text: str,
    response: ChatResponse,
) -> None:
    """Store one finished exchange; persistence failures never fail the reply."""
    if not client_id:
        return
    try:
        persistence.record_turn(
            client_id,
            thread_id,
            turn_id,
            user_text,
            response.response,
            sources=[source.model_dump() for source in (response.sources or [])],
            rag_used=bool(response.rag_used),
        )
    except Exception as exc:
        print(f"[PERSIST ERROR] {type(exc).__name__}", file=sys.stderr)


def _request_thread_id(payload: ChatRequest) -> str:
    return (payload.thread_id or "").strip() or f"study-session-{uuid.uuid4().hex[:8]}"


def _request_turn_id(payload: ChatRequest) -> str:
    return (payload.turn_id or "").strip() or uuid.uuid4().hex


def _turn_scope(session_token: Optional[str]) -> str:
    return (session_token or "").strip() or "__local__"


def _turn_fingerprint(payload: ChatRequest) -> str:
    return _json_hash({
        "message": payload.message,
        "mode": payload.mode,
        "rag_enabled": bool(payload.rag_enabled),
    })


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _prune_turn_records() -> None:
    cutoff = time.monotonic() - TURN_TTL_SECONDS
    stale = [key for key, record in _turn_records.items() if record.updated_at < cutoff]
    for key in stale:
        _turn_records.pop(key, None)


def _turn_key(session_token: Optional[str], thread_id: str, turn_id: str) -> tuple[str, str, str]:
    return (_turn_scope(session_token), thread_id, turn_id)


def _get_or_start_turn(
    session_token: Optional[str],
    thread_id: str,
    turn_id: str,
    fingerprint: str,
) -> tuple[tuple[str, str, str], _TurnRecord, bool]:
    _prune_turn_records()
    key = _turn_key(session_token, thread_id, turn_id)
    existing = _turn_records.get(key)
    if existing is not None:
        existing.updated_at = time.monotonic()
        if existing.fingerprint != fingerprint:
            raise HTTPException(
                status_code=409,
                detail="This turn ID was already used for different request content.",
            )
        if existing.status == TURN_COMPLETED:
            return key, existing, True
    record = _TurnRecord(
        status=TURN_PENDING,
        fingerprint=fingerprint,
        updated_at=time.monotonic(),
        response=None,
    )
    _turn_records[key] = record
    return key, record, False


def _mark_turn(record: _TurnRecord, status: str, response: Optional[dict[str, Any]] = None) -> None:
    record.status = status
    record.response = response
    record.updated_at = time.monotonic()


@asynccontextmanager
async def _conversation_guard(session_token: Optional[str], thread_id: str):
    """Serialize state changes for one session/thread without blocking others."""
    key = (_turn_scope(session_token), thread_id)
    entry = _thread_locks.get(key)
    if entry is None:
        entry = _ThreadLockEntry(lock=asyncio.Lock())
        _thread_locks[key] = entry
    entry.references += 1
    acquired = False
    try:
        await entry.lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            entry.lock.release()
        entry.references -= 1
        if entry.references == 0 and not entry.lock.locked() and _thread_locks.get(key) is entry:
            _thread_locks.pop(key, None)


def _restore_checkpoint(execution: Optional[dict[str, Any]]) -> None:
    """Restore the exact pre-turn state after cancellation/failure.

    This is called while the conversation lock is held, so another request
    cannot observe the intermediate checkpoint. InMemorySaver supports the
    delete-and-recreate sequence used here; other graph doubles may omit it.
    """
    if not execution:
        return
    graph = execution.get("graph")
    config = execution.get("config")
    if graph is None or config is None:
        return
    checkpointer = getattr(graph, "checkpointer", None)
    thread_id = execution.get("thread_id")
    snapshot_values = execution.get("snapshot_values")
    try:
        if checkpointer is not None and thread_id and hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(thread_id)
        if snapshot_values and hasattr(graph, "update_state"):
            graph.update_state(config, copy.deepcopy(snapshot_values))
    except Exception as exc:
        print(f"[CHECKPOINT RESTORE ERROR] {type(exc).__name__}", file=sys.stderr)


def _replay_response(record: _TurnRecord) -> ChatResponse:
    if not record.response:
        raise HTTPException(status_code=500, detail="Completed turn has no stored response.")
    return ChatResponse(**copy.deepcopy(record.response))


def _langsmith_key_configured() -> bool:
    """Return only whether a server-side LangSmith credential is configured."""
    return bool(
        os.getenv("LANGSMITH_API_KEY", "").strip()
        or os.getenv("LANGCHAIN_API_KEY", "").strip()
    )


def _langsmith_project() -> str:
    # TISW uses one deliberate project namespace. Do not let an inherited
    # process setting silently route application traces to another project.
    return "tisw"


def _mock_streaming_enabled() -> bool:
    """Enable only the explicit local browser streaming fixture."""
    return (
        is_offline_mode()
        and os.getenv("TISW_MOCK_STREAMING", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _stream_diagnostics_enabled() -> bool:
    return os.getenv("TISW_STREAM_DIAGNOSTICS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _safe_trace_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


_SECRET_LOG_PATTERN = re.compile(r"(?i)(?:gsk_[A-Za-z0-9_-]+|lsv2_[A-Za-z0-9_-]+)")


def _log_safe_exception(
    label: str,
    exc: BaseException,
    *,
    execution: Optional[dict[str, Any]] = None,
) -> None:
    """Log enough diagnostic detail to identify stream failures without secrets."""
    request_id = (execution or {}).get("request_id")
    thread_id = (execution or {}).get("thread_id")
    context = {
        "request_id": request_id if request_id else None,
        "thread_id_hash": _safe_trace_id(thread_id) if thread_id else None,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    print(
        f"[{label}] " + _SECRET_LOG_PATTERN.sub("[REDACTED]", json.dumps(context, ensure_ascii=True)),
        file=sys.stderr,
    )
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(_SECRET_LOG_PATTERN.sub("[REDACTED]", formatted), file=sys.stderr, end="")


def _log_stream_exception(exc: BaseException, *, execution: Optional[dict[str, Any]] = None) -> None:
    _log_safe_exception("STREAM ERROR", exc, execution=execution)


def _stream_diagnostic(
    stage: str,
    *,
    request_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    message: Optional[str] = None,
    response: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Emit opt-in request correlation without tokens or conversation history."""
    if not _stream_diagnostics_enabled():
        return
    payload: dict[str, Any] = {
        "stage": stage,
        "request_id": request_id,
        "turn_id": turn_id,
        "thread_id_hash": _safe_trace_id(thread_id) if thread_id else None,
        "source": source,
    }
    if message is not None:
        payload["message"] = _SECRET_LOG_PATTERN.sub("[REDACTED]", message[:1000])
    if response is not None:
        payload["response"] = _SECRET_LOG_PATTERN.sub("[REDACTED]", response[:1000])
    print(
        "[STREAM DIAGNOSTIC] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        file=sys.stderr,
    )


@contextmanager
def _nonfatal_tracing_context(trace_context: Any):
    """Keep optional tracing setup/cleanup failures from failing generation.

    LangGraph/model exceptions still propagate to the caller. Only exceptions
    raised while entering or leaving the observability context are isolated,
    because a trace must never turn a completed answer into a failed turn.
    """
    try:
        trace_context.__enter__()
    except Exception as exc:
        _log_safe_exception("TRACE ERROR", exc)
        yield
        return

    try:
        yield
    except BaseException:
        try:
            trace_context.__exit__(*sys.exc_info())
        except Exception as exc:
            _log_safe_exception("TRACE ERROR", exc)
        raise
    else:
        try:
            trace_context.__exit__(None, None, None)
        except Exception as exc:
            _log_safe_exception("TRACE ERROR", exc)


def _request_tracing_context(
    requested: bool,
    *,
    thread_id: str,
    mode: str,
    rag_used: bool,
    streaming: bool,
    request_id: str,
):
    """Create a request-scoped LangSmith context without changing process env."""
    if is_offline_mode():
        return nullcontext()

    from langsmith import tracing_context

    if not requested or not _langsmith_key_configured():
        # Explicitly disable ambient environment tracing for this request.
        return _nonfatal_tracing_context(tracing_context(enabled=False))

    return _nonfatal_tracing_context(tracing_context(
        enabled=True,
        project_name=_langsmith_project(),
        tags=["TISW", "streaming" if streaming else "non-streaming", mode],
        metadata={
            "application": "TISW",
            "environment": "local",
            "mode": mode,
            "request_id": request_id,
            "thread_id_hash": _safe_trace_id(thread_id),
            "rag_enabled": rag_used,
            "streaming": streaming,
        },
    ))


def _resolve_chat_graph(session_token: Optional[str], *, streaming: bool):
    """Resolve an existing graph or create the per-session graph once per request."""
    if session_token and session_token.strip():
        session = _resolve_session(session_token)
        llm = _build_llm_for_session(session)
        graph_factory = create_tutor_streaming_graph if streaming else create_tutor_graph
        return graph_factory(llm, session.checkpointer)

    graph = tutor_stream_app if streaming else tutor_app
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="No API key configured. Please verify your Groq API key using the sidebar.",
        )
    return graph


def _prepare_chat_execution(
    payload: ChatRequest,
    session_token: Optional[str],
    *,
    streaming: bool,
    thread_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    request_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> dict[str, Any]:
    if is_offline_mode():
        raise HTTPException(status_code=503, detail="Groq access is disabled while TISW_OFFLINE=true.")

    user_text = payload.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = thread_id or _request_thread_id(payload)
    turn_id = turn_id or _request_turn_id(payload)
    graph = _resolve_chat_graph(session_token, streaming=streaming)
    config_base = {"configurable": {"thread_id": thread_id}}

    current_mode = "normal"
    snapshot_values: Optional[dict[str, Any]] = None
    try:
        values = getattr(graph.get_state(config_base), "values", {}) or {}
        snapshot_values = copy.deepcopy(values)
        current_mode = values.get("active_mode", "normal") or "normal"
    except Exception:
        pass
    if client_id and not (snapshot_values or {}).get("messages"):
        _seed_graph_history(graph, config_base, client_id, thread_id)
    expected_mode = detect_mode_request(user_text, current_mode)
    request_id = request_id or uuid.uuid4().hex
    enriched = enrich_message(user_text)
    rag_enabled = bool(payload.rag_enabled)
    rag_used = False
    sources: List[SourceItem] = []
    rag_context_str: Optional[str] = None

    if detect_mode_command(user_text):
        invoke_payload = {"messages": [HumanMessage(content=enriched)], "rag_context": None}
    elif rag_enabled:
        session_id = _require_session_token(session_token)
        _restore_documents_for_session(session_id, _resolve_session(session_id))
        doc_index = get_session_document_index(session_id)
        if len(doc_index.documents) == 0:
            early = ChatResponse(
                response="⚠️ **RAG is enabled, but no documents have been uploaded yet.**\n\n"
                         "Please upload a PDF, DOCX, or TXT document using the paperclip or document manager in the sidebar before querying your documents.",
                thread_id=thread_id,
                mode=current_mode,
                rag_used=False,
                sources=[],
            )
            return {
                "graph": graph,
                "config": config_base,
                "invoke_payload": None,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "mode": current_mode,
                "rag_used": False,
                "sources": [],
                "request_id": request_id,
                "snapshot_values": snapshot_values,
                "early_response": early,
                "trace_requested": bool(payload.tracing_enabled),
            }

        retrieval_query = _build_retrieval_query(graph, config_base, user_text)
        retrieved_chunks = doc_index.hybrid_search(query=retrieval_query, final_top_k=FINAL_TOP_K)
        if retrieved_chunks:
            rag_used = True
            context_blocks = []
            for i, chunk in enumerate(retrieved_chunks, start=1):
                page_info = f", Page {chunk.page_number}" if chunk.page_number is not None else ""
                context_blocks.append(
                    f"--- EXCERPT {i} ---\n[Source: {chunk.filename}{page_info} | Chunk ID: {chunk.chunk_id}]\n{chunk.text}"
                )
                snippet = chunk.text[:220].strip() + ("…" if len(chunk.text) > 220 else "")
                sources.append(SourceItem(
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    snippet=snippet,
                ))
            rag_context_str = "\n\n".join(context_blocks)
        else:
            rag_context_str = "[No relevant excerpts were found in the uploaded documents for this question.]"
        invoke_payload = {"messages": [HumanMessage(content=enriched)], "rag_context": rag_context_str}
    else:
        invoke_payload = {"messages": [HumanMessage(content=enriched)], "rag_context": None}

    metadata = {
        "application": "TISW",
        "environment": "local",
        "mode": expected_mode,
        "request_id": request_id,
        "thread_id_hash": _safe_trace_id(thread_id),
        "rag_enabled": rag_used,
        "streaming": streaming,
    }
    config = {
        **config_base,
        "run_name": "tisw.chat.stream" if streaming else "tisw.chat",
        "tags": ["TISW", "streaming" if streaming else "non-streaming", expected_mode],
        "metadata": metadata,
    }
    return {
        "graph": graph,
        "config": config,
        "invoke_payload": invoke_payload,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "mode": expected_mode,
        "rag_used": rag_used,
        "sources": sources,
        "request_id": request_id,
        "snapshot_values": snapshot_values,
        "early_response": None,
        "trace_requested": bool(payload.tracing_enabled),
    }


def _sse_event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _final_response_from_output(output: Any) -> str:
    if not isinstance(output, dict):
        return ""
    messages = output.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, (AIMessage, AIMessageChunk)):
        content = last.content
    elif isinstance(last, dict) and last.get("type") in {"ai", "AIMessage", "AIMessageChunk"}:
        content = last.get("content", "")
    else:
        return ""
    return content if isinstance(content, str) else _message_chunk_text(last)


def _extract_tutor_delta(event: Any) -> str:
    """Extract only text from an explicitly tagged tutor model event."""
    if not isinstance(event, dict) or event.get("event") != "on_chat_model_stream":
        return ""
    metadata = event.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "tutor":
        return ""
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    chunk = data.get("chunk")
    if not isinstance(chunk, AIMessageChunk):
        return ""
    return _message_chunk_text(chunk)


async def _replay_stream_events(record: _TurnRecord) -> AsyncIterator[str]:
    response = _replay_response(record).model_dump()
    yield _sse_event("start", {
        "request_id": uuid.uuid4().hex,
        "thread_id": response["thread_id"],
        "turn_id": response.get("turn_id"),
        "mode": response.get("mode", "normal"),
        "tracing_enabled": False,
        "replay": True,
    })
    if response.get("response"):
        yield _sse_event("delta", {"text": response["response"]})
    yield _sse_event("done", {**response, "replay": True})


async def _stream_chat_events(
    payload: ChatRequest,
    session_token: Optional[str],
    client_id: Optional[str] = None,
) -> AsyncIterator[str]:
    if _mock_streaming_enabled():
        mock_request_id = uuid.uuid4().hex
        mock_turn_id = _request_turn_id(payload)
        mock_thread_id = _request_thread_id(payload)
        _stream_diagnostic(
            "request_received",
            request_id=mock_request_id,
            turn_id=mock_turn_id,
            thread_id=mock_thread_id,
            message=payload.message,
            source="mock",
        )
        async for event in _mock_stream_events(payload, request_id=mock_request_id):
            yield event
        return

    thread_id = _request_thread_id(payload)
    turn_id = _request_turn_id(payload)
    request_id = uuid.uuid4().hex
    _stream_diagnostic(
        "request_received",
        request_id=request_id,
        turn_id=turn_id,
        thread_id=thread_id,
        message=payload.message,
        source="live_or_replay",
    )
    fingerprint = _turn_fingerprint(payload)
    async with _conversation_guard(session_token, thread_id):
        _key, record, replay = _get_or_start_turn(session_token, thread_id, turn_id, fingerprint)
        if replay:
            _stream_diagnostic(
                "replayed",
                request_id=request_id,
                turn_id=turn_id,
                thread_id=thread_id,
                response=(record.response or {}).get("response"),
                source="completed_replay",
            )
            async for event in _replay_stream_events(record):
                yield event
            return
        async for event in _stream_chat_events_locked(
            payload,
            session_token,
            thread_id=thread_id,
            turn_id=turn_id,
            request_id=request_id,
            record=record,
            client_id=client_id,
        ):
            yield event


async def _mock_stream_events(
    payload: ChatRequest,
    *,
    request_id: Optional[str] = None,
) -> AsyncIterator[str]:
    thread_id = (payload.thread_id or "").strip() or "mock-stream-session"
    turn_id = (payload.turn_id or "").strip() or uuid.uuid4().hex
    request_id = request_id or uuid.uuid4().hex
    answer = (
        "This is a local streaming fixture.\n\n"
        "It sends deterministic chunks through the same SSE protocol "
        "without initializing Groq or LangSmith."
    )
    yield _sse_event("start", {
        "request_id": request_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "mode": "normal",
        "tracing_enabled": False,
        "mock": True,
    })
    for chunk in (
        "This is a local ",
        "streaming fixture.\n\n",
        "It sends deterministic chunks ",
        "through the same SSE protocol ",
        "without initializing Groq or LangSmith.",
    ):
        yield _sse_event("delta", {"text": chunk})
    yield _sse_event("done", {
        "response": answer,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "mode": "normal",
        "rag_used": False,
        "sources": [],
        "mock": True,
    })


async def _stream_chat_events_locked(
    payload: ChatRequest,
    session_token: Optional[str],
    *,
    thread_id: str,
    turn_id: str,
    request_id: str,
    record: _TurnRecord,
    client_id: Optional[str] = None,
) -> AsyncIterator[str]:
    execution = None
    emitted_text = False
    accumulated: list[str] = []
    final_response = ""
    try:
        execution = _prepare_chat_execution(
            payload,
            session_token,
            streaming=True,
            thread_id=thread_id,
            turn_id=turn_id,
            request_id=request_id,
            client_id=client_id,
        )
        graph_message = (
            execution["invoke_payload"]["messages"][-1].content
            if execution["invoke_payload"]
            else None
        )
        _stream_diagnostic(
            "graph_input",
            request_id=execution["request_id"],
            turn_id=turn_id,
            thread_id=thread_id,
            message=str(graph_message) if graph_message is not None else None,
            source="live_generation",
        )
        yield _sse_event("start", {
            "request_id": execution["request_id"],
            "thread_id": execution["thread_id"],
            "turn_id": turn_id,
            "mode": execution["mode"],
            "tracing_enabled": bool(
                execution["trace_requested"]
                and not is_offline_mode()
                and _langsmith_key_configured()
            ),
        })

        if execution["early_response"] is not None:
            final_response = execution["early_response"].response
            early_response = execution["early_response"]
            early_response.turn_id = turn_id
            _mark_turn(record, TURN_COMPLETED, early_response.model_dump())
            _persist_completed_turn(client_id, thread_id, turn_id, payload.message.strip(), early_response)
            if final_response:
                emitted_text = True
                yield _sse_event("delta", {"text": final_response})
            yield _sse_event("done", early_response.model_dump())
            return

        trace_context = _request_tracing_context(
            execution["trace_requested"],
            thread_id=execution["thread_id"],
            mode=execution["mode"],
            rag_used=execution["rag_used"],
            streaming=True,
            request_id=execution["request_id"],
        )
        with trace_context:
            async for event in execution["graph"].astream_events(
                execution["invoke_payload"],
                config=execution["config"],
                version="v2",
            ):
                text = _extract_tutor_delta(event)
                if text:
                    if not accumulated:
                        _stream_diagnostic(
                            "first_delta",
                            request_id=execution["request_id"],
                            turn_id=turn_id,
                            thread_id=thread_id,
                            response=text,
                            source="live_generation",
                        )
                    accumulated.append(text)
                    emitted_text = True
                    yield _sse_event("delta", {"text": text})
                elif (
                    event.get("event") == "on_chain_end"
                    and event.get("name") in {"LangGraph", execution["config"].get("run_name")}
                ):
                    final_response = _final_response_from_output((event.get("data") or {}).get("output"))

        if not final_response:
            final_response = "".join(accumulated)
        if not emitted_text and final_response:
            yield _sse_event("delta", {"text": final_response})
        if not final_response:
            raise RuntimeError("Empty response from LangGraph.")

        response = ChatResponse(
            response=final_response,
            thread_id=execution["thread_id"],
            turn_id=turn_id,
            mode=execution["mode"],
            rag_used=execution["rag_used"],
            sources=execution["sources"],
        )
        _stream_diagnostic(
            "completed",
            request_id=execution["request_id"],
            turn_id=turn_id,
            thread_id=thread_id,
            response=final_response,
            source="live_generation",
        )
        _mark_turn(record, TURN_COMPLETED, response.model_dump())
        _persist_completed_turn(client_id, thread_id, turn_id, payload.message.strip(), response)
        if execution["sources"]:
            yield _sse_event("sources", {"sources": [source.model_dump() for source in execution["sources"]]})
        yield _sse_event("done", response.model_dump())
    except (asyncio.CancelledError, GeneratorExit):
        _restore_checkpoint(execution)
        _mark_turn(record, TURN_CANCELLED)
        raise
    except HTTPException as exc:
        _restore_checkpoint(execution)
        _mark_turn(record, TURN_FAILED)
        yield _sse_event("error", {"message": str(exc.detail), "status": exc.status_code})
    except Exception as exc:
        _restore_checkpoint(execution)
        _mark_turn(record, TURN_FAILED)
        _log_stream_exception(exc, execution=execution)
        yield _sse_event("error", {"message": "Generation failed. Please try again."})


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/session", response_model=SessionResponse)
def create_local_session(
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Issue an isolated session for the server-configured local Groq key."""
    if is_offline_mode():
        raise HTTPException(status_code=503, detail="Groq access is disabled while TISW_OFFLINE=true.")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key in {"your_groq_api_key_here", "gsk_...", "changeme"}:
        raise HTTPException(
            status_code=503,
            detail="No server Groq API key is configured. Add one or verify a visitor key.",
        )

    token = _issue_session_token(api_key, GROQ_DEFAULT_MODEL, _client_id_from_header(x_client_id))
    return SessionResponse(
        session_token=token,
        model=GROQ_DEFAULT_MODEL,
        message="Server-configured Groq session created.",
    )


@app.post("/api/verify-key", response_model=VerifyKeyResponse)
async def verify_key(
    payload: VerifyKeyRequest,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """
    Verify a visitor-supplied Groq API key with a real test request.
    Returns an opaque session token on success.
    The raw key is NEVER returned and is stored only server-side.
    """
    if is_offline_mode():
        raise HTTPException(status_code=503, detail="Groq access is disabled while TISW_OFFLINE=true.")

    raw_key = (payload.api_key or "").strip()
    if not raw_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    if not raw_key.startswith("gsk_"):
        raise HTTPException(status_code=400, detail="That doesn't look like a Groq API key (expected gsk_...).")

    model = (payload.model or "").strip() or GROQ_DEFAULT_MODEL

    # ── Real verification: make a minimal Groq API call ────────────────────────
    try:
        probe_llm = ChatGroq(model=model, api_key=raw_key, temperature=0.0)
        probe_result = probe_llm.invoke([SystemMessage(content="You are a test assistant."),
                                          HumanMessage(content=_VERIFY_PROBE)])
        reply = str(probe_result.content).strip()
        # A valid key will produce any coherent response
        if not reply:
            raise ValueError("Empty response from Groq during key verification.")
    except HTTPException:
        raise
    except Exception as exc:
        # Never log the key or a partial key. The response is deliberately generic.
        raise _classify_groq_error(str(exc))

    # ── Verification passed — issue session token ──────────────────────────────
    token = _issue_session_token(raw_key, model, _client_id_from_header(x_client_id))
    return VerifyKeyResponse(
        session_token=token,
        model=model,
        message="Groq API key verified. Session active.",
    )


@app.get("/api/runtime", response_model=RuntimeStatusResponse)
def runtime_status():
    """Expose safe local capability flags without returning credentials."""
    offline = is_offline_mode()
    key_configured = _langsmith_key_configured()
    return RuntimeStatusResponse(
        offline=offline,
        langsmith_key_configured=key_configured,
        tracing_available=bool(key_configured and not offline),
        tracing_enabled=False,
        project=_langsmith_project(),
    )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    payload: ChatRequest,
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Stream one LangGraph turn as user-facing Server-Sent Events."""
    return StreamingResponse(
        _stream_chat_events(payload, x_session_token, _client_id_from_header(x_client_id)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Backwards-compatible non-streaming LangGraph chat endpoint."""
    client_id = _client_id_from_header(x_client_id)
    thread_id = _request_thread_id(payload)
    turn_id = _request_turn_id(payload)
    async with _conversation_guard(x_session_token, thread_id):
        _key, record, replay = _get_or_start_turn(
            x_session_token,
            thread_id,
            turn_id,
            _turn_fingerprint(payload),
        )
        if replay:
            return _replay_response(record)

        execution = None
        try:
            execution = _prepare_chat_execution(
                payload,
                x_session_token,
                streaming=False,
                thread_id=thread_id,
                turn_id=turn_id,
                client_id=client_id,
            )
            if execution["early_response"] is not None:
                response = execution["early_response"]
            else:
                with _request_tracing_context(
                    execution["trace_requested"],
                    thread_id=execution["thread_id"],
                    mode=execution["mode"],
                    rag_used=execution["rag_used"],
                    streaming=False,
                    request_id=execution["request_id"],
                ):
                    result = execution["graph"].invoke(
                        execution["invoke_payload"],
                        config=execution["config"],
                    )
                msgs = result.get("messages", [])
                if not msgs:
                    raise HTTPException(status_code=500, detail="Empty response from LangGraph.")
                response = ChatResponse(
                    response=str(msgs[-1].content),
                    thread_id=execution["thread_id"],
                    turn_id=turn_id,
                    mode=result.get("active_mode", execution["mode"]),
                    rag_used=execution["rag_used"],
                    sources=execution["sources"],
                )
            response.turn_id = turn_id
            _mark_turn(record, TURN_COMPLETED, response.model_dump())
            _persist_completed_turn(client_id, thread_id, turn_id, payload.message.strip(), response)
            return response
        except HTTPException:
            _restore_checkpoint(execution)
            _mark_turn(record, TURN_FAILED)
            raise
        except Exception as exc:
            _restore_checkpoint(execution)
            _mark_turn(record, TURN_FAILED)
            print(f"[CHAT ERROR] {type(exc).__name__}", file=sys.stderr)
            raise _classify_groq_error(str(exc))


@app.get("/api/health")
def health_check():
    """Backend health — safe to call publicly, never returns keys."""
    env_configured = bool(os.getenv("GROQ_API_KEY"))
    active_sessions = sum(1 for v in _session_store.values() if not v.is_expired())
    return {
        "status": "healthy",
        "env_key_configured": env_configured,
        "active_visitor_sessions": active_sessions,
        "model": GROQ_DEFAULT_MODEL,
    }


@app.post("/api/reset", response_model=ResetResponse)
def reset_session():
    """Issues a fresh thread_id for a new conversation."""
    new_thread_id = f"study-session-{uuid.uuid4().hex[:8]}"
    return ResetResponse(message="Fresh study session initialized", thread_id=new_thread_id)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTED CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/threads", response_model=ThreadListResponse)
def list_threads(
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Lists this browser's stored conversations, newest first."""
    client_id = _client_id_from_header(x_client_id)
    if not client_id:
        return ThreadListResponse(threads=[])
    return ThreadListResponse(
        threads=[ThreadItem(**row) for row in persistence.list_threads(client_id)]
    )


@app.get("/api/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
def list_thread_messages(
    thread_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Returns the stored transcript for one conversation."""
    client_id = _require_client_id(x_client_id)
    _validate_thread_id(thread_id)
    stored = persistence.list_messages(client_id, thread_id)
    return ThreadMessagesResponse(
        thread_id=thread_id,
        messages=[StoredMessage(**row) for row in stored],
    )


@app.delete("/api/threads/{thread_id}")
def delete_thread(
    thread_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Deletes one stored conversation and its messages."""
    client_id = _require_client_id(x_client_id)
    _validate_thread_id(thread_id)
    if not persistence.delete_thread(client_id, thread_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "success", "message": "Conversation deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT MANAGEMENT & INGESTION (SESSION ISOLATED)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
):
    """
    Accepts and indexes a document for the current session's hybrid RAG store.
    Supports PDF, DOCX, and TXT files only.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    safe_filename = os.path.basename(file.filename.replace("\\", "/"))
    if not safe_filename or safe_filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    ext = os.path.splitext(safe_filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    session_id = _require_session_token(x_session_token)
    session_entry = _resolve_session(session_id)
    _restore_documents_for_session(session_id, session_entry)
    doc_index = get_session_document_index(session_id)

    try:
        content = await file.read(MAX_FILE_SIZE_BYTES + 1)
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB upload limit.",
            )
        meta = doc_index.add_document(safe_filename, content)
        if session_entry.client_id:
            try:
                persistence.save_document(
                    session_entry.client_id,
                    doc_id=meta.doc_id,
                    filename=meta.filename,
                    file_type=meta.file_type,
                    size_kb=meta.size_kb,
                    chunk_count=meta.chunk_count,
                    uploaded_at=meta.uploaded_at,
                    content=content,
                )
            except Exception as exc:
                print(f"[PERSIST ERROR] {type(exc).__name__}", file=sys.stderr)
        return UploadDocumentResponse(
            status="success",
            doc_id=meta.doc_id,
            filename=meta.filename,
            file_type=meta.file_type,
            size_kb=meta.size_kb,
            chunk_count=meta.chunk_count,
            message=f"Document '{meta.filename}' ({meta.size_kb} KB, {meta.chunk_count} chunks) indexed successfully.",
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[UPLOAD ERROR] {type(exc).__name__}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Failed to process the document.")


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
):
    """Lists all documents indexed for the current session."""
    session_id = _require_session_token(x_session_token)
    _restore_documents_for_session(session_id, _resolve_session(session_id))
    doc_index = get_session_document_index(session_id)
    return DocumentListResponse(documents=doc_index.list_documents())


@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
):
    """Deletes a document from the session index and purges its vector and BM25 entries."""
    session_id = _require_session_token(x_session_token)
    session_entry = _resolve_session(session_id)
    _restore_documents_for_session(session_id, session_entry)
    doc_index = get_session_document_index(session_id)
    removed = doc_index.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found or already removed.")
    if session_entry.client_id:
        try:
            persistence.delete_document(session_entry.client_id, doc_id)
        except Exception as exc:
            print(f"[PERSIST ERROR] {type(exc).__name__}", file=sys.stderr)
    return {"status": "success", "message": "Document removed and index updated."}


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n[SERVER] 🚀 TISW Backend → http://localhost:{port}")
    print(f"[SERVER] 🔑 Verify Key:    POST /api/verify-key")
    print(f"[SERVER] 💬 Chat:          POST /api/chat")
    print(f"[SERVER] 🔄 Reset:         POST /api/reset")
    print(f"[SERVER] 💚 Health:        GET  /api/health\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
