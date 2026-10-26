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

LOCAL DEV FALLBACK
══════════════════
If no X-Session-Token header is present and GROQ_API_KEY is set in the
.env file, the backend uses that environment-variable key transparently
(preserving the existing local development workflow).

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
  POST /api/verify-key  — Verify a visitor key, receive session token
  POST /api/chat        — Invoke LangGraph tutor (per-request LLM)
  POST /api/reset       — Issue fresh thread_id
  GET  /api/health      — Backend health
  POST /api/upload      — Document upload
"""

import os
import sys
import uuid
import time
import hmac
import hashlib
import traceback
from typing import Optional, Dict

# ── Windows UTF-8 console fix ──────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from main import (
    tutor_app,
    load_and_validate_groq_llm,
    create_tutor_graph,
    TEACHING_METHODOLOGY_PROMPT,
    build_system_message,
    detect_persistent_preference,
)

# ── Fallback: try to init global tutor_app if main.py left it as None ─────────
if tutor_app is None:
    try:
        _llm = load_and_validate_groq_llm()
        tutor_app = create_tutor_graph(_llm, InMemorySaver())
        print("[INIT] ✅ LangGraph Tutor Graph initialized (env key).")
    except Exception as exc:
        print(f"[INIT] ⚠️  No env key — visitors must supply their own Groq key. ({exc})",
              file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════════════
# PER-VISITOR SESSION STORE
# ═══════════════════════════════════════════════════════════════════════════════

SESSION_TTL_SECONDS = 7200   # 2 hours; token expires after this

class _SessionEntry:
    __slots__ = ("api_key", "model", "expires_at", "checkpointer")

    def __init__(self, api_key: str, model: str):
        self.api_key: str = api_key
        self.model: str = model
        self.expires_at: float = time.monotonic() + SESSION_TTL_SECONDS
        # Each visitor session gets its own isolated checkpointer so conversation
        # history from one user cannot bleed into another's.
        self.checkpointer: InMemorySaver = InMemorySaver()

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at

    def refresh(self) -> None:
        self.expires_at = time.monotonic() + SESSION_TTL_SECONDS


# Dict[session_token -> _SessionEntry]
_session_store: Dict[str, _SessionEntry] = {}


def _prune_expired_sessions() -> None:
    """Remove expired entries (called on every verify-key to avoid unbounded growth)."""
    expired = [k for k, v in _session_store.items() if v.is_expired()]
    for k in expired:
        del _session_store[k]


def _resolve_session(token: str) -> _SessionEntry:
    """
    Look up a session token. Raises 401 if missing or expired.
    Refreshes the TTL on every successful lookup.
    """
    entry = _session_store.get(token)
    if entry is None or entry.is_expired():
        if entry is not None:
            del _session_store[token]
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


# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="TISW AI Tutor Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    mode: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class ResetResponse(BaseModel):
    message: str
    thread_id: str


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
    """Prepend mode-specific turn instructions (unchanged from original)."""
    mode = (mode or "default").lower()
    prefix = ""
    if mode == "brief":
        prefix = "[Instruction: explain briefly / TL;DR, direct and concise]\n"
    elif mode == "deep":
        prefix = "[Instruction: go deeper / under the hood / architectural deep dive]\n"
    elif mode == "socratic":
        prefix = "[Instruction: teach me through Socratic questions]\n"
    elif mode == "confused":
        prefix = (
            "[Instruction: The student is confused. Never repeat the same explanation. "
            "Step back, drop abstraction down one level, switch to a fresh everyday analogy.]\n"
        )
    return prefix + user_text


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/verify-key", response_model=VerifyKeyResponse)
async def verify_key(payload: VerifyKeyRequest):
    """
    Verify a visitor-supplied Groq API key with a real test request.
    Returns an opaque session token on success.
    The raw key is NEVER returned and is stored only server-side.
    """
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
        # Log only a non-secret fingerprint, not the key itself
        key_hint = f"{raw_key[:6]}...{raw_key[-4:]}" if len(raw_key) > 10 else "****"
        print(f"[VERIFY] Key verification failed for hint={key_hint}: {exc}", file=sys.stderr)
        raise _classify_groq_error(str(exc))

    # ── Verification passed — issue session token ──────────────────────────────
    _prune_expired_sessions()
    token = str(uuid.uuid4())   # cryptographically random opaque token
    _session_store[token] = _SessionEntry(api_key=raw_key, model=model)

    print(f"[SESSION] New visitor session created (token_prefix={token[:8]})")
    return VerifyKeyResponse(
        session_token=token,
        model=model,
        message="Groq API key verified. Session active.",
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
):
    """
    Core chat endpoint.

    Key resolution order:
      1. If X-Session-Token is present → resolve visitor's key from session store.
      2. Else → fall back to server env GROQ_API_KEY (local dev only).

    In both cases a per-request ChatGroq instance is built so no key ever
    leaks across different visitors or requests.
    """
    user_text = payload.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = (payload.thread_id or "").strip() or f"study-session-{uuid.uuid4().hex[:8]}"
    enriched = enrich_message(user_text, payload.mode)

    # ── Resolve which key/checkpointer/graph to use ────────────────────────────
    if x_session_token:
        # Visitor-supplied key path
        session = _resolve_session(x_session_token)
        llm = _build_llm_for_session(session)
        graph = create_tutor_graph(llm, session.checkpointer)
    elif tutor_app is not None:
        # Local dev / server env-key path
        graph = tutor_app
    else:
        raise HTTPException(
            status_code=503,
            detail="No API key configured. Please verify your Groq API key using the sidebar.",
        )

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=enriched)]},
            config=config,
        )
        msgs = result.get("messages", [])
        if not msgs:
            raise HTTPException(status_code=500, detail="Empty response from LangGraph.")

        tutor_reply = str(msgs[-1].content)
        return ChatResponse(response=tutor_reply, thread_id=thread_id)

    except HTTPException:
        raise
    except Exception as exc:
        # Log traceback without the key
        print(f"[CHAT ERROR] {traceback.format_exc()}", file=sys.stderr)
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


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts document uploads for contextual tutor ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    content = await file.read()
    size_kb = round(len(content) / 1024, 1)
    return {
        "status": "success",
        "filename": file.filename,
        "size_kb": size_kb,
        "message": f"Document '{file.filename}' ({size_kb} KB) uploaded successfully.",
    }


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
