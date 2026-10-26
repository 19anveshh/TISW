"""
server.py - FastAPI Backend for TISW (Teaching in a Structured Way).

Integrates the LangGraph tutor workflow from main.py and exposes:
  POST /api/chat    - Invoke tutor_app workflow and return { "response": tutor_reply, "thread_id": thread_id }
  POST /api/reset   - Spawn a fresh conversation thread_id
  GET  /api/health  - Backend health & Groq model status
"""

import os
import sys
import uuid
import traceback
from typing import Optional

# Force UTF-8 encoding for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# 1. Import LangGraph workflow (tutor_app) from main.py
from main import tutor_app, load_and_validate_groq_llm, create_tutor_graph

# Fallback: ensure tutor_app is initialized if main.py couldn't load it at import time
if tutor_app is None:
    try:
        _llm = load_and_validate_groq_llm()
        tutor_app = create_tutor_graph(_llm, InMemorySaver())
        print("[INIT] ✅ LangGraph Tutor Graph initialized successfully.")
    except Exception as exc:
        print(f"[INIT] ⚠️ Failed to initialize LangGraph Tutor Graph: {exc}", file=sys.stderr)

# =====================================================================
# APP INITIALIZATION & CORS CONFIGURATION
# =====================================================================

app = FastAPI(title="TISW AI Study Tutor Backend", version="2.0.0")

# Enable CORS middleware so the Vite/React dev server can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# REQUEST & RESPONSE SCHEMAS
# =====================================================================

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


# =====================================================================
# HELPER: Pedagogical Mode Instruction
# =====================================================================

def enrich_message(user_text: str, mode: Optional[str] = None) -> str:
    """Prepends temporary turn instructions based on teaching mode."""
    mode = (mode or "default").lower()
    prefix = ""
    if mode == "brief":
        prefix = "[Instruction: explain briefly / TL;DR, direct and concise]\n"
    elif mode == "deep":
        prefix = "[Instruction: go deeper / under the hood / architectural deep dive]\n"
    elif mode == "socratic":
        prefix = "[Instruction: teach me through Socratic questions — present a scenario and ask guiding questions]\n"
    elif mode == "confused":
        prefix = "[Instruction: The student is confused. Never repeat the same explanation. Step back, drop abstraction down one level, switch to a completely fresh everyday analogy, and trace with tiny input (x = 2).]\n"

    return prefix + user_text


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend and Groq LLM status."""
    return {
        "status": "healthy" if tutor_app is not None else "degraded",
        "model": os.getenv("GROQ_MODEL", "unknown"),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Core Chat Endpoint:
    - Accepts { "thread_id": string, "message": string }
    - Calls tutor_app.invoke() with LangGraph config thread_id
    - Returns { "response": tutor_reply, "thread_id": thread_id }
    """
    global tutor_app
    if not tutor_app:
        try:
            _llm = load_and_validate_groq_llm()
            tutor_app = create_tutor_graph(_llm, InMemorySaver())
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Tutor graph is not initialized. Please check GROQ_API_KEY in .env. ({e})"
            )

    user_text = payload.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Generate or reuse thread_id
    thread_id = payload.thread_id.strip() if payload.thread_id else f"study-session-{uuid.uuid4().hex[:8]}"
    enriched_input = enrich_message(user_text, payload.mode)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Invoke the LangGraph workflow
        result = tutor_app.invoke(
            {"messages": [HumanMessage(content=enriched_input)]},
            config=config,
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="Empty response returned from LangGraph.")

        tutor_reply = str(messages[-1].content)

        return ChatResponse(
            response=tutor_reply,
            thread_id=thread_id,
        )

    except HTTPException:
        raise
    except Exception as exc:
        error_msg = str(exc)
        print(f"[CHAT ERROR] {traceback.format_exc()}", file=sys.stderr)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise HTTPException(status_code=429, detail="Groq rate limit reached. Please wait a moment and try again.")
        if "401" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY. Please check your .env file.")
        raise HTTPException(status_code=500, detail=f"Tutor error: {error_msg}")


@app.post("/api/reset", response_model=ResetResponse)
def reset_session():
    """Generates a fresh thread_id for a new conversation."""
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
        "message": f"Document '{file.filename}' ({size_kb} KB) uploaded successfully."
    }


# =====================================================================
# SERVER RUNNER
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n[SERVER] 🚀 TISW Backend -> http://localhost:{port}")
    print(f"[SERVER] 💬 Chat Endpoint: POST /api/chat")
    print(f"[SERVER] 🔄 Reset Session:  POST /api/reset")
    print(f"[SERVER] 💚 Health Check:   GET  /api/health\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
