/**
 * api.js — TISW Backend API client.
 *
 * Session token model:
 *   1. Visitor calls verifyGroqKey(apiKey) → backend verifies with real Groq call
 *      → returns opaque session_token.
 *   2. All subsequent sendChatMessage calls include X-Session-Token header.
 *   3. The raw API key is NEVER stored here — only the opaque token lives in
 *      React state (in-memory, cleared on page refresh).
 */

export const API_BASE = "http://localhost:8000";

// ─── Key Verification ──────────────────────────────────────────────────────────

/**
 * Verify a visitor-supplied Groq API key with the backend.
 * Returns { session_token, model, message } on success.
 * Throws a user-friendly Error on failure.
 *
 * @param {string} apiKey  — raw gsk_... key
 * @returns {Promise<{ session_token: string, model: string, message: string }>}
 */
export async function verifyGroqKey(apiKey) {
  const res = await fetch(`${API_BASE}/api/verify-key`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Key verification failed (${res.status})`);
  }
  // data.session_token is opaque — it is NOT the api key
  return data;
}

// ─── Chat ──────────────────────────────────────────────────────────────────────

/**
 * Sends a chat message to the FastAPI/LangGraph backend.
 *
 * @param {Object} params
 * @param {string} params.message         — user's prompt
 * @param {string} params.threadId        — current conversation thread ID
 * @param {string} [params.mode]          — pedagogical mode ('normal' | 'tisw')
 * @param {string|null} [params.sessionToken]    — visitor session token
 * @param {string|null} [params.clientSessionId] — isolated client session ID
 * @param {boolean} [params.ragEnabled]   — whether document RAG is active
 * @returns {Promise<{ response: string, thread_id: string, mode: string, rag_used: boolean, sources?: Array }>}
 */
export async function sendChatMessage({
  message,
  threadId,
  mode,
  sessionToken,
  clientSessionId,
  ragEnabled = false,
}) {
  const headers = { "Content-Type": "application/json" };
  if (sessionToken) {
    headers["X-Session-Token"] = sessionToken;
  }
  if (clientSessionId) {
    headers["X-Client-Session-Id"] = clientSessionId;
  }

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      message,
      thread_id: threadId,
      mode: mode || "normal",
      rag_enabled: Boolean(ragEnabled),
    }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Server error (${res.status})`);
  }

  return res.json();
}

// ─── File & Document Management ───────────────────────────────────────────────

export const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"];

/**
 * Uploads a study document for hybrid RAG indexing.
 * Supports PDF, DOCX, and TXT files only.
 *
 * @param {File} file
 * @param {string|null} [sessionToken]
 * @param {string|null} [clientSessionId]
 */
export async function uploadFile(file, sessionToken, clientSessionId) {
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    throw new Error(`Unsupported file type '${ext}'. Please upload a PDF, DOCX, or TXT file.`);
  }

  const formData = new FormData();
  formData.append("file", file);

  const headers = {};
  if (sessionToken) headers["X-Session-Token"] = sessionToken;
  if (clientSessionId) headers["X-Client-Session-Id"] = clientSessionId;

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Upload failed (${res.status})`);
  }

  return res.json();
}

/**
 * Fetches the list of uploaded documents for the active session.
 *
 * @param {string|null} [sessionToken]
 * @param {string|null} [clientSessionId]
 * @returns {Promise<{ documents: Array }>}
 */
export async function fetchDocuments(sessionToken, clientSessionId) {
  const headers = {};
  if (sessionToken) headers["X-Session-Token"] = sessionToken;
  if (clientSessionId) headers["X-Client-Session-Id"] = clientSessionId;

  const res = await fetch(`${API_BASE}/api/documents`, {
    method: "GET",
    headers,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Failed to fetch documents (${res.status})`);
  }

  return res.json();
}

/**
 * Deletes a document from the session index.
 *
 * @param {string} docId
 * @param {string|null} [sessionToken]
 * @param {string|null} [clientSessionId]
 */
export async function deleteDocument(docId, sessionToken, clientSessionId) {
  const headers = {};
  if (sessionToken) headers["X-Session-Token"] = sessionToken;
  if (clientSessionId) headers["X-Client-Session-Id"] = clientSessionId;

  const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
    method: "DELETE",
    headers,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Failed to delete document (${res.status})`);
  }

  return res.json();
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Requests a fresh thread ID.
 */
export async function resetThread() {
  try {
    const res = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    if (!res.ok) throw new Error("Server reset failed");
    return res.json();
  } catch {
    return {
      message: "Client fallback session",
      thread_id: `study-session-${crypto.randomUUID().slice(0, 8)}`,
    };
  }
}

/**
 * Backend health check.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return { status: "unreachable" };
    return res.json();
  } catch {
    return { status: "unreachable" };
  }
}
