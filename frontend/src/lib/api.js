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

export const API_BASE = import.meta.env?.VITE_API_BASE_URL || "http://localhost:8000";
const STREAM_DIAGNOSTICS = import.meta.env?.VITE_TISW_STREAM_DIAGNOSTICS === "true";
const CLIENT_ID_KEY = "tisw.client_id";
const CLIENT_ID_PATTERN = /^[A-Za-z0-9_-]{8,64}$/;

function streamDiagnostic(stage, details = {}) {
  if (STREAM_DIAGNOSTICS) {
    console.debug("[TISW stream]", stage, details);
  }
}

/**
 * Browser-stable id that lets the backend reload this visitor's saved
 * conversations and documents after a refresh. It scopes local data only;
 * it is never used for authentication.
 */
export function getClientId() {
  try {
    const stored = localStorage.getItem(CLIENT_ID_KEY);
    if (stored && CLIENT_ID_PATTERN.test(stored)) return stored;
    const generated = crypto.randomUUID().replace(/-/g, "");
    localStorage.setItem(CLIENT_ID_KEY, generated);
    return generated;
  } catch {
    return null;
  }
}

export function isActiveStreamRequest(activeRequest, requestId, threadId) {
  return Boolean(
    activeRequest &&
    activeRequest.requestId === requestId &&
    activeRequest.threadId === threadId
  );
}

function sessionHeaders(sessionToken, headers = {}) {
  if (sessionToken) headers["X-Session-Token"] = sessionToken;
  const clientId = getClientId();
  if (clientId) headers["X-Client-Id"] = clientId;
  return headers;
}

async function parseResponse(res, fallbackMessage) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${fallbackMessage} (${res.status})`);
  return data;
}

const NETWORK_HINT =
  'Could not reach the tutor server. It may still be starting up — wait a few seconds and try again.';

/**
 * fetch() rejects with a bare "Failed to fetch" when the request never gets a
 * response: a CORS rejection, a dropped connection, or the host waking from a
 * cold start. A sleeping host usually accepts the retried request, so retry
 * once before surfacing a message the visitor can act on.
 */
async function fetchReachable(url, options) {
  try {
    return await fetch(url, options);
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    await new Promise((resolve) => setTimeout(resolve, 2500));
    try {
      return await fetch(url, options);
    } catch (retryErr) {
      if (retryErr?.name === 'AbortError') throw retryErr;
      throw new Error(NETWORK_HINT);
    }
  }
}

/** Create a server-issued session for the local .env Groq key. */
export async function createServerSession() {
  const res = await fetchReachable(`${API_BASE}/api/session`, {
    method: "POST",
    headers: sessionHeaders(null),
  });
  return parseResponse(res, 'Could not create a tutor session');
}

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
  const res = await fetchReachable(`${API_BASE}/api/verify-key`, {
    method: "POST",
    headers: sessionHeaders(null, { "Content-Type": "application/json" }),
    body: JSON.stringify({ api_key: apiKey }),
  });

  const data = await parseResponse(res, 'Key verification failed');
  // data.session_token is opaque — it is NOT the api key
  return data;
}

// ─── Persisted Conversation History ───────────────────────────────────────────

/** Lists this browser's stored conversations, newest first. */
export async function fetchThreads() {
  const res = await fetch(`${API_BASE}/api/threads`, {
    headers: sessionHeaders(null),
  });
  return parseResponse(res, 'Could not load conversation history');
}

/** Returns the stored transcript for one conversation. */
export async function fetchThreadMessages(threadId) {
  const res = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}/messages`, {
    headers: sessionHeaders(null),
  });
  return parseResponse(res, 'Could not load that conversation');
}

/** Deletes one stored conversation. */
export async function deleteThread(threadId) {
  const res = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}`, {
    method: "DELETE",
    headers: sessionHeaders(null),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const error = new Error(data.detail || `Could not delete that conversation (${res.status})`);
    error.status = res.status;
    throw error;
  }
  return res.json().catch(() => ({}));
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
 * @param {boolean} [params.ragEnabled]   — whether document RAG is active
 * @returns {Promise<{ response: string, thread_id: string, mode: string, rag_used: boolean, sources?: Array }>}
 */
export async function sendChatMessage({
  message,
  threadId,
  turnId,
  mode,
  sessionToken,
  ragEnabled = false,
}) {
  const headers = sessionHeaders(sessionToken, { "Content-Type": "application/json" });

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      message,
      thread_id: threadId,
      turn_id: turnId,
      mode: mode || "normal",
      rag_enabled: Boolean(ragEnabled),
    }),
  });

  return parseResponse(res, 'Server error');
}

/**
 * Stream one chat turn using the backend's SSE protocol.
 * The callbacks receive parsed events; the response body is never buffered.
 */
export async function streamChatMessage({
  message,
  threadId,
  turnId,
  sessionToken,
  ragEnabled = false,
  tracingEnabled = false,
  signal,
  onStart,
  onDelta,
  onSources,
  onDone,
  onError,
}) {
  const headers = sessionHeaders(sessionToken, { "Content-Type": "application/json" });
  streamDiagnostic("request", {
    message,
    threadId,
    turnId,
    tracingEnabled: Boolean(tracingEnabled),
    hasSessionToken: Boolean(sessionToken),
  });
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers,
    signal,
    body: JSON.stringify({
      message,
      thread_id: threadId,
      turn_id: turnId,
      rag_enabled: Boolean(ragEnabled),
      tracing_enabled: Boolean(tracingEnabled),
    }),
  });

  if (!res.ok) return parseResponse(res, 'Server error');
  if (!res.body) throw new Error('Streaming is unavailable in this browser.');

  const reader = res.body.getReader();
  return consumeSseReader(reader, {
    onStart: (payload) => {
      streamDiagnostic("start", payload);
      onStart?.(payload);
    },
    onDelta: (text) => {
      streamDiagnostic("delta", { text });
      onDelta?.(text);
    },
    onSources: (sources) => onSources?.(sources),
    onDone: (payload) => {
      streamDiagnostic("done", payload);
      onDone?.(payload);
    },
    onError: (error) => {
      streamDiagnostic("error", { message: error.message, status: error.status });
      onError?.(error);
    },
  });
}

/** Consume an already-open SSE reader; exported for deterministic browserless tests. */
export async function consumeSseReader(reader, { onStart, onDelta, onSources, onDone, onError } = {}) {
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';
  let dataLines = [];
  let sawDone = false;

  const dispatch = (frame) => {
    const lines = frame.split(/\r?\n/);
    currentEvent = 'message';
    dataLines = [];
    for (const line of lines) {
      if (line.startsWith('event:')) currentEvent = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) return;
    let payload;
    try {
      payload = JSON.parse(dataLines.join('\n'));
    } catch {
      throw new Error('The tutor returned an invalid streaming event.');
    }
    if (currentEvent === 'start') onStart?.(payload);
    else if (currentEvent === 'delta') onDelta?.(payload.text || '');
    else if (currentEvent === 'sources') onSources?.(payload.sources || []);
    else if (currentEvent === 'done') {
      sawDone = true;
      onDone?.(payload);
    }
    else if (currentEvent === 'error') {
      const error = new Error(payload.message || 'Generation failed.');
      error.status = payload.status;
      onError?.(error);
      throw error;
    }
  };

  const processFrames = () => {
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() || '';
    for (const frame of frames) if (frame.trim()) dispatch(frame);
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      processFrames();
    }
    buffer += decoder.decode();
    // A trailing frame without the SSE blank-line terminator is incomplete.
    // Do not promote it to a successful done event at EOF.
    if (!sawDone) throw new Error('The tutor stream ended before completion.');
    return { streamed: true };
  } catch (error) {
    try { await reader.cancel(); } catch { /* preserve original error */ }
    throw error;
  } finally {
    try { reader.releaseLock(); } catch { /* reader may already be released */ }
  }
}

/** Safe local capability flags for the tracing control. */
export async function getRuntimeStatus() {
  const res = await fetch(`${API_BASE}/api/runtime`);
  return parseResponse(res, 'Could not read backend runtime status');
}

// ─── File & Document Management ───────────────────────────────────────────────

export const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"];

/**
 * Uploads a study document for hybrid RAG indexing.
 * Supports PDF, DOCX, and TXT files only.
 *
 * @param {File} file
 * @param {string|null} [sessionToken]
 */
export async function uploadFile(file, sessionToken) {
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    throw new Error(`Unsupported file type '${ext}'. Please upload a PDF, DOCX, or TXT file.`);
  }

  const formData = new FormData();
  formData.append("file", file);

  const headers = sessionHeaders(sessionToken);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    headers,
    body: formData,
  });

  return parseResponse(res, 'Upload failed');
}

/**
 * Fetches the list of uploaded documents for the active session.
 *
 * @param {string|null} [sessionToken]
 * @returns {Promise<{ documents: Array }>}
 */
export async function fetchDocuments(sessionToken) {
  const headers = sessionHeaders(sessionToken);

  const res = await fetch(`${API_BASE}/api/documents`, {
    method: "GET",
    headers,
  });

  return parseResponse(res, 'Failed to fetch documents');
}

/**
 * Deletes a document from the session index.
 *
 * @param {string} docId
 * @param {string|null} [sessionToken]
 */
export async function deleteDocument(docId, sessionToken) {
  const headers = sessionHeaders(sessionToken);

  const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
    method: "DELETE",
    headers,
  });

  return parseResponse(res, 'Failed to delete document');
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
