/**
 * api.js — Backend API client for TISW (Teaching in a Structured Way).
 * Handles chat requests to FastAPI backend, health checks, thread resets, and uploads.
 */

export const API_BASE = "http://localhost:8000";

/**
 * Sends a chat message to the FastAPI backend.
 * @param {Object} params
 * @param {string} params.message - The user's input prompt
 * @param {string} params.threadId - Current conversation thread ID
 * @param {string} [params.mode] - Optional pedagogical mode
 * @returns {Promise<{ response: string, thread_id: string }>}
 */
export async function sendChatMessage({ message, threadId, mode }) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      thread_id: threadId,
      mode: mode || "default",
    }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Server error (${res.status})`);
  }

  return res.json();
}

/**
 * Uploads a file for study document ingestion / context.
 * @param {File} file
 * @returns {Promise<{ status: string, filename: string, size_kb: number, message: string }>}
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Upload failed (${res.status})`);
  }

  return res.json();
}

/**
 * Requests a fresh thread ID from the backend.
 * @returns {Promise<{ message: string, thread_id: string }>}
 */
export async function resetThread() {
  try {
    const res = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to reset session on server");
    return res.json();
  } catch {
    // Client-side fallback if server is momentarily unreachable
    return {
      message: "Client fallback session",
      thread_id: `study-session-${crypto.randomUUID().slice(0, 8)}`,
    };
  }
}

/**
 * Checks backend health status.
 * @returns {Promise<{ status: string, model?: string }>}
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
