"""
persistence.py — SQLite storage for chat history and uploaded documents.

Rows are scoped by a browser-stable client id (X-Client-Id) because session
tokens are short-lived and re-issued on every reload. The client id is not
authentication: it only partitions this local workspace so a returning visitor
sees the same threads, messages, and documents.

The database lives at TISW_DB_PATH (default: .data/tisw.db) and is safe to
delete while the server is stopped.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_DB_PATH = os.path.abspath(
    os.getenv("TISW_DB_PATH")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "tisw.db")
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    client_id  TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT 'New Session',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (client_id, thread_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    turn_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources     TEXT,
    rag_used    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    UNIQUE (client_id, thread_id, turn_id, role)
);

CREATE INDEX IF NOT EXISTS idx_messages_thread
    ON messages (client_id, thread_id, id);

CREATE TABLE IF NOT EXISTS documents (
    client_id   TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    size_kb     REAL NOT NULL,
    chunk_count INTEGER NOT NULL,
    uploaded_at REAL NOT NULL,
    content     BLOB NOT NULL,
    PRIMARY KEY (client_id, doc_id)
);
"""

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None


def db_path() -> str:
    return _DB_PATH


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    directory = os.path.dirname(_DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    _conn = conn
    return conn


def _thread_label(user_text: str) -> str:
    collapsed = " ".join((user_text or "").split())
    return collapsed[:48] or "New Session"


def record_turn(
    client_id: str,
    thread_id: str,
    turn_id: str,
    user_text: str,
    assistant_text: str,
    *,
    sources: Optional[List[Dict[str, Any]]] = None,
    rag_used: bool = False,
) -> None:
    """Persist one completed turn. Safe to call twice for the same turn id."""
    if not client_id or not thread_id or not turn_id:
        return
    now = time.time()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    with _lock:
        conn = _connection()
        with conn:
            conn.execute(
                "INSERT INTO threads (client_id, thread_id, label, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(client_id, thread_id) DO UPDATE SET updated_at = excluded.updated_at",
                (client_id, thread_id, _thread_label(user_text), now, now),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO messages "
                "(client_id, thread_id, turn_id, role, content, sources, rag_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (client_id, thread_id, turn_id, "user", user_text, "[]", 0, now),
                    (
                        client_id, thread_id, turn_id, "assistant", assistant_text,
                        sources_json, 1 if rag_used else 0, now,
                    ),
                ),
            )


def list_threads(client_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    if not client_id:
        return []
    with _lock:
        rows = _connection().execute(
            "SELECT t.thread_id, t.label, t.created_at, t.updated_at, "
            "  (SELECT COUNT(*) FROM messages m "
            "   WHERE m.client_id = t.client_id AND m.thread_id = t.thread_id) AS message_count "
            "FROM threads t WHERE t.client_id = ? "
            "ORDER BY t.updated_at DESC LIMIT ?",
            (client_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def list_messages(client_id: str, thread_id: str) -> List[Dict[str, Any]]:
    if not client_id or not thread_id:
        return []
    with _lock:
        rows = _connection().execute(
            "SELECT id, role, content, sources, rag_used, created_at "
            "FROM messages WHERE client_id = ? AND thread_id = ? ORDER BY id ASC",
            (client_id, thread_id),
        ).fetchall()
    messages: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["sources"] = json.loads(row["sources"] or "[]")
        except (TypeError, ValueError):
            item["sources"] = []
        item["rag_used"] = bool(row["rag_used"])
        messages.append(item)
    return messages


def load_history_for_graph(client_id: str, thread_id: str, limit: int = 24) -> List[Dict[str, str]]:
    """Latest turns in chronological order, for seeding a fresh checkpointer."""
    if not client_id or not thread_id:
        return []
    with _lock:
        rows = _connection().execute(
            "SELECT role, content FROM messages "
            "WHERE client_id = ? AND thread_id = ? ORDER BY id DESC LIMIT ?",
            (client_id, thread_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def delete_thread(client_id: str, thread_id: str) -> bool:
    if not client_id or not thread_id:
        return False
    with _lock:
        conn = _connection()
        with conn:
            conn.execute(
                "DELETE FROM messages WHERE client_id = ? AND thread_id = ?",
                (client_id, thread_id),
            )
            cursor = conn.execute(
                "DELETE FROM threads WHERE client_id = ? AND thread_id = ?",
                (client_id, thread_id),
            )
    return cursor.rowcount > 0


def save_document(
    client_id: str,
    *,
    doc_id: str,
    filename: str,
    file_type: str,
    size_kb: float,
    chunk_count: int,
    uploaded_at: float,
    content: bytes,
) -> None:
    if not client_id:
        return
    with _lock:
        conn = _connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(client_id, doc_id, filename, file_type, size_kb, chunk_count, uploaded_at, content) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    client_id, doc_id, filename, file_type, float(size_kb),
                    int(chunk_count), float(uploaded_at), sqlite3.Binary(content),
                ),
            )


def list_document_metadata(client_id: str) -> List[Dict[str, Any]]:
    if not client_id:
        return []
    with _lock:
        rows = _connection().execute(
            "SELECT doc_id, filename, file_type, size_kb, chunk_count, uploaded_at "
            "FROM documents WHERE client_id = ? ORDER BY uploaded_at ASC",
            (client_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_documents_for_restore(client_id: str) -> List[Dict[str, Any]]:
    if not client_id:
        return []
    with _lock:
        rows = _connection().execute(
            "SELECT doc_id, filename, content FROM documents "
            "WHERE client_id = ? ORDER BY uploaded_at ASC",
            (client_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(client_id: str, doc_id: str) -> bool:
    if not client_id or not doc_id:
        return False
    with _lock:
        conn = _connection()
        with conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE client_id = ? AND doc_id = ?",
                (client_id, doc_id),
            )
    return cursor.rowcount > 0
