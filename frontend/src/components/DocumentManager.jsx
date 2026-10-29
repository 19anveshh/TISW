import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FileText, Trash2, Database, Plus, RefreshCw } from '../icons';
import { fetchDocuments, deleteDocument, uploadFile } from '../lib/api';

// Mirrors rag_engine's MAX_FILE_SIZE_BYTES and MAX_FILES_PER_SESSION; the server
// still enforces both, this only stops a doomed upload from leaving the browser.
const MAX_FILE_BYTES = 15 * 1024 * 1024;
const MAX_DOCS_PER_SESSION = 10;

/**
 * DocumentManager — Sidebar section for toggling RAG and managing uploaded documents.
 * Strict per-session isolation via a server-issued sessionToken.
 */
export default function DocumentManager({
  ragEnabled,
  onToggleRag,
  sessionToken,
  onToast,
  documents,
  setDocuments,
}) {
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const statusSeenRef = useRef({});

  // Load documents when session changes
  const loadDocs = useCallback(async () => {
    if (!sessionToken) {
      setDocuments([]);
      return;
    }
    try {
      setLoading(true);
      const data = await fetchDocuments(sessionToken);
      const next = data.documents || [];
      const seen = statusSeenRef.current;
      next.forEach((doc) => {
        const before = seen[doc.doc_id];
        if (before === 'indexing' && doc.status === 'ready') {
          onToast?.(`"${doc.filename}" is ready — ${doc.chunk_count} chunks indexed`, 'success');
        } else if (before === 'indexing' && doc.status === 'failed') {
          onToast?.(`"${doc.filename}" could not be indexed. Remove it and upload again.`, 'error');
        }
        seen[doc.doc_id] = doc.status;
      });
      const liveIds = new Set(next.map((doc) => doc.doc_id));
      Object.keys(seen).forEach((id) => {
        if (!liveIds.has(id)) delete seen[id];
      });
      setDocuments(next);
    } catch (err) {
      // Don't show loud error on initial load if backend is booting
      console.warn('Could not load session documents:', err);
    } finally {
      setLoading(false);
    }
  }, [sessionToken, setDocuments, onToast]);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  // Indexing runs in a server-side worker, so keep refreshing while any document
  // is still reported as 'indexing'. Each refresh re-arms the timer until none remain.
  useEffect(() => {
    if (!sessionToken) return undefined;
    if (!documents.some((doc) => doc.status === 'indexing')) return undefined;
    const timer = setInterval(loadDocs, 3000);
    return () => clearInterval(timer);
  }, [sessionToken, documents, loadDocs]);

  // Handle single file upload
  const handleUpload = async (file) => {
    if (!file) return;
    if (!sessionToken) {
      onToast?.('Configure a Groq API key before uploading documents.', 'error');
      return;
    }

    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!['.pdf', '.docx', '.txt'].includes(ext)) {
      onToast?.(`Unsupported file type '${ext}'. Please upload a .pdf, .docx, or .txt file.`, 'error');
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      onToast?.(`'${file.name}' is ${(file.size / 1048576).toFixed(1)} MB — the limit is 15 MB.`, 'error');
      return;
    }
    if (documents.length >= MAX_DOCS_PER_SESSION) {
      onToast?.('Upload limit reached (10 documents). Remove one before adding another.', 'error');
      return;
    }

    setUploading(true);
    try {
      const res = await uploadFile(file, sessionToken);
      onToast?.(
        `Added ${file.name} — indexing ${res.chunk_count} ${res.chunk_count === 1 ? 'chunk' : 'chunks'} in the background`,
        'info',
      );
      statusSeenRef.current[res.doc_id] = res.index_status || 'indexing';
      // Automatically enable RAG if not already on
      if (!ragEnabled) {
        onToggleRag(true);
      }
      await loadDocs();
    } catch (err) {
      onToast?.(err.message || 'Failed to upload document', 'error');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) await handleUpload(file);
  };

  const handleDelete = async (docId, filename) => {
    try {
      await deleteDocument(docId, sessionToken);
      setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
      onToast?.(`Removed "${filename}"`, 'info');
    } catch (err) {
      onToast?.(err.message || 'Failed to remove document', 'error');
    }
  };

  return (
    <div className="doc-manager-section">
      {/* ── RAG Toggle Header ── */}
      <div className="rag-toggle-card">
        <div className="rag-toggle-info">
          <div className="rag-toggle-title-row">
            <Database className="w-3.5 h-3.5 text-cyan" />
            <span className="rag-toggle-label">Knowledge Base (RAG)</span>
          </div>
          <span className="rag-toggle-desc">
            {ragEnabled
              ? 'Active — answering from documents with citations'
              : 'Off — standard AI tutor mode'}
          </span>
        </div>

        <button
          type="button"
          role="switch"
          aria-checked={ragEnabled}
          className={`rag-switch ${ragEnabled ? 'rag-switch--active' : ''}`}
          onClick={() => onToggleRag(!ragEnabled)}
          title={ragEnabled ? 'Disable document RAG' : 'Enable document RAG'}
        >
          <span className="rag-switch-handle" />
        </button>
      </div>

      {/* ── Upload & Document List ── */}
      <div className="doc-manager-body">
        <div className="doc-manager-header">
          <span className="sidebar-section-label">
            Documents ({documents.length}/10)
          </span>
          <button
            type="button"
            className="doc-refresh-btn"
            onClick={loadDocs}
            disabled={loading}
            title="Refresh document list"
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'spin' : ''}`} />
          </button>
        </div>

        {/* Drop / Browse upload button */}
        <div
          className={`doc-upload-box ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={(e) => { e.preventDefault(); setDragOver(false); }}
          onDrop={handleDrop}
          onClick={() => !uploading && sessionToken && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files?.[0]) {
                handleUpload(e.target.files[0]);
                e.target.value = '';
              }
            }}
          />
          {uploading ? (
            <div className="doc-upload-inner">
              <RefreshCw className="w-3.5 h-3.5 spin text-cyan" />
              <span className="doc-upload-text">Extracting & indexing…</span>
            </div>
          ) : (
            <div className="doc-upload-inner">
              <Plus className="w-3.5 h-3.5 text-cyan" />
              <span className="doc-upload-text">
                {sessionToken ? 'Add PDF, DOCX, or TXT' : 'Configure Groq key to upload'}
              </span>
            </div>
          )}
        </div>

        {/* Document items list */}
        {documents.length === 0 && !loading && (
          <div className="doc-empty-state">
            <FileText className="w-5 h-5 text-muted" />
            <p className="doc-empty-title">No documents yet</p>
            <p className="doc-empty-sub">
              Upload study guides, slides, or notes to ground your answers.
            </p>
          </div>
        )}

        <div className="doc-list">
          {documents.map((doc) => {
            const extUpper = (doc.file_type || 'txt').toUpperCase();
            return (
              <div key={doc.doc_id} className="doc-item">
                <div className="doc-item-left">
                  <span className={`doc-type-badge doc-type--${extUpper.toLowerCase()}`}>
                    {extUpper}
                  </span>
                  <div className="doc-item-details">
                    <span className="doc-item-name" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <span className="doc-item-meta">
                      {doc.size_kb} KB · {doc.chunk_count} chunks
                      {doc.status === 'indexing' && (
                        <span className="doc-status doc-status--indexing">
                          <RefreshCw className="w-2.5 h-2.5 spin" /> Indexing
                        </span>
                      )}
                      {doc.status === 'failed' && (
                        <span className="doc-status doc-status--failed">Not indexed</span>
                      )}
                    </span>
                  </div>
                </div>

                <button
                  type="button"
                  className="doc-delete-btn"
                  onClick={() => handleDelete(doc.doc_id, doc.filename)}
                  title={`Delete ${doc.filename}`}
                  aria-label={`Delete ${doc.filename}`}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
