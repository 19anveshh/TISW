import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FileText, Trash2, Database, Plus, RefreshCw, X } from '../icons';
import { fetchDocuments, deleteDocument, uploadFile } from '../lib/api';

/**
 * DocumentManager — Sidebar section for toggling RAG and managing uploaded documents.
 * Strict per-session isolation via sessionToken or clientSessionId.
 */
export default function DocumentManager({
  ragEnabled,
  onToggleRag,
  sessionToken,
  clientSessionId,
  onToast,
  documents,
  setDocuments,
}) {
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Load documents when session changes
  const loadDocs = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchDocuments(sessionToken, clientSessionId);
      setDocuments(data.documents || []);
    } catch (err) {
      // Don't show loud error on initial load if backend is booting
      console.warn('Could not load session documents:', err);
    } finally {
      setLoading(false);
    }
  }, [sessionToken, clientSessionId, setDocuments]);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  // Handle single file upload
  const handleUpload = async (file) => {
    if (!file) return;

    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!['.pdf', '.docx', '.txt'].includes(ext)) {
      onToast?.(`Unsupported file type '${ext}'. Please upload a .pdf, .docx, or .txt file.`, 'error');
      return;
    }

    setUploading(true);
    try {
      const res = await uploadFile(file, sessionToken, clientSessionId);
      onToast?.(`Uploaded ${file.name} (${res.chunk_count} chunks indexed)`, 'success');
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
      await deleteDocument(docId, sessionToken, clientSessionId);
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
          onClick={() => !uploading && fileInputRef.current?.click()}
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
              <span className="doc-upload-text">Add PDF, DOCX, or TXT</span>
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
