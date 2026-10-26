import React, { useCallback, useState } from 'react';
import { uploadFile } from '../lib/api';

/**
 * FileDropZone — Drag-and-drop file upload area for RAG document ingestion.
 * Supports: PDF, TXT, MD, DOCX, PY, JS, TS, CSV, JSON
 */
export default function FileDropZone({ onUploadSuccess, onUploadError, disabled }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const file = e.dataTransfer?.files?.[0];
    if (!file) return;

    await processFile(file);
  }, []);

  const handleFileSelect = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processFile(file);
    e.target.value = '';
  }, []);

  const processFile = async (file) => {
    setIsUploading(true);
    setUploadedFile(null);

    try {
      const result = await uploadFile(file);
      setUploadedFile({ name: file.name, size: result.size_kb });
      onUploadSuccess?.(result);
    } catch (err) {
      onUploadError?.(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      className={`file-drop-zone ${isDragOver ? 'file-drop-zone--active' : ''} ${isUploading ? 'file-drop-zone--uploading' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
      onDrop={handleDrop}
    >
      {isUploading ? (
        <div className="file-drop-content">
          <div className="file-drop-spinner" />
          <span className="file-drop-text">Uploading...</span>
        </div>
      ) : uploadedFile ? (
        <div className="file-drop-content">
          <span className="file-drop-icon">✅</span>
          <span className="file-drop-text">
            {uploadedFile.name} ({uploadedFile.size}KB)
          </span>
          <button
            type="button"
            className="file-drop-clear"
            onClick={() => setUploadedFile(null)}
          >
            ×
          </button>
        </div>
      ) : (
        <label className="file-drop-content file-drop-label">
          <span className="file-drop-icon">📎</span>
          <span className="file-drop-text">
            {isDragOver ? 'Drop file here...' : 'Drop file or click to upload'}
          </span>
          <input
            type="file"
            onChange={handleFileSelect}
            accept=".pdf,.txt,.md,.docx,.py,.js,.ts,.csv,.json"
            className="file-drop-input"
            disabled={disabled}
          />
        </label>
      )}
    </div>
  );
}
