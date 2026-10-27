import React, { useCallback, useState } from 'react';
import { uploadFile } from '../lib/api';

/**
 * FileDropZone — Drag-and-drop file upload area for RAG document ingestion.
 * Supports: PDF (.pdf), DOCX (.docx), TXT (.txt)
 */
export default function FileDropZone({
  onUploadSuccess,
  onUploadError,
  disabled,
  sessionToken,
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);

  const processFile = useCallback(async (file) => {
    setIsUploading(true);
    setUploadedFile(null);

    try {
      const result = await uploadFile(file, sessionToken);
      setUploadedFile({ name: file.name, size: result.size_kb, chunks: result.chunk_count });
      onUploadSuccess?.(result);
    } catch (err) {
      onUploadError?.(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, [onUploadError, onUploadSuccess, sessionToken]);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const file = e.dataTransfer?.files?.[0];
    if (file) await processFile(file);
  }, [processFile]);

  const handleFileSelect = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (file) await processFile(file);
    e.target.value = '';
  }, [processFile]);

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
          <span className="file-drop-text">Extracting & indexing document chunks...</span>
        </div>
      ) : uploadedFile ? (
        <div className="file-drop-content">
          <span className="file-drop-icon">📄</span>
          <span className="file-drop-text">
            {uploadedFile.name} ({uploadedFile.size} KB, {uploadedFile.chunks} chunks)
          </span>
          <button
            type="button"
            className="file-drop-clear"
            onClick={() => setUploadedFile(null)}
            title="Clear"
          >
            ×
          </button>
        </div>
      ) : (
        <label className="file-drop-content file-drop-label">
          <span className="file-drop-icon">📎</span>
          <div className="file-drop-text-group">
            <span className="file-drop-text">
              {isDragOver ? 'Drop PDF, DOCX, or TXT here...' : 'Drop PDF, DOCX, or TXT file or browse'}
            </span>
            <span className="file-drop-subtext">Supported: .pdf, .docx, .txt (up to 15MB)</span>
          </div>
          <input
            type="file"
            onChange={handleFileSelect}
            accept=".pdf,.docx,.txt"
            className="file-drop-input"
            disabled={disabled}
          />
        </label>
      )}
    </div>
  );
}
