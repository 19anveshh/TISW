import React, { useState } from 'react';
import { ChevronDown, FileText, Database } from '../icons';

/**
 * SourceCitations — Renders verified source citations for document-grounded RAG responses.
 *
 * @param {Object} props
 * @param {Array<{ filename: string, page_number?: number|null, chunk_id: string, snippet: string }>} props.sources
 */
export default function SourceCitations({ sources }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="citations-container">
      <button
        type="button"
        className="citations-toggle-btn"
        onClick={() => setIsOpen((v) => !v)}
        aria-expanded={isOpen}
      >
        <div className="citations-toggle-left">
          <Database className="w-3.5 h-3.5 text-cyan" />
          <span className="citations-heading">Sources & Grounding</span>
          <span className="citations-count-pill">{sources.length}</span>
        </div>
        <div className={`citations-chevron ${isOpen ? 'open' : ''}`}>
          <ChevronDown className="w-3.5 h-3.5" />
        </div>
      </button>

      {isOpen && (
        <div className="citations-list animate-fadeIn">
          {sources.map((src, idx) => {
            const ext = (src.filename.split('.').pop() || '').toUpperCase();
            return (
              <div key={src.chunk_id || idx} className="citation-card">
                <div className="citation-header">
                  <div className="citation-file-info">
                    <FileText className="w-3.5 h-3.5 citation-icon" />
                    <span className="citation-filename" title={src.filename}>
                      {src.filename}
                    </span>
                    <span className={`citation-badge citation-badge--${ext.toLowerCase()}`}>
                      {ext}
                    </span>
                  </div>
                  {src.page_number != null ? (
                    <span className="citation-page">Page {src.page_number}</span>
                  ) : (
                    <span className="citation-page">Excerpt</span>
                  )}
                </div>
                <div className="citation-snippet">
                  "{src.snippet}"
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
