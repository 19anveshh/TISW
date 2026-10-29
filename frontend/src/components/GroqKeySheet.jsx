import React from 'react';
import GlassSheet from './GlassSheet';
import GroqKeyWidget from './GroqKeyWidget';
import { ExternalLink } from '../icons';

export default function GroqKeySheet({
  sessionToken,
  onVerified,
  onClear,
  onOpenDocuments,
  onClose,
}) {
  return (
    <GlassSheet
      title="Groq API key"
      onClose={onClose}
      closeLabel="Close Groq key and session panel"
    >
      <p className="sheet-lead">
        TISW uses Groq for model access. Add your own free key to start a session — the key is
        verified with a live request and stays on the server, not in this browser.
      </p>

      <GroqKeyWidget
        sessionToken={sessionToken}
        onVerified={onVerified}
        onClear={onClear}
        autoExpand={!sessionToken}
      />

      <div className="sheet-status-row">
        <span className={`sheet-status ${sessionToken ? 'sheet-status--on' : ''}`}>
          {sessionToken ? 'Session active' : 'No key configured yet'}
        </span>
      </div>

      <div className="sheet-note">
        <strong>How your key is handled.</strong> It is checked with a real Groq call, kept in
        server memory only, and never returned to the page or written to browser storage. The
        browser receives an opaque session token instead. Sessions expire after about two hours
        without use, and logs redact key-shaped strings.
      </div>

      <div className="sheet-links">
        <a
          className="sheet-btn"
          href="https://console.groq.com/keys"
          target="_blank"
          rel="noopener noreferrer"
        >
          Open Groq Console
          <ExternalLink />
        </a>
        <button type="button" className="sheet-btn" onClick={onOpenDocuments}>
          Documents &amp; history
        </button>
      </div>
    </GlassSheet>
  );
}
