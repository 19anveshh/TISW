import React, { useRef, useState } from 'react';
import { RefreshCw, X } from '../icons';
import { verifyGroqKey } from '../lib/api';

function EyeIcon({ open }) {
  return open ? (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ) : (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  );
}

/**
 * GroqKeyWidget — the single key-configuration form.
 * Rendered inside the Groq Key sheet (and the sidebar status card links here),
 * so there is exactly one place where a key is entered and verified.
 *
 *   sessionToken  string|null   — opaque token held in App state
 *   onVerified    fn(token)     — called when the key passes backend verification
 *   onClear       fn()          — called when the user removes the key
 *   autoExpand    bool          — open the form immediately when no key is set
 */
export default function GroqKeyWidget({
  sessionToken,
  onVerified,
  onClear,
  autoExpand = false,
}) {
  const [expanded, setExpanded] = useState(autoExpand);
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const connected = Boolean(sessionToken);

  const handleExpand = () => {
    setExpanded(true);
    setError('');
    setTimeout(() => inputRef.current?.focus(), 60);
  };

  const handleCancel = () => {
    setExpanded(false);
    setApiKey('');
    setError('');
  };

  const handleVerify = async () => {
    const trimmed = apiKey.trim();
    if (!trimmed) { setError('Please paste your Groq API key.'); return; }
    setError('');
    setVerifying(true);
    try {
      const data = await verifyGroqKey(trimmed);
      // Only the opaque token leaves this function; the raw key is discarded.
      setApiKey('');
      setExpanded(false);
      onVerified(data.session_token);
    } catch (err) {
      setError(err.message || 'Verification failed.');
    } finally {
      setVerifying(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleVerify();
    if (e.key === 'Escape') handleCancel();
  };

  // ── Connected state ────────────────────────────────────────────────────────
  if (connected && !expanded) {
    return (
      <div className="groq-widget">
        <div className="groq-widget-row">
          <span className="groq-badge">G</span>
          <span className="groq-widget-status connected">Groq connected</span>
          <button
            type="button"
            className="groq-widget-action"
            onClick={() => { onClear(); setShowKey(false); }}
            title="Remove this key and verify another"
          >
            Change
          </button>
        </div>
      </div>
    );
  }

  // ── Collapsed (no key yet) ─────────────────────────────────────────────────
  if (!expanded) {
    return (
      <div className="groq-widget">
        <div className="groq-widget-row">
          <span className="groq-badge">G</span>
          <span className="groq-widget-label">Groq AI</span>
          <button
            type="button"
            className="groq-widget-action"
            onClick={handleExpand}
            title="Add your Groq API key"
          >
            Add key
          </button>
        </div>
      </div>
    );
  }

  // ── Expanded form ──────────────────────────────────────────────────────────
  return (
    <div className="groq-widget groq-widget--expanded">
      <div className="groq-form-header">
        <span className="groq-badge">G</span>
        <span className="groq-form-title">Paste your Groq API key</span>
        <button type="button" className="groq-form-cancel-x" onClick={handleCancel} aria-label="Cancel">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="groq-input-wrap">
        <input
          ref={inputRef}
          type={showKey ? 'text' : 'password'}
          className={`groq-key-input ${error ? 'groq-key-input--error' : ''}`}
          value={apiKey}
          onChange={(e) => { setApiKey(e.target.value); setError(''); }}
          onKeyDown={handleKeyDown}
          placeholder="gsk_..."
          autoComplete="off"
          spellCheck={false}
          disabled={verifying}
          aria-label="Groq API key"
          aria-invalid={error ? 'true' : undefined}
        />
        <button
          type="button"
          className="groq-eye-btn"
          onClick={() => setShowKey((v) => !v)}
          tabIndex={-1}
          aria-label={showKey ? 'Hide key' : 'Show key'}
        >
          <EyeIcon open={showKey} />
        </button>
      </div>

      {error && <p className="groq-error-msg" role="alert">{error}</p>}

      <p className="groq-hint">
        Free API key at{' '}
        <a
          href="https://console.groq.com/keys"
          target="_blank"
          rel="noopener noreferrer"
          className="groq-link"
        >
          console.groq.com/keys
        </a>
        . Verified server-side, never stored in browser.
      </p>

      <div className="groq-form-actions">
        <button type="button" className="groq-cancel-btn" onClick={handleCancel} disabled={verifying}>
          Cancel
        </button>
        <button
          type="button"
          className="groq-verify-btn"
          onClick={handleVerify}
          disabled={verifying || !apiKey.trim()}
        >
          {verifying ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 spin" />
              Verifying…
            </>
          ) : (
            'Verify & Save'
          )}
        </button>
      </div>
    </div>
  );
}
