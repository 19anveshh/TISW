import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Layers, Plus, X, Send, Paperclip, RefreshCw, BookOpen
} from './icons';
import EarthBackground from './EarthBackground';
import MarkdownRenderer from './components/MarkdownRenderer';
import FileDropZone from './components/FileDropZone';
import Toast from './components/Toast';
import { sendChatMessage, checkHealth, verifyGroqKey } from './lib/api';

// ============================================================================
// Eye icon (show/hide password)
// ============================================================================
function EyeIcon({ open }) {
  return open ? (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  ) : (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
      <line x1="1" y1="1" x2="23" y2="23"/>
    </svg>
  );
}

// ============================================================================
// GroqKeyWidget — inline sidebar or hero section
// Props:
//   sessionToken  string|null   — opaque token held in App state
//   onVerified    fn(token)     — called when key passes backend verification
//   onClear       fn()          — called when user removes the key
//   variant       'sidebar'|'hero'
// ============================================================================
function GroqKeyWidget({ sessionToken, onVerified, onClear, variant = 'sidebar' }) {
  const isHero = variant === 'hero';
  const [expanded, setExpanded]   = useState(isHero);
  const [apiKey, setApiKey]       = useState('');
  const [showKey, setShowKey]     = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError]         = useState('');
  const inputRef                  = useRef(null);

  const connected = Boolean(sessionToken);

  const handleExpand = () => {
    setExpanded(true);
    setError('');
    setTimeout(() => inputRef.current?.focus(), 60);
  };

  const handleCancel = () => {
    if (!isHero) {
      setExpanded(false);
    }
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
      // Store only the opaque token — raw key is discarded here
      setApiKey('');
      if (!isHero) setExpanded(false);
      onVerified(data.session_token);
    } catch (err) {
      setError(err.message || 'Verification failed.');
    } finally {
      setVerifying(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleVerify();
    if (e.key === 'Escape' && !isHero) handleCancel();
  };

  // ── Connected state ────────────────────────────────────────────────────────
  if (connected && !expanded) {
    return (
      <div className={`groq-widget ${isHero ? 'groq-widget--hero' : ''}`}>
        <div className="groq-widget-row">
          <span className="groq-badge">G</span>
          <span className="groq-widget-status connected">Groq connected</span>
          <button
            type="button"
            className="groq-widget-action"
            onClick={() => { onClear(); if (!isHero) setExpanded(false); }}
            title="Remove key"
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
    <div className={`groq-widget groq-widget--expanded ${isHero ? 'groq-widget--hero' : ''}`}>
      <div className="groq-form-header">
        <span className="groq-badge">G</span>
        <span className="groq-form-title">
          {isHero ? 'Enter Groq API Key to Start' : 'Paste your Groq API key'}
        </span>
        {!isHero && (
          <button type="button" className="groq-form-cancel-x" onClick={handleCancel} aria-label="Cancel">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <div className="groq-input-wrap">
        <input
          ref={inputRef}
          type={showKey ? 'text' : 'password'}
          className={`groq-key-input ${error ? 'groq-key-input--error' : ''}`}
          value={apiKey}
          onChange={(e) => { setApiKey(e.target.value); setError(''); }}
          onKeyDown={handleKeyDown}
          placeholder="Paste your Groq key (gsk_...)"
          autoComplete="off"
          spellCheck={false}
          disabled={verifying}
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

      {error && <p className="groq-error-msg">{error}</p>}

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
        {(!isHero || apiKey) && (
          <button type="button" className="groq-cancel-btn" onClick={handleCancel} disabled={verifying}>
            {isHero ? 'Clear' : 'Cancel'}
          </button>
        )}
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
            'Verify & Connect'
          )}
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// TISW — Teaching in a Structured Way
// ============================================================================
export default function App() {
  // --- Core conversation state ---
  const [messages, setMessages]   = useState([]);
  const [prompt, setPrompt]       = useState('');
  const [threadId, setThreadId]   = useState(() => `study-session-${crypto.randomUUID().slice(0, 8)}`);
  const [isLoading, setIsLoading] = useState(false);

  // --- Session token (opaque, in memory only — never persisted) ---
  // null  → no visitor key configured (falls back to server env key if present)
  // str   → visitor's opaque session token from /api/verify-key
  const [sessionToken, setSessionToken] = useState(null);

  // --- UI state ---
  const [sidebarOpen, setSidebarOpen]   = useState(false);
  const [toast, setToast]               = useState(null);
  const [toastType, setToastType]       = useState('info');
  const [backendStatus, setBackendStatus] = useState('connected');
  const [showFileZone, setShowFileZone] = useState(false);

  // --- Thread history ---
  const [threads, setThreads] = useState(() => [{
    id: `study-session-${crypto.randomUUID().slice(0, 8)}`,
    label: 'New Session',
    messageCount: 0,
    createdAt: Date.now(),
  }]);

  const chatEndRef  = useRef(null);
  const textareaRef = useRef(null);

  // --- Toast helper ---
  const triggerToast = useCallback((msg, type = 'info') => {
    setToast(msg);
    setToastType(type);
    setTimeout(() => setToast((prev) => (prev === msg ? null : prev)), 3500);
  }, []);

  // --- Auto-scroll ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // --- Backend health check ---
  useEffect(() => {
    checkHealth().then((data) => {
      setBackendStatus(data.status === 'healthy' ? 'connected' : 'degraded');
    });
  }, []);

  // --- Auto-resize textarea ---
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [prompt]);

  // =========================================================================
  // SEND MESSAGE
  // =========================================================================
  const handleSend = useCallback(async (customText = null) => {
    const textToSend = (customText !== null ? customText : prompt).trim();
    if (!textToSend || isLoading) return;

    if (customText === null) setPrompt('');

    const userMsg = { role: 'user', content: textToSend, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    setThreads((prev) =>
      prev.map((t) =>
        t.id === threadId
          ? { ...t, messageCount: t.messageCount + 2, label: textToSend.slice(0, 36) }
          : t
      )
    );

    try {
      const data = await sendChatMessage({
        message: textToSend,
        threadId,
        mode: 'default',
        sessionToken,   // opaque token — null if using server env key
      });

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response, timestamp: Date.now() },
      ]);

      if (data.thread_id) setThreadId(data.thread_id);
      setBackendStatus('connected');
    } catch (err) {
      const errorMsg = err.message || 'Failed to reach tutor backend.';

      // Detect expired session
      const isSessionExpired = errorMsg.toLowerCase().includes('session expired') ||
                               errorMsg.includes('401');

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: isSessionExpired
            ? `> ⚠️ **Session Expired**\n> Your Groq session has expired. Please re-enter your API key in the sidebar.`
            : `> ⚠️ **Error**\n> ${errorMsg}`,
          timestamp: Date.now(),
          isError: true,
        },
      ]);

      if (isSessionExpired) {
        setSessionToken(null);   // clear expired token so widget reverts to "Add key"
      }

      setBackendStatus('offline');
      triggerToast(errorMsg, 'error');
    } finally {
      setIsLoading(false);
    }
  }, [prompt, threadId, isLoading, triggerToast, sessionToken]);

  // =========================================================================
  // NEW CHAT / RESET SESSION
  // =========================================================================
  const handleNewSession = useCallback(() => {
    const newId = `study-session-${crypto.randomUUID().slice(0, 8)}`;
    setThreadId(newId);
    setMessages([]);
    setPrompt('');
    setIsLoading(false);
    setThreads((prev) => [
      { id: newId, label: 'New Session', messageCount: 0, createdAt: Date.now() },
      ...prev,
    ]);
    triggerToast('Started fresh session', 'success');
  }, [triggerToast]);

  const hasMessages = messages.length > 0;

  return (
    <div className="app-root">
      {/* 3D Earth Background */}
      <EarthBackground overlayOpacity={0.28} glowOpacity={0.4} />

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-wordmark">TISW</div>
            <div className="brand-tagline">Teaching in a Structured Way</div>
          </div>
          <button
            type="button"
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
            title="Close sidebar"
            aria-label="Close sidebar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* New Chat */}
        <button type="button" className="new-session-btn" onClick={handleNewSession}>
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>

        {/* Groq API Key Widget */}
        <GroqKeyWidget
          sessionToken={sessionToken}
          onVerified={(token) => {
            setSessionToken(token);
            triggerToast('Groq API key verified — you\'re connected!', 'success');
          }}
          onClear={() => {
            setSessionToken(null);
            triggerToast('API key removed', 'info');
          }}
        />

        {/* Session History */}
        <div className="sidebar-threads">
          <div className="sidebar-section-label">History</div>
          {threads.map((thread) => (
            <button
              key={thread.id}
              type="button"
              className={`thread-item ${thread.id === threadId ? 'thread-item--active' : ''}`}
              onClick={() => {
                setThreadId(thread.id);
                setSidebarOpen(false);
              }}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span className="thread-label">{thread.label}</span>
            </button>
          ))}
        </div>
      </aside>

      {/* Sidebar backdrop */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main workspace */}
      <div className="main-wrapper">
        <header className="top-header">
          <div className="header-left">
            <button
              type="button"
              className="header-icon-btn"
              onClick={() => setSidebarOpen((prev) => !prev)}
              title="Chat History"
              aria-label="Chat History"
            >
              <Layers className="w-4 h-4" />
            </button>
          </div>
          <div className="header-right">
            <button
              type="button"
              className="header-icon-btn"
              onClick={handleNewSession}
              title="New Chat"
              aria-label="New Chat"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </header>

        <main className="chat-area">
          {!hasMessages && (
            <div className="hero-minimal">
              <h1 className="hero-wordmark">TISW</h1>
              <p className="hero-tagline">Teaching in a Structured Way</p>
              {!sessionToken && (
                <div className="hero-groq-container">
                  <GroqKeyWidget
                    sessionToken={sessionToken}
                    onVerified={(token) => {
                      setSessionToken(token);
                      triggerToast("Groq API key verified — you're connected!", 'success');
                    }}
                    onClear={() => {
                      setSessionToken(null);
                      triggerToast('API key removed', 'info');
                    }}
                    variant="hero"
                  />
                </div>
              )}
            </div>
          )}

          {hasMessages && (
            <div className="messages-container">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`message-row ${msg.role === 'user' ? 'message-row--user' : 'message-row--assistant'} animate-fadeIn`}
                >
                  {msg.role === 'user' ? (
                    <div className="message-bubble message-bubble--user">
                      {msg.content}
                    </div>
                  ) : (
                    <div className={`message-bubble message-bubble--assistant ${msg.isError ? 'message-bubble--error' : ''}`}>
                      <div className="assistant-header">
                        <span className="assistant-label">TISW</span>
                      </div>
                      <div className="assistant-content">
                        <MarkdownRenderer content={msg.content} />
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="message-row message-row--assistant animate-fadeIn">
                  <div className="message-bubble message-bubble--assistant">
                    <div className="assistant-header">
                      <span className="assistant-label">TISW</span>
                    </div>
                    <div className="thinking-text">
                      <RefreshCw className="w-3.5 h-3.5 spin" />
                      <span>Structuring response…</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>
          )}
        </main>

        {/* Floating Input */}
        <div className="input-area">
          {showFileZone && (
            <div className="file-zone-container">
              <FileDropZone
                onUploadSuccess={(result) => {
                  triggerToast(`${result.filename} uploaded`, 'success');
                  setShowFileZone(false);
                }}
                onUploadError={(err) => triggerToast(err, 'error')}
                disabled={isLoading}
              />
            </div>
          )}

          <div className="input-card">
            <textarea
              ref={textareaRef}
              rows={1}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={sessionToken ? 'Ask anything…' : 'Add your Groq API key in the sidebar to start…'}
              className="input-textarea"
              disabled={isLoading}
            />

            <div className="input-controls">
              <button
                type="button"
                className="input-icon-btn"
                onClick={() => setShowFileZone((prev) => !prev)}
                title="Attach document"
                aria-label="Attach document"
              >
                <Paperclip className="w-4 h-4" />
              </button>

              <button
                type="button"
                className="send-btn"
                onClick={() => handleSend()}
                disabled={isLoading || !prompt.trim()}
                title="Send message"
                aria-label="Send message"
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      <Toast message={toast} type={toastType} />
    </div>
  );
}
