import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Layers, Plus, X, Send, Paperclip, RefreshCw, BookOpen, ChevronDown, ArrowUpRight, Database, FileText
} from './icons';
import EarthBackground from './EarthBackground';
import MarkdownRenderer from './components/MarkdownRenderer';
import FileDropZone from './components/FileDropZone';
import DocumentManager from './components/DocumentManager';
import SourceCitations from './components/SourceCitations';
import Toast from './components/Toast';
import { sendChatMessage, checkHealth, verifyGroqKey, fetchDocuments } from './lib/api';

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
// ============================================================================
// GroqKeyWidget — inline sidebar section
// Props:
//   sessionToken  string|null   — opaque token held in App state
//   onVerified    fn(token)     — called when key passes backend verification
//   onClear       fn()          — called when user removes the key
// ============================================================================
function GroqKeyWidget({ sessionToken, onVerified, onClear }) {
  const [expanded, setExpanded]   = useState(false);
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
      // Store only the opaque token — raw key is discarded here
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
            onClick={() => { onClear(); setExpanded(false); }}
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
  const [activeMode, setActiveMode]     = useState('normal');

  // --- Document RAG state (isolated per visitor/session) ---
  const [clientSessionId]               = useState(() => `client-${crypto.randomUUID().slice(0, 10)}`);
  const [ragEnabled, setRagEnabled]     = useState(false);
  const [documents, setDocuments]       = useState([]);

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
        mode: activeMode,
        sessionToken,       // opaque token — null if using server env key
        clientSessionId,    // isolated client session for RAG document store
        ragEnabled,
      });

      if (data.mode) {
        setActiveMode(data.mode);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          mode: data.mode || 'normal',
          rag_used: Boolean(data.rag_used),
          sources: data.sources || [],
          timestamp: Date.now()
        },
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
  }, [prompt, threadId, isLoading, triggerToast, sessionToken, clientSessionId, ragEnabled, activeMode]);

  // =========================================================================
  // NEW CHAT / RESET SESSION
  // =========================================================================
  const handleNewSession = useCallback(() => {
    const newId = `study-session-${crypto.randomUUID().slice(0, 8)}`;
    setThreadId(newId);
    setMessages([]);
    setPrompt('');
    setIsLoading(false);
    setActiveMode('normal');
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

        {/* Document RAG & Knowledge Base Manager */}
        <DocumentManager
          ragEnabled={ragEnabled}
          onToggleRag={(val) => {
            setRagEnabled(val);
            triggerToast(
              val
                ? `Knowledge Base (RAG) enabled (${documents.length} document${documents.length === 1 ? '' : 's'})`
                : 'Knowledge Base (RAG) disabled',
              'info'
            );
          }}
          sessionToken={sessionToken}
          clientSessionId={clientSessionId}
          onToast={triggerToast}
          documents={documents}
          setDocuments={setDocuments}
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
          <div className="header-brand" onClick={() => setSidebarOpen(false)}>
            <span className="brand-logo-text">TISW</span>
          </div>

          <nav className="header-nav">
            <button type="button" className="nav-link" onClick={() => triggerToast('TISW — Teaching in a Structured Way', 'info')}>
              About
            </button>
            <button
              type="button"
              className={`nav-link nav-link--mode ${activeMode === 'tisw' ? 'nav-link--tisw' : ''}`}
              onClick={() => triggerToast(
                activeMode === 'tisw'
                  ? 'TISW Mode is active. Type "answer normally" to return to Normal Mode.'
                  : 'Normal Mode is default. Type "explain using TISW" to activate structured teaching.',
                'info'
              )}
              title="Active Mode (Normal vs TISW)"
            >
              <span className={`mode-dot ${activeMode === 'tisw' ? 'mode-dot--tisw' : ''}`} />
              <span>{activeMode === 'tisw' ? 'TISW Mode' : 'Normal Mode'}</span>
            </button>
            <button
              type="button"
              className={`nav-link nav-link--rag ${ragEnabled ? 'nav-link--rag-active' : ''}`}
              onClick={() => {
                setRagEnabled((v) => !v);
                triggerToast(
                  !ragEnabled
                    ? `Knowledge Base (RAG) enabled (${documents.length} document${documents.length === 1 ? '' : 's'})`
                    : 'Knowledge Base (RAG) disabled — answering from general tutor knowledge',
                  'info'
                );
              }}
              title="Toggle Knowledge Base (RAG)"
            >
              <span className={`mode-dot ${ragEnabled ? 'mode-dot--rag' : ''}`} />
              <span>{ragEnabled ? `RAG: ON (${documents.length})` : 'RAG: OFF'}</span>
            </button>
            <button type="button" className="nav-link" onClick={() => triggerToast('Active Curriculum: Computer Science, Mathematics, Engineering, Sciences', 'info')}>
              Curriculum
            </button>
            <button type="button" className="nav-link" onClick={() => triggerToast('Normal Mode gives direct answers; TISW Mode gives deep structured lessons', 'info')}>
              FAQ
            </button>
            <button type="button" className="nav-link nav-link--dropdown" onClick={() => setSidebarOpen(true)}>
              <span>{sessionToken ? 'Groq Active' : 'Groq Key'}</span>
              <ChevronDown className="w-3.5 h-3.5" />
            </button>
          </nav>

          <div className="header-actions">
            <button
              type="button"
              className="nav-btn-ghost"
              onClick={() => setSidebarOpen((prev) => !prev)}
              title="Sessions & Groq Key"
              aria-label="Sessions"
            >
              Sessions
            </button>
            <button
              type="button"
              className="nav-btn-pill"
              onClick={handleNewSession}
              title="Start a new chat session"
              aria-label="New Chat"
            >
              <span>New Chat</span>
            </button>
          </div>
        </header>

        <main className="chat-area">
          {!hasMessages && (
            <div className="hero-container">
              <div className="hero-content">
                <h1 className="hero-title">
                  Elevate Your<br />
                  Learning Experience
                </h1>
                <p className="hero-subtitle">
                  Unlock your learning potential in a structured AI environment, powered by TISW & Groq
                </p>
                <button
                  type="button"
                  className="hero-cta-btn"
                  onClick={() => textareaRef.current?.focus()}
                >
                  <span>Start Learning & Chat</span>
                </button>
              </div>

              {/* Floating Feature Glass Cards matching reference screenshot */}
              <div className="hero-floating-card hero-floating-card--left">
                <div className="floating-card-header">
                  <span className="floating-card-tag">Structured Path</span>
                  <div className="floating-card-badge">
                    <ArrowUpRight className="w-3 h-3" />
                  </div>
                </div>
                <div className="floating-card-title">Adaptive Concept Mastery</div>
                <div className="floating-card-meta">Step-by-Step AI</div>
              </div>

              <div className="hero-floating-card hero-floating-card--right">
                <div className="floating-card-header">
                  <span className="floating-card-tag">Concept Mastery</span>
                  <div className="floating-card-badge">
                    <ArrowUpRight className="w-3 h-3" />
                  </div>
                </div>
                <div className="floating-card-value">98%</div>
                <div className="floating-card-bar">
                  <div className="floating-card-bar-fill" />
                </div>
              </div>
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
                        {msg.mode === 'tisw' && (
                          <span className="assistant-mode-badge tisw">TISW Structured</span>
                        )}
                        {msg.rag_used && (
                          <span className="assistant-mode-badge rag">RAG Grounded</span>
                        )}
                      </div>
                      <div className="assistant-content">
                        <MarkdownRenderer content={msg.content} />
                      </div>
                      {msg.sources && msg.sources.length > 0 && (
                        <SourceCitations sources={msg.sources} />
                      )}
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
                  triggerToast(`${result.filename} uploaded & indexed`, 'success');
                  setRagEnabled(true);
                  fetchDocuments(sessionToken, clientSessionId)
                    .then((data) => setDocuments(data.documents || []))
                    .catch(() => {});
                  setShowFileZone(false);
                }}
                onUploadError={(err) => triggerToast(err, 'error')}
                disabled={isLoading}
                sessionToken={sessionToken}
                clientSessionId={clientSessionId}
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
              placeholder={
                !sessionToken
                  ? 'Open the sidebar to configure your Groq API key...'
                  : ragEnabled
                  ? `RAG Active (${documents.length} doc${documents.length === 1 ? '' : 's'}) — ask anything about your documents...`
                  : activeMode === 'tisw'
                  ? 'TISW Mode Active — ask a follow-up or type "answer normally"...'
                  : 'Ask anything… (type "use TISW" for structured teaching)'
              }
              className="input-textarea"
              disabled={isLoading}
            />

            <div className="input-controls">
              <button
                type="button"
                className={`input-icon-btn ${documents.length > 0 ? 'input-icon-btn--has-docs' : ''}`}
                onClick={() => setShowFileZone((prev) => !prev)}
                title={
                  documents.length > 0
                    ? `${documents.length} document(s) uploaded — click to add more`
                    : 'Attach PDF, DOCX, or TXT document'
                }
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
