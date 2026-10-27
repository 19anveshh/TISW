import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Plus, X, Send, Paperclip, RefreshCw, BookOpen, ChevronDown, ArrowUpRight, Activity
} from './icons';
import EarthBackground from './EarthBackground';
import MarkdownRenderer from './components/MarkdownRenderer';
import FileDropZone from './components/FileDropZone';
import DocumentManager from './components/DocumentManager';
import SourceCitations from './components/SourceCitations';
import Toast from './components/Toast';
import CinematicIntro from './components/CinematicIntro';
import { streamChatMessage, checkHealth, getRuntimeStatus, verifyGroqKey, createServerSession, fetchDocuments, isActiveStreamRequest } from './lib/api';
import { createTurnMessages, updateAssistantMessage } from './lib/messageState';

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
  const [introRevealed, setIntroRevealed] = useState(false);
  const [introComplete, setIntroComplete] = useState(false);

  const handleIntroReveal = useCallback(() => setIntroRevealed(true), []);
  const handleIntroComplete = useCallback(() => setIntroComplete(true), []);

  // --- Core conversation state ---
  const [messages, setMessages]   = useState([]);
  const [prompt, setPrompt]       = useState('');
  const [initialThreadId] = useState(() => `study-session-${crypto.randomUUID().slice(0, 8)}`);
  const [threadId, setThreadId]   = useState(initialThreadId);
  const [isLoading, setIsLoading] = useState(false);
  const [tracingPreference, setTracingPreference] = useState(() => localStorage.getItem('tisw.tracing') === 'true');
  const [runtimeStatus, setRuntimeStatus] = useState({ offline: true, tracingAvailable: false, status: 'checking' });

  // --- Session token (opaque, in memory only — never persisted) ---
  // null  → no server-issued session yet
  // str   → opaque token issued by /api/session or /api/verify-key
  const [sessionToken, setSessionToken] = useState(null);
  const sessionTokenRef = useRef(null);

  // --- Document RAG state (isolated per visitor/session) ---
  const [ragEnabled, setRagEnabled]     = useState(false);
  const [documents, setDocuments]       = useState([]);

  // --- UI state ---
  const [sidebarOpen, setSidebarOpen]   = useState(false);
  const [toast, setToast]               = useState(null);
  const [toastType, setToastType]       = useState('info');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [showFileZone, setShowFileZone] = useState(false);

  // --- Thread history ---
  const [threads, setThreads] = useState(() => [{
    id: initialThreadId,
    label: 'New Session',
    messageCount: 0,
    createdAt: Date.now(),
  }]);

  const chatEndRef  = useRef(null);
  const textareaRef = useRef(null);
  const activeRequestRef = useRef(null);
  const retryTurnRef = useRef(null);
  const messagesByThreadRef = useRef(new Map([[initialThreadId, []]]));

  useEffect(() => {
    messagesByThreadRef.current.set(threadId, messages);
  }, [threadId, messages]);

  // --- Toast helper ---
  const triggerToast = useCallback((msg, type = 'info') => {
    setToast(msg);
    setToastType(type);
    setTimeout(() => setToast((prev) => (prev === msg ? null : prev)), 3500);
  }, []);

  const updateSessionToken = useCallback((token) => {
    sessionTokenRef.current = token || null;
    setSessionToken(token || null);
  }, []);

  const establishServerSession = useCallback(async () => {
    try {
      const data = await createServerSession();
      if (!sessionTokenRef.current) updateSessionToken(data.session_token);
    } catch {
      // A missing server key is valid when the visitor uses the sidebar key flow.
    }
  }, [updateSessionToken]);

  // --- Auto-scroll ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // --- Backend health check ---
  useEffect(() => {
    checkHealth().then((data) => {
      setBackendStatus(data.status === 'healthy' ? 'connected' : 'degraded');
    });
    getRuntimeStatus()
      .then((data) => setRuntimeStatus({
        offline: Boolean(data.offline),
        tracingAvailable: Boolean(data.tracing_available),
        status: 'ready',
      }))
      .catch(() => setRuntimeStatus({ offline: false, tracingAvailable: false, status: 'unavailable' }));
    establishServerSession();
  }, [establishServerSession]);

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
  const handleStopGeneration = useCallback((restorePrompt = true) => {
    const request = activeRequestRef.current;
    if (!request) return;
    activeRequestRef.current = null;
    request.controller.abort();
    retryTurnRef.current = {
      threadId: request.threadId,
      turnId: request.turnId,
      message: request.message,
      clientRequestId: request.requestId,
    };
    setIsLoading(false);
    setMessages((prev) => prev.filter((msg) => msg.clientRequestId !== request.requestId));
    const stored = messagesByThreadRef.current.get(request.threadId);
    if (stored) {
      messagesByThreadRef.current.set(
        request.threadId,
        stored.filter((msg) => msg.clientRequestId !== request.requestId),
      );
    }
    setThreads((prev) => prev.map((thread) => (
      thread.id === request.threadId
        ? { ...thread, messageCount: Math.max(0, thread.messageCount - 2) }
        : thread
    )));
    if (restorePrompt) setPrompt(request.message);
    triggerToast('Generation stopped', 'info');
  }, [triggerToast]);

  const handleSend = useCallback(async (customText = null, retryTurn = null) => {
    const textToSend = (customText !== null ? customText : prompt).trim();
    if (!textToSend || isLoading) return;

    if (customText === null) setPrompt('');

    const requestThreadId = threadId;
    const pendingRetry = retryTurn || retryTurnRef.current;
    const reuseRetry = Boolean(
      pendingRetry &&
      pendingRetry.threadId === requestThreadId &&
      pendingRetry.message === textToSend
    );
    const requestId = crypto.randomUUID();
    const turnId = reuseRetry ? pendingRetry.turnId : requestId;
    if (reuseRetry && pendingRetry.clientRequestId) {
      setMessages((prev) => prev.filter((msg) => msg.clientRequestId !== pendingRetry.clientRequestId));
    }
    retryTurnRef.current = null;
    const controller = new AbortController();
    const { userMessageId, assistantMessageId, messages: turnMessages } = createTurnMessages({
      requestId,
      text: textToSend,
    });
    activeRequestRef.current = {
      requestId,
      turnId,
      threadId: requestThreadId,
      message: textToSend,
      userMessageId,
      assistantMessageId,
      controller,
    };
    if (import.meta.env?.VITE_TISW_STREAM_DIAGNOSTICS === 'true') {
      console.debug('[TISW stream] submit', {
        message: textToSend,
        threadId: requestThreadId,
        turnId,
        requestId,
        tracingEnabled: Boolean(tracingPreference && runtimeStatus.tracingAvailable && !runtimeStatus.offline),
        hasSessionToken: Boolean(sessionToken),
      });
    }
    setMessages((prev) => [...prev, ...turnMessages]);
    setIsLoading(true);

    if (!reuseRetry) {
      setThreads((prev) =>
        prev.map((t) =>
          t.id === threadId
            ? { ...t, messageCount: t.messageCount + 2, label: textToSend.slice(0, 36) }
            : t
        )
      );
    }

    let completed = false;
    const updateAssistant = (update) => {
      if (!isActiveStreamRequest(activeRequestRef.current, requestId, requestThreadId)) return;
      setMessages((prev) => updateAssistantMessage(prev, assistantMessageId, update));
    };

    try {
      await streamChatMessage({
        message: textToSend,
        threadId: requestThreadId,
        turnId,
        sessionToken,       // opaque server-issued token
        ragEnabled,
        tracingEnabled: Boolean(tracingPreference && runtimeStatus.tracingAvailable && !runtimeStatus.offline),
        signal: controller.signal,
        onDelta: (text) => updateAssistant((msg) => ({ content: `${msg.content}${text}` })),
        onSources: (sources) => updateAssistant({ sources, rag_used: sources.length > 0 }),
        onDone: (data) => {
          completed = true;
          updateAssistant({
            content: typeof data.response === 'string' ? data.response : '',
            isStreaming: false,
            retryable: false,
            rag_used: Boolean(data.rag_used),
            sources: data.sources || [],
          });
          if (data.thread_id) setThreadId(data.thread_id);
          setBackendStatus('connected');
        },
      });
      if (!completed && isActiveStreamRequest(activeRequestRef.current, requestId, requestThreadId)) {
        throw new Error('The tutor stream ended before completion.');
      }
    } catch (err) {
      if (controller.signal.aborted || err.name === 'AbortError') return;
      const errorMsg = err.message || 'Failed to reach tutor backend.';
      retryTurnRef.current = {
        threadId: requestThreadId,
        turnId,
        message: textToSend,
        clientRequestId: requestId,
      };

      // Detect expired session
      const isSessionExpired = errorMsg.toLowerCase().includes('session expired') ||
                               errorMsg.includes('401');

      updateAssistant({
        content: isSessionExpired
          ? `> ⚠️ **Session Expired**\n> Your Groq session has expired. Please re-enter your API key in the sidebar.`
          : `> ⚠️ **Error**\n> ${errorMsg}`,
        isStreaming: false,
        isError: true,
        retryable: true,
        retryMessage: textToSend,
        retryThreadId: requestThreadId,
        retryTurnId: turnId,
      });

      if (isSessionExpired) {
        updateSessionToken(null);
        setDocuments([]);
        establishServerSession();
      }

      setBackendStatus('offline');
      triggerToast(errorMsg, 'error');
    } finally {
      if (activeRequestRef.current?.requestId === requestId) {
        activeRequestRef.current = null;
        setIsLoading(false);
      }
    }
  }, [prompt, threadId, isLoading, triggerToast, sessionToken, ragEnabled, tracingPreference, runtimeStatus, updateSessionToken, establishServerSession]);

  // =========================================================================
  // NEW CHAT / RESET SESSION
  // =========================================================================
  const handleNewSession = useCallback(() => {
    handleStopGeneration(false);
    const newId = `study-session-${crypto.randomUUID().slice(0, 8)}`;
    setThreadId(newId);
    setMessages([]);
    messagesByThreadRef.current.set(newId, []);
    setPrompt('');
    setIsLoading(false);
    setThreads((prev) => [
      { id: newId, label: 'New Session', messageCount: 0, createdAt: Date.now() },
      ...prev,
    ]);
    triggerToast('Started fresh session', 'success');
  }, [handleStopGeneration, triggerToast]);

  const handleTracingToggle = useCallback(() => {
    if (runtimeStatus.offline || !runtimeStatus.tracingAvailable) {
      triggerToast(
        runtimeStatus.offline
          ? 'LangSmith tracing is unavailable while offline.'
          : 'LangSmith tracing is unavailable because the backend has no configured key.',
        'info'
      );
      return;
    }
    setTracingPreference((current) => {
      const next = !current;
      localStorage.setItem('tisw.tracing', String(next));
      triggerToast(`LangSmith tracing ${next ? 'enabled' : 'disabled'} for new requests.`, 'success');
      return next;
    });
  }, [runtimeStatus, triggerToast]);

  const hasMessages = messages.length > 0;

  return (
    <>
      <CinematicIntro onReveal={handleIntroReveal} onComplete={handleIntroComplete} />
      <div
        className={`app-root ${introRevealed ? 'app-root--intro-revealed' : 'app-root--intro-hidden'} ${!introComplete ? 'app-root--intro-transition' : ''}`}
        data-backend-status={backendStatus}
      >
      {/* 3D Earth Background */}
      <EarthBackground overlayOpacity={0.28} glowOpacity={0.4} />

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-wordmark">TISW</div>
             <div className="brand-tagline">AI Study Assistant</div>
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
            updateSessionToken(token);
            triggerToast('Groq API key verified — you\'re connected!', 'success');
          }}
          onClear={() => {
            updateSessionToken(null);
            setDocuments([]);
            establishServerSession();
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
                handleStopGeneration(false);
                const nextMessages = messagesByThreadRef.current.get(thread.id) || [];
                setThreadId(thread.id);
                setMessages(nextMessages);
                setPrompt('');
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
            <button type="button" className="nav-link" onClick={() => triggerToast('A clear, conversational AI study assistant powered by Groq.', 'info')}>
              About
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
            <button type="button" className="nav-link" onClick={() => triggerToast('Ask naturally, or upload documents for grounded answers.', 'info')}>
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
              className={`tracing-toggle ${tracingPreference && runtimeStatus.tracingAvailable && !runtimeStatus.offline ? 'tracing-toggle--on' : ''} ${runtimeStatus.offline || !runtimeStatus.tracingAvailable ? 'tracing-toggle--unavailable' : ''}`}
              onClick={handleTracingToggle}
              disabled={runtimeStatus.offline || !runtimeStatus.tracingAvailable}
              title={runtimeStatus.offline ? 'LangSmith tracing is unavailable while offline.' : runtimeStatus.tracingAvailable ? (tracingPreference ? 'LangSmith tracing is enabled for new requests.' : 'LangSmith tracing is disabled.') : 'LangSmith tracing is unavailable because no server key is configured.'}
              aria-label={runtimeStatus.offline || !runtimeStatus.tracingAvailable ? 'LangSmith tracing unavailable' : `LangSmith tracing ${tracingPreference ? 'on' : 'off'}`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>{runtimeStatus.offline || !runtimeStatus.tracingAvailable ? 'Trace unavailable' : `Trace: ${tracingPreference ? 'ON' : 'OFF'}`}</span>
            </button>
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
                  A clear, conversational AI study assistant powered by Groq
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
                  <span className="floating-card-tag">Learning Focus</span>
                  <div className="floating-card-badge">
                    <ArrowUpRight className="w-3 h-3" />
                  </div>
                </div>
                <div className="floating-card-title">Adaptive Concept Mastery</div>
                <div className="floating-card-meta">Personalized AI</div>
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
                  key={msg.id || index}
                  className={`message-row ${msg.role === 'user' ? 'message-row--user' : 'message-row--assistant'} animate-fadeIn`}
                >
                  {msg.role === 'user' ? (
                    <div className="message-bubble message-bubble--user">
                      {msg.content}
                    </div>
                  ) : (
                    <div className={`message-bubble message-bubble--assistant ${msg.isError ? 'message-bubble--error' : ''}`}>
                      <div className="assistant-header">
                        <span className="assistant-label">Tutor</span>
                        {msg.rag_used && (
                          <span className="assistant-mode-badge rag">RAG Grounded</span>
                        )}
                      </div>
                       <div className="assistant-content">
                         {msg.isStreaming && !msg.content ? (
                           <div className="thinking-text">
                             <RefreshCw className="w-3.5 h-3.5 spin" />
                             <span>Thinking…</span>
                           </div>
                         ) : (
                           <>
                             <MarkdownRenderer content={msg.content || ''} />
                             {msg.isStreaming && <span className="streaming-cursor" aria-label="Response streaming" />}
                         </>
                       )}
                       </div>
                       {msg.retryable && !isLoading && msg.retryMessage && (
                         <button
                           type="button"
                           className="message-retry-btn"
                           onClick={() => handleSend(msg.retryMessage, {
                             threadId: msg.retryThreadId || threadId,
                             turnId: msg.retryTurnId,
                             message: msg.retryMessage,
                             clientRequestId: msg.clientRequestId,
                           })}
                         >
                           Retry response
                         </button>
                       )}
                      {msg.sources && msg.sources.length > 0 && (
                        <SourceCitations sources={msg.sources} />
                      )}
                    </div>
                  )}
                </div>
              ))}

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
                  fetchDocuments(sessionToken)
                    .then((data) => setDocuments(data.documents || []))
                    .catch(() => {});
                  setShowFileZone(false);
                }}
                onUploadError={(err) => triggerToast(err, 'error')}
                disabled={isLoading}
                sessionToken={sessionToken}
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
                   : 'Ask anything…'
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
                className={`send-btn ${isLoading ? 'send-btn--stop' : ''}`}
                onClick={() => (isLoading ? handleStopGeneration() : handleSend())}
                disabled={!isLoading && !prompt.trim()}
                title={isLoading ? 'Stop generation' : 'Send message'}
                aria-label={isLoading ? 'Stop generation' : 'Send message'}
              >
                {isLoading ? (
                  <X className="w-4 h-4" />
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
    </>
  );
}
