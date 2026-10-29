import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Plus, X, Send, Paperclip, RefreshCw, BookOpen, ArrowUpRight, Activity, Trash2,
  Sparkles, History, GitHub, Menu
} from './icons';
import EarthBackground from './EarthBackground';
import MarkdownRenderer from './components/MarkdownRenderer';
import FileDropZone from './components/FileDropZone';
import DocumentManager from './components/DocumentManager';
import SourceCitations from './components/SourceCitations';
import Toast from './components/Toast';
import AboutSheet from './components/AboutSheet';
import StartHereSheet from './components/StartHereSheet';
import FaqSheet from './components/FaqSheet';
import GroqKeySheet from './components/GroqKeySheet';
import MobileNavSheet from './components/MobileNavSheet';
import CinematicIntro from './components/CinematicIntro';
import {
  streamChatMessage,
  checkHealth,
  getRuntimeStatus,
  createServerSession,
  fetchDocuments,
  fetchThreads,
  fetchThreadMessages,
  deleteThread,
  isActiveStreamRequest,
} from './lib/api';
import { createTurnMessages, updateAssistantMessage, messagesFromHistory } from './lib/messageState';
import { createStreamPacer } from './lib/streamPacer';

const ACTIVE_THREAD_KEY = 'tisw.active_thread';
const GITHUB_URL = 'https://github.com/19anveshh/TISW';

function toThreadEntry(row) {
  return {
    id: row.thread_id,
    label: row.label || 'New Session',
    messageCount: row.message_count || 0,
    createdAt: Math.round((row.created_at || 0) * 1000),
    updatedAt: Math.round((row.updated_at || 0) * 1000),
  };
}

/** A restored thread is in TISW mode when its latest answer was a walkthrough part. */
function modeFromMessages(rows = []) {
  const lastAssistant = [...rows].reverse().find((msg) => msg.role === 'assistant');
  return lastAssistant?.paging ? 'tisw' : 'normal';
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
  const [activeMode, setActiveMode] = useState('normal');
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
  const [activeSheet, setActiveSheet]   = useState(null); // start | about | faq | groq | menu
  const [backendStatus, setBackendStatus] = useState('checking');
  const [showFileZone, setShowFileZone] = useState(false);

  // --- Thread history ---
  const [threads, setThreads] = useState(() => [{
    id: initialThreadId,
    label: 'New Session',
    messageCount: 0,
    createdAt: Date.now(),
  }]);

  const messagesScrollRef = useRef(null);
  const anchoredUserMessageRef = useRef(null);
  const textareaRef = useRef(null);
  const activeRequestRef = useRef(null);
  const pacerRef = useRef(null);
  const retryTurnRef = useRef(null);
  const messagesByThreadRef = useRef(new Map([[initialThreadId, []]]));
  const loadedThreadsRef = useRef(new Set([initialThreadId]));
  const threadIdRef = useRef(initialThreadId);
  const hasLocalActivityRef = useRef(false);
  // Captured before the threadId effect writes the freshly generated id.
  const [preferredThreadId] = useState(() => {
    try {
      return localStorage.getItem(ACTIVE_THREAD_KEY);
    } catch {
      return null;
    }
  });

  useEffect(() => {
    messagesByThreadRef.current.set(threadId, messages);
  }, [threadId, messages]);

  useEffect(() => {
    threadIdRef.current = threadId;
    try {
      localStorage.setItem(ACTIVE_THREAD_KEY, threadId);
    } catch {
      // Private-mode storage failures only disable restore, never the chat.
    }
  }, [threadId]);

  // Restore the most recent stored conversation after a reload.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchThreads();
        if (cancelled) return;
        const stored = (data.threads || []).map(toThreadEntry);
        if (stored.length === 0) return;
        const active = stored.find((thread) => thread.id === preferredThreadId) || stored[0];
        const transcript = await fetchThreadMessages(active.id);
        if (cancelled || hasLocalActivityRef.current) return;
        const restored = messagesFromHistory(transcript.messages || []);
        messagesByThreadRef.current.set(active.id, restored);
        loadedThreadsRef.current.add(active.id);
        setThreads(stored);
        setThreadId(active.id);
        setMessages(restored);
        setActiveMode(modeFromMessages(restored));
      } catch {
        // History is optional: keep the fresh local session if it is unavailable.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [preferredThreadId]);

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

  // --- Sheets (About, Start Here, FAQ, Groq Key, mobile menu) ---
  const activeSheetRef = useRef(null);
  const sheetTriggerRef = useRef(null);

  const openSheet = useCallback((name) => {
    // Keep the original trigger when one sheet opens another inside the overlay.
    if (!activeSheetRef.current) {
      sheetTriggerRef.current = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    }
    activeSheetRef.current = name;
    setActiveSheet(name);
  }, []);

  const closeSheet = useCallback(() => {
    activeSheetRef.current = null;
    const target = sheetTriggerRef.current;
    sheetTriggerRef.current = null;
    setActiveSheet(null);
    if (target && document.contains(target)) {
      window.setTimeout(() => target.focus?.(), 40);
    }
  }, []);

  const handleOpenDocuments = useCallback(() => {
    activeSheetRef.current = null;
    sheetTriggerRef.current = null;
    setActiveSheet(null);
    setSidebarOpen(true);
  }, []);

  const handleKeyVerified = useCallback((token) => {
    updateSessionToken(token);
    triggerToast("Groq API key verified — you're connected!", 'success');
  }, [updateSessionToken, triggerToast]);

  const handleKeyCleared = useCallback(() => {
    updateSessionToken(null);
    setDocuments([]);
    establishServerSession();
    triggerToast('API key removed', 'info');
  }, [updateSessionToken, establishServerSession, triggerToast]);

  const handleToggleRag = useCallback((value) => {
    setRagEnabled(value);
    if (value) {
      triggerToast(
        `Knowledge Base (RAG) enabled (${documents.length} document${documents.length === 1 ? '' : 's'})`,
        'info'
      );
    } else {
      triggerToast('Knowledge Base (RAG) disabled — answering from general tutor knowledge', 'info');
    }
  }, [documents.length, triggerToast]);

  // --- Scroll anchor ---
  // Position a new question once, then leave the scroll position alone while the
  // answer streams: following every delta is what made the page crawl and jitter.
  useEffect(() => {
    const lastUser = [...messages].reverse().find((msg) => msg.role === 'user');
    if (!lastUser || anchoredUserMessageRef.current === lastUser.id) return;
    anchoredUserMessageRef.current = lastUser.id;
    const container = messagesScrollRef.current;
    const row = document.querySelector(`[data-message-id="${lastUser.id}"]`);
    if (!container || !row) return;
    const containerRect = container.getBoundingClientRect();
    const rowRect = row.getBoundingClientRect();
    const offsetFromTop = rowRect.top - containerRect.top;
    if (offsetFromTop <= container.clientHeight * 0.4) return;
    // One instant move at send time; nothing animates while the answer streams.
    container.scrollTo({ top: container.scrollTop + offsetFromTop - 24 });
  }, [messages]);

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
    pacerRef.current?.cancel();
    pacerRef.current = null;
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

  // Reveal any text still being paced so the reader never leaves a half-typed turn.
  const flushPacing = useCallback(() => {
    const pacer = pacerRef.current;
    pacerRef.current = null;
    pacer?.flush();
  }, []);

  const handleSend = useCallback(async (customText = null, retryTurn = null) => {
    const textToSend = (customText !== null ? customText : prompt).trim();
    if (!textToSend || isLoading) return;
    hasLocalActivityRef.current = true;
    flushPacing();

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

    let streamCompleted = false;
    const updateAssistant = (update) => {
      if (!isActiveStreamRequest(activeRequestRef.current, requestId, requestThreadId)) return;
      setMessages((prev) => updateAssistantMessage(prev, assistantMessageId, update));
    };
    // Message ids are request-scoped, so a late update after a thread switch is a no-op.
    const applyAssistant = (update) => {
      setMessages((prev) => updateAssistantMessage(prev, assistantMessageId, update));
    };

    let donePayload = null;
    const pacer = createStreamPacer({
      onReveal: (text) => applyAssistant({ content: text }),
      onDone: () => {
        if (pacerRef.current === pacer) pacerRef.current = null;
        const data = donePayload;
        donePayload = null;
        if (!data) {
          applyAssistant({ isStreaming: false });
          return;
        }
        applyAssistant({
          content: typeof data.response === 'string' ? data.response : '',
          isStreaming: false,
          retryable: false,
          rag_used: Boolean(data.rag_used),
          sources: data.sources || [],
          paging: data.paging || null,
        });
      },
    });
    pacerRef.current = pacer;

    try {
      await streamChatMessage({
        message: textToSend,
        threadId: requestThreadId,
        turnId,
        sessionToken,       // opaque server-issued token
        ragEnabled,
        tracingEnabled: Boolean(tracingPreference && runtimeStatus.tracingAvailable && !runtimeStatus.offline),
        signal: controller.signal,
        onStart: (data) => setActiveMode(data.mode === 'tisw' ? 'tisw' : 'normal'),
        onDelta: (text) => pacer.push(text),
        onSources: (sources) => updateAssistant({ sources, rag_used: sources.length > 0 }),
        onDone: (data) => {
          streamCompleted = true;
          donePayload = data;
          if (data.mode) setActiveMode(data.mode === 'tisw' ? 'tisw' : 'normal');
          if (data.thread_id) setThreadId(data.thread_id);
          setBackendStatus('connected');
          pacer.finish(typeof data.response === 'string' ? data.response : undefined);
        },
      });
      if (!streamCompleted && isActiveStreamRequest(activeRequestRef.current, requestId, requestThreadId)) {
        throw new Error('The tutor stream ended before completion.');
      }
    } catch (err) {
      if (controller.signal.aborted || err.name === 'AbortError') return;
      pacer.cancel();
      if (pacerRef.current === pacer) pacerRef.current = null;
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
          ? `> ⚠️ **Session Expired**\n> Your Groq session has expired. Open **Groq Key** and verify your API key again.`
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
  }, [prompt, threadId, isLoading, triggerToast, sessionToken, ragEnabled, tracingPreference, runtimeStatus, updateSessionToken, establishServerSession, flushPacing]);

  // =========================================================================
  // NEW CHAT / RESET SESSION
  // =========================================================================
  const handleNewSession = useCallback(() => {
    hasLocalActivityRef.current = true;
    handleStopGeneration(false);
    flushPacing();
    const newId = `study-session-${crypto.randomUUID().slice(0, 8)}`;
    setThreadId(newId);
    setMessages([]);
    setActiveMode('normal');
    messagesByThreadRef.current.set(newId, []);
    loadedThreadsRef.current.add(newId);
    setPrompt('');
    setIsLoading(false);
    setThreads((prev) => [
      { id: newId, label: 'New Session', messageCount: 0, createdAt: Date.now() },
      ...prev,
    ]);
    triggerToast('Started fresh session', 'success');
  }, [handleStopGeneration, flushPacing, triggerToast]);

  const handleSelectThread = useCallback(async (thread) => {
    setSidebarOpen(false);
    if (thread.id === threadIdRef.current) return;
    hasLocalActivityRef.current = true;
    handleStopGeneration(false);
    flushPacing();
    setPrompt('');
    const cached = loadedThreadsRef.current.has(thread.id)
      ? messagesByThreadRef.current.get(thread.id)
      : null;
    setThreadId(thread.id);
    setMessages(cached || []);
    setActiveMode(modeFromMessages(cached || []));
    if (cached) return;
    try {
      const data = await fetchThreadMessages(thread.id);
      if (threadIdRef.current !== thread.id) return;
      const restored = messagesFromHistory(data.messages || []);
      messagesByThreadRef.current.set(thread.id, restored);
      loadedThreadsRef.current.add(thread.id);
      setMessages(restored);
      setActiveMode(modeFromMessages(restored));
    } catch {
      if (threadIdRef.current === thread.id) {
        triggerToast('Could not load that conversation', 'error');
      }
    }
  }, [handleStopGeneration, flushPacing, triggerToast]);

  const handleDeleteThread = useCallback(async (thread) => {
    try {
      await deleteThread(thread.id);
    } catch (err) {
      // A locally created thread that never reached the server is still removable.
      if (err.status !== 404) {
        triggerToast('Could not delete that conversation', 'error');
        return;
      }
    }
    messagesByThreadRef.current.delete(thread.id);
    loadedThreadsRef.current.delete(thread.id);
    setThreads((prev) => prev.filter((item) => item.id !== thread.id));
    if (threadIdRef.current === thread.id) {
      handleNewSession();
    } else {
      triggerToast('Conversation deleted', 'info');
    }
  }, [handleNewSession, triggerToast]);

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
  const openWalkthrough = [...messages].reverse().find((msg) => msg.role === 'assistant')?.paging;

  const composerPlaceholder = (() => {
    if (!sessionToken) return 'Add your Groq API key to start chatting...';
    if (activeMode === 'tisw') {
      return openWalkthrough && !openWalkthrough.complete
        ? `/continue for part ${openWalkthrough.part + 1}, or ask a doubt…`
        : 'Ask your topic — /normal to exit…';
    }
    if (ragEnabled) {
      return `RAG Active (${documents.length} doc${documents.length === 1 ? '' : 's'}) — ask anything about your documents...`;
    }
    return 'Ask anything… or /tisw <topic>';
  })();

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

        {/* Groq key status — opens the single key-configuration sheet */}
        <div className="sidebar-key-card">
          <span className="groq-badge">G</span>
          <div className="sidebar-key-info">
            <span className="sidebar-key-label">Groq API key</span>
            <span className={`sidebar-key-status ${sessionToken ? 'sidebar-key-status--on' : ''}`}>
              {sessionToken ? 'Connected for this session' : 'Not configured yet'}
            </span>
          </div>
          <button type="button" className="sidebar-key-btn" onClick={() => openSheet('groq')}>
            {sessionToken ? 'Manage' : 'Add key'}
          </button>
        </div>

        {/* Document RAG & Knowledge Base Manager */}
        <DocumentManager
          ragEnabled={ragEnabled}
          onToggleRag={handleToggleRag}
          sessionToken={sessionToken}
          onToast={triggerToast}
          documents={documents}
          setDocuments={setDocuments}
        />

        {/* Session History */}
        <div className="sidebar-threads">
          <div className="sidebar-section-label">History</div>
          {threads.map((thread) => {
            const isActive = thread.id === threadId;
            return (
              <div
                key={thread.id}
                className={`thread-item ${isActive ? 'thread-item--active' : ''}`}
              >
                <button
                  type="button"
                  className="thread-open"
                  onClick={() => handleSelectThread(thread)}
                  aria-current={isActive ? 'true' : undefined}
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  <span className="thread-label">{thread.label}</span>
                </button>
                {thread.messageCount > 0 && (
                  <button
                    type="button"
                    className="thread-delete"
                    onClick={() => handleDeleteThread(thread)}
                    title="Delete conversation"
                    aria-label={`Delete ${thread.label}`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            );
          })}
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

          <nav className="header-nav" aria-label="Primary">
            <button
              type="button"
              className="nav-cta"
              onClick={() => openSheet('start')}
              aria-haspopup="dialog"
              aria-label="Start Here"
              title="New here? Start with the three-step setup"
            >
              <Sparkles />
              <span>Start Here</span>
            </button>
            <button
              type="button"
              className="nav-link"
              onClick={() => openSheet('about')}
              aria-haspopup="dialog"
            >
              About
            </button>
            <button
              type="button"
              className="nav-link"
              onClick={() => openSheet('faq')}
              aria-haspopup="dialog"
            >
              FAQ
            </button>
            <button
              type="button"
              className="nav-link"
              onClick={() => openSheet('groq')}
              aria-haspopup="dialog"
              title={sessionToken ? 'Groq key connected — manage your session' : 'Add your Groq API key'}
            >
              <span>Groq Key</span>
              {sessionToken && <span className="mode-dot mode-dot--connected" aria-hidden="true" />}
            </button>
          </nav>

          <div className="header-actions">
            <button
              type="button"
              className={`nav-link nav-link--rag rag-toggle-btn ${ragEnabled ? 'nav-link--rag-active' : ''}`}
              onClick={() => handleToggleRag(!ragEnabled)}
              title="Toggle Knowledge Base (RAG)"
              aria-pressed={ragEnabled}
            >
              <span className={`mode-dot ${ragEnabled ? 'mode-dot--rag' : ''}`} />
              <span>{ragEnabled ? `RAG: ON (${documents.length})` : 'RAG: OFF'}</span>
            </button>
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
              className="header-icon-btn header-icon-btn--history"
              onClick={() => setSidebarOpen(true)}
              title="Documents and chat history"
              aria-label="Documents and chat history"
            >
              <History />
            </button>
            <a
              className="header-icon-btn header-icon-btn--github"
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              title="Star TISW on GitHub"
              aria-label="Star TISW on GitHub"
            >
              <GitHub />
            </a>
            <button
              type="button"
              className="nav-btn-pill"
              onClick={handleNewSession}
              title="Start a new chat session"
              aria-label="New Chat"
            >
              <Plus />
              <span>New Chat</span>
            </button>
            <button
              type="button"
              className="header-icon-btn mobile-nav-toggle"
              onClick={() => openSheet('menu')}
              aria-haspopup="dialog"
              aria-label="Open menu"
              title="Menu"
            >
              <Menu />
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
                  onClick={() => openSheet('start')}
                  aria-haspopup="dialog"
                >
                  <span>Start Here</span>
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
            <div className="messages-container" ref={messagesScrollRef}>
              {messages.map((msg, index) => (
                <div
                  key={msg.id || index}
                  data-message-id={msg.id}
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
                        {msg.paging && (
                          <span className="assistant-mode-badge tisw">TISW</span>
                        )}
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
                       {msg.paging && (
                         <div className={`walkthrough-status ${msg.paging.complete ? 'walkthrough-status--complete' : ''}`}>
                           <span className="walkthrough-part">
                             Part {msg.paging.part} of {msg.paging.parts_total}
                           </span>
                           {msg.paging.title && !msg.paging.complete && (
                             <span className="walkthrough-title">{msg.paging.title}</span>
                           )}
                           {msg.paging.complete && (
                             <span className="walkthrough-title">Walkthrough complete</span>
                           )}
                         </div>
                       )}
                       {msg.paging && !msg.paging.complete && index === messages.length - 1 && !msg.isStreaming && !isLoading && (
                         <div className="walkthrough-controls">
                           <button
                             type="button"
                             className="walkthrough-btn walkthrough-btn--primary"
                             onClick={() => handleSend('/continue')}
                           >
                             Continue
                           </button>
                           <button
                             type="button"
                             className="walkthrough-btn"
                             onClick={() => handleSend('/all')}
                           >
                             Give me all
                           </button>
                         </div>
                       )}
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

          {activeMode === 'tisw' && (
            <div className="input-mode-row">
              <span className="input-mode-pill">
                <span className="input-mode-dot" />
                TISW walkthrough · 4 parts
              </span>
              <button
                type="button"
                className="input-mode-exit"
                onClick={() => handleSend('/normal')}
                disabled={isLoading}
                title="Leave TISW mode"
              >
                Exit
              </button>
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
              placeholder={composerPlaceholder}
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
        {activeSheet === 'start' && (
          <StartHereSheet
            sessionToken={sessionToken}
            documentCount={documents.length}
            onOpenGroq={() => setActiveSheet('groq')}
            onOpenDocuments={handleOpenDocuments}
            onOpenFaq={() => setActiveSheet('faq')}
            onClose={closeSheet}
          />
        )}
        {activeSheet === 'about' && <AboutSheet onClose={closeSheet} />}
        {activeSheet === 'faq' && <FaqSheet onClose={closeSheet} />}
        {activeSheet === 'groq' && (
          <GroqKeySheet
            sessionToken={sessionToken}
            onVerified={handleKeyVerified}
            onClear={handleKeyCleared}
            onOpenDocuments={handleOpenDocuments}
            onClose={closeSheet}
          />
        )}
        {activeSheet === 'menu' && (
          <MobileNavSheet
            onClose={closeSheet}
            onOpen={(name) => setActiveSheet(name)}
            onOpenHistory={handleOpenDocuments}
            onNewChat={() => {
              closeSheet();
              handleNewSession();
            }}
            onToggleRag={() => handleToggleRag(!ragEnabled)}
            ragEnabled={ragEnabled}
            documentCount={documents.length}
            tracingPreference={tracingPreference}
            tracingAvailable={runtimeStatus.tracingAvailable}
            offline={runtimeStatus.offline}
            onToggleTracing={handleTracingToggle}
            sessionToken={sessionToken}
          />
        )}
      </div>
    </>
  );
}
