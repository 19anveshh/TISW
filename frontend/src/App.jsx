import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Layers, Plus, X, Send, Paperclip, RefreshCw, BookOpen
} from './icons';
import EarthBackground from './EarthBackground';
import MarkdownRenderer from './components/MarkdownRenderer';
import FileDropZone from './components/FileDropZone';
import Toast from './components/Toast';
import { sendChatMessage, checkHealth } from './lib/api';

// ============================================================================
// Groq Config Modal — purely local display; key must be set in server .env
// ============================================================================
function GroqConfigModal({ onClose }) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="groq-modal-backdrop" onClick={onClose}>
      <div className="groq-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Groq API Key Setup">
        <div className="groq-modal-header">
          <span className="groq-modal-title">
            <span className="groq-badge">G</span>
            Groq API Key
          </span>
          <button className="groq-modal-close" onClick={onClose} aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="groq-modal-body">
          <p className="groq-modal-desc">
            This project uses a <strong>server-side</strong> Groq API key stored in a <code>.env</code> file.
            The key is never exposed to the browser or committed to GitHub.
          </p>

          <div className="groq-steps">
            <div className="groq-step">
              <span className="groq-step-num">1</span>
              <span>Get your free key at{' '}
                <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer" className="groq-link">
                  console.groq.com/keys
                </a>
              </span>
            </div>
            <div className="groq-step">
              <span className="groq-step-num">2</span>
              <span>Copy <code>.env.example</code> → <code>.env</code> in the project root</span>
            </div>
            <div className="groq-step">
              <span className="groq-step-num">3</span>
              <span>Paste your key as <code>GROQ_API_KEY=gsk_...</code></span>
            </div>
            <div className="groq-step">
              <span className="groq-step-num">4</span>
              <span>Restart the backend: <code>uv run python server.py</code></span>
            </div>
          </div>

          <div className="groq-env-preview">
            <span className="groq-env-label">.env</span>
            <pre className="groq-env-code">{`GROQ_API_KEY=gsk_your_key_here\nGROQ_MODEL=openai/gpt-oss-20b`}</pre>
          </div>

          <p className="groq-modal-note">
            ⚠️ Never commit <code>.env</code> to Git. It is already listed in <code>.gitignore</code>.
          </p>
        </div>

        <div className="groq-modal-footer">
          <button className="groq-modal-done" onClick={onClose}>Got it</button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// TISW — Teaching in a Structured Way
// Minimal, transparent AI tutoring workspace
// ============================================================================
export default function App() {
  // --- Core conversation state ---
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [threadId, setThreadId] = useState(() => `study-session-${crypto.randomUUID().slice(0, 8)}`);
  const [isLoading, setIsLoading] = useState(false);

  // --- UI state ---
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toast, setToast] = useState(null);
  const [toastType, setToastType] = useState('info');
  const [backendStatus, setBackendStatus] = useState('connected');
  const [showFileZone, setShowFileZone] = useState(false);
  const [showGroqModal, setShowGroqModal] = useState(false);

  // --- Thread history ---
  const [threads, setThreads] = useState(() => [{
    id: `study-session-${crypto.randomUUID().slice(0, 8)}`,
    label: 'New Session',
    messageCount: 0,
    createdAt: Date.now(),
  }]);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // --- Toast helper ---
  const triggerToast = useCallback((msg, type = 'info') => {
    setToast(msg);
    setToastType(type);
    setTimeout(() => setToast((prev) => (prev === msg ? null : prev)), 3000);
  }, []);

  // --- Auto-scroll to bottom of chat ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // --- Backend health check ---
  useEffect(() => {
    checkHealth().then((data) => {
      if (data.status === 'healthy') {
        setBackendStatus('connected');
      } else if (data.status === 'degraded') {
        setBackendStatus('degraded');
      } else {
        setBackendStatus('offline');
      }
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

    // Append user message immediately
    const userMsg = { role: 'user', content: textToSend, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    // Update session label in history
    setThreads((prev) =>
      prev.map((t) =>
        t.id === threadId ? { ...t, messageCount: t.messageCount + 2, label: textToSend.slice(0, 36) } : t
      )
    );

    try {
      // Real API call to FastAPI / LangGraph backend
      const data = await sendChatMessage({
        message: textToSend,
        threadId: threadId,
        mode: 'default',
      });

      // Append assistant's response
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          timestamp: Date.now(),
        },
      ]);

      if (data.thread_id) {
        setThreadId(data.thread_id);
      }
      setBackendStatus('connected');
    } catch (err) {
      const errorMsg = err.message || 'Failed to reach tutor backend.';
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `> ⚠️ **Backend Error**\n> ${errorMsg}\n\nPlease verify that the FastAPI backend server is running on \`http://localhost:8000\`.`,
          timestamp: Date.now(),
          isError: true,
        },
      ]);
      setBackendStatus('offline');
      triggerToast(errorMsg, 'error');
    } finally {
      setIsLoading(false);
    }
  }, [prompt, threadId, isLoading, triggerToast]);

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

      {/* Minimal Translucent Sidebar for Chat History */}
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

        {/* Clean New Chat Action */}
        <button type="button" className="new-session-btn" onClick={handleNewSession}>
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>

        {/* Groq API Key section */}
        <div className="groq-config-section">
          <div className="groq-config-row">
            <span className="groq-config-badge">G</span>
            <span className="groq-config-label">Groq AI</span>
            <button
              type="button"
              className="groq-config-btn"
              onClick={() => setShowGroqModal(true)}
              title="Configure Groq API Key"
            >
              Configure key
            </button>
          </div>
        </div>

        {/* Session History List */}
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

      {/* Sidebar overlay for mobile / backdrop click */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main Workspace Interface */}
      <div className="main-wrapper">
        {/* Minimal Transparent Header */}
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

        {/* Chat / Content Area */}
        <main className="chat-area">
          {/* Minimalist Empty State when no conversation has started */}
          {!hasMessages && (
            <div className="hero-minimal">
              <h1 className="hero-wordmark">TISW</h1>
              <p className="hero-tagline">Teaching in a Structured Way</p>
            </div>
          )}

          {/* Conversation messages */}
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

              {/* Minimal Loading / Thinking indicator */}
              {isLoading && (
                <div className="message-row message-row--assistant animate-fadeIn">
                  <div className="message-bubble message-bubble--assistant">
                    <div className="assistant-header">
                      <span className="assistant-label">TISW</span>
                    </div>
                    <div className="thinking-text">
                      <RefreshCw className="w-3.5 h-3.5 spin" />
                      <span>Structuring response...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>
          )}
        </main>

        {/* Minimal Transparent Floating Input Area */}
        <div className="input-area">
          {/* Optional file drop zone */}
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

          {/* Message Composer Card */}
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
              placeholder="Ask anything..."
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

      {/* Global Toast Alerts */}
      <Toast message={toast} type={toastType} />

      {/* Groq API Key Modal */}
      {showGroqModal && <GroqConfigModal onClose={() => setShowGroqModal(false)} />}
    </div>
  );
}
