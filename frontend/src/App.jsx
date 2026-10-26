import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Sparkles, Search, Bell, Paperclip, Globe, RefreshCw, Send, Zap,
  Lightbulb, Wand2, Box, Plus, X, Volume2, BookOpen, Layers
} from './icons';
import EarthBackground from './EarthBackground';
import MarkdownRenderer from './components/MarkdownRenderer';
import QuickActionChips from './components/QuickActionChips';
import FileDropZone from './components/FileDropZone';
import Toast from './components/Toast';
import { sendChatMessage, checkHealth } from './lib/api';

// ============================================================================
// MAIN APPLICATION — StudyFlow AI (Personalized AI Study Tutor)
// ============================================================================
export default function App() {
  // --- Core state ---
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [threadId, setThreadId] = useState(() => `study-session-${crypto.randomUUID().slice(0, 8)}`);
  const [mode, setMode] = useState('default');
  const [isLoading, setIsLoading] = useState(false);

  // --- UI state ---
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toast, setToast] = useState(null);
  const [toastType, setToastType] = useState('info');
  const [backendStatus, setBackendStatus] = useState('connected');
  const [showFileZone, setShowFileZone] = useState(false);

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
    setTimeout(() => setToast((prev) => (prev === msg ? null : prev)), 3200);
  }, []);

  // --- Auto-scroll to bottom of chat ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // --- Periodic backend health check ---
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
  // SEND MESSAGE (Direct fetch to /api/chat with Groq LangGraph Backend)
  // =========================================================================
  const handleSend = useCallback(async (customText = null, customMode = null) => {
    const textToSend = (customText !== null ? customText : prompt).trim();
    if (!textToSend || isLoading) return;

    if (customText === null) setPrompt('');

    const effectiveMode = customMode || mode;

    // Append user message immediately
    const userMsg = { role: 'user', content: textToSend, mode: effectiveMode, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    // Update session label in history
    setThreads((prev) =>
      prev.map((t) =>
        t.id === threadId ? { ...t, messageCount: t.messageCount + 2, label: textToSend.slice(0, 36) } : t
      )
    );

    try {
      // Direct call to FastAPI backend
      const data = await sendChatMessage({
        message: textToSend,
        threadId: threadId,
        mode: effectiveMode,
      });

      // Append assistant's LangGraph tutor response
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
      if (effectiveMode !== 'default') {
        setMode('default');
      }
    }
  }, [prompt, mode, threadId, isLoading, triggerToast]);

  // =========================================================================
  // NEW CHAT / RESET SESSION
  // =========================================================================
  const handleNewSession = useCallback(() => {
    const newId = `study-session-${crypto.randomUUID().slice(0, 8)}`;
    setThreadId(newId);
    setMessages([]);
    setPrompt('');
    setMode('default');
    setIsLoading(false);
    setThreads((prev) => [
      { id: newId, label: 'New Session', messageCount: 0, createdAt: Date.now() },
      ...prev,
    ]);
    triggerToast('Fresh study session started', 'success');
  }, [triggerToast]);

  // =========================================================================
  // QUICK CHIP HANDLER
  // =========================================================================
  const handleChipClick = useCallback((chip) => {
    if (messages.length === 0) {
      setMode(chip.id);
      triggerToast(`Mode: ${chip.label}`, 'info');
    } else {
      handleSend(chip.message, chip.mode);
    }
  }, [messages, handleSend, triggerToast]);

  // Status indicator helpers
  const statusDot = backendStatus === 'connected' ? 'status-dot--green' :
    backendStatus === 'degraded' ? 'status-dot--amber' : 'status-dot--red';
  const statusText = backendStatus === 'connected' ? 'Connected to StudyFlow Backend (Groq)' :
    backendStatus === 'degraded' ? 'Backend running (Groq degraded)' :
    'Backend offline — run: uvicorn server:app --reload --port 8000';

  // Starter Prompts
  const starterPrompts = [
    { text: 'Why do we need Recursion?', icon: '🔄' },
    { text: 'Explain Gradient Descent from first principles', icon: '📉' },
    { text: 'How do Neural Networks actually learn?', icon: '🧠' },
    { text: 'What is a Hash Table and why is it O(1)?', icon: '🗂️' },
  ];

  const hasMessages = messages.length > 0;

  return (
    <div className="app-root">
      {/* Dynamic 3D Earth Background */}
      <EarthBackground overlayOpacity={0.32} glowOpacity={0.45} />

      {/* Sidebar for Thread History & Controls */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-orb">
              <div className="brand-orb-inner">
                <div className="brand-play" />
              </div>
            </div>
            <span className="brand-name">StudyFlow AI</span>
          </div>
          <button
            type="button"
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* New Chat Button */}
        <button type="button" className="new-session-btn" onClick={handleNewSession}>
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>

        <div className="sidebar-threads">
          <div className="sidebar-section-label">Recent Sessions</div>
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
              {thread.messageCount > 0 && (
                <span className="thread-badge">{thread.messageCount}</span>
              )}
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <span className={`status-dot ${statusDot}`} />
            <span className="status-text">{statusText}</span>
          </div>
        </div>
      </aside>

      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main Chat Interface */}
      <div className="main-wrapper">
        {/* Top Header */}
        <header className="top-header">
          <div className="header-left">
            <button
              type="button"
              className="menu-btn"
              onClick={() => setSidebarOpen(true)}
            >
              <Layers className="w-5 h-5" />
            </button>
            <div className="header-brand" onClick={handleNewSession} role="button" tabIndex={0}>
              <div className="brand-orb brand-orb--sm">
                <div className="brand-orb-inner">
                  <div className="brand-play" />
                </div>
              </div>
              <span className="brand-name">StudyFlow AI</span>
            </div>
          </div>

          <div className="header-right">
            {/* Thread ID badge */}
            <div className="thread-badge-header">
              <span className={`status-dot ${statusDot}`} />
              <span>{threadId.slice(-8)}</span>
            </div>

            {/* Reset / New Chat icon button */}
            <button
              type="button"
              className="header-icon-btn"
              onClick={handleNewSession}
              title="Reset / New Chat"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Chat Messages Area */}
        <main className="chat-area">
          {/* Hero greeting when empty */}
          {!hasMessages && (
            <div className="hero-section">
              <div className="hero-badge">
                <div className="hero-badge-inner">
                  <div className="brand-play" />
                </div>
              </div>
              <h1 className="hero-title">Hi, I'm StudyFlow AI</h1>
              <p className="hero-subtitle">
                How can we build your intuition from first principles today?
              </p>

              {/* Starter prompts */}
              <div className="starter-grid">
                {starterPrompts.map((sp, i) => (
                  <button
                    key={i}
                    type="button"
                    className="starter-card"
                    onClick={() => handleSend(sp.text)}
                  >
                    <span className="starter-icon">{sp.icon}</span>
                    <span className="starter-text">{sp.text}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Conversation history */}
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
                      {msg.mode && msg.mode !== 'default' && (
                        <span className="message-mode-tag">{msg.mode}</span>
                      )}
                    </div>
                  ) : (
                    <div className={`message-bubble message-bubble--assistant ${msg.isError ? 'message-bubble--error' : ''}`}>
                      <div className="assistant-header">
                        <span className="assistant-badge">
                          <Sparkles className="w-3.5 h-3.5" />
                          <span>First-Principles Mentor</span>
                        </span>
                      </div>
                      <div className="assistant-content">
                        <MarkdownRenderer content={msg.content} />
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* Loading / Thinking Spinner */}
              {isLoading && (
                <div className="message-row message-row--assistant animate-fadeIn">
                  <div className="message-bubble message-bubble--assistant">
                    <div className="assistant-header">
                      <span className="assistant-badge">
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>First-Principles Mentor</span>
                      </span>
                    </div>
                    <div className="assistant-content">
                      <div className="thinking-text">
                        <RefreshCw className="w-3.5 h-3.5 spin" />
                        <span>Deriving from first principles with Groq...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>
          )}
        </main>

        {/* Bottom Input Controls */}
        <div className="input-area">
          {/* Quick pedagogical mode chips */}
          <QuickActionChips
            onChipClick={handleChipClick}
            activeMode={mode}
            disabled={isLoading}
          />

          {/* Optional file drop zone */}
          {showFileZone && (
            <FileDropZone
              onUploadSuccess={(result) => {
                triggerToast(`${result.filename} uploaded`, 'success');
                setShowFileZone(false);
              }}
              onUploadError={(err) => triggerToast(err, 'error')}
              disabled={isLoading}
            />
          )}

          {/* Prompt input card */}
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
              placeholder="Ask anything — 'Why do we need this?', 'Go deeper', 'Teach me through questions'..."
              className="input-textarea"
              disabled={isLoading}
            />

            <div className="input-controls">
              <div className="input-controls-left">
                <button
                  type="button"
                  className="input-icon-btn"
                  onClick={() => setShowFileZone((prev) => !prev)}
                  title="Upload document for context"
                >
                  <Paperclip className="w-4 h-4" />
                </button>
              </div>

              <button
                type="button"
                className="send-btn"
                onClick={() => handleSend()}
                disabled={isLoading || !prompt.trim()}
                title="Send message"
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          {/* Footer status label */}
          <div className="input-footer">
            <span className={`status-dot ${statusDot}`} />
            <span className="input-footer-text">{statusText}</span>
          </div>
        </div>
      </div>

      {/* Global Toast Alerts */}
      <Toast message={toast} type={toastType} />
    </div>
  );
}
