import React from 'react';
import GlassSheet from './GlassSheet';
import {
  Activity,
  BookOpen,
  Database,
  GitHub,
  HelpCircle,
  History,
  KeyRound,
  Plus,
  Sparkles,
} from '../icons';

const GITHUB_URL = 'https://github.com/19anveshh/TISW';

export default function MobileNavSheet({
  onClose,
  onOpen,
  onOpenHistory,
  onNewChat,
  onToggleRag,
  ragEnabled,
  documentCount,
  tracingPreference,
  tracingAvailable,
  offline,
  onToggleTracing,
  sessionToken,
}) {
  const traceUnavailable = offline || !tracingAvailable;

  return (
    <GlassSheet title="Menu" onClose={onClose} closeLabel="Close menu">
      <nav className="sheet-menu" aria-label="Mobile navigation">
        <button
          type="button"
          className="sheet-menu-item sheet-menu-item--primary"
          onClick={() => onOpen('start')}
        >
          <Sparkles /> Start Here
        </button>
        <button type="button" className="sheet-menu-item" onClick={() => onOpen('about')}>
          <BookOpen /> About
        </button>
        <button type="button" className="sheet-menu-item" onClick={() => onOpen('faq')}>
          <HelpCircle /> FAQ
        </button>
        <button type="button" className="sheet-menu-item" onClick={() => onOpen('groq')}>
          <KeyRound /> Groq Key
          <span className={`sheet-menu-meta ${sessionToken ? 'sheet-menu-meta--on' : ''}`}>
            {sessionToken ? 'Connected' : 'Set up'}
          </span>
        </button>
      </nav>

      <div className="sheet-divider" />

      <nav className="sheet-menu" aria-label="Session tools">
        <button type="button" className="sheet-menu-item" onClick={onOpenHistory}>
          <History /> Documents &amp; history
        </button>
        <button type="button" className="sheet-menu-item" onClick={onToggleRag}>
          <Database /> Knowledge Base (RAG)
          <span className={`sheet-menu-meta ${ragEnabled ? 'sheet-menu-meta--on' : ''}`}>
            {ragEnabled ? `ON (${documentCount})` : 'OFF'}
          </span>
        </button>
        <button
          type="button"
          className="sheet-menu-item"
          onClick={onToggleTracing}
          disabled={traceUnavailable}
          aria-disabled={traceUnavailable}
        >
          <Activity /> LangSmith tracing
          <span className={`sheet-menu-meta ${tracingPreference && !traceUnavailable ? 'sheet-menu-meta--on' : ''}`}>
            {traceUnavailable ? 'Unavailable' : tracingPreference ? 'ON' : 'OFF'}
          </span>
        </button>
        <button type="button" className="sheet-menu-item" onClick={onNewChat}>
          <Plus /> New Chat
        </button>
        <a
          className="sheet-menu-item"
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Star TISW on GitHub"
        >
          <GitHub /> Star on GitHub
        </a>
      </nav>
    </GlassSheet>
  );
}
