import React from 'react';
import GlassSheet from './GlassSheet';
import { Database, ExternalLink, GitHub, HelpCircle, KeyRound, Upload } from '../icons';

const GITHUB_URL = 'https://github.com/19anveshh/TISW';

export default function StartHereSheet({
  sessionToken,
  documentCount,
  onOpenGroq,
  onOpenDocuments,
  onOpenFaq,
  onClose,
}) {
  return (
    <GlassSheet title="Start here" onClose={onClose} closeLabel="Close start here" wide>
      <p className="sheet-lead">
        Three short steps and you are ready: connect a free Groq key, then study from your own
        documents or straight from a question.
      </p>

      <ol className="sheet-steps">
        <li className="sheet-step">
          <span className="sheet-step-index">1</span>
          <div>
            <h3 className="sheet-step-title">Get a free Groq API key</h3>
            <p className="sheet-step-text">
              TISW runs on Groq&apos;s API. Sign in at the Groq Console, create a key, and copy
              it — it starts with <strong>gsk_</strong>. The free tier is enough to try a session.
            </p>
            <div className="sheet-step-actions">
              <a
                className="sheet-btn"
                href="https://console.groq.com/keys"
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink /> Open Groq Console
              </a>
            </div>
          </div>
        </li>

        <li className="sheet-step">
          <span className="sheet-step-index">2</span>
          <div>
            <h3 className="sheet-step-title">Add it in Groq Key</h3>
            <p className="sheet-step-text">
              Open <strong>Groq Key</strong> in the header, paste the key, and choose{' '}
              <strong>Verify &amp; Save</strong>. The key is verified server-side and never
              stored in this browser.
            </p>
            <div className="sheet-step-actions">
              {sessionToken ? (
                <>
                  <span className="sheet-status sheet-status--on">Key connected</span>
                  <button type="button" className="sheet-btn" onClick={onOpenGroq}>
                    <KeyRound /> Manage key
                  </button>
                </>
              ) : (
                <button type="button" className="sheet-btn sheet-btn--primary" onClick={onOpenGroq}>
                  <KeyRound /> Add Groq key
                </button>
              )}
            </div>
          </div>
        </li>

        <li className="sheet-step">
          <span className="sheet-step-index">3</span>
          <div>
            <h3 className="sheet-step-title">Upload your documents</h3>
            <p className="sheet-step-text">
              Add <strong>PDF, DOCX, or TXT</strong> files from the sidebar — up to 15 MB each
              and 10 documents per session. Turn on <strong>Knowledge Base (RAG)</strong> for
              document-based questions, complex queries, and answers that cite your material.
            </p>
            <div className="sheet-step-actions">
              <button type="button" className="sheet-btn" onClick={onOpenDocuments}>
                <Upload /> Open documents
              </button>
              {documentCount > 0 && (
                <span className="sheet-status sheet-status--on">
                  {documentCount} document{documentCount === 1 ? '' : 's'} ready
                </span>
              )}
            </div>
          </div>
        </li>
      </ol>

      <section className="sheet-section">
        <h3 className="sheet-section-title">Structured learning with /tisw</h3>
        <p className="sheet-step-text">
          TISW means <strong>Teaching in a Structured Way</strong>. Normal chat is the default;
          the structured walkthrough starts only when you ask for it. Use{' '}
          <strong>/tisw &lt;topic&gt;</strong> and TISW teaches it in four parts:
        </p>
        <ul className="sheet-list sheet-spaced">
          <li>Intuition and the problem — a familiar entry point before any jargon.</li>
          <li>The real concept and how it works — the accurate idea, explained in plain English.</li>
          <li>A worked example — the idea applied step by step.</li>
          <li>Depth, limits, and practice — when it applies, where it breaks, and what to try next.</li>
        </ul>
        <p className="sheet-step-text sheet-spaced">
          Each part arrives only when you ask. Press <strong>Continue</strong> under the answer,
          or type <strong>/continue</strong> / <strong>/next</strong>. Use{' '}
          <strong>/all</strong> for the remaining parts in one answer and{' '}
          <strong>/normal</strong> to leave the walkthrough. Doubts asked between parts are
          answered without advancing or restarting it.
        </p>
        <p className="sheet-hint">
          As it teaches, TISW checks your understanding: when appropriate it asks one short
          prediction or checking question and adapts to your answer — no separate quiz or score
          sheet.
        </p>
      </section>

      <div className="sheet-links">
        <button type="button" className="sheet-btn" onClick={onOpenFaq}>
          <HelpCircle /> Read the FAQ
        </button>
        <a
          className="sheet-btn"
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Star TISW on GitHub"
        >
          <GitHub /> Star on GitHub
        </a>
        <button type="button" className="sheet-btn" onClick={onOpenDocuments}>
          <Database /> Documents
        </button>
      </div>
    </GlassSheet>
  );
}
