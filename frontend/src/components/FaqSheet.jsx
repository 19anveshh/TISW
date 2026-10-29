import React, { useState } from 'react';
import GlassSheet from './GlassSheet';
import { ChevronDown } from '../icons';

const FAQ_ITEMS = [
  {
    q: 'What is TISW?',
    a: (
      <>
        <p>
          TISW stands for <strong>Teaching in a Structured Way</strong>. It is an AI study
          assistant: ask questions normally, or add your own documents so answers are grounded
          in your material. The structured walkthrough is optional — normal chat stays the
          default.
        </p>
      </>
    ),
  },
  {
    q: 'How do I get started?',
    a: (
      <>
        <p>
          <strong>1.</strong> Create a free key at{' '}
          <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer">
            console.groq.com/keys
          </a>
          . <strong>2.</strong> Open <strong>Groq Key</strong> and
          paste it — it is verified server-side. <strong>3.</strong> Upload PDF, DOCX, or TXT
          files from the sidebar and switch on Knowledge Base (RAG). Then ask a question.
        </p>
      </>
    ),
  },
  {
    q: 'How does the /tisw shortcut work?',
    a: (
      <>
        <p>
          Type <strong>/tisw</strong> followed by a topic — for example{' '}
          <strong>/tisw binary search</strong> — to start part 1 immediately. <strong>/tisw</strong>{' '}
          alone turns the mode on, and your next question starts part 1.
        </p>
        <p>
          Every walkthrough has four parts: intuition and the problem, the real concept and how
          it works, a worked example, and depth, limits, and practice. Use{' '}
          <strong>/continue</strong> or <strong>/next</strong> (or the Continue button) for the
          next part, <strong>/all</strong> for the rest in one answer, and{' '}
          <strong>/normal</strong> to leave the walkthrough. A doubt asked between parts is
          answered without advancing or restarting it, and a new chat starts in normal mode.
        </p>
      </>
    ),
  },
  {
    q: 'What is the structured study evaluation?',
    a: (
      <>
        <p>
          TISW evaluates understanding inside the lesson rather than with a separate exam. The
          four parts move from intuition to depth, and when appropriate the tutor asks one short
          prediction or checking question, then uses your reply to correct misunderstandings and
          adjust the depth. There is no separate scored quiz or report card.
        </p>
      </>
    ),
  },
  {
    q: 'Which documents can I upload?',
    a: (
      <>
        <p>
          <strong>PDF</strong>, <strong>DOCX</strong>, and <strong>TXT</strong>. Each file can be
          up to <strong>15 MB</strong>, and a session holds up to <strong>10 documents</strong>.
          PDFs keep one-indexed page numbers when the parser provides them; DOCX and TXT are
          treated as having no page number, so none is invented.
        </p>
        <p>
          The upload returns as soon as the file is accepted, then the document is chunked and
          embedded in the background. The sidebar marks it <strong>Indexing</strong> until it
          turns searchable, so a large PDF never has to wait inside one request.
        </p>
      </>
    ),
  },
  {
    q: 'How do document answers stay grounded?',
    a: (
      <>
        <p>
          With <strong>Knowledge Base (RAG)</strong> on, TISW retrieves relevant excerpts from
          your documents, answers from them, and lists the sources under the answer. Turn RAG
          off to answer from general tutor knowledge instead.
        </p>
        <p>
          If RAG is on but nothing is indexed yet, TISW tells you to upload a document — or waits
          for the ones still indexing — rather than pretending retrieval happened. Walkthrough
          continuation parts keep using the walkthrough&apos;s own topic and citations.
        </p>
      </>
    ),
  },
  {
    q: 'Is my Groq API key private?',
    a: (
      <>
        <p>
          Your key is sent only to the TISW backend, checked with a real Groq request, and kept
          in server memory for the session. It is never returned to the page and never written to
          localStorage or sessionStorage — the browser holds only an opaque session token.
          Key-shaped strings are redacted from server logs, and sessions expire after about two
          hours without use.
        </p>
      </>
    ),
  },
  {
    q: 'Do I have to upload documents?',
    a: (
      <>
        <p>
          No. TISW works as a normal tutor without any uploads. Documents are optional and only
          affect document-based questions and citations.
        </p>
      </>
    ),
  },
  {
    q: 'What does "Failed to fetch" mean?',
    a: (
      <>
        <p>
          The browser could not reach the tutor server. On the free hosted backend, the first
          request after idle can be slow while the service wakes up; TISW retries once
          automatically after a couple of seconds.
        </p>
        <p>
          If it still fails, wait a little and send again. If you self-host, confirm the backend
          answers at <strong>/api/health</strong> and that your site origin is allowed by{' '}
          <strong>TISW_ALLOWED_ORIGINS</strong> or <strong>TISW_ALLOWED_ORIGIN_REGEX</strong>.
        </p>
      </>
    ),
  },
  {
    q: 'Why does TISW say no API key is configured?',
    a: (
      <>
        <p>
          The backend can run without its own Groq key. When that happens it reports that no key
          is configured, and you add your own through <strong>Groq Key</strong>. A key added that
          way belongs to your session only and is not shared with other visitors.
        </p>
      </>
    ),
  },
  {
    q: 'Where is the source code?',
    a: (
      <>
        <p>
          TISW is open source on GitHub:{' '}
          <a
            href="https://github.com/19anveshh/TISW"
            target="_blank"
            rel="noopener noreferrer"
          >
            github.com/19anveshh/TISW
          </a>
          .
        </p>
      </>
    ),
  },
];

export default function FaqSheet({ onClose }) {
  const [openIndex, setOpenIndex] = useState(null);

  return (
    <GlassSheet title="Frequently asked questions" onClose={onClose} closeLabel="Close FAQ" wide>
      <p className="sheet-lead">
        Short answers, checked against how the app actually behaves. Click a question to expand
        it.
      </p>

      <div className="faq-list">
        {FAQ_ITEMS.map((item, index) => {
          const open = openIndex === index;
          const answerId = `faq-answer-${index}`;
          return (
            <div key={item.q} className={`faq-item ${open ? 'faq-item--open' : ''}`}>
              <button
                type="button"
                className="faq-question"
                aria-expanded={open}
                aria-controls={answerId}
                onClick={() => setOpenIndex(open ? null : index)}
              >
                <span>{item.q}</span>
                <ChevronDown />
              </button>
              <div className="faq-answer" id={answerId} hidden={!open}>
                {item.a}
              </div>
            </div>
          );
        })}
      </div>
    </GlassSheet>
  );
}
