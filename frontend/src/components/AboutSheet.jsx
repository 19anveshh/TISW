import React, { useCallback, useEffect, useRef, useState } from 'react';
import { X } from '../icons';
import './AboutSheet.css';

const EXIT_MS = 380;

function prefersReducedMotion() {
  return typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

export default function AboutSheet({ onClose }) {
  const [closing, setClosing] = useState(false);
  const closeRef = useRef(null);
  const closedRef = useRef(false);
  const exitTimerRef = useRef(null);

  const requestClose = useCallback(() => {
    if (closedRef.current) return;
    closedRef.current = true;

    if (prefersReducedMotion()) {
      onClose();
      return;
    }
    setClosing(true);
    exitTimerRef.current = setTimeout(onClose, EXIT_MS);
  }, [onClose]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();

    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        requestClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      clearTimeout(exitTimerRef.current);
      document.body.style.overflow = previousOverflow;
    };
  }, [requestClose]);

  return (
    <div
      className={`about-overlay ${closing ? 'about-overlay--closing' : ''}`}
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
    >
      <div
        className={`about-sheet ${closing ? 'about-sheet--closing' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label="About TISW"
      >
        <button
          ref={closeRef}
          type="button"
          className="about-close"
          onClick={requestClose}
          aria-label="Close About"
        >
          <X className="w-4 h-4" />
        </button>

        <header className="about-head">
          <span className="about-eyebrow">About TISW</span>
          <h2 className="about-title">Teaching in a Structured Way</h2>
        </header>

        <div className="about-body">
          <p>
            TISW turns your documents into an interactive learning and knowledge platform.
            Instead of spending hours searching through lengthy notes, textbooks, or PDFs, you
            can ask questions in natural language and get clear, relevant answers based on your
            study materials.
          </p>
          <p>
            What makes TISW different is its combination of AI-powered document search,
            intelligent retrieval, and structured study evaluation. Beyond answering questions,
            it helps students evaluate their understanding through structured assessments based
            on their learning content, making it easier to identify knowledge gaps and focus on
            areas that need improvement.
          </p>
          <p>
            Whether you're preparing for exams, revising complex topics, or learning something
            new, TISW helps you study smarter, save time, and understand concepts more
            effectively.
          </p>
        </div>

        <footer className="about-foot">
          <p className="about-tagline">
            Learn smarter. Evaluate your progress. Improve with confidence.
          </p>
          <button type="button" className="about-btn" onClick={requestClose}>
            Got it
          </button>
        </footer>
      </div>
    </div>
  );
}
