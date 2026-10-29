import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { X } from '../icons';
import './GlassSheet.css';

const EXIT_MS = 260;

const FOCUSABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'textarea:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

function prefersReducedMotion() {
  return typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

/**
 * GlassSheet — the single modal surface used by About, Start Here, FAQ,
 * Groq Key, and the compact mobile menu. It owns the About-section visual
 * language, scroll locking, Escape handling, focus trapping, and focus return.
 */
export default function GlassSheet({
  title,
  onClose,
  children,
  className = '',
  closeLabel = 'Close',
  wide = false,
}) {
  const [closing, setClosing] = useState(false);
  const panelRef = useRef(null);
  const closedRef = useRef(false);
  const timerRef = useRef(null);
  const returnFocusRef = useRef(null);
  const titleId = useId();

  const requestClose = useCallback(() => {
    if (closedRef.current) return;
    closedRef.current = true;

    if (prefersReducedMotion()) {
      onClose();
      return;
    }
    setClosing(true);
    timerRef.current = setTimeout(onClose, EXIT_MS);
  }, [onClose]);

  useEffect(() => {
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    panelRef.current?.focus();

    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        requestClose();
        return;
      }
      if (event.key !== 'Tab') return;
      const panel = panelRef.current;
      if (!panel) return;
      const nodes = [...panel.querySelectorAll(FOCUSABLE)]
        .filter((node) => node.getClientRects().length > 0);
      if (nodes.length === 0) {
        event.preventDefault();
        panel.focus();
        return;
      }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === panel)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      clearTimeout(timerRef.current);
      document.body.style.overflow = previousOverflow;
      const target = returnFocusRef.current;
      // If another sheet replaced this one, let that sheet keep focus.
      window.setTimeout(() => {
        const active = document.activeElement;
        const insideDialog = active && active !== document.body
          && typeof active.closest === 'function' && active.closest('[role="dialog"]');
        if (!insideDialog) target?.focus?.();
      }, 0);
    };
  }, [requestClose]);

  return (
    <div
      className={`sheet-overlay ${closing ? 'sheet-overlay--closing' : ''}`}
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
    >
      <div className={`sheet-shell ${wide ? 'sheet-shell--wide' : ''} ${closing ? 'sheet-shell--closing' : ''}`}>
        <div
          ref={panelRef}
          tabIndex={-1}
          className={`sheet-panel ${className}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby={title ? titleId : undefined}
        >
          <h2 id={titleId} className="sheet-title">{title}</h2>
          {children}
        </div>
        <button
          type="button"
          className="sheet-close"
          onClick={requestClose}
          aria-label={closeLabel}
        >
          <X />
        </button>
      </div>
    </div>
  );
}
