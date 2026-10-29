/**
 * Stream pacer — reveals provider chunks at a readable speed.
 *
 * Groq delivers ~300-350 chunks per second (~1,100 chars/s), and rendering each
 * chunk collapses into a handful of 450-char repaints. The pacer buffers text
 * and releases it on a steady tick, so the transcript flows instead of snapping.
 */

const DEFAULT_BASE_CHARS_PER_SECOND = 70;
const DEFAULT_MAX_CHARS_PER_SECOND = 140;
const DEFAULT_CATCH_UP_BACKLOG = 400;
const DEFAULT_TICK_MS = 33;

export function createStreamPacer({
  onReveal,
  onDone,
  baseCharsPerSecond = DEFAULT_BASE_CHARS_PER_SECOND,
  maxCharsPerSecond = DEFAULT_MAX_CHARS_PER_SECOND,
  catchUpBacklog = DEFAULT_CATCH_UP_BACKLOG,
  tickMs = DEFAULT_TICK_MS,
  now = () => performance.now(),
} = {}) {
  let received = '';
  let revealed = 0;
  let finished = false;
  let closed = false;
  let timer = null;
  let last = now();

  const stopTimer = () => {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };

  const close = () => {
    if (closed) return;
    closed = true;
    stopTimer();
    onDone?.();
  };

  // Reading pace until the backlog is clearly ahead of the reader, then ease up
  // to at most double speed so long answers still finish in a reasonable time.
  const revealRate = () => {
    const backlog = received.length - revealed;
    if (backlog <= catchUpBacklog) return baseCharsPerSecond;
    const accelerated = baseCharsPerSecond + (backlog - catchUpBacklog) * 0.1;
    return Math.min(maxCharsPerSecond, accelerated);
  };

  const tick = () => {
    const stamp = now();
    const elapsed = Math.max(0, (stamp - last) / 1000);
    last = stamp;
    if (revealed >= received.length) {
      if (finished) close();
      return;
    }
    const step = Math.max(1, Math.round(revealRate() * elapsed));
    revealed = Math.min(received.length, revealed + step);
    onReveal?.(received.slice(0, revealed), revealed === received.length);
    if (revealed >= received.length && finished) close();
  };

  const startTimer = () => {
    if (timer === null && !closed) {
      last = now();
      timer = setInterval(tick, tickMs);
    }
  };

  return {
    push(text) {
      if (!text || closed) return;
      received += text;
      startTimer();
    },

    /** Marks the stream complete; `finalText` is authoritative when provided. */
    finish(finalText) {
      if (closed) return;
      if (typeof finalText === 'string') {
        received = finalText;
        revealed = Math.min(revealed, received.length);
      }
      finished = true;
      if (revealed >= received.length) close();
      else startTimer();
    },

    /** Reveals everything immediately, e.g. when the reader moves on. */
    flush() {
      if (closed) return;
      revealed = received.length;
      if (revealed > 0) onReveal?.(received, true);
      close();
    },

    cancel() {
      closed = true;
      stopTimer();
    },
  };
}
