import { createStreamPacer } from './src/lib/streamPacer.js';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

// Deterministic scheduler so pacing behaviour is independent of wall-clock time.
let virtualNow = 0;
let timerId = 0;
const timers = new Map();
globalThis.setInterval = (fn) => {
  const id = ++timerId;
  timers.set(id, fn);
  return id;
};
globalThis.clearInterval = (id) => timers.delete(id);
const advance = (ms) => {
  virtualNow += ms;
  for (const fn of [...timers.values()]) fn();
};
const drain = (stepMs = 33, maxTicks = 4000) => {
  let ticks = 0;
  while (timers.size > 0 && ticks < maxTicks) {
    advance(stepMs);
    ticks += 1;
  }
  return ticks;
};

// 1. A burst of text is revealed at a readable pace, not all at once.
{
  const startedAt = virtualNow;
  const reveals = [];
  const pacer = createStreamPacer({
    now: () => virtualNow,
    onReveal: (text, complete) => reveals.push({ text, complete, at: virtualNow }),
  });
  pacer.push('x'.repeat(700));
  drain();
  const first = reveals[0].text.length;
  const last = reveals[reveals.length - 1].text;
  const elapsedSeconds = (reveals[reveals.length - 1].at - startedAt) / 1000;
  const rate = last.length / elapsedSeconds;
  assert(first <= 5, `first reveal must be a small step, got ${first} chars`);
  assert(last.length === 700, 'all received text must eventually be revealed');
  assert(rate > 40 && rate < 110, `base pace should stay near 70 chars/s, got ${rate.toFixed(1)}`);
  const maxStep = reveals.reduce((max, item, i) => Math.max(max, item.text.length - (i ? reveals[i - 1].text.length : 0)), 0);
  assert(maxStep <= 6, `reveals must not arrive as large jumps, got ${maxStep} chars`);
  pacer.cancel();
}

// 2. A large backlog speeds up but is capped, so long answers still finish.
{
  const startedAt = virtualNow;
  const reveals = [];
  const pacer = createStreamPacer({
    now: () => virtualNow,
    onReveal: (text) => reveals.push({ len: text.length, at: virtualNow }),
  });
  pacer.push('y'.repeat(3500));
  drain();
  const elapsedSeconds = (reveals[reveals.length - 1].at - startedAt) / 1000;
  const averageRate = 3500 / elapsedSeconds;
  assert(reveals[reveals.length - 1].len === 3500, 'long answer must be revealed in full');
  assert(averageRate > 70, `long backlog should accelerate past base pace, got ${averageRate.toFixed(1)}`);
  const maxStep = reveals.reduce((max, item, i) => Math.max(max, item.len - (i ? reveals[i - 1].len : 0)), 0);
  const maxRate = (maxStep / 33) * 1000;
  assert(maxRate <= 160, `reveal rate must stay capped near 140 chars/s, got ${maxRate.toFixed(1)}`);
  pacer.cancel();
}

// 3. The final server response wins over accumulated deltas.
{
  const reveals = [];
  let doneCalls = 0;
  const pacer = createStreamPacer({
    now: () => virtualNow,
    onReveal: (text) => reveals.push(text),
    onDone: () => {
      doneCalls += 1;
    },
  });
  pacer.push('partial');
  pacer.finish('partial answer, final');
  drain();
  assert(reveals[reveals.length - 1] === 'partial answer, final', 'authoritative final text must be revealed');
  assert(doneCalls === 1, 'onDone must fire exactly once');
  advance(33);
  assert(doneCalls === 1, 'onDone must not fire again after completion');
}

// 4. flush() reveals everything immediately and closes the pacer once.
{
  const reveals = [];
  let doneCalls = 0;
  const pacer = createStreamPacer({
    now: () => virtualNow,
    onReveal: (text) => reveals.push(text),
    onDone: () => {
      doneCalls += 1;
    },
  });
  pacer.push('hello world');
  pacer.flush();
  assert(reveals[reveals.length - 1] === 'hello world', 'flush must reveal the entire buffer');
  assert(doneCalls === 1, 'flush must complete the pacer once');
  pacer.push(' late');
  assert(reveals[reveals.length - 1] === 'hello world', 'no text may be revealed after flush');
  assert(timers.size === 0, 'flush must stop the reveal timer');
}

// 5. cancel() stops work and never calls onDone.
{
  let doneCalls = 0;
  const pacer = createStreamPacer({
    now: () => virtualNow,
    onReveal: () => {},
    onDone: () => {
      doneCalls += 1;
    },
  });
  pacer.push('text');
  pacer.cancel();
  drain(33, 10);
  assert(doneCalls === 0, 'cancelled pacer must not complete');
  assert(timers.size === 0, 'cancel must stop the reveal timer');
}

console.log('OK: stream pacer reveals at a readable pace, accelerates on backlog, and closes once');
