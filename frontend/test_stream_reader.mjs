import { consumeSseReader, isActiveStreamRequest, streamChatMessage } from './src/lib/api.js';

function fakeReader(chunks, { error = null } = {}) {
  let index = 0;
  return {
    cancelled: false,
    released: false,
    async read() {
      if (error && index === chunks.length) throw error;
      if (index >= chunks.length) return { value: undefined, done: true };
      return { value: new TextEncoder().encode(chunks[index++]), done: false };
    },
    async cancel() { this.cancelled = true; },
    releaseLock() { this.released = true; },
  };
}

async function run() {
  if (!isActiveStreamRequest({ requestId: 'old', threadId: 'thread-a' }, 'old', 'thread-b')) {
    // An old stream must not update a newly selected conversation.
  } else {
    throw new Error('old stream was accepted for the wrong conversation');
  }
  if (!isActiveStreamRequest({ requestId: 'current', threadId: 'thread-a' }, 'current', 'thread-a')) {
    throw new Error('current stream was rejected');
  }

  const received = [];
  const normalReader = fakeReader([
    'event: start\r\ndata: {"ok":true}\r\n\r\n',
    'event: delta\ndata: {\n',
    'data: "text":"\u4e16\u754c"\n',
    'data: }\n\n',
    'event: done\ndata: {"response":"\u4e16\u754c"}\n\n',
  ]);
  await consumeSseReader(normalReader, {
    onStart: (value) => received.push(['start', value.ok]),
    onDelta: (value) => received.push(['delta', value]),
    onDone: (value) => received.push(['done', value.response]),
  });
  if (JSON.stringify(received) !== JSON.stringify([['start', true], ['delta', '世界'], ['done', '世界']])) {
    throw new Error(`unexpected parsed events: ${JSON.stringify(received)}`);
  }
  if (!normalReader.released || normalReader.cancelled) throw new Error('normal reader cleanup incorrect');

  const originalFetch = globalThis.fetch;
  let requestBody;
  globalThis.fetch = async (_url, options) => {
    requestBody = JSON.parse(options.body);
    return {
      ok: true,
      body: { getReader: () => fakeReader(['event: done\ndata: {"response":"echo"}\n\n']) },
    };
  };
  try {
    await streamChatMessage({
      message: 'What is Python?',
      threadId: 'thread-a',
      turnId: 'python-turn',
      sessionToken: 'opaque-test-token',
      tracingEnabled: false,
      onDone: () => {},
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
  if (JSON.stringify(requestBody) !== JSON.stringify({
    message: 'What is Python?',
    thread_id: 'thread-a',
    turn_id: 'python-turn',
    rag_enabled: false,
    tracing_enabled: false,
  })) {
    throw new Error(`frontend request payload changed: ${JSON.stringify(requestBody)}`);
  }

  const malformedReader = fakeReader(['event: delta\ndata: {bad}\n\n']);
  try { await consumeSseReader(malformedReader); throw new Error('malformed event was accepted'); }
  catch (error) {
    if (!String(error.message).includes('invalid streaming event')) throw error;
    if (!malformedReader.cancelled || !malformedReader.released) throw new Error('malformed cleanup missing');
  }

  const eofReader = fakeReader(['event: start\ndata: {}\n\n']);
  try { await consumeSseReader(eofReader); throw new Error('incomplete EOF was accepted'); }
  catch (error) {
    if (!String(error.message).includes('before completion')) throw error;
    if (!eofReader.cancelled || !eofReader.released) throw new Error('EOF cleanup missing');
  }

  const unterminatedDoneReader = fakeReader(['event: done\ndata: {"response":"partial"}']);
  try { await consumeSseReader(unterminatedDoneReader); throw new Error('unterminated done was accepted'); }
  catch (error) {
    if (!String(error.message).includes('before completion')) throw error;
    if (!unterminatedDoneReader.cancelled || !unterminatedDoneReader.released) throw new Error('unterminated EOF cleanup missing');
  }

  const abortReader = fakeReader([], { error: Object.assign(new Error('aborted'), { name: 'AbortError' }) });
  try { await consumeSseReader(abortReader); throw new Error('abort was accepted'); }
  catch (error) {
    if (error.name !== 'AbortError') throw error;
    if (!abortReader.cancelled || !abortReader.released) throw new Error('abort cleanup missing');
  }

  console.log('OK: frontend SSE partial chunks, multiline data, Unicode, malformed events, EOF, and abort cleanup passed');
}

run();
