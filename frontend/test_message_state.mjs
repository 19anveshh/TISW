import { createTurnMessages, updateAssistantMessage } from './src/lib/messageState.js';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const turn = createTurnMessages({
  requestId: 'request-hi',
  text: 'hi',
  timestamp: 1,
});

assert(turn.messages.length === 2, 'a turn must create exactly two messages');
assert(turn.messages[0].role === 'user', 'first message must be the user message');
assert(turn.messages[0].content === 'hi', 'user message must retain the submitted text');
assert(turn.messages[1].role === 'assistant', 'second message must be the assistant message');
assert(turn.messages[0].id !== turn.messages[1].id, 'user and assistant IDs must differ');

let messages = turn.messages;
const userContent = () => messages.find((message) => message.role === 'user').content;

messages = updateAssistantMessage(messages, turn.assistantMessageId, (message) => ({
  content: `${message.content}Hello`,
}));
assert(userContent() === 'hi', 'first assistant delta changed the user message');

messages = updateAssistantMessage(messages, turn.assistantMessageId, (message) => ({
  content: `${message.content}! How can I help you today?`,
}));
assert(userContent() === 'hi', 'second assistant delta changed the user message');

messages = updateAssistantMessage(messages, turn.assistantMessageId, {
  content: 'Hello! How can I help you today?',
  isStreaming: false,
});

assert(userContent() === 'hi', 'done event changed the user message');
assert(
  messages.find((message) => message.role === 'assistant').content ===
    'Hello! How can I help you today?',
  'assistant content was not finalized correctly',
);
assert(messages.filter((message) => message.role === 'user').length === 1, 'user message duplicated');
assert(messages.filter((message) => message.role === 'assistant').length === 1, 'assistant message duplicated');

console.log('OK: user content remains hi while assistant deltas and done finalize the separate assistant message');
