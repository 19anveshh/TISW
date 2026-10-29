/**
 * Message identity helpers used by the streaming conversation UI.
 * User and assistant messages share a request cleanup key, but never share
 * the message ID used for content updates.
 */

export function createTurnMessages({ requestId, text, timestamp = Date.now() }) {
  const userMessageId = `${requestId}:user`;
  const assistantMessageId = `${requestId}:assistant`;

  return {
    userMessageId,
    assistantMessageId,
    messages: [
      {
        id: userMessageId,
        role: 'user',
        content: text,
        clientRequestId: requestId,
        timestamp,
      },
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        clientRequestId: requestId,
        timestamp,
      },
    ],
  };
}

export function updateAssistantMessage(messages, assistantMessageId, update) {
  return messages.map((message) => (
    message.id === assistantMessageId
      ? {
          ...message,
          ...(typeof update === 'function' ? update(message) : update),
        }
      : message
  ));
}

/** Converts stored transcript rows into renderable message objects. */
export function messagesFromHistory(rows = []) {
  return rows.map((row) => ({
    id: `history:${row.id}`,
    role: row.role,
    content: row.content || '',
    sources: Array.isArray(row.sources) ? row.sources : [],
    rag_used: Boolean(row.rag_used),
    paging: row.paging || null,
    timestamp: row.created_at ? Math.round(row.created_at * 1000) : undefined,
  }));
}
