# TISW — Teaching in a Structured Way

TISW is an AI tutoring application built with **React + Vite**, **FastAPI**,
**LangGraph**, and **Groq**. It preserves a natural Normal Mode and an
explicitly activated TISW structured-teaching mode.

## Features

- Minimal transparent glass UI with animated Earth background and sidebar.
- Multi-turn LangGraph conversations with Normal and TISW modes.
- Groq API key configuration through a local `.env` or the secure sidebar flow.
- Optional document-only RAG for PDF, DOCX, and TXT study material.
- Structure-aware chunking, FastEmbed semantic search, BM25 keyword search,
  weighted Reciprocal Rank Fusion, and source citations.
- Per-session document lists, deletion, index updates, upload limits, and
  server-issued session isolation.
- Native LangGraph token streaming over Server-Sent Events.
- Request-scoped, opt-in LangSmith tracing with an offline-safe UI toggle.

## Architecture and data flow

1. The React client requests a short-lived, opaque session token from the
   backend when a server `.env` key is configured. A visitor-supplied Groq key
   is verified server-side and receives the same kind of token. Raw keys never
   enter browser storage or API responses.
2. Normal chat requests go through the existing LangGraph tutor workflow. The
   web client uses `POST /api/chat/stream` for progressive output; the existing
   `POST /api/chat` response remains available for non-streaming consumers.
3. With RAG disabled, no document retrieval or embedding work is performed.
4. With RAG enabled, the backend resolves the server-issued session, retrieves
   only that session's chunks, and adds up to six grounded excerpts to the
   existing graph prompt.
5. The response returns source metadata generated from the selected chunks;
   the frontend renders it in the Sources & Grounding panel.
6. Streaming emits `start`, ordered user-visible `delta`, optional `sources`,
   and one `done` SSE event. The completed assistant message is checkpointed
   once by the streaming graph; partial chunks are not persisted separately.
7. Each request carries a client-generated `turn_id`. The backend serializes
   turns per session/thread and replays completed turns with the same ID
   instead of appending them again. Cancelled or failed turns restore the
   pre-turn checkpoint while the lock is held.

The current index registry is in memory. It is safe for per-process local
sessions, but documents disappear when the process restarts or the session
expires. A multi-instance deployment needs shared authenticated sessions and a
shared, tenant-aware index store before enabling persistent document storage.

## Setup

### Prerequisites

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- A Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### Install

```bash
git clone https://github.com/19anveshh/TISW.git
cd TISW
cp .env.example .env
uv sync
cd frontend
npm install
cd ..
```

Edit `.env` and set the real key locally:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-20b
PORT=8000
```

Never commit `.env` or place a real key in frontend code.

## Run locally

Use two terminals:

```bash
# Terminal 1 — backend
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2 — frontend
cd frontend
npm run dev
```

Open <http://localhost:5173>. The backend health endpoint is
<http://localhost:8000/api/health>.

The chat header includes a `Trace: OFF` / `Trace: ON` control. The preference
is stored per browser in `localStorage` and is sent only as a request-scoped
boolean; it never changes process-wide environment variables. The backend
requires a server-side LangSmith key and an explicit online opt-in. When
`TISW_OFFLINE=true`, the control is disabled as `Trace unavailable` and every
request remains untraced.

The stream protocol uses JSON data frames:

```text
event: start   data: {request_id, thread_id, mode, tracing_enabled}
event: delta   data: {text}
event: sources data: {sources: [...]}
event: done    data: {response, thread_id, mode, rag_used, sources, paging?}
event: error   data: {message, status?}
```

`paging` appears only while a TISW walkthrough is in play:
`{part, parts_total, title, topic, complete}`.

Only `on_chat_model_stream` events from the `tutor` node are forwarded. Final
graph state is used for completion, so final state updates cannot duplicate
already emitted deltas. Hidden prompts, tool arguments, and credentials are
not forwarded.

Providers stream far faster than a person reads (Groq delivers roughly
1,100 characters per second), and repainting on every chunk collapses into a
few large updates. The web client therefore buffers deltas in
`frontend/src/lib/streamPacer.js` and reveals them at ~70 characters per
second, rising to at most ~140 while the buffer is ahead of the reader. The
pacer never changes what is stored: `done` remains authoritative, a new
message or thread switch reveals the remaining text immediately, and stopping
generation cancels it. `frontend/test_stream_pacer.mjs` covers the pacing,
finish, flush, and cancel paths.

The transcript does not follow the stream. A new question is positioned once
when it is sent, and the viewport then stays still while the answer grows, so
long paragraphs no longer drag the page down or make the layout jitter. The
shell is a single fixed viewport: the document itself never scrolls, only the
message pane does.

Message density follows a chat-transcript layout: questions keep a compact
glass bubble, while answers render as plain text under a small `Tutor` label
instead of a full-width card. Body text is 14.5 px at a 1.58 line height with
tight paragraph, heading, and list margins, so a long answer shows about 15%
more text per screen than the card layout it replaced.

Conversation serialization and turn idempotency are process-local because the
current sessions and `InMemorySaver` checkpointer are process-local. A
multi-worker or multi-instance deployment needs shared durable checkpoint,
lock, and turn-record storage before it can provide cross-process guarantees.

For a browser-only streaming demonstration without Groq, start the backend
with the explicit local fixture enabled:

```powershell
$env:TISW_OFFLINE = "true"
$env:TISW_MOCK_STREAMING = "true"
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

Then start the frontend normally. The fixture emits deterministic chunks over
the real SSE route; it does not checkpoint a chatbot answer and cannot enable
LangSmith tracing. Remove `TISW_MOCK_STREAMING` for ordinary offline behavior.

The sidebar also supports verifying a visitor's own Groq key when the server
does not have one. That key is held server-side in a short-lived in-memory
session and is never returned to the browser.

## Hidden teaching methodology

The application starts as a normal conversational chatbot. There is no visible
teaching-mode control. TISW activates only after an explicit request such as
`Use TISW to explain Python`, `Please use TISW for this topic`, `Let's use TISW
in this conversation`, or `Explain this using TISW`. Incidental mentions such
as `What does TISW stand for?` do not activate it. The command updates
`active_mode` in the current LangGraph thread and is not sent to Groq as a
normal question. Follow-ups inherit that thread's mode.

To deactivate it, say `Stop using TISW`, `Switch back to normal chat`,
`Disable TISW for this conversation`, or `Answer normally from now on`. A new
thread starts in normal mode, and separate server-issued sessions keep their
mode state isolated.

### Slash commands and paged answers

TISW answers are delivered one quarter at a time instead of dumping the whole
walkthrough at once. The walkthrough always has four beats: intuition and the
problem, the real concept and how it works, a worked example, and depth,
limits, and practice. Each part is capped at roughly 250–350 words, and the
next quarter is only generated when the student asks for it.

| Command | Effect |
| --- | --- |
| `/tisw <topic>` | Activate TISW and deliver part 1 immediately |
| `/tisw` | Activate TISW without answering (the next question starts part 1) |
| `/continue` or `/next` | Generate the next quarter |
| `/all` | Give the remaining walkthrough in one complete answer |
| `/normal` | Leave TISW mode and clear the walkthrough |

Bare `continue`, `next`, `give me all`, and `give me everything` work too while
a part is open. Any other question asked between parts is treated as a doubt:
it is answered fully in TISW style without advancing or restarting the
walkthrough. After part 4, follow-up questions are answered in full; start a
new paged walkthrough with `/tisw <topic>`. The web client shows a
`Part k of 4` label with **Continue** and **Give me all** controls on the latest
answer, persists that label per message, and restores it after a reload.

The command state lives in `tisw_part` and `tisw_topic` in the LangGraph
thread, so it follows reloads and re-issued sessions that replay history.
Page-level details are also kept server-side; no extra planning model call is
made, so paging costs no additional input tokens.

Paged answers work with RAG. `/continue`, `/next`, and `/all` carry no topic of
their own, so they retrieve against the walkthrough's stored topic and keep the
same citations as part 1. If the session has no documents at that moment, the
part is still delivered — ungrounded — rather than replaced by an upload
warning.

## Document RAG

### Enable and disable

1. Open the left sidebar.
2. Add a PDF, DOCX, or TXT file with the document manager or paperclip button.
3. Turn **Knowledge Base (RAG)** on to use document retrieval.
4. Turn it off to return to the normal chatbot workflow. RAG-off requests do
   not run retrieval and clear any previous document context from the graph.

If RAG is enabled without an uploaded document, TISW asks the user to upload
one and does not call the LLM pretending that retrieval occurred.

### Processing and retrieval

- PDF text keeps one-indexed page numbers when the parser provides them.
- DOCX and TXT metadata has a `None` page number; no page is invented.
- Paragraphs and sentence boundaries are preserved where possible.
- The defaults target roughly 400–500 tokens per chunk and 60–75 tokens of
  overlap. Chunk size and overlap are configurable in `.env`.
- FastEmbed (`BAAI/bge-small-en-v1.5`) supplies semantic vectors.
- BM25 supplies keyword candidates.
- Each retriever contributes up to 10 candidates.
- Weighted RRF uses semantic weight `0.5`, keyword weight `0.5`, and `k=60`:

  ```text
  score = 0.5 / (60 + semantic_rank)
        + 0.5 / (60 + keyword_rank)
  ```

  Ranks start at 1, missing ranks contribute nothing, duplicates collapse by
  stable chunk ID, and the top 6 fused chunks reach the Groq prompt.

### Citations

The backend returns the exact filename, optional PDF page number, stable chunk
ID, and a snippet for every selected chunk. The model is instructed to cite
only those sources and to say when the uploaded documents do not cover an
answer. Normal Mode remains natural; TISW Mode applies the existing structured
teaching behavior to grounded material.

## Persistence

Conversations and uploaded documents survive a page reload and a backend
restart:

- Completed turns are stored in SQLite (`.data/tisw.db`, override with
  `TISW_DB_PATH`) along with their citations and walkthrough part metadata, so
  the sidebar transcript, source lists, and `Part k of 4` labels are restored
  exactly as they were. Existing databases gain the new column automatically.
- Uploaded files are stored byte-for-byte and re-indexed into the hybrid
  retriever the first time a new session asks for its documents.
- When a reloaded thread is used again, its last 12 exchanges are replayed into
  the fresh LangGraph checkpointer so the tutor keeps the conversation context.
- History is scoped by a random `tisw.client_id` kept in `localStorage`. It is
  a workspace partition, not an authentication credential: it never grants
  access to API keys, and deleting the browser's local storage starts a clean
  workspace.
- Delete a conversation from the sidebar, or stop the server and remove
  `.data/tisw.db` to clear everything.

## Security and isolation

- `.env` is ignored by Git and `.env.example` contains placeholders only.
- API keys are never returned, logged, placed in frontend source, or stored in
  localStorage/sessionStorage.
- Document routes require a server-issued opaque session token. Arbitrary
  browser-generated IDs are not accepted as document ownership.
- The `X-Client-Id` used for persistence is an opaque browser identifier, not a
  credential. It only selects rows in the local SQLite store.
- Each token has its own LangGraph checkpointer and document index. Deleted
  documents are removed from metadata, vectors, and BM25 indexes.
- Uploads are limited by extension, basic file signature, size, and document
  count. Filenames are reduced to safe basenames before indexing.
- Configure public browser origins with `TISW_ALLOWED_ORIGINS` before a public
  deployment. `TISW_ALLOWED_ORIGIN_REGEX` covers hosts that change per deploy or
  per run; its default allows the TISW Vercel project (production and previews)
  and any `localhost`/`127.0.0.1` port. Use HTTPS and a real
  authenticated/shared session service for multi-user production deployments.

## Tests and checks

```bash
uv run python test_rag_features.py
uv run python test_rag_pipeline.py
uv run python test_server_rag_api.py
uv run python test_security.py
uv run python test_modes.py
uv run python test_paging.py
uv run python test_evaluations.py
uv run python test_streaming_tracing.py

cd frontend
npm run build
npm run lint
```

The tests cover all supported extractors, invalid files, chunk overlap,
semantic and BM25 retrieval, RRF math, deduplication, six-result limiting,
server-issued session boundaries, RAG-off behavior, RAG-on citations, and
document removal. External Groq calls are mocked in API behavior tests.

## Evaluation framework

The project includes a separate, fixed 150-case evaluation framework under
`evaluations/`. It measures deterministic mode/state behavior, optional actual
RAG retrieval, response quality through an opt-in structured LLM judge, coding
fixture execution, latency, provider token usage, bounded evaluation retries,
and baseline regressions. It does not change production prompts or automate
prompt optimization.

```bash
# Deterministic checks; no Groq response-quality claim
uv run python -m evaluations.runner --offline

# Include the actual FastEmbed/BM25 RAG fixture evaluation
uv run python -m evaluations.runner --offline --include-rag

# Invoke the actual Groq-backed graph and optional judge
uv run python -m evaluations.runner --live --judge --output evaluation_results
```

Each run writes machine-readable JSON and a Markdown report. Measured,
not-run, and blocked metrics remain distinct. The fixed dataset, rubric,
judge-calibration template, and learning-effectiveness protocol are kept
separate from application prompts. A valid `GROQ_API_KEY` is required for live
response evaluation; no live model score is claimed when it is unavailable.

## Known limitations

- FastEmbed downloads its model on first use and currently runs in-process.
- Session tokens and in-memory retrieval indexes are lost on process restart.
  Persisted chats and document bytes are reloaded from SQLite, but restoring a
  document index re-embeds it on first use.
- There is no persistent user account system yet; tokens are bearer sessions
  with a two-hour TTL, and history is scoped per browser rather than per login.
- The live Groq response path still requires a valid configured key and was not
  substituted with a fake model in production.

## Project structure

```text
TISW/
├── main.py                    # LangGraph tutor and response modes
├── server.py                  # FastAPI routes and session boundary
├── persistence.py             # SQLite store for chats and document bytes
├── rag_engine.py              # Extraction, chunking, indexes, hybrid RAG
├── test_rag_features.py       # Offline RAG unit coverage
├── test_rag_pipeline.py      # End-to-end RAG pipeline checks
├── test_server_rag_api.py    # FastAPI RAG integration checks
├── test_paging.py            # Four-part TISW walkthrough and slash commands
├── .env.example              # Safe local configuration template
└── frontend/
    └── src/
        ├── App.jsx
        ├── components/DocumentManager.jsx
        ├── components/SourceCitations.jsx
        └── lib/api.js
```
