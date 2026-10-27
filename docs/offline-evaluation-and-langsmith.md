# Offline evaluation and optional LangSmith tracing

## Safety defaults

The custom evaluator is offline by default when invoked without `--live`.
`evaluations.runner` applies `TISW_OFFLINE=true` before importing the
application. In this mode:

- `main.py` does not construct the global Groq client.
- FastAPI session, verification, and chat routes return a clear 503 instead
  of attempting provider access.
- LangSmith tracing environment switches are forced off.
- No LangSmith package client is created by the runtime policy.
- A missing compatible response cache is `NOT_RUN`; it is never a mock `PASS`.
- The streaming endpoint remains testable with fakes, but offline requests
  cannot initialize Groq or emit LangSmith traces.

Run deterministic checks with no provider access:

```bash
uv run --no-sync python -m evaluations.runner --offline --include-rag
uv run --no-sync python -u test_evaluations.py
uv run --no-sync python -u test_modes.py
uv run --no-sync python -u test_security.py
```

`--offline` may use compatible, fingerprinted response and judge cache entries
when both are available. Cached records are marked `response_source=cache`.
Deterministic state checks use `response_source=mock_provider`, local fixture
checks use `deterministic_fixture`, live provider answers use
`live_provider`, and missing model responses use `missing_cache` plus
`NOT_RUN`. Cache keys include the evaluator, rubric, dataset, prompt, model,
temperature, case, and retrieval context fingerprints.

The web UI stores its tracing preference per browser and sends it with future
chat requests. `GET /api/runtime` reports only boolean capability flags. The
backend applies tracing through a request-scoped LangSmith context, never by
mutating process-wide tracing environment variables per click. The streaming
route is `POST /api/chat/stream` and emits `start`, `delta`, optional `sources`,
`done`, or `error` SSE events.

Streaming requests carry a distinct `turn_id`. The backend uses a process-local
session/thread lock and turn record: completed turns are replayed for repeated
IDs, while cancelled or failed turns restore the pre-turn in-memory checkpoint.
This protects one worker only; shared locking and durable idempotency are still
required for multi-worker or multi-instance deployment.

For a local visual streaming check, explicitly set `TISW_MOCK_STREAMING=true`
alongside `TISW_OFFLINE=true` before starting the backend. This fixture emits
deterministic SSE chunks and never initializes Groq, checkpoints an answer, or
contacts LangSmith. It is disabled unless the additional variable is set.

## LangSmith configuration

The installed versions are:

- `langsmith==0.14.12`
- `langchain==1.4.7`
- `langchain-core==1.7.0`
- `langgraph==1.2.17`
- `langchain-groq==1.1.3`

The installed LangSmith SDK recognizes `LANGSMITH_TRACING_V2` and the
`LANGCHAIN_TRACING_V2` compatibility name. Project selection uses
`LANGSMITH_PROJECT` (with the SDK's legacy project fallback available). The
project accepts `LANGSMITH_TRACING` as a human-readable opt-in setting and
centralizes it through `runtime_config.py`.

Tracing is disabled unless explicitly enabled. Offline mode always wins over
all tracing settings. No key is required for offline work.

For a future online session, configure locally without committing secrets:

```env
TISW_OFFLINE=false
LANGSMITH_TRACING=true
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=tisw
LANGSMITH_API_KEY=<local secret, never commit>
```

When enabled, LangChain/LangGraph tracing can transmit graph inputs,
outputs, metadata, prompts, responses, and other run information to the
configured LangSmith service. This repository has not verified the privacy
policy or server-side retention of that data. Do not enable tracing for
sensitive conversations unless that transmission is approved.

No custom exporter or hosted client is required: the future online process
should set these variables before graph execution and use the official
LangChain/LangSmith integration.

## Future evaluation mapping

`evaluations/langsmith_adapter.py` performs only local conversion. It does not
create a hosted dataset or run an experiment. Each fixed dataset case maps to:

- `inputs.question`, `inputs.history`, and `inputs.expected_mode`;
- reference facts and evaluation criteria as reference outputs;
- stable case ID, category, subject, dataset version, paired group/mode, and
  rubric groups as metadata.

Local evaluator records map to experiment-result-shaped rows containing the
answer, status, deterministic/judge metrics, response source, latency, and
usage. The five paired groups can be grouped by `build_pair_rows` for a
future Normal/TISW experiment. A scoring mechanism remains necessary;
LangSmith is observability and experiment infrastructure, not a quality
judge by itself.

## Validation status

Completed locally: evaluator unit tests, mocked provider/judge failure tests,
Python compilation, mode/state tests, security tests, and offline selector
checks. No Groq request, LangSmith request, hosted experiment, deployment, or
network-dependent dependency installation was performed for this work.
