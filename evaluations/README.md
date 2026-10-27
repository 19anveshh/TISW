# TISW evaluation framework

The framework evaluates the existing LangGraph workflow without changing
production prompts or frontend behavior.

## Architecture summary

- `main.py` builds a single-node graph: `START -> tutor -> END`.
- `TutorState.messages` uses LangGraph `add_messages`; `active_mode`, session
  preferences, and optional `rag_context` are checkpointed with
  `InMemorySaver`.
- `server.py` creates a per-session graph/checkpointer for server-issued opaque
  tokens. It performs optional hybrid RAG retrieval before invoking the graph.
- `rag_engine.py` uses FastEmbed vectors, BM25, and weighted reciprocal rank
  fusion, returning at most six chunks with source metadata.
- Normal and TISW system prompts are selected in `build_system_message`.
- The production code has no retry/alternative-answer loop. LangSmith tracing
  is optional and centrally controlled by `runtime_config.py`; the evaluator
  records provider usage metadata when a live Groq/LangChain response supplies
  it.

## Highest-priority evaluation gaps now made visible

- Semantic response quality, TISW teaching quality, coding explanations, and
  Normal/TISW paired differences require a live run and judge; offline mode
  intentionally leaves them `not_run`.
- Judge agreement is unmeasured until the calibration template contains real
  blinded human labels.
- Exact cost is unmeasured unless provider token usage and explicit prices are
  available.
- Production latency/error tracing is not persistent or distributed; this
  runner measures the invoked evaluation process only.
- Learning gain, delayed retention, and transfer require the human protocol in
  `learning_effectiveness_plan.md`.

## Evaluation layers

1. `dataset.py` contains 150 fixed expanded cases with stable IDs, history,
   expected mode, reference facts, criteria, and category metadata.
2. `rubrics.py` contains versioned general, TISW, coding, and RAG rubrics with
   observable low/medium/high anchors.
3. `metrics.py` provides conservative deterministic coverage, retrieval
   Precision@k, Recall@k, MRR, nDCG, percentile, and aggregation functions.
4. `coding.py` executes only fixed/generated snippets in a separate process
   with a timeout, minimal environment, isolated Python mode, and import
   deny-list. CI/container sandboxing is still recommended for hostile code.
5. `runner.py` runs deterministic checks offline or invokes the actual graph in
    live mode. An optional judge produces structured scores; a retry is opt-in
    and capped at one additional independent attempt. Top-level outcomes are
    exactly `PASS`, `FAIL`, `ERROR`, or `NOT_RUN`.
6. Response and judge stages use persistent fingerprinted caches under
   `evaluation_results/.cache` by default. Atomic checkpoints preserve partial
   records and `--resume` skips only compatible PASS/FAIL records.
7. `--offline` is the safe default, forces `TISW_OFFLINE=true`, disables
   LangSmith tracing, and never initializes the application Groq client.

## Commands

```bash
# No provider call: dataset, mode-state, fixture, and compatible-cache checks
uv run python -m evaluations.runner --offline

# Also exercise the actual FastEmbed/BM25 fixture retrieval
uv run python -m evaluations.runner --offline --include-rag

# Actual Groq-backed graph; requires GROQ_API_KEY in the environment/.env
uv run python -m evaluations.runner --live --judge --output evaluation_results

# Run an explicit small representative sample
uv run python -m evaluations.runner --live --judge \
  --case-id FACT-001 --case-id EXPL-001 --case-id DEEP-001 \
  --max-external-calls 24 --max-total-tokens 20000 \
  --max-rate-limit-errors 1 --output evaluation_results

# Resume a compatible interrupted run with bounded provider use
uv run python -m evaluations.runner --live --judge \
  --resume evaluation_results/<run>.checkpoint.json \
  --max-external-calls 300 --max-total-tokens 180000

# Critical mode failures return non-zero; compare against an earlier JSON run
uv run python -m evaluations.runner --live --baseline evaluation_results/<run>.json --fail-on-regression
```

Each run writes one JSON result and one Markdown report. Reports contain the
model, dataset/rubric/prompt hashes, timestamp, measured/not-run/blocked
statuses, outputs, traces, latency, provider usage when available, failures,
and baseline comparison. API keys are scrubbed and are never read into report
metadata.

Provider 429s, timeouts, missing responses, invalid judge JSON, and budget
exhaustion are recorded as `ERROR` or `NOT_RUN`; they cannot become a quality
PASS. Repeated rate-limit errors stop the run and mark remaining cases
`NOT_RUN`.

To audit an archived run without making provider calls:

```bash
uv run python -m evaluations.audit \
  --source evaluation_results/<original>.json \
  --quota-source evaluation_results/<quota-limited>.json \
  --validation "uv run python -u test_evaluations.py|PASS"
```

## Interpretation

Offline results do not measure model response quality. A live run without
`--judge` measures workflow output and deterministic checks but leaves semantic
quality judge metrics unmeasured. Exact cost is unmeasured unless the provider
returns token usage and `EVAL_*_USD_PER_MILLION` prices are configured.

The learning-effectiveness protocol is in
`learning_effectiveness_plan.md`; no learner improvement claim is made by this
framework.
