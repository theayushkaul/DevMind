# DevMind

**An event-driven, RAG-grounded AI agent that reviews GitHub pull requests like a senior engineer would.**

DevMind listens for PR events on GitHub, queues them for async processing, retrieves relevant context from the rest of the codebase using RAG, runs a multi-node LangGraph pipeline against the diff, and posts structured, prioritized review comments back on the PR — all on a fully free-tier infrastructure stack.

[![CI](https://img.shields.io/badge/tests-292%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Table of Contents

- [Why DevMind](#why-devmind)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [The Agent Pipeline](#the-agent-pipeline)
- [Key Design Decisions](#key-design-decisions)
- [Testing](#testing)
- [Deployment](#deployment)
- [Eval Results](#eval-results)
- [What Breaks at Scale](#what-breaks-at-scale)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Why DevMind

Most "AI code review" demos are a diff pasted into a single LLM prompt. That approach breaks down for three reasons DevMind is explicitly designed around:

1. **A diff in isolation lacks context.** If new code calls `self.db.execute(query)`, whether that's a SQL-injection risk depends on how `db.execute` is implemented — information that lives in a different file. DevMind retrieves that context via RAG before review.
2. **Webhooks have a hard SLA.** GitHub expects a response in under ~10 seconds. A real review (diff fetch + retrieval + several sequential LLM calls) takes 15–60 seconds. Any synchronous design either times out or triggers GitHub's retry storm. DevMind decouples ingestion from processing with a durable queue.
3. **One bad LLM call shouldn't kill the whole review.** Security, bug, and style checking are independent, failure-isolated pipeline stages — a malformed JSON response from one checker doesn't take down the others.

The result is a system that maps cleanly onto canonical distributed-systems problems — idempotency under at-least-once delivery, decoupling via queues, stateless serverless constraints, retrieval trade-offs, and bulkhead-style failure isolation — rather than a thin LLM wrapper.

---

## Architecture

```
GitHub PR opened / synchronize
        │
        ▼
┌───────────────────────────────────────────┐
│ FastAPI Webhook Receiver (Railway)         │
│  1. hmac.compare_digest() signature check  │
│  2. Redis SET NX idempotency claim         │
│  3. Publish to QStash                      │
│  4. Return 200 OK   (<10s, GitHub SLA)     │
└───────────────────────────────────────────┘
        │  async, decoupled
        ▼
┌───────────────────────────────────────────┐
│ Upstash QStash (durable queue + retries)   │
│  - exponential backoff on failure          │
│  - HTTP callback into Lambda Function URL  │
│  - dead-letter path → Supabase dlq_events  │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ AWS Lambda — processor/handler.py          │
│  asyncio.run() bridges Lambda's sync entry │
│  point to the async pipeline underneath    │
└───────────────────────────────────────────┘
        │
        ├──► pgvector (Supabase) — RAG retrieval of related code
        │
        ├──► LangGraph pipeline (Groq-backed):
        │      DiffParser → RAGContext → SecurityChecker
        │      → BugDetector → StyleChecker → CommentSynthesizer
        │
        ├──► GitHub API — post inline review comments
        │
        └──► Supabase Postgres — persist review + findings
```

**Why not a simple synchronous request/response?** GitHub's webhook SLA and multi-second LLM latency are fundamentally incompatible without a queue in between. The queue isn't an optimization — it's a correctness requirement.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Webhook receiver | FastAPI on Railway | Fast, async-native, cheap to run 24/7 on free tier |
| Queue | Upstash QStash | Durable delivery, built-in exponential backoff, DLQ support |
| Compute | AWS Lambda | Effectively-free serverless compute, scales to zero |
| Agent orchestration | LangGraph | Typed shared state across pipeline nodes, explicit graph structure |
| LLM | Groq API | Fast inference, generous free tier, purpose-built code models |
| Embeddings | HuggingFace Inference API | Free, hosted — avoids blowing Lambda's 250MB package limit |
| Vector store | Supabase pgvector | Postgres + vectors in one free-tier service |
| Cache / idempotency | Upstash Redis | Atomic `SET NX`, low-latency dedup |
| Persistence | Supabase Postgres | Reviews, findings, repos, DLQ — one relational store |
| GitHub integration | GitHub App + PyGithub | Better rate limits than OAuth apps |
| Testing | pytest, pytest-asyncio, pytest-mock | 292 passing unit tests across all modules |
| Containerization | Docker (multi-stage) | `python:3.12-slim-bookworm` base — avoids musl/Alpine wheel issues |
| CI | GitHub Actions | Tests run on every push |

---

## Quick Start

> Get a diff reviewed locally in under 5 minutes, no cloud services required.

```bash
# 1. Clone and install
git clone https://github.com/<you>/devmind.git
cd devmind
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in at minimum: GROQ_API_KEY, HUGGINGFACE_API_KEY

# 3. Run the test suite to confirm everything's wired correctly
pytest

# 4. Run the agent pipeline against a sample diff
python scripts/index_repo.py --repo ./sample_repo   # one-time local indexing
python -m app.agent.graph --diff sample_diffs/example.diff
```

For the full webhook → queue → Lambda flow, see [Deployment](#deployment).

---

## Configuration

All configuration is via environment variables (see `.env.example` for the complete template):

```bash
# LLM
GROQ_API_KEY=                 # console.groq.com
HUGGINGFACE_API_KEY=          # embeddings via Inference API

# GitHub App
GITHUB_APP_ID=
GITHUB_PRIVATE_KEY=           # PEM key from GitHub App settings
GITHUB_WEBHOOK_SECRET=        # used for HMAC signature validation

# Supabase
SUPABASE_URL=
SUPABASE_ANON_KEY=
DATABASE_URL=                 # postgres://... direct connection, used by SQLAlchemy async

# Upstash
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
QSTASH_URL=
QSTASH_TOKEN=
QSTASH_CURRENT_SIGNING_KEY=
QSTASH_NEXT_SIGNING_KEY=

# App
LAMBDA_FUNCTION_URL=           # called by QStash after enqueue
MAX_COMMENTS_PER_PR=15
MAX_CHUNK_TOKENS=6000
ENVIRONMENT=development        # development | production
```

---

## Project Structure

```
devmind/
├── app/
│   ├── agent/
│   │   ├── state.py               # AgentState TypedDict — shared across all nodes
│   │   ├── graph.py                # LangGraph definition and wiring
│   │   └── nodes/
│   │       ├── diff_parser.py          # pure Python, no LLM
│   │       ├── rag_context.py          # embedding + pgvector retrieval, no LLM
│   │       ├── security_checker.py
│   │       ├── bug_detector.py
│   │       ├── style_checker.py
│   │       ├── comment_synthesizer.py  # dedup, prioritize, cap at 15, no LLM
│   │       └── _finding_parser.py      # shared JSON parsing/validation for all checkers
│   ├── llm/
│   │   └── client.py               # Groq SDK wrapper — retry logic, JSON parsing
│   ├── webhook/
│   │   ├── router.py               # /webhook endpoint
│   │   ├── validator.py            # hmac.compare_digest() signature check
│   │   ├── idempotency.py          # Redis SET NX dedup
│   │   └── qstash_publisher.py
│   ├── github/
│   │   ├── auth.py                 # GitHub App JWT + installation tokens
│   │   ├── diff_fetcher.py
│   │   └── comment_poster.py
│   ├── rag/
│   │   ├── embedder.py             # HuggingFace Inference API wrapper
│   │   ├── indexer.py              # repo indexing
│   │   └── retriever.py            # pgvector similarity search, wired as LangGraph node
│   └── db/
│       ├── models.py                # SQLAlchemy 2.0 async models
│       ├── session.py                # async session helpers
│       └── migrations/               # Alembic
├── processor/
│   └── handler.py                  # AWS Lambda entry point (asyncio.run() wrapper)
├── tests/
│   ├── unit/                        # 292 tests — nodes, validators, idempotency, chunking
│   ├── integration/                 # full LangGraph run with mocked LLM
│   └── eval/                        # eval dataset + metrics runner
├── scripts/
│   ├── index_repo.py
│   └── run_evals.py
├── .github/workflows/ci.yml
├── Dockerfile                       # multi-stage, python:3.12-slim-bookworm
├── railway.toml
├── .dockerignore
├── requirements.txt
├── .env.example
├── pytest.ini
└── README.md
```

> **Note on naming:** the Lambda processor package is named `processor/`, not `lambda/` — `lambda` is a reserved Python keyword and breaks import resolution if used as a package name.

---

## The Agent Pipeline

Six sequential LangGraph nodes operating on a shared, typed `AgentState`:

```
DiffParserNode          → structured DiffChunk objects (no LLM — unidiff parsing)
RAGContextNode          → diff chunks + relevant repo context (no LLM — embed + search)
SecurityCheckerNode     → SQL injection, hardcoded secrets, insecure deserialization,
                            path traversal, auth bypass, XSS
BugDetectorNode         → off-by-ones, bad null checks, wrong operators, race conditions,
                            swallowed exceptions, missing edge cases
StyleCheckerNode        → naming, SRP violations, magic numbers, missing docstrings,
                            dead code, convention drift
CommentSynthesizerNode  → dedup, prioritize (critical > warning > suggestion),
                            cap at 15, format with severity emoji (no LLM)
```

Each checker node calls the Groq SDK directly (not via LangChain's `ChatGroq` wrapper) and outputs strict JSON:

```json
{
  "findings": [
    {
      "file_path": "src/api/routes.py",
      "line_number": 42,
      "severity": "critical",
      "comment": "SQL injection risk: user input directly interpolated into query string."
    }
  ]
}
```

A single `_finding_parser.py` module handles JSON parsing/validation for all three checkers — extracted specifically because "parse untrusted LLM output" (malformed JSON, missing fields, wrong types, markdown fences) is subtle enough that duplicating it three times would mean fixing the same bug three times.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Webhook → queue → Lambda, not synchronous** | GitHub's ~10s SLA vs. 15–60s real review time makes async a correctness requirement, not an optimization |
| **Redis `SET NX` over GET-then-SET** | Atomic claim avoids the check-then-act race where two near-simultaneous webhook deliveries both pass a naive check |
| **Fail-open on Redis, fail-closed on queue failures** | Deliberate asymmetry: a duplicate review is annoying; a silently dropped PR is broken product |
| **RAG over full codebase, not just the diff** | A diff alone can't reveal violations of conventions defined elsewhere (e.g., an auth pattern used by every other route) |
| **pgvector over a dedicated vector DB** | Free tier, one service instead of two, sufficient at portfolio scale — explicit migration trigger: retrieval latency or embedding count |
| **Direct Groq SDK over LangChain's `ChatGroq`** | Direct token-usage visibility and full control over JSON error handling at the exact failure point |
| **Separate node per concern (security/bug/style)** | Independent testability and failure isolation — a bulkhead pattern applied to LLM pipeline nodes |
| **`hmac.compare_digest()` over `==`** | Closes a timing side-channel that a naive short-circuiting string comparison leaves open |
| **`python:3.12-slim-bookworm` over Alpine** | Avoids musl libc wheel incompatibilities in the multi-stage Docker build |
| **HuggingFace Inference API over local embeddings** | A local `sentence-transformers` + PyTorch install would blow past Lambda's 250MB deployment package limit |
| **Comment cap at 15, prioritized** | Noise destroys trust — a PR flooded with 40 minor suggestions gets ignored wholesale |

---

## Testing

**292 passing unit tests** across nodes, validators, idempotency logic, and chunking — built without needing live LLM, GitHub, Postgres, or Redis dependencies in CI.

```bash
pytest                          # full suite
pytest tests/unit -v            # fast, deterministic, no external deps
pytest tests/integration        # full LangGraph run, LLM mocked
pytest --cov=app                # coverage report
```

Key testing lessons baked into the suite:

- **Mock at the import site, not the definition site.** `call_llm_json` is patched at `app.agent.nodes.security_checker.call_llm_json`, not `app.llm.client.call_llm_json` — patching the source module does nothing to a node's already-imported local binding.
- **Pure-Python nodes need no mocks.** `DiffParserNode` and `CommentSynthesizerNode` make no LLM calls, so dedup logic, the 15-comment cap, and severity ordering are tested as ordinary deterministic functions.
- **Idempotency is tested in isolation.** Redis key generation and `SET NX` semantics are verified against a real or fake Redis without needing an actual webhook delivery.

Representative test cases:

| Area | Case |
|---|---|
| Diff parser | Empty diff, binary-only diff, `package-lock.json` filtering, 2000-line file chunked with overlap |
| Idempotency | Same key twice → second call short-circuits; new SHA on same PR → treated as new event |
| LLM output parsing | Valid JSON, invalid JSON → graceful empty-findings fallback, extra fields ignored |
| Comment synthesizer | 30 findings → capped at 15; duplicate same-line findings → deduped; critical sorts first |
| Load | 20 concurrent webhooks all enqueued; QStash retry processes an event exactly once |

---

## Deployment

1. **Supabase** — create all tables (`repositories`, `reviews`, `review_comments`, `dlq_events`, `code_embeddings`) and enable the `pgvector` extension. Run Alembic migrations.
2. **Upstash** — provision a Redis instance (idempotency + caching) and a QStash queue (message delivery + retries).
3. **Railway** — deploy the FastAPI webhook receiver using `railway.toml`; this is the public URL GitHub calls.
4. **AWS Lambda** — deploy `processor/handler.py` with a Function URL; this is the endpoint QStash calls after enqueue.
5. **GitHub App** — create the App, configure the webhook URL (pointing at Railway), generate a private key, and note the webhook secret.
6. **CI** — `.github/workflows/ci.yml` runs the full test suite on every push.

```bash
# Build and run locally with Docker
docker build -t devmind .
docker run --env-file .env -p 8000:8000 devmind
```

---

## Eval Results

DevMind is evaluated against a hand-labeled dataset of real open-source PRs with known human review comments, categorized by type (security/bug/style) and severity.

| Metric | Target |
|---|---|
| Security precision | > 70% |
| Bug detection recall | > 50% |
| Avg latency (p50), webhook → posted comment | < 30s |
| False positive rate | < 20% |

```bash
python scripts/run_evals.py   # runs the full pipeline against tests/eval/eval_dataset
```

---

## What Breaks at Scale

| Component | Breaks when... | Production fix |
|---|---|---|
| Groq API | Rate limits hit under concurrent PR bursts | Paid tier, request queuing/throttling, fallback model |
| pgvector | Embedding count reaches millions; query latency rises | Migrate to Qdrant/Pinecone; consider hierarchical retrieval |
| QStash | 500 msg/day free tier exhausted | Move to SQS/Kafka for higher throughput and finer-grained DLQ control |
| Lambda | Cold starts under bursty, infrequent traffic | Provisioned concurrency, or a long-running worker pool |
| DB connections | Concurrent Lambda invocations exhaust the Postgres connection limit | Connection pooler (PgBouncer / Supabase pooler) |
| Comment quality | Precision drops as codebase grows/diversifies | Expand eval dataset, add human-in-the-loop feedback |

---

## Roadmap

- [ ] Incremental re-indexing (only re-embed changed files on new commits, not the whole repo)
- [ ] Relevance threshold on RAG retrieval to avoid feeding low-quality "related" chunks to checkers
- [ ] Prototype hierarchical retrieval (PageIndex-style) as a comparative alternative to flat pgvector search
- [ ] Per-node timeout with graceful degradation (partial findings) instead of full-invocation failure
- [ ] Human-in-the-loop feedback loop to refine checker prompts over time
- [ ] Provisioned concurrency for Lambda to reduce cold-start latency variance

---

## Contributing

Issues and PRs are welcome. Before opening a PR:

1. Run `pytest` locally — all 292 tests must pass.
2. Add unit tests for any new node or module (see `tests/unit` for the existing patterns).
3. Keep checker-node system prompts JSON-only — no preamble, no markdown fences — since `_finding_parser.py` expects strict JSON.
4. Run `docker build -t devmind .` to confirm the multi-stage build still succeeds.

---

*Built by Ayush Kaul — a distributed-systems portfolio project covering event-driven architecture, idempotent processing, RAG retrieval, and multi-agent LLM orchestration.*
