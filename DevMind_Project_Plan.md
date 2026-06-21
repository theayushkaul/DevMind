# DevMind — AI-Powered GitHub Code Review Agent
## Complete Project Plan & Context Document

> **Purpose of this document:** This is a living reference document for building DevMind. It is structured to be fed directly into Claude AI as context for any coding session, architecture decision, or debugging task. Always include this document at the start of any Claude conversation related to this project.

---

## 1. Project Overview

### What is DevMind?
DevMind is a GitHub-integrated AI agent that automatically reviews pull requests the moment they are opened or updated. It reads the code diff, understands the broader codebase context using RAG (Retrieval-Augmented Generation), and posts structured, actionable review comments directly on the PR — mimicking what a senior engineer would flag.

### Core Value Proposition
- Zero human intervention needed for initial code review pass
- Catches bugs, security issues, code smells, and style violations instantly
- Context-aware: understands project conventions, not just the isolated diff
- Fully open-source LLM powered — zero API cost during development and demo

### Target Audience (for resume/portfolio)
- Interviewers at product companies evaluating SWE candidates
- Demonstrates: agentic AI design, backend infra, cloud deployment, system design thinking, testing discipline

---

## 2. Technology Stack (Zero Cost)

### LLM
| Component | Choice | Model ID | Reason |
|---|---|---|---|
| Primary LLM | Groq API | `qwen-2.5-coder-32b` | Purpose-built for code, 128K context, ~535 tok/sec, free tier |
| Fallback LLM | Google Gemini API | `gemini-1.5-flash` | Free tier backup if Groq rate limits hit |
| Embeddings | HuggingFace Inference API | `BAAI/bge-small-en-v1.5` | Free, excellent quality, small payload |
| Local Dev LLM | Ollama | `qwen2.5-coder:7b` | Zero API calls during development |

**Groq Free Tier Limits:** 14,400 requests/day, 6,000 tokens/min — sufficient for a portfolio project with active usage.

### Infrastructure (All Free Tier)
| Component | Service | Free Limit |
|---|---|---|
| API / Webhook Server | Railway | 500 hrs/month |
| Serverless Processing | AWS Lambda | 1M requests/month (forever free) |
| Database | Supabase (PostgreSQL) | 500MB, 2 projects |
| Vector Store | Supabase pgvector | Included with Postgres |
| Cache + Queue | Upstash Redis | 10,000 commands/day |
| Message Queue | Upstash QStash | 500 messages/day |
| Version Control | GitHub | Free |
| Monitoring | Betterstack | Free tier (logs + uptime) |

### Application Stack
- **Language:** Python 3.11+
- **Framework:** FastAPI (webhook receiver + REST API)
- **Agent Framework:** LangChain + LangGraph (stateful multi-node agent)
- **GitHub Integration:** PyGithub + GitHub Webhooks
- **ORM:** SQLAlchemy + asyncpg
- **Testing:** pytest, pytest-asyncio, pytest-mock
- **Containerization:** Docker (local dev), Railway (prod)
- **Version Control:** Git (GitHub)

---

## 3. System Architecture

### High-Level Data Flow

```
GitHub PR Opened/Updated
        │
        ▼
[Railway] FastAPI Webhook Receiver
        │  (validates GitHub signature, idempotency check)
        │
        ▼
[Upstash QStash] Message Queue
        │  (buffers burst events, retries on failure)
        │
        ▼
[AWS Lambda] PR Processor Function
        │
        ├──► [Supabase pgvector] RAG: Fetch repo context
        │         (embeddings of repo codebase)
        │
        ├──► [Groq API] qwen-2.5-coder-32b
        │         LangGraph Agent:
        │         Node 1: Diff Parser
        │         Node 2: Security Checker
        │         Node 3: Logic Bug Detector
        │         Node 4: Style/Convention Checker
        │         Node 5: Comment Synthesizer
        │
        ▼
[GitHub API] Post Review Comments on PR
        │
        ▼
[Supabase Postgres] Store review history, metadata
[Upstash Redis] Cache repo embeddings, dedup events
```

### Component Responsibilities

#### 1. Webhook Receiver (FastAPI on Railway)
- Receives `pull_request` events from GitHub via webhook
- Validates `X-Hub-Signature-256` HMAC header — reject invalid requests
- Checks Redis for duplicate event IDs (idempotency key = `{repo}:{pr_number}:{head_sha}`)
- Enqueues valid events to QStash
- Returns `200 OK` immediately (GitHub expects < 10s response)

#### 2. Message Queue (Upstash QStash)
- Decouples webhook receiver from processing (burst protection)
- Provides automatic retry with exponential backoff on Lambda failures
- Dead Letter Queue (DLQ): events that fail 3x go to a DLQ table in Supabase for manual inspection
- QStash calls Lambda via HTTP — Lambda URL exposed via Function URL

#### 3. PR Processor (AWS Lambda)
- Triggered by QStash HTTP call
- Fetches full PR diff from GitHub API
- Chunks diff into reviewable segments (see Chunking Strategy below)
- Retrieves repo context from pgvector (RAG)
- Runs LangGraph agent pipeline
- Posts structured comments back to GitHub
- Logs review to Supabase

#### 4. LangGraph Agent (Inside Lambda)
Five sequential nodes, each independently testable:

```
[DiffParserNode]
    ↓ structured diff chunks
[RAGContextNode]
    ↓ diff chunks + relevant repo context
[SecurityCheckerNode]
    ↓ security findings
[LogicBugDetectorNode]
    ↓ logic findings
[StyleCheckerNode]
    ↓ style findings
[CommentSynthesizerNode]
    ↓ final structured review comments
[GitHubPosterNode]
```

#### 5. RAG Pipeline (Supabase pgvector)
- **Indexing:** On first PR from a repo, clone repo and embed all `.py`, `.js`, `.ts`, `.java` files using `BAAI/bge-small-en-v1.5`
- **Chunking for indexing:** 512 token chunks with 64 token overlap
- **Retrieval:** For each diff chunk, embed it and fetch top-5 semantically similar code segments
- **Cache:** Store repo embeddings in Redis (TTL: 24hrs) to avoid re-embedding on every PR

---

## 4. Data Models

### Supabase Tables

```sql
-- Repositories tracked by DevMind
CREATE TABLE repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    github_repo_id BIGINT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,            -- e.g. "ayushkaul/my-repo"
    installation_id BIGINT,
    indexed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- PR review records
CREATE TABLE reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID REFERENCES repositories(id),
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    status TEXT NOT NULL,               -- 'queued' | 'processing' | 'completed' | 'failed'
    comments_posted INTEGER DEFAULT 0,
    tokens_used INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    UNIQUE(repo_id, pr_number, head_sha) -- idempotency constraint
);

-- Individual review comments
CREATE TABLE review_comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID REFERENCES reviews(id),
    file_path TEXT NOT NULL,
    line_number INTEGER,
    category TEXT NOT NULL,             -- 'security' | 'bug' | 'style' | 'performance'
    severity TEXT NOT NULL,             -- 'critical' | 'warning' | 'suggestion'
    comment_body TEXT NOT NULL,
    github_comment_id BIGINT,           -- ID returned by GitHub API
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dead letter queue for failed processing
CREATE TABLE dlq_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payload JSONB NOT NULL,
    failure_reason TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector embeddings (pgvector)
CREATE TABLE code_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID REFERENCES repositories(id),
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384),              -- BAAI/bge-small-en-v1.5 dimension
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX ON code_embeddings USING ivfflat (embedding vector_cosine_ops);
```

---

## 5. LangGraph Agent — Detailed Node Specs

### Agent State Schema
```python
from typing import TypedDict, List, Optional
from dataclasses import dataclass

@dataclass
class DiffChunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str

@dataclass
class ReviewFinding:
    file_path: str
    line_number: int
    category: str       # 'security' | 'bug' | 'style' | 'performance'
    severity: str       # 'critical' | 'warning' | 'suggestion'
    comment: str
    confidence: float   # 0.0 - 1.0

class AgentState(TypedDict):
    raw_diff: str
    repo_full_name: str
    pr_number: int
    diff_chunks: List[DiffChunk]
    repo_context: dict              # chunk -> relevant code snippets
    security_findings: List[ReviewFinding]
    bug_findings: List[ReviewFinding]
    style_findings: List[ReviewFinding]
    final_comments: List[ReviewFinding]
    error: Optional[str]
```

### Node Prompts

#### DiffParserNode
- Input: raw unified diff string
- Task: Parse into structured `DiffChunk` objects, detect programming language per file, filter out auto-generated files (e.g. `package-lock.json`, `*.min.js`)
- Output: `List[DiffChunk]`
- **No LLM call** — pure Python parsing with `unidiff` library

#### RAGContextNode
- Input: `List[DiffChunk]`
- Task: For each chunk, embed it and query pgvector for top-5 related code segments from the repo
- Output: dict mapping chunk → retrieved snippets
- **No LLM call** — embedding + vector search

#### SecurityCheckerNode — System Prompt
```
You are a senior security engineer performing a focused security review.
Analyze the following code diff and retrieved codebase context.
Flag ONLY confirmed or highly probable security issues — do not guess.

Categories to check:
- SQL injection (raw string queries, f-strings in DB calls)
- Hardcoded secrets (API keys, passwords, tokens in code)
- Insecure deserialization (pickle.loads on untrusted input)
- Path traversal (user-controlled file paths)
- Authentication bypass (missing auth checks on new routes)
- XSS in template rendering

Output ONLY valid JSON — no preamble, no explanation outside JSON:
{
  "findings": [
    {
      "file_path": "src/api/routes.py",
      "line_number": 42,
      "severity": "critical",
      "comment": "SQL injection risk: user input directly interpolated into query string. Use parameterized queries."
    }
  ]
}
If no issues found, return: {"findings": []}
```

#### LogicBugDetectorNode — System Prompt
```
You are a senior software engineer reviewing a code diff for logical bugs.
You also have context from related files in the same codebase.

Look for:
- Off-by-one errors in loops and array indexing
- Incorrect null/None checks (checking wrong variable, missing null check)
- Wrong operator (= vs ==, & vs &&, etc.)
- Race conditions in async/concurrent code
- Incorrect error handling (catching Exception silently, swallowing errors)
- Missing edge cases (empty list, zero division, negative input)
- API contract violations (calling function with wrong argument types/order)

Output ONLY valid JSON:
{
  "findings": [
    {
      "file_path": "src/processor.py",
      "line_number": 87,
      "severity": "warning",
      "comment": "Off-by-one: loop runs to len(items) but items is 0-indexed. Last element is never processed."
    }
  ]
}
If no issues found, return: {"findings": []}
```

#### StyleCheckerNode — System Prompt
```
You are a code reviewer focused on maintainability and style.
Review this diff against the project conventions visible in the codebase context.

Check for:
- Naming inconsistencies (camelCase vs snake_case mixed, unclear variable names)
- Functions doing more than one thing (violates SRP)
- Magic numbers/strings without constants
- Missing docstrings on public functions/classes
- Overly complex conditionals that should be extracted
- Dead code (variables assigned but never used)
- Inconsistency with patterns used in the rest of the codebase

Keep suggestions constructive and actionable.
Output ONLY valid JSON:
{
  "findings": [
    {
      "file_path": "src/utils.py",
      "line_number": 23,
      "severity": "suggestion",
      "comment": "Magic number 86400 should be extracted to a constant: SECONDS_IN_DAY = 86400"
    }
  ]
}
If no issues found, return: {"findings": []}
```

#### CommentSynthesizerNode
- Merges all findings from the three checker nodes
- Deduplicates overlapping comments on the same line
- Prioritizes: critical > warning > suggestion
- Caps total comments at 15 per PR (avoid noise)
- Formats each comment with severity emoji prefix:
  - 🚨 Critical
  - ⚠️ Warning
  - 💡 Suggestion
- **No LLM call** — pure Python logic

---

## 6. Diff Chunking Strategy

Large PRs can exceed the LLM context window. Strategy:

```python
MAX_CHUNK_TOKENS = 6000     # Leave room for system prompt + response
OVERLAP_TOKENS = 200        # Context continuity across chunks

def chunk_diff(raw_diff: str) -> List[DiffChunk]:
    # 1. Split by file (each hunk is a natural chunk boundary)
    # 2. If a single file diff > MAX_CHUNK_TOKENS, split by function/class boundary
    # 3. If still too large, split by line count with overlap
    # 4. Filter: skip files matching patterns:
    #    - *.lock, *.min.js, *.min.css, package-lock.json
    #    - Auto-generated files (look for "DO NOT EDIT" header)
    #    - Binary files
    pass
```

**Interview talking point:** This chunking strategy and the trade-offs involved (losing cross-chunk context vs. exceeding token limits) is an excellent system design discussion point.

---

## 7. Key System Design Decisions & Rationale

### Decision 1: Webhook → Queue → Lambda (not synchronous)
**Why:** GitHub webhook expects a response in < 10 seconds. Code review takes 15-60 seconds. Decoupling via queue means we respond immediately and process async. Also handles traffic spikes (multiple PRs opened at once).

### Decision 2: Idempotency via Redis
**Why:** GitHub retries webhooks on no-response. Without idempotency, one PR gets reviewed multiple times. Key: `{repo_full_name}:{pr_number}:{head_sha}` — unique per PR state.

### Decision 3: RAG over full codebase (not just diff)
**Why:** A diff in isolation lacks context. If a new function calls `self.db.execute(query)`, the reviewer needs to know what `db` is and whether it uses parameterized queries — that's in another file. RAG retrieves that context.

### Decision 4: pgvector over dedicated vector DB
**Why:** Free tier. Supabase gives us Postgres + pgvector in one service. For a portfolio project with moderate load, this is sufficient. Good interview answer: "For production scale I'd migrate to Pinecone or Qdrant with dedicated infra."

### Decision 5: Separate Lambda nodes per concern
**Why:** Each checker (security, bugs, style) is an independent Lambda node in LangGraph — independently testable, independently failure-isolated. If the security checker errors, the rest of the review still proceeds.

### Decision 6: Comment cap at 15 per PR
**Why:** Noise is worse than silence. A PR flooded with 40 minor suggestions will be ignored. 15 prioritized, high-confidence comments are actionable. This also controls LLM token cost.

---

## 8. Week-by-Week Build Plan

### Week 1-2: Foundation & Local Dev Setup
**Goal:** End-to-end skeleton running locally with mocked components

- [ ] Set up Python project structure (FastAPI + LangChain + LangGraph)
- [ ] Install and run Ollama locally with `qwen2.5-coder:7b`
- [ ] Build minimal FastAPI webhook receiver (no queue yet)
- [ ] Parse GitHub webhook payload, extract PR diff
- [ ] Write `DiffParserNode` — pure Python, no LLM
- [ ] Write one LangGraph node (SecurityCheckerNode) with local Ollama
- [ ] Print findings to console (no GitHub posting yet)
- [ ] Write unit tests for DiffParserNode

**Milestone:** Given a hardcoded diff, agent prints security findings to terminal.

---

### Week 3-4: Full Agent Pipeline
**Goal:** All 5 nodes working end-to-end locally

- [ ] Build `RAGContextNode` with HuggingFace embeddings + local Chroma (temporary)
- [ ] Build `LogicBugDetectorNode`
- [ ] Build `StyleCheckerNode`
- [ ] Build `CommentSynthesizerNode` (dedup + prioritize + format)
- [ ] Wire all nodes in LangGraph with proper state passing
- [ ] Switch from Ollama to Groq API (`qwen-2.5-coder-32b`)
- [ ] Write unit tests for each node independently
- [ ] Write integration test: full pipeline on a sample diff

**Milestone:** Full agent pipeline runs end-to-end on a sample diff and outputs formatted comments.

---

### Week 5-6: GitHub Integration
**Goal:** Real PRs trigger real review comments

- [ ] Create GitHub App (not OAuth app — Apps have better rate limits)
- [ ] Set up webhook endpoint with signature validation
- [ ] Implement GitHub API client (PyGithub) to post inline PR comments
- [ ] Test with a dummy repo: open PR → see comments appear
- [ ] Handle GitHub API rate limiting (exponential backoff)
- [ ] Handle edge cases: empty diff, binary files, draft PRs

**Milestone:** Open a real PR on a test repo → DevMind posts real review comments.

---

### Week 7-8: Infrastructure & Persistence
**Goal:** Production-grade infra on free tier

- [ ] Set up Supabase: create all tables + pgvector extension
- [ ] Migrate RAG from local Chroma to Supabase pgvector
- [ ] Set up Upstash Redis (idempotency + caching)
- [ ] Set up Upstash QStash (message queue)
- [ ] Deploy FastAPI to Railway (webhook receiver)
- [ ] Deploy processor to AWS Lambda (Function URL for QStash)
- [ ] Configure GitHub webhook to point to Railway URL
- [ ] Implement DLQ: failed events → Supabase `dlq_events` table

**Milestone:** End-to-end system running on free cloud infra. Open PR → get review within 60 seconds.

---

### Week 9-10: Testing, Evals & Hardening
**Goal:** Prove it works reliably, build eval dataset

- [ ] Build eval dataset: 50 open-source PRs with known bugs (manually labelled)
- [ ] Run eval: measure precision/recall per category (security/bug/style)
- [ ] Latency benchmark: webhook received → first comment posted
- [ ] Load test: simulate 20 concurrent PR webhooks
- [ ] Test idempotency: send same webhook 5 times, assert only 1 review posted
- [ ] Test DLQ: kill Lambda mid-processing, assert event lands in DLQ
- [ ] Test chunking: generate a 2000-line diff, assert no context window errors
- [ ] Cost tracking: log tokens used per review, calculate average cost

**Milestone:** Documented eval results + benchmark numbers to put in README.

---

### Week 11-12: Polish, README & Demo Prep
**Goal:** Portfolio-ready project

- [ ] Write architecture diagram (Mermaid or Excalidraw)
- [ ] Write comprehensive README with:
  - Problem statement
  - Architecture diagram
  - Tech stack table
  - Key design decisions
  - Eval results (precision/recall numbers)
  - Latency benchmarks
  - How to run locally
  - How to deploy
- [ ] Record a 2-minute demo video (open PR → watch comments appear live)
- [ ] Clean up code: type hints everywhere, docstrings on all public functions
- [ ] Add GitHub Actions CI: run tests on every push
- [ ] Set up Betterstack for uptime monitoring

**Milestone:** Project is interview-ready. Can demo live in any interview.

---

## 9. Testing Strategy

### Unit Tests (pytest)

```
tests/
├── unit/
│   ├── test_diff_parser.py          # DiffParserNode — no mocks needed
│   ├── test_comment_synthesizer.py  # dedup logic, priority sorting, cap at 15
│   ├── test_idempotency.py          # Redis key generation logic
│   ├── test_chunking.py             # edge cases: empty diff, huge file, binary
│   └── test_webhook_validator.py    # HMAC signature validation
├── integration/
│   ├── test_agent_pipeline.py       # full LangGraph run with mocked LLM
│   ├── test_github_posting.py       # mock GitHub API, assert correct calls
│   └── test_rag_retrieval.py        # embed + retrieve from test pgvector instance
└── eval/
    ├── eval_dataset/                # 50 PR diffs with ground truth labels
    ├── run_evals.py                 # runs agent on all 50, computes metrics
    └── eval_results.json            # stores results for README
```

### Key Test Cases to Write

**DiffParser:**
- Empty diff → returns empty list
- Diff with only binary files → all filtered out
- Diff with `package-lock.json` → filtered out
- 2000-line single-file diff → chunked correctly with overlap

**Idempotency:**
- Same `{repo}:{pr}:{sha}` key twice → second call returns early, no queue enqueue
- Different SHA same PR number → treated as new event (PR updated)

**LLM Output Parsing:**
- Valid JSON → parsed correctly
- Invalid JSON response from LLM → graceful fallback, log error, return empty findings
- JSON with extra fields → ignored without crashing
- Empty findings list → no comments posted

**Comment Synthesizer:**
- 30 findings → capped at 15
- Duplicate findings on same line → deduped to 1
- Mixed severities → critical first

**Load / Stress:**
- 20 concurrent webhooks → all enqueued, none dropped
- QStash retry on Lambda timeout → event processed exactly once

---

## 10. Eval Framework

### Eval Dataset Construction
1. Find 50 open-source PRs on GitHub that have human review comments
2. Record the diff + the human reviewer's actual comments (ground truth)
3. Label each ground truth comment with category (security/bug/style) and severity

### Metrics
- **Precision:** Of all comments DevMind posts, what % are valid findings?
- **Recall:** Of all real issues in ground truth, what % did DevMind catch?
- **False Positive Rate:** Comments posted on code that has no real issue
- **Latency:** webhook receipt → last comment posted (p50, p95)
- **Token cost per review:** avg tokens used × Groq pricing

### Target Numbers (aspirational)
| Metric | Target |
|---|---|
| Security precision | > 70% |
| Bug detection recall | > 50% |
| Avg latency (p50) | < 30 seconds |
| False positive rate | < 20% |

---

## 11. Directory Structure

```
devmind/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── webhook/
│   │   ├── router.py               # /webhook endpoint
│   │   ├── validator.py            # HMAC signature validation
│   │   └── idempotency.py          # Redis dedup logic
│   ├── agent/
│   │   ├── graph.py                # LangGraph definition
│   │   ├── state.py                # AgentState TypedDict
│   │   └── nodes/
│   │       ├── diff_parser.py
│   │       ├── rag_context.py
│   │       ├── security_checker.py
│   │       ├── bug_detector.py
│   │       ├── style_checker.py
│   │       └── comment_synthesizer.py
│   ├── rag/
│   │   ├── embedder.py             # HuggingFace embedding wrapper
│   │   ├── indexer.py              # Repo codebase indexing
│   │   └── retriever.py            # pgvector similarity search
│   ├── github/
│   │   ├── client.py               # PyGithub wrapper
│   │   ├── diff_fetcher.py         # Fetch PR diffs
│   │   └── comment_poster.py       # Post inline review comments
│   ├── db/
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── session.py              # Async DB session
│   │   └── migrations/             # Alembic migrations
│   └── config.py                   # Environment variables, settings
├── lambda/
│   └── handler.py                  # AWS Lambda entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
├── scripts/
│   ├── index_repo.py               # One-off: index a repo's codebase
│   └── run_evals.py
├── .github/
│   └── workflows/
│       └── ci.yml                  # Run tests on push
├── docker-compose.yml              # Local dev: Postgres + Redis
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 12. Environment Variables

```bash
# LLM
GROQ_API_KEY=                       # From console.groq.com
GEMINI_API_KEY=                     # Fallback LLM
HUGGINGFACE_API_KEY=                # For embeddings

# GitHub App
GITHUB_APP_ID=
GITHUB_PRIVATE_KEY=                 # PEM key from GitHub App settings
GITHUB_WEBHOOK_SECRET=              # Random string, set in GitHub App webhook config

# Supabase
SUPABASE_URL=
SUPABASE_ANON_KEY=
DATABASE_URL=                       # postgres://... (direct connection)

# Upstash
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
QSTASH_URL=
QSTASH_TOKEN=
QSTASH_CURRENT_SIGNING_KEY=
QSTASH_NEXT_SIGNING_KEY=

# App Config
LAMBDA_FUNCTION_URL=                # AWS Lambda Function URL (called by QStash)
MAX_COMMENTS_PER_PR=15
MAX_CHUNK_TOKENS=6000
ENVIRONMENT=development             # development | production
```

---

## 13. How to Use This Document with Claude AI

When starting any Claude conversation about DevMind, paste this document and then ask your specific question. Useful prompts:

### For coding sessions:
```
[paste this document]

I'm working on the SecurityCheckerNode (app/agent/nodes/security_checker.py).
Here's what I have so far: [paste code]
Help me implement the LLM call with proper JSON output parsing and error handling.
```

### For debugging:
```
[paste this document]

I'm getting this error when running the LangGraph pipeline: [paste error]
The error happens in the RAGContextNode. Here's the node code: [paste code]
```

### For architecture decisions:
```
[paste this document]

I'm deciding between using QStash vs a simple Redis queue for the message queue.
Given my free tier constraints and the architecture above, which should I use and why?
```

### For test writing:
```
[paste this document]

Write pytest unit tests for the CommentSynthesizerNode.
The node's code is: [paste code]
Focus on: deduplication logic, priority sorting, and the 15-comment cap.
```

---

## 14. Interview Talking Points

When asked about this project in an interview, be ready to discuss:

1. **Why async webhook processing?** GitHub expects < 10s response. Review takes 30-60s. Queue decouples them.

2. **How do you handle duplicate webhooks?** Redis idempotency key `{repo}:{pr}:{sha}`. GitHub retries on no-response — without this, same PR gets reviewed multiple times.

3. **Why RAG instead of just sending the diff?** A diff without codebase context misses patterns. E.g., if a new route skips auth, you only know it's wrong if you've seen how other routes implement auth — that's in another file.

4. **How do you handle PRs larger than the context window?** Chunk by file boundary first, then by function/class boundary, then by token count with overlap. Each chunk is reviewed independently and findings merged.

5. **Why pgvector over Pinecone?** Free tier, single service (no extra dependency), sufficient for portfolio scale. In production I'd evaluate Qdrant or Pinecone based on query latency at scale.

6. **How do you measure if the reviews are actually good?** Built an eval dataset of 50 real PRs with known bugs. Measure precision/recall per category. Also track false positive rate — too many wrong comments destroys trust.

7. **What breaks at scale?** Groq rate limits, pgvector query latency at 1M+ embeddings, QStash message limits. Would move to paid Groq tier, dedicated vector DB, and Kafka for high scale.

---

*Last updated: May 2026 | Built by Ayush Kaul*
