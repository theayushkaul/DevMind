"""
app/agent/nodes/security_checker.py
─────────────────────────────────────
The SecurityCheckerNode: the first LLM-dependent node in the LangGraph pipeline.

Reads:  state["diff_chunks"]   (from DiffParserNode)
        state["repo_context"]  (from RAGContextNode, optional — may be {})
Writes: state["security_findings"]
        state["tokens_used"]   (incremented, not overwritten)
        state["error"]         (on failure, without raising)

Design decisions:
- One LLM call PER CHUNK, not one call for the whole diff. This keeps each
  call well within the context window regardless of PR size, and means a
  malformed response on chunk 3 of 5 doesn't lose findings from the other 4 —
  each chunk's call/parse/failure is isolated.
- All LLM interaction goes through app.llm.client.call_llm_json(). This node
  owns ONLY the system prompt and the security-specific confidence mapping.
  Response parsing (JSON shape validation, line-number coercion, severity
  validation) is shared across all three checker nodes via
  app.agent.nodes._finding_parser — see that module for why.
- A chunk that fails to parse contributes ZERO findings and is logged, but
  does NOT abort the node. One bad chunk shouldn't blank out findings from
  every other chunk in the same PR.
- Confidence score: the system prompt below (per the project plan) does not
  ask the model to self-report a confidence float — only severity. We derive
  confidence deterministically from severity as a starting heuristic
  (critical findings are usually unambiguous; suggestions are softer
  judgment calls). This can be replaced later with a model-reported
  confidence if the prompt is extended.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.agent.nodes._finding_parser import build_user_content, extract_findings
from app.agent.state import AgentState, DiffChunk, ReviewFinding
from app.llm.client import LLMCallResult, call_llm_json

logger = logging.getLogger(__name__)

NODE_NAME = "security_checker"
NODE_LABEL = "SecurityCheckerNode"

# ---------------------------------------------------------------------------
# System prompt — verbatim from the DevMind project plan, Section 5
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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
"""

# Heuristic confidence-by-severity, used because the prompt above does not
# elicit a model-reported confidence score (see module docstring).
CONFIDENCE_BY_SEVERITY: Dict[str, float] = {
    "critical": 0.9,
    "warning": 0.75,
    "suggestion": 0.6,
}


# ---------------------------------------------------------------------------
# Public node entry point
# ---------------------------------------------------------------------------

def run(state: AgentState) -> AgentState:
    """
    LangGraph node entry point for the security checker.

    Iterates every diff chunk, calls the LLM once per chunk, and accumulates
    findings. Designed to never raise — any per-chunk failure is logged and
    skipped so the rest of the review can still complete.
    """
    diff_chunks: List[DiffChunk] = state.get("diff_chunks", [])
    repo_context: Dict[str, List[str]] = state.get("repo_context", {})

    if not diff_chunks:
        logger.info("%s: no diff chunks to review, skipping", NODE_LABEL)
        return {"security_findings": []}  # type: ignore[return-value]

    all_findings: List[ReviewFinding] = []
    tokens_used = state.get("tokens_used", 0)
    chunk_errors: List[str] = []

    for chunk in diff_chunks:
        result = _review_chunk(chunk, repo_context)
        tokens_used += result.total_tokens

        if not result.success:
            logger.warning(
                "%s: chunk %s failed: %s", NODE_LABEL, chunk.dedup_key, result.error,
            )
            chunk_errors.append(f"{chunk.dedup_key}: {result.error}")
            continue

        findings = extract_findings(
            result.data, chunk,
            category="security",
            source_node=NODE_NAME,
            confidence_by_severity=CONFIDENCE_BY_SEVERITY,
            node_label=NODE_LABEL,
        )
        all_findings.extend(findings)

    logger.info(
        "%s: reviewed %d chunks, found %d issues, %d chunk failures",
        NODE_LABEL, len(diff_chunks), len(all_findings), len(chunk_errors),
    )

    update: AgentState = {  # type: ignore[assignment]
        "security_findings": all_findings,
        "tokens_used": tokens_used,
    }

    # Only surface an error if EVERY chunk failed — partial failure is
    # expected and tolerable, total failure suggests something systemic
    # (e.g. GROQ_API_KEY misconfigured) and is worth flagging upstream.
    if chunk_errors and len(chunk_errors) == len(diff_chunks):
        update["error"] = (
            f"{NODE_LABEL}: all {len(diff_chunks)} chunks failed. "
            f"First error: {chunk_errors[0]}"
        )

    return update


# ---------------------------------------------------------------------------
# Per-chunk review
# ---------------------------------------------------------------------------

def _review_chunk(
    chunk: DiffChunk,
    repo_context: Dict[str, List[str]],
) -> LLMCallResult:
    """
    Build the user prompt for a single chunk and call the LLM.

    Separated from run() so it can be unit tested directly with a single
    chunk, without needing a full AgentState.
    """
    user_content = build_user_content(chunk, repo_context)
    return call_llm_json(SYSTEM_PROMPT, user_content)
