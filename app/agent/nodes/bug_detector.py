"""
app/agent/nodes/bug_detector.py
─────────────────────────────────
The LogicBugDetectorNode: flags logical bugs in the diff.

Reads:  state["diff_chunks"]   (from DiffParserNode)
        state["repo_context"]  (from RAGContextNode, optional — may be {})
Writes: state["bug_findings"]
        state["tokens_used"]   (incremented, not overwritten)
        state["error"]         (on failure, without raising)

This node is structurally identical to SecurityCheckerNode — same per-chunk
LLM call pattern, same shared response parsing via _finding_parser. The only
things that differ between the three checker nodes are:
    1. The system prompt (the actual domain expertise)
    2. The `category` stamped onto each ReviewFinding ("bug" here)
    3. Possibly the confidence-by-severity mapping, if a node's findings
       warrant different calibration (kept identical to SecurityChecker's
       for now — no evidence yet that bug findings need different confidence
       weighting than security findings of the same nominal severity)

See app/agent/nodes/_finding_parser.py for why response parsing isn't
duplicated here, and app/agent/nodes/security_checker.py for the more
heavily-commented version of this same pattern.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.agent.nodes._finding_parser import build_user_content, extract_findings
from app.agent.state import AgentState, DiffChunk, ReviewFinding
from app.llm.client import LLMCallResult, call_llm_json

logger = logging.getLogger(__name__)

NODE_NAME = "bug_detector"
NODE_LABEL = "LogicBugDetectorNode"

# ---------------------------------------------------------------------------
# System prompt — verbatim from the DevMind project plan, Section 5
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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
"""

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
    LangGraph node entry point for the logic bug detector.

    Iterates every diff chunk, calls the LLM once per chunk, and accumulates
    findings. Never raises — a failed chunk is logged and skipped.
    """
    diff_chunks: List[DiffChunk] = state.get("diff_chunks", [])
    repo_context: Dict[str, List[str]] = state.get("repo_context", {})

    if not diff_chunks:
        logger.info("%s: no diff chunks to review, skipping", NODE_LABEL)
        return {"bug_findings": []}  # type: ignore[return-value]

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
            category="bug",
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
        "bug_findings": all_findings,
        "tokens_used": tokens_used,
    }

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
    """Build the user prompt for a single chunk and call the LLM."""
    user_content = build_user_content(chunk, repo_context)
    return call_llm_json(SYSTEM_PROMPT, user_content)
