"""
app/agent/nodes/style_checker.py
───────────────────────────────────
The StyleCheckerNode: flags maintainability and style issues in the diff.

Reads:  state["diff_chunks"]   (from DiffParserNode)
        state["repo_context"]  (from RAGContextNode, optional — may be {})
Writes: state["style_findings"]
        state["tokens_used"]   (incremented, not overwritten)
        state["error"]         (on failure, without raising)

Structurally identical to SecurityCheckerNode and LogicBugDetectorNode —
see security_checker.py for the heavily-commented version of this pattern,
and _finding_parser.py for why response parsing isn't duplicated per node.

One deliberate difference from the other two checkers: style findings are
inherently softer judgment calls than security or correctness findings — a
naming inconsistency is rarely "critical" in the way a SQL injection is. The
CONFIDENCE_BY_SEVERITY mapping below is intentionally calibrated slightly
lower across the board to reflect that a "critical" style finding still
shouldn't carry the same weight as a "critical" security finding when the
CommentSynthesizerNode later sorts/caps findings across all three categories.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.agent.nodes._finding_parser import build_user_content, extract_findings
from app.agent.state import AgentState, DiffChunk, ReviewFinding
from app.llm.client import LLMCallResult, call_llm_json

logger = logging.getLogger(__name__)

NODE_NAME = "style_checker"
NODE_LABEL = "StyleCheckerNode"

# ---------------------------------------------------------------------------
# System prompt — verbatim from the DevMind project plan, Section 5
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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
"""

# Calibrated lower than SecurityChecker/BugDetector — see module docstring.
CONFIDENCE_BY_SEVERITY: Dict[str, float] = {
    "critical": 0.7,
    "warning": 0.6,
    "suggestion": 0.5,
}


# ---------------------------------------------------------------------------
# Public node entry point
# ---------------------------------------------------------------------------

def run(state: AgentState) -> AgentState:
    """
    LangGraph node entry point for the style checker.

    Iterates every diff chunk, calls the LLM once per chunk, and accumulates
    findings. Never raises — a failed chunk is logged and skipped.
    """
    diff_chunks: List[DiffChunk] = state.get("diff_chunks", [])
    repo_context: Dict[str, List[str]] = state.get("repo_context", {})

    if not diff_chunks:
        logger.info("%s: no diff chunks to review, skipping", NODE_LABEL)
        return {"style_findings": []}  # type: ignore[return-value]

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
            category="style",
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
        "style_findings": all_findings,
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
