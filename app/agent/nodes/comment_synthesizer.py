"""
app/agent/nodes/comment_synthesizer.py
─────────────────────────────────────────
The CommentSynthesizerNode: the last processing step before GitHubPosterNode.

Reads:  state["security_findings"]  (from SecurityCheckerNode)
        state["bug_findings"]       (from LogicBugDetectorNode)
        state["style_findings"]     (from StyleCheckerNode)
Writes: state["final_comments"]
        state["error"]              (on failure, without raising)

No LLM call — pure Python logic, per the project plan. This node's job is
entirely mechanical: merge three lists, remove true duplicates, prioritize,
and cap. That mechanical nature is exactly why it's NOT an LLM node — this
kind of deterministic list processing is something plain Python does more
reliably, more cheaply, and more testably than an LLM call ever would.

Pipeline position:
    [SecurityCheckerNode]  ─┐
    [LogicBugDetectorNode] ─┼──► [CommentSynthesizerNode] ──► [GitHubPosterNode]
    [StyleCheckerNode]     ─┘

Three responsibilities, applied in this order:

1. DEDUPLICATE — "overlapping comments on the same line" (project plan,
   Section 5). We interpret "overlapping" as same file_path + line_number +
   category (ReviewFinding.dedup_key). Two checkers can legitimately flag
   DIFFERENT issues on the same line (e.g. SecurityChecker flags a SQL
   injection on line 42 while BugDetector flags a missing null check on the
   same line) — those are not duplicates and both should survive. But if
   SecurityChecker and BugDetector both happen to flag the SAME category of
   issue on the same line (rare, but possible when two prompts overlap in
   scope), that genuinely is the same finding seen twice, and only the
   higher-confidence version should be kept.

2. PRIORITIZE — sort by severity (critical > warning > suggestion) per the
   project plan, with confidence as the tiebreaker within a severity tier.
   This ordering matters because of the cap below: when there are more than
   15 findings, we want the cap to discard the WEAKEST ones, not an
   arbitrary slice. Confidence as a secondary sort key is also why the
   per-checker confidence calibration in security_checker.py / bug_detector.py
   / style_checker.py exists — see style_checker.py's module docstring for
   why a "critical" style finding is calibrated below a "critical" security
   finding. That calibration is invisible until this node sorts by it.

3. CAP — hard limit of MAX_COMMENTS_PER_PR (15, matching the project plan's
   env var default). Applied AFTER dedup and sort, so it always drops the
   least valuable findings, never an arbitrary subset.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.agent.state import AgentState, ReviewFinding

logger = logging.getLogger(__name__)

NODE_LABEL = "CommentSynthesizerNode"

# Matches the project plan's MAX_COMMENTS_PER_PR env var default (Section 12).
MAX_COMMENTS_PER_PR: int = 15


# ---------------------------------------------------------------------------
# Public node entry point
# ---------------------------------------------------------------------------

def run(state: AgentState) -> AgentState:
    """
    LangGraph node entry point for the comment synthesizer.

    Merges security_findings + bug_findings + style_findings, deduplicates,
    sorts by priority, and caps the result. Never raises — if all three
    input lists are empty or missing, this simply produces an empty
    final_comments list rather than treating that as an error (an empty
    review is a valid, if uneventful, outcome).
    """
    security_findings: List[ReviewFinding] = state.get("security_findings", [])
    bug_findings: List[ReviewFinding] = state.get("bug_findings", [])
    style_findings: List[ReviewFinding] = state.get("style_findings", [])

    all_findings = [*security_findings, *bug_findings, *style_findings]

    if not all_findings:
        logger.info("%s: no findings from any checker, nothing to synthesize", NODE_LABEL)
        return {"final_comments": []}  # type: ignore[return-value]

    deduped = deduplicate(all_findings)
    prioritized = sort_by_priority(deduped)
    capped = cap_at_limit(prioritized, limit=MAX_COMMENTS_PER_PR)

    logger.info(
        "%s: merged %d raw findings (security=%d, bug=%d, style=%d) -> "
        "%d after dedup -> %d after cap",
        NODE_LABEL, len(all_findings),
        len(security_findings), len(bug_findings), len(style_findings),
        len(deduped), len(capped),
    )

    return {"final_comments": capped}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 1: Deduplication
# ---------------------------------------------------------------------------

def deduplicate(findings: List[ReviewFinding]) -> List[ReviewFinding]:
    """
    Remove true duplicates — findings that collide on
    ReviewFinding.dedup_key (file_path + line_number + category).

    When two findings collide, the higher-confidence one is kept. If
    confidence is tied, the first one encountered wins (stable behaviour —
    input order is security, then bug, then style, so ties favour whichever
    checker ran first; this is an arbitrary but deterministic tiebreak,
    which is what matters for reproducible output).

    This does NOT deduplicate across categories: a security finding and a
    bug finding on the same line are different issues by definition (they
    came from checkers looking for different things) and both survive.
    """
    best_by_key: Dict[str, ReviewFinding] = {}

    for finding in findings:
        key = finding.dedup_key
        existing = best_by_key.get(key)

        if existing is None or finding.confidence > existing.confidence:
            best_by_key[key] = finding

    # Dict preserves insertion order in Python 3.7+, but insertion order
    # here reflects "first time this key was seen," not necessarily the
    # winning finding's original position. That's fine — sort_by_priority()
    # imposes the real ordering next; dedup's job is only to pick winners.
    return list(best_by_key.values())


# ---------------------------------------------------------------------------
# Step 2: Prioritization
# ---------------------------------------------------------------------------

def sort_by_priority(findings: List[ReviewFinding]) -> List[ReviewFinding]:
    """
    Sort findings by severity (critical first), then by confidence
    descending within the same severity tier.

    Uses ReviewFinding.severity_rank (lower = higher priority: critical=0,
    warning=1, suggestion=2) so this stays in sync with that property's
    definition rather than re-encoding the severity ordering here.

    Stable sort (Python's sort always is) means findings that are equal on
    both keys retain their relative input order — deterministic output for
    identical input, which matters for testing and for not having review
    comments shuffle randomly between runs on an unchanged diff.
    """
    return sorted(
        findings,
        key=lambda f: (f.severity_rank, -f.confidence),
    )


# ---------------------------------------------------------------------------
# Step 3: Capping
# ---------------------------------------------------------------------------

def cap_at_limit(findings: List[ReviewFinding], limit: int) -> List[ReviewFinding]:
    """
    Truncate to at most `limit` findings.

    Assumes `findings` is already sorted by priority (sort_by_priority) —
    this function just takes the first `limit` items. Kept as a separate
    function (rather than inlined as findings[:limit] in run()) so the
    "what gets dropped and why" decision is named and independently testable.
    """
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    return findings[:limit]
