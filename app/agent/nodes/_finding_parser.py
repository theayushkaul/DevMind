"""
app/agent/nodes/_finding_parser.py
─────────────────────────────────────
Shared response-parsing logic for the three LLM checker nodes:
SecurityCheckerNode, LogicBugDetectorNode, StyleCheckerNode.

Why this module exists:
All three checker nodes share an identical response contract from the LLM:
    {"findings": [{"file_path": ..., "line_number": ..., "severity": ...,
                    "comment": ...}]}
and all three need the same defensive parsing — the model is asked to follow
this schema but cannot be trusted to always comply. Rather than duplicate
~80 lines of "is this a dict, is severity valid, can line_number be coerced
to an int" logic three times (once per node, with three chances to drift out
of sync), that logic lives here once.

What does NOT live here, by design:
- The system prompt (each node owns its own — that's the actual domain logic
  that differentiates a security review from a style review)
- The `category` value (passed in by the caller, since it's fixed per node)
- The confidence heuristic mapping (passed in by the caller — different
  checkers could reasonably weight severities differently in the future)
- Building the user-facing prompt content (each node's chunk + RAG context
  formatting is similar but not identical, kept local to each node for now)

This is a leading-underscore module (`_finding_parser`, not `finding_parser`)
to signal it's an internal implementation detail of the nodes package, not
part of the public node interface that graph.py wires up.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

from app.agent.state import Category, DiffChunk, ReviewFinding, Severity

logger = logging.getLogger(__name__)

VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "warning", "suggestion"})


def extract_findings(
    data: Any,
    chunk: DiffChunk,
    *,
    category: Category,
    source_node: str,
    confidence_by_severity: Mapping[str, float],
    node_label: str,
) -> List[ReviewFinding]:
    """
    Convert a parsed LLM JSON response into a list of validated ReviewFindings.

    Args:
        data:                    The parsed JSON dict from LLMCallResult.data.
        chunk:                   The DiffChunk this response is for (used for
                                  fallback file_path/line_number values).
        category:                Fixed category for every finding this node
                                  produces, e.g. "security", "bug", "style".
        source_node:             Value stamped into ReviewFinding.source_node,
                                  e.g. "security_checker".
        confidence_by_severity:  Maps "critical"/"warning"/"suggestion" to a
                                  confidence float. Each node supplies its own
                                  mapping so checkers can diverge later if
                                  warranted (e.g. style suggestions might
                                  always be lower-confidence than security
                                  findings of the same nominal severity).
        node_label:               Human-readable name used in log messages,
                                  e.g. "SecurityCheckerNode".

    Returns:
        A list of valid ReviewFinding objects. Malformed entries are skipped
        and logged, never raised — one bad entry must not discard the rest
        of an otherwise-valid findings array.
    """
    if not isinstance(data, dict):
        logger.warning(
            "%s: expected dict response for %s, got %s",
            node_label, chunk.dedup_key, type(data).__name__,
        )
        return []

    raw_findings = data.get("findings")
    if raw_findings is None:
        logger.warning(
            "%s: response for %s missing 'findings' key",
            node_label, chunk.dedup_key,
        )
        return []

    if not isinstance(raw_findings, list):
        logger.warning(
            "%s: 'findings' for %s is not a list (got %s)",
            node_label, chunk.dedup_key, type(raw_findings).__name__,
        )
        return []

    results: List[ReviewFinding] = []
    for i, item in enumerate(raw_findings):
        finding = _parse_single_finding(
            item, chunk, index=i,
            category=category,
            source_node=source_node,
            confidence_by_severity=confidence_by_severity,
            node_label=node_label,
        )
        if finding is not None:
            results.append(finding)

    return results


def _parse_single_finding(
    item: Any,
    chunk: DiffChunk,
    *,
    index: int,
    category: Category,
    source_node: str,
    confidence_by_severity: Mapping[str, float],
    node_label: str,
) -> ReviewFinding | None:
    """
    Parse and validate one entry from the "findings" array.

    Returns None for any entry that doesn't conform — logged at debug level,
    since an occasional malformed entry is expected LLM behaviour, not a bug
    in our code.
    """
    if not isinstance(item, dict):
        logger.debug(
            "%s: finding[%d] for %s is not an object, skipping",
            node_label, index, chunk.dedup_key,
        )
        return None

    file_path = item.get("file_path")
    line_number = item.get("line_number")
    severity = item.get("severity")
    comment = item.get("comment")

    # file_path: fall back to the chunk's own file_path if the model omitted
    # or mangled it — we already know which file this chunk belongs to.
    if not isinstance(file_path, str) or not file_path.strip():
        file_path = chunk.file_path

    line_number = coerce_line_number(line_number, fallback=chunk.start_line)

    if severity not in VALID_SEVERITIES:
        logger.debug(
            "%s: finding[%d] for %s has invalid severity %r, skipping",
            node_label, index, chunk.dedup_key, severity,
        )
        return None

    if not isinstance(comment, str) or not comment.strip():
        logger.debug(
            "%s: finding[%d] for %s has empty/missing comment, skipping",
            node_label, index, chunk.dedup_key,
        )
        return None

    try:
        return ReviewFinding(
            file_path=file_path,
            line_number=line_number,
            category=category,
            severity=severity,  # type: ignore[arg-type]
            comment=comment.strip(),
            confidence=confidence_by_severity[severity],
            source_node=source_node,
        )
    except (ValueError, KeyError) as exc:
        # ValueError: ReviewFinding.__post_init__ validation failed (defense
        #   in depth — shouldn't happen given the checks above).
        # KeyError: confidence_by_severity didn't have an entry for this
        #   severity — a caller configuration bug, not a model output bug,
        #   but we still don't want it to crash the whole node.
        logger.debug(
            "%s: finding[%d] for %s failed validation: %s",
            node_label, index, chunk.dedup_key, exc,
        )
        return None


def coerce_line_number(value: Any, fallback: int) -> int:
    """
    Best-effort conversion of the model's line_number field to an int.

    Handles the realistic range of malformed-but-recoverable outputs:
    42, 42.0, "42" all become 42. Anything else falls back to the chunk's
    start_line rather than discarding the finding entirely.

    `bool` is explicitly rejected despite being an int subclass in Python —
    we don't want a stray `True`/`False` silently becoming line 1/0.
    """
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return fallback
    return fallback


def build_user_content(
    chunk: DiffChunk,
    repo_context: Dict[str, List[str]],
) -> str:
    """
    Format a diff chunk (plus any retrieved RAG context) into the user
    message sent to the LLM.

    Shared across all three checker nodes because the input shape they need
    is identical — file metadata, the diff itself, and optional related
    snippets. If a node ever needs to diverge (e.g. StyleChecker wanting to
    see the whole file rather than just the diff), it can stop calling this
    and build its own — nothing else depends on this being the only path.

    If repo_context has no entry for this chunk (RAGContextNode hasn't run,
    or pgvector returned nothing), we omit the "Related code" section
    entirely rather than sending an empty placeholder header.
    """
    sections = [
        f"File: {chunk.file_path}",
        f"Language: {chunk.language}",
        f"Lines {chunk.start_line}-{chunk.end_line}",
        "",
        "Diff:",
        chunk.content,
    ]

    related_snippets = repo_context.get(chunk.dedup_key, [])
    if related_snippets:
        sections.append("")
        sections.append("Related code from this repository (for context only):")
        for i, snippet in enumerate(related_snippets, start=1):
            sections.append(f"--- Related snippet {i} ---")
            sections.append(snippet)

    return "\n".join(sections)
