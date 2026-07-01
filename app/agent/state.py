"""
app/agent/state.py
──────────────────
Defines the data contracts for the entire LangGraph agent pipeline.

Everything here answers one question: "What shape does data take as it
flows through the graph?"

Design principles applied:
- DiffChunk and ReviewFinding are dataclasses (not TypedDicts) because they
  are value objects with behaviour-free, immutable data. Dataclasses give us
  __repr__, __eq__, and type safety for free.
- AgentState is a TypedDict because LangGraph's state management requires
  dict-compatible types. LangGraph merges state updates via dict operations
  under the hood — a dataclass would break that.
- All string-enumerated fields (category, severity, status) use Literal types
  so mypy catches typos like "cirital" at type-check time, not at 2am during
  a production incident.
- Optional[str] on `error` is intentional — a None error means success.
  Nodes should write to this field on failure but NOT raise exceptions, so
  the graph continues to the next node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# Literal type aliases — single source of truth for valid string values
# ---------------------------------------------------------------------------

Category = Literal["security", "bug", "style", "performance"]
Severity = Literal["critical", "warning", "suggestion"]


# ---------------------------------------------------------------------------
# DiffChunk
# ---------------------------------------------------------------------------

@dataclass
class DiffChunk:
    """
    A single reviewable unit of a pull request diff.

    The DiffParserNode produces a list of these. Every downstream LLM node
    receives this structured form — never the raw unified diff string — so
    each node can reference exact file paths and line numbers when generating
    findings.

    Attributes:
        file_path:   Repo-relative path, e.g. "src/api/routes.py".
                     Always uses forward slashes regardless of OS.
        start_line:  First line number of this chunk in the *new* file
                     (post-diff line numbers, not original). Used by GitHub
                     API to anchor inline PR comments.
        end_line:    Last line number of this chunk in the new file.
        content:     The raw diff hunk text for this chunk, including the
                     +/- prefix characters. LLMs understand this format.
        language:    Detected programming language, e.g. "python", "typescript".
                     Defaults to "unknown" for unrecognised extensions.
                     Used by LLM nodes to calibrate language-specific checks.
        chunk_index: Zero-based position of this chunk within its file's chunks.
                     Used to reconstruct ordering and for deduplication keys.
    """

    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    chunk_index: int = 0

    def __post_init__(self) -> None:
        # Normalise path separators so Windows dev environments don't produce
        # paths like "src\\api\\routes.py" which wouldn't match GitHub's API.
        self.file_path = self.file_path.replace("\\", "/")

        if self.start_line < 0:
            raise ValueError(
                f"start_line must be >= 0, got {self.start_line} for {self.file_path}"
            )
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line}) "
                f"for {self.file_path}"
            )

    @property
    def line_count(self) -> int:
        """Number of lines spanned by this chunk."""
        return self.end_line - self.start_line + 1

    @property
    def dedup_key(self) -> str:
        """
        Stable identity key for this chunk.
        Used by CommentSynthesizerNode to attribute findings to their source chunk
        and by RAGContextNode as the dict key in repo_context.
        """
        return f"{self.file_path}:{self.chunk_index}"


# ---------------------------------------------------------------------------
# ReviewFinding
# ---------------------------------------------------------------------------

@dataclass
class ReviewFinding:
    """
    A single issue identified by a checker node.

    Produced by: SecurityCheckerNode, LogicBugDetectorNode, StyleCheckerNode.
    Consumed by: CommentSynthesizerNode, GitHubPosterNode.

    Attributes:
        file_path:   Same format as DiffChunk.file_path.
        line_number: The specific line the finding refers to. May be -1 if the
                     issue is file-level (e.g. missing module docstring).
        category:    One of: "security" | "bug" | "style" | "performance".
        severity:    One of: "critical" | "warning" | "suggestion".
        comment:     The human-readable review comment. Should be actionable —
                     explain what's wrong AND how to fix it.
        confidence:  Float in [0.0, 1.0]. LLM nodes should set this based on
                     how certain they are. The synthesizer can filter low-
                     confidence findings when the comment cap is tight.
        source_node: Which node produced this finding. Used for debugging and
                     for aggregation metrics ("how many bugs did SecurityChecker
                     catch vs LogicBugDetector?").
    """

    file_path: str
    line_number: int
    category: Category
    severity: Severity
    comment: str
    confidence: float
    source_node: str = "unknown"

    def __post_init__(self) -> None:
        self.file_path = self.file_path.replace("\\", "/")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

        if not self.comment.strip():
            raise ValueError("comment cannot be empty or whitespace")

    @property
    def severity_rank(self) -> int:
        """
        Numeric rank for sorting. Lower = higher priority.
        Used by CommentSynthesizerNode to enforce: critical > warning > suggestion.
        """
        return {"critical": 0, "warning": 1, "suggestion": 2}[self.severity]

    @property
    def dedup_key(self) -> str:
        """
        Two findings at the same file+line with the same category are considered
        duplicates — even if their comments differ. This happens when the security
        checker and the bug checker both flag a SQL injection on line 42.
        The higher-confidence one wins (handled in synthesizer).
        """
        return f"{self.file_path}:{self.line_number}:{self.category}"

    @property
    def formatted_comment(self) -> str:
        """
        Returns the comment prefixed with the severity emoji, ready to post to GitHub.
        """
        prefix = {
            "critical": "🚨 **Critical**",
            "warning":  "⚠️ **Warning**",
            "suggestion": "💡 **Suggestion**",
        }[self.severity]
        return f"{prefix}: {self.comment}"


# ---------------------------------------------------------------------------
# AgentState — the graph's shared mutable state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    The complete state object that flows through the LangGraph pipeline.

    LangGraph passes this dict to each node. Each node returns a *partial*
    dict — only the keys it modified. LangGraph merges the partial update
    into the full state. This is why AgentState must be a TypedDict, not a
    dataclass.

    `total=False` means all keys are optional at the TypedDict level. In
    practice, the first node (DiffParserNode) receives `raw_diff`,
    `repo_full_name`, and `pr_number` as required inputs. Downstream nodes
    progressively populate the rest.

    Why not use a single flat dict?
    Type safety. With TypedDict, mypy knows that `state["security_findings"]`
    is `List[ReviewFinding]`, not `Any`. Bugs from mis-keyed state access
    become compile-time errors.

    Fields:
        raw_diff:           The complete unified diff string from GitHub API.
        repo_full_name:     GitHub repo identifier, e.g. "ayushkaul/devmind".
        pr_number:          The pull request number (integer).
        head_sha:           The commit SHA at the tip of the PR branch.
                            Used for idempotency key and for posting comments
                            to the right commit.
        diff_chunks:        Produced by DiffParserNode. List of parsed, filtered
                            chunks ready for LLM consumption.
        repo_context:       Produced by RAGContextNode. Maps chunk.dedup_key →
                            list of relevant code snippets retrieved from pgvector.
        security_findings:  Produced by SecurityCheckerNode.
        bug_findings:       Produced by LogicBugDetectorNode.
        style_findings:     Produced by StyleCheckerNode.
        final_comments:     Produced by CommentSynthesizerNode. The deduplicated,
                            prioritised, capped list ready for GitHub posting.
        error:              Set by any node on failure. Does NOT stop execution —
                            nodes check this field to skip work if upstream failed.
        tokens_used:        Cumulative token count across all LLM calls. Used for
                            cost tracking and logged to Supabase after completion.
    """

    raw_diff: str
    repo_full_name: str
    pr_number: int
    head_sha: str
    diff_chunks: List[DiffChunk]
    repo_context: Dict[str, List[str]]   # dedup_key → list of retrieved snippets
    security_findings: List[ReviewFinding]
    bug_findings: List[ReviewFinding]
    style_findings: List[ReviewFinding]
    final_comments: List[ReviewFinding]
    error: Optional[str]
    tokens_used: int


def initial_state(
    raw_diff: str,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> AgentState:
    """
    Factory function: creates a fully-initialised AgentState with safe defaults.

    Always use this instead of constructing the dict manually. It ensures
    every list field starts empty (not None), so nodes can safely call
    `state["security_findings"].extend(new_findings)` without None checks.

    Example:
        state = initial_state(
            raw_diff=diff_text,
            repo_full_name="ayushkaul/my-repo",
            pr_number=42,
            head_sha="abc123",
        )
    """
    return AgentState(
        raw_diff=raw_diff,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
        diff_chunks=[],
        repo_context={},
        security_findings=[],
        bug_findings=[],
        style_findings=[],
        final_comments=[],
        error=None,
        tokens_used=0,
    )
