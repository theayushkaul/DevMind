"""
app/agent/graph.py
───────────────────
Wires all agent nodes into a compiled LangGraph pipeline.

Pipeline topology:

    START
      │
      ▼
  [diff_parser]  ── no chunks ──► [comment_synthesizer] ──► END
      │
      │ (chunks exist)
      ▼
  [rag_context]          ← NEW: fills state["repo_context"] via pgvector
      │
      ▼
  [security_checker]
      │
      ▼
  [bug_detector]
      │
      ▼
  [style_checker]
      │
      ▼
  [comment_synthesizer]
      │
      ▼
     END

The RAGContextNode sits between diff_parser and the three checker nodes.
It's on the "chunks exist" branch only — no point running a vector search
when there's nothing to review. If RAG retrieval fails (DB unavailable,
repo not indexed), it writes an empty repo_context and the checkers proceed
without codebase context. Degraded quality, but not a crash.

Why sequential for checker nodes (not parallel)?
See original comment — sequential is easier to reason about and debug.
The latency budget (< 60s total) is met by three sequential ~10s LLM calls.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    bug_detector,
    comment_synthesizer,
    diff_parser,
    security_checker,
    style_checker,
)
from app.rag import retriever as rag_context
from app.agent.state import AgentState, initial_state

logger = logging.getLogger(__name__)

NODE_DIFF_PARSER          = "diff_parser"
NODE_RAG_CONTEXT          = "rag_context"
NODE_SECURITY_CHECKER     = "security_checker"
NODE_BUG_DETECTOR         = "bug_detector"
NODE_STYLE_CHECKER        = "style_checker"
NODE_COMMENT_SYNTHESIZER  = "comment_synthesizer"


def build_graph() -> StateGraph:
    """Construct and compile the DevMind agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node(NODE_DIFF_PARSER,         diff_parser.run)
    builder.add_node(NODE_RAG_CONTEXT,         rag_context.run)
    builder.add_node(NODE_SECURITY_CHECKER,    security_checker.run)
    builder.add_node(NODE_BUG_DETECTOR,        bug_detector.run)
    builder.add_node(NODE_STYLE_CHECKER,       style_checker.run)
    builder.add_node(NODE_COMMENT_SYNTHESIZER, comment_synthesizer.run)

    builder.add_edge(START, NODE_DIFF_PARSER)

    # Conditional edge: skip LLM nodes if no chunks to review
    builder.add_conditional_edges(
        NODE_DIFF_PARSER,
        _route_after_diff_parser,
        {
            "run_checkers": NODE_RAG_CONTEXT,        # goes to RAG first now
            "skip_to_synthesizer": NODE_COMMENT_SYNTHESIZER,
        },
    )

    # RAG → checkers (sequential)
    builder.add_edge(NODE_RAG_CONTEXT,      NODE_SECURITY_CHECKER)
    builder.add_edge(NODE_SECURITY_CHECKER, NODE_BUG_DETECTOR)
    builder.add_edge(NODE_BUG_DETECTOR,     NODE_STYLE_CHECKER)
    builder.add_edge(NODE_STYLE_CHECKER,    NODE_COMMENT_SYNTHESIZER)
    builder.add_edge(NODE_COMMENT_SYNTHESIZER, END)

    return builder.compile()


def _route_after_diff_parser(
    state: AgentState,
) -> Literal["run_checkers", "skip_to_synthesizer"]:
    chunks = state.get("diff_chunks", [])
    if not chunks:
        logger.info("graph: no diff chunks — skipping LLM checker nodes")
        return "skip_to_synthesizer"
    return "run_checkers"


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(
    raw_diff: str,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> AgentState:
    """
    Run the full agent pipeline for a single PR and return the final state.

    Args:
        raw_diff:        Unified diff string from GitHub API.
        repo_full_name:  e.g. "ayushkaul/my-repo"
        pr_number:       PR number (integer).
        head_sha:        Commit SHA at PR tip.

    Returns:
        Final AgentState. Callers care about:
            state["final_comments"]  — list of ReviewFindings to post
            state["tokens_used"]     — total tokens for cost tracking
            state["error"]           — non-None if something went wrong
    """
    state = initial_state(
        raw_diff=raw_diff,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
    )

    graph = get_graph()

    logger.info(
        "graph: starting pipeline for %s PR #%d (sha=%s, diff_len=%d)",
        repo_full_name, pr_number, head_sha[:8], len(raw_diff),
    )

    final_state: AgentState = graph.invoke(state)

    logger.info(
        "graph: pipeline complete for %s PR #%d — "
        "%d final comments, %d tokens used, error=%s",
        repo_full_name, pr_number,
        len(final_state.get("final_comments", [])),
        final_state.get("tokens_used", 0),
        final_state.get("error"),
    )

    return final_state
