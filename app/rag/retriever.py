"""
app/rag/retriever.py
─────────────────────
Two things live here:

1. retrieve_context() — given a diff chunk, find the top-K most semantically
   similar code snippets from the indexed repository using pgvector cosine
   similarity search. This fills state["repo_context"].

2. run() — the LangGraph RAGContextNode. Sits between DiffParserNode and the
   three checker nodes. For each diff chunk, it calls retrieve_context() and
   stores the results in state["repo_context"] keyed by chunk.dedup_key.

Why pgvector cosine similarity (not dot product or L2)?
  Cosine similarity measures the angle between vectors, ignoring magnitude.
  For text embeddings, magnitude correlates with text length — a long
  function embedding and a short function embedding shouldn't be penalised
  for length differences when measuring semantic similarity. Cosine
  similarity normalises this away.

  L2 distance (Euclidean) is magnitude-sensitive — wrong for text.
  Dot product is equivalent to cosine only when vectors are unit-normalised.
  bge-small-en-v1.5 does output normalised vectors, so dot product would
  also work, but cosine is explicit and correct by construction.

pgvector operator: <=> (cosine distance, lower = more similar)
  The ivfflat index in the migration uses vector_cosine_ops, so the
  index is used automatically when we use <=>.

Raw SQL, not ORM:
  pgvector's cosine similarity operator (<=>) is not natively understood
  by SQLAlchemy's ORM expression layer. We use session.execute() with
  raw SQL text for the similarity search, which is simpler and more
  readable than fighting SQLAlchemy's type system to register custom
  operators. The ORM is used for inserts (in indexer.py); raw SQL for
  the vector search query.

Top-K: 5 nearest neighbours per diff chunk, as specified in the project plan.
  This is a hyperparameter worth tuning — too few and context is sparse,
  too many and the LLM prompt gets noisy. 5 is a reasonable starting point.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import AgentState, DiffChunk
from app.db.session import get_session
from app.rag.embedder import EmbeddingError, embed_single

logger = logging.getLogger(__name__)

TOP_K = 5  # number of similar snippets to retrieve per diff chunk
NODE_LABEL = "RAGContextNode"


# ---------------------------------------------------------------------------
# LangGraph node entry point
# ---------------------------------------------------------------------------

def run(state: AgentState) -> AgentState:
    """
    LangGraph node: populate state["repo_context"] with retrieved snippets.

    For each diff chunk, embeds the chunk content and queries pgvector for
    the top-5 most similar code segments from the indexed repository.

    This node is sync (like the other nodes) but internally calls
    asyncio.run() for the DB query — same pattern as handler.py.

    Fails gracefully: if the repo isn't indexed yet, or if the embedding
    API is unavailable, repo_context remains empty and the checker nodes
    proceed without RAG context. This degrades quality but doesn't crash
    the pipeline.
    """
    import asyncio

    diff_chunks: List[DiffChunk] = state.get("diff_chunks", [])
    repo_full_name: str = state.get("repo_full_name", "")

    if not diff_chunks:
        return {"repo_context": {}}  # type: ignore[return-value]

    # Check if DATABASE_URL is configured — if not, skip RAG silently
    if not os.environ.get("DATABASE_URL"):
        logger.info(
            "%s: DATABASE_URL not set — skipping RAG retrieval", NODE_LABEL
        )
        return {"repo_context": {}}  # type: ignore[return-value]

    repo_context: Dict[str, List[str]] = {}

    for chunk in diff_chunks:
        try:
            snippets = asyncio.run(
                retrieve_context(
                    chunk_text=chunk.content,
                    repo_full_name=repo_full_name,
                    top_k=TOP_K,
                )
            )
            if snippets:
                repo_context[chunk.dedup_key] = snippets
                logger.debug(
                    "%s: retrieved %d snippets for %s",
                    NODE_LABEL, len(snippets), chunk.dedup_key,
                )
        except Exception as exc:
            # Non-fatal — log and continue without context for this chunk
            logger.warning(
                "%s: retrieval failed for %s: %s",
                NODE_LABEL, chunk.dedup_key, exc,
            )

    logger.info(
        "%s: retrieved context for %d/%d chunks",
        NODE_LABEL, len(repo_context), len(diff_chunks),
    )

    return {"repo_context": repo_context}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Core retrieval function — public for direct testing
# ---------------------------------------------------------------------------

async def retrieve_context(
    chunk_text: str,
    repo_full_name: str,
    top_k: int = TOP_K,
) -> List[str]:
    """
    Find the top-K most semantically similar code chunks for a given text.

    Args:
        chunk_text:      The diff chunk content to find context for.
        repo_full_name:  Used to scope the search to a single repo's embeddings.
        top_k:           Number of similar snippets to return.

    Returns:
        List of chunk_text strings from the indexed codebase, ordered by
        cosine similarity (most similar first). Empty list if the repo
        isn't indexed or no embeddings exist.
    """
    # Embed the query chunk
    try:
        query_vector = embed_single(chunk_text)
    except EmbeddingError as exc:
        logger.warning("retriever: embedding failed for query: %s", exc)
        return []

    # Format as pgvector literal: '[0.1,0.2,...]'
    vector_literal = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"

    async with get_session() as session:
        results = await _cosine_search(
            session=session,
            vector_literal=vector_literal,
            repo_full_name=repo_full_name,
            top_k=top_k,
        )

    return results


async def _cosine_search(
    session: AsyncSession,
    vector_literal: str,
    repo_full_name: str,
    top_k: int,
) -> List[str]:
    """
    Execute a pgvector cosine similarity search.

    SQL pattern:
        SELECT ce.chunk_text
        FROM code_embeddings ce
        JOIN repositories r ON ce.repo_id = r.id
        WHERE r.full_name = :repo_full_name
          AND ce.embedding IS NOT NULL
        ORDER BY ce.embedding <=> :query_vector
        LIMIT :top_k

    The <=> operator is pgvector's cosine distance (0 = identical, 2 = opposite).
    The ivfflat index on ce.embedding makes this an approximate nearest-
    neighbour search — not exact, but fast enough for retrieval use cases.

    Why JOIN on repositories.full_name rather than repo_id?
        The handler passes repo_full_name (from the webhook payload) not
        repo_id (a DB UUID). Joining on full_name avoids an extra lookup.
        full_name has no index here, but repositories is a small table
        (one row per repo) so a sequential scan is fine.
    """
    sql = text("""
        SELECT ce.chunk_text
        FROM code_embeddings ce
        JOIN repositories r ON ce.repo_id = r.id
        WHERE r.full_name = :repo_full_name
          AND ce.embedding IS NOT NULL
        ORDER BY ce.embedding <=> CAST(:query_vector AS vector)
        LIMIT :top_k
    """)

    try:
        result = await session.execute(
            sql,
            {
                "repo_full_name": repo_full_name,
                "query_vector": vector_literal,
                "top_k": top_k,
            },
        )
        rows = result.fetchall()
        return [row[0] for row in rows]
    except Exception as exc:
        # pgvector not installed, repo not indexed, or DB error
        logger.warning("retriever: cosine search failed: %s", exc)
        return []
