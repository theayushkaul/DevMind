"""
app/rag/embedder.py
────────────────────
Thin wrapper around the HuggingFace Inference API for text embeddings.

Model: BAAI/bge-small-en-v1.5
  - 384 dimensions — matches the vector(384) column in code_embeddings
  - Trained specifically for code retrieval (code + docstrings)
  - ~33M parameters — fast inference, low memory, strong quality/size ratio
  - Free on HuggingFace Inference API (no GPU needed for inference)

Why the Inference API, not running the model locally?
  Lambda has a 250MB deployment package limit. The transformers library +
  model weights for bge-small-en-v1.5 alone would exceed this. The
  Inference API offloads computation to HuggingFace's servers — we send
  text, get back a vector. One HTTPS call, no local model files.

  Trade-off: latency (~200-400ms per call vs ~50ms local), external
  dependency. For a portfolio project with free-tier constraints, this
  is the right call. At production scale you'd run the model in a
  dedicated container or use a managed embedding service.

API contract:
  POST https://api-inference.huggingface.co/pipeline/feature-extraction/{model}
  Body: {"inputs": ["text1", "text2", ...], "options": {"wait_for_model": true}}
  Response: [[float x 384], [float x 384], ...]

  One API call can embed multiple texts (batch). We batch per-file chunks
  to minimise round-trips during indexing, but embed one diff chunk at a
  time during retrieval (low latency is more important than throughput there).

Rate limits (free tier):
  ~30,000 tokens/month for inference. Each embedding call uses roughly
  1 token per word. For a portfolio project with occasional PR reviews
  this is sufficient. Track usage in the HuggingFace console.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List

import httpx

logger = logging.getLogger(__name__)

HF_API_BASE = "https://api-inference.huggingface.co/pipeline/feature-extraction"
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# HuggingFace Inference API can be slow to respond when a model is cold
# ("loading" state). wait_for_model=true tells HF to wait up to 60s for
# the model to load rather than returning a 503 immediately.
_DEFAULT_TIMEOUT = 60.0
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


class EmbeddingError(Exception):
    """Raised when the HuggingFace API call fails after all retries."""


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings into 384-dimensional vectors.

    Batches all texts into a single API call. For large batches, HuggingFace
    may split them internally; the response always matches the input order.

    Args:
        texts: List of strings to embed. Each should be under ~512 tokens
               (bge-small-en-v1.5's context window). Longer texts are
               silently truncated by the model.

    Returns:
        List of 384-dimensional float vectors, one per input text.
        Order matches the input list.

    Raises:
        EmbeddingError: if the API returns an error after all retries.
        RuntimeError:   if HUGGINGFACE_API_KEY is not set.
    """
    if not texts:
        return []

    api_key = _require_env("HUGGINGFACE_API_KEY")
    url = f"{HF_API_BASE}/{MODEL_ID}"

    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},
    }

    last_error: str = ""
    backoff = _RETRY_BACKOFF

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_DEFAULT_TIMEOUT,
            )
        except httpx.TimeoutException as exc:
            last_error = f"Timeout on attempt {attempt}: {exc}"
            logger.warning("embedder: %s", last_error)
            if attempt < _MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            continue
        except httpx.RequestError as exc:
            last_error = f"Network error on attempt {attempt}: {exc}"
            logger.warning("embedder: %s", last_error)
            if attempt < _MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            continue

        if response.status_code == 503:
            # Model is loading — wait_for_model=true should handle this,
            # but sometimes HF still returns 503 on first call
            last_error = f"HF model loading (503) on attempt {attempt}"
            logger.warning("embedder: %s", last_error)
            if attempt < _MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            continue

        if not response.is_success:
            last_error = f"HF API {response.status_code}: {response.text[:200]}"
            logger.error("embedder: %s", last_error)
            raise EmbeddingError(last_error)

        vectors = response.json()

        # Validate response shape
        if not isinstance(vectors, list) or not vectors:
            raise EmbeddingError(f"Unexpected HF response shape: {type(vectors)}")

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"HF returned {len(vectors)} vectors for {len(texts)} inputs"
            )

        # Validate first vector dimension
        first = vectors[0]
        if not isinstance(first, list) or len(first) != EMBEDDING_DIM:
            raise EmbeddingError(
                f"Expected {EMBEDDING_DIM}-dim vector, got {len(first) if isinstance(first, list) else type(first)}"
            )

        logger.debug(
            "embedder: embedded %d texts → %d×%d vectors",
            len(texts), len(vectors), EMBEDDING_DIM,
        )
        return vectors

    raise EmbeddingError(
        f"HuggingFace API failed after {_MAX_RETRIES} attempts. Last error: {last_error}"
    )


def embed_single(text: str) -> List[float]:
    """
    Embed a single text string. Convenience wrapper around embed_texts().

    Used by the retriever at query time — one diff chunk at a time,
    where latency matters more than throughput.
    """
    vectors = embed_texts([text])
    return vectors[0]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} environment variable is not set. "
            f"Get a free key at https://huggingface.co/settings/tokens"
        )
    return value
