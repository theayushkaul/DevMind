"""
app/llm/client.py
──────────────────
A thin, shared wrapper around the Groq SDK.

Why this exists (see also: conversation with Ayush on LangChain vs direct SDK):
SecurityCheckerNode, LogicBugDetectorNode, and StyleCheckerNode all do the
exact same low-level dance:
    1. Send a system prompt + user content to Groq
    2. Get back a chat completion
    3. Parse the response text as JSON
    4. Handle every way that can go wrong (timeouts, rate limits, malformed
       JSON, empty responses, wrong schema)

Rather than re-implement that four times with four slightly different bugs,
every checker node calls `call_llm_json()` from this module. The node only
owns its system prompt and its domain-specific parsing of the "findings" key.

Design decisions:
- We use the raw `groq` SDK, not LangChain's ChatGroq wrapper. This gives us
  direct access to `response.usage` for token tracking (your `reviews` table
  needs `tokens_used`), and keeps the JSON-repair logic fully visible rather
  than buried inside an abstraction layer.
- Retries are handled explicitly here (not just relying on the SDK's built-in
  `max_retries`) because we want different behaviour for different failure
  classes: a rate limit (429) should back off and retry; a malformed JSON
  response should NOT retry the network call, just fail fast and let the
  calling node fall back to empty findings.
- `call_llm_json()` returns a `LLMCallResult` dataclass rather than raising
  on parse failure. Nodes check `result.success` and act accordingly. This
  mirrors the same philosophy as DiffParserNode: errors are data, not control
  flow, so one failed node doesn't crash the LangGraph pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from groq import APIError, APIStatusError, APITimeoutError, Groq, RateLimitError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen-2.5-coder-32b"

# Conservative defaults. Code review prompts are long (diff + RAG context),
# completions are short (structured JSON), so we bias the token budget
# toward input.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1  # low temperature: we want consistent, deterministic
                           # findings, not creative variation between runs

# Retry policy for transient failures (rate limits, timeouts, 5xx).
# We do NOT retry on malformed JSON — that's a model output problem, not a
# transient network problem, and retrying won't fix it.
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class LLMCallResult:
    """
    The outcome of a single call_llm_json() invocation.

    Exactly one of (data, error) is meaningful depending on `success`:
        success=True  → `data` holds the parsed JSON dict, `error` is None
        success=False → `data` is None, `error` describes what went wrong

    `raw_response` is kept even on failure — it's invaluable for debugging
    a checker node that keeps producing empty findings: was the LLM call
    itself broken, or did the model just return malformed JSON?
    """

    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    raw_response: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[Groq] = None


def get_client() -> Groq:
    """
    Lazily construct a singleton Groq client.

    Lazy construction matters in the Lambda context: the API key is read
    from the environment at call time, not at import time, which avoids
    import-time crashes in test environments where GROQ_API_KEY isn't set
    and lets us monkeypatch `get_client` cleanly in tests.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it in .env (local dev) or Lambda environment config (prod)."
            )
        _client = Groq(api_key=api_key)
    return _client


def reset_client() -> None:
    """Reset the singleton. Used by tests to inject a mock client cleanly."""
    global _client
    _client = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def call_llm_json(
    system_prompt: str,
    user_content: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LLMCallResult:
    """
    Call Groq with a system + user prompt, expecting a JSON object back.

    This is the ONLY function checker nodes should call to talk to the LLM.
    It handles:
        - Network-level retries with exponential backoff (rate limits, timeouts)
        - Stripping markdown code fences models sometimes wrap JSON in
          (e.g. "```json\\n{...}\\n```") even when told not to
        - JSON parse failures (returned as a failed LLMCallResult, not raised)
        - Token usage extraction for cost tracking

    Args:
        system_prompt: The role + task description (e.g. SecurityCheckerNode's
                        system prompt from the project plan).
        user_content:  The diff chunk + RAG context to review.
        model:         Groq model ID. Defaults to qwen-2.5-coder-32b.
        max_tokens:    Response token budget.
        temperature:   Sampling temperature. Low by default for consistency.

    Returns:
        LLMCallResult — always returned, never raises for LLM-level failures.
        May raise RuntimeError if GROQ_API_KEY is missing (a config error,
        not a runtime error, so it's appropriate to fail loudly).
    """
    client = get_client()

    last_error: Optional[str] = None
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return _parse_response(response)

        except RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            logger.warning(
                "Groq rate limit hit (attempt %d/%d), backing off %.1fs",
                attempt, MAX_RETRIES, backoff,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
            continue

        except APITimeoutError as exc:
            last_error = f"APITimeoutError: {exc}"
            logger.warning(
                "Groq request timed out (attempt %d/%d), retrying",
                attempt, MAX_RETRIES,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
            continue

        except APIStatusError as exc:
            # 5xx server errors are transient and worth retrying.
            # 4xx (other than rate limit, handled above) are not — e.g. a
            # 400 means our request was malformed and retrying won't help.
            if 500 <= exc.status_code < 600 and attempt < MAX_RETRIES:
                last_error = f"APIStatusError {exc.status_code}: {exc}"
                logger.warning(
                    "Groq server error %d (attempt %d/%d), retrying",
                    exc.status_code, attempt, MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue
            else:
                return LLMCallResult(
                    success=False,
                    data=None,
                    error=f"APIStatusError {exc.status_code}: {exc}",
                    raw_response=None,
                )

        except APIError as exc:
            # Catch-all for other SDK-raised API errors. Not retried — if the
            # SDK itself is rejecting the request shape, retrying is futile.
            return LLMCallResult(
                success=False,
                data=None,
                error=f"APIError: {exc}",
                raw_response=None,
            )

    # Exhausted all retries
    return LLMCallResult(
        success=False,
        data=None,
        error=f"Exhausted {MAX_RETRIES} retries. Last error: {last_error}",
        raw_response=None,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(response: Any) -> LLMCallResult:
    """
    Extract and parse the JSON content from a successful Groq API response.

    Separated from call_llm_json() so it's independently unit-testable with
    a fake response object — no network mocking required.
    """
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    if not response.choices:
        return LLMCallResult(
            success=False,
            data=None,
            error="Groq response contained no choices",
            raw_response=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    raw_text = response.choices[0].message.content or ""

    if not raw_text.strip():
        return LLMCallResult(
            success=False,
            data=None,
            error="Groq returned an empty response",
            raw_response=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return LLMCallResult(
            success=False,
            data=None,
            error=f"JSONDecodeError: {exc}",
            raw_response=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    if not isinstance(data, dict):
        return LLMCallResult(
            success=False,
            data=None,
            error=f"Expected a JSON object, got {type(data).__name__}",
            raw_response=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return LLMCallResult(
        success=True,
        data=data,
        error=None,
        raw_response=raw_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# Matches a ```json ... ``` or plain ``` ... ``` fence wrapping the whole response.
_FENCE_PATTERN = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```$",
    re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    """
    Models are instructed to output raw JSON, but qwen-2.5-coder and similar
    code-tuned models frequently wrap output in markdown code fences anyway
    (a habit from code-completion training data). Rather than fight this with
    prompt engineering alone, we defensively strip fences if present.

    Only strips a fence that wraps the ENTIRE response (anchored ^...$) to
    avoid mangling JSON that legitimately contains backtick characters in a
    string value (e.g. a comment that mentions inline code).
    """
    stripped = text.strip()
    match = _FENCE_PATTERN.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped
