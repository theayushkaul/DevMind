"""
app/github/client.py
─────────────────────
GitHub API client factory using GitHub App authentication.

Why GitHub App auth, not a Personal Access Token (PAT):
  PATs are tied to a specific user account. If that user leaves the org or
  revokes the token, DevMind stops working. GitHub Apps are first-class
  principals: they have their own identity, configurable permissions, and
  generate short-lived installation tokens (1 hour TTL) rather than
  long-lived secrets. They also have higher API rate limits than PATs.

The GitHub App auth flow (two-step):
  1. App authenticates AS ITSELF using a JWT signed with its private key:
       JWT = sign({"iss": app_id, "exp": now+300}, private_key, RS256)
     This JWT proves "I am DevMind app #12345".

  2. App uses that JWT to request an installation access token for a specific
     repo installation:
       POST /app/installations/{installation_id}/access_tokens
     Response: {"token": "ghs_xxx", "expires_at": "..."}
     This token is scoped to that installation's repos with the permissions
     the user granted when installing the app.

  3. All subsequent API calls use the installation token as a Bearer token.

The installation_id comes from the webhook payload
(payload["installation"]["id"]) — GitHub includes it in every webhook
event so the receiver knows which installation to auth as.

Token TTL and caching:
  Installation tokens expire after 1 hour. Lambda invocations are short
  (well under a minute), so we generate a fresh token per invocation.
  For higher throughput, tokens could be cached in Redis with a TTL of
  50 minutes — left as a future optimization since the current free-tier
  volume doesn't hit rate limits.
"""

from __future__ import annotations

import logging
import os

from github import Auth, Github, GithubIntegration

logger = logging.getLogger(__name__)


def get_github_client(installation_id: int) -> Github:
    """
    Build an authenticated PyGithub client for a specific installation.

    Args:
        installation_id: The GitHub App installation ID from the webhook
                         payload. Scopes the token to that installation's
                         repos.

    Returns:
        An authenticated Github client ready to make API calls.

    Raises:
        RuntimeError: if GITHUB_APP_ID or GITHUB_PRIVATE_KEY env vars are
                      missing.
        github.GithubException: if the GitHub API rejects the auth request
                                (wrong app ID, invalid key, etc.).
    """
    app_id = _require_env("GITHUB_APP_ID")
    private_key = _require_env("GITHUB_PRIVATE_KEY")

    # PyGithub 2.x App auth flow
    app_auth = Auth.AppAuth(app_id=app_id, private_key=private_key)
    integration = GithubIntegration(auth=app_auth)

    # Exchange the app JWT for an installation access token
    installation_auth = integration.get_access_token(installation_id)

    logger.debug(
        "github_client: obtained installation token for installation %d "
        "(expires: %s)",
        installation_id,
        installation_auth.expires_at,
    )

    return Github(auth=Auth.Token(installation_auth.token))


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} environment variable is not set. "
            f"Set it in .env (local dev) or Lambda environment config (prod)."
        )
    return value
