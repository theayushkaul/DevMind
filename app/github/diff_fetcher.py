"""
app/github/diff_fetcher.py
───────────────────────────
Fetches the unified diff for a pull request from the GitHub API.

Why fetch the diff here rather than in the webhook receiver:
  The webhook payload contains metadata about the PR (repo, PR number, SHA)
  but NOT the actual diff. GitHub keeps the diff payload out of the webhook
  for size reasons — a large PR diff could be megabytes. We fetch it lazily
  in Lambda after dequeuing the event, using the installation token generated
  at that point.

  This design also means the diff is always fetched at processing time, not
  at webhook receipt time. If processing is delayed by a few seconds due to
  queue backlog, we still get the correct diff (GitHub diffs are stable for
  a given head SHA).

API choice — why not PyGithub's get_files()?
  PyGithub's PR.get_files() returns a list of file objects with patch strings
  (individual file diffs), but NOT a unified diff. The unified diff format
  (what `git diff` produces) is what `unidiff` parses. We need the raw
  unified diff, which GitHub returns when you request:
    GET /repos/{owner}/{repo}/pulls/{pull_number}
    Accept: application/vnd.github.v3.diff

  PyGithub doesn't expose this directly, so we drop down to the underlying
  requester for this one call.

Size constraints:
  GitHub caps PR diffs at 300 files and 20,000 lines. Beyond that, the API
  returns a truncated diff. We log a warning when the diff is suspiciously
  large but don't reject it — the DiffParserNode's chunking handles large
  diffs correctly.
"""

from __future__ import annotations

import logging

from github import Github, GithubException

logger = logging.getLogger(__name__)

# Warn if diff exceeds this size — may indicate truncation
LARGE_DIFF_WARN_BYTES = 500_000  # 500 KB


def fetch_pr_diff(
    client: Github,
    repo_full_name: str,
    pr_number: int,
) -> str:
    """
    Fetch the unified diff for a pull request.

    Args:
        client:          Authenticated PyGithub client (from get_github_client).
        repo_full_name:  e.g. "ayushkaul/devmind"
        pr_number:       The PR number (integer).

    Returns:
        The unified diff as a string.

    Raises:
        GithubException: if the API returns an error (not found, rate limited,
                         permission denied, etc.).
        ValueError:      if the diff is empty (PR has no changed files).
    """
    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
    except GithubException as exc:
        logger.error(
            "diff_fetcher: failed to fetch PR %s #%d: %s %s",
            repo_full_name, pr_number, exc.status, exc.data,
        )
        raise

    # Drop down to the raw requester to get the unified diff format.
    # PyGithub's requester handles auth headers — we only override Accept.
    headers, raw_diff = pr._requester.requestBlobAndCheck(  # noqa: SLF001
        "GET",
        pr.url,
        headers={"Accept": "application/vnd.github.v3.diff"},
        input=None,
    )

    if isinstance(raw_diff, bytes):
        diff_text = raw_diff.decode("utf-8", errors="replace")
    else:
        diff_text = raw_diff

    if not diff_text or not diff_text.strip():
        raise ValueError(
            f"PR {repo_full_name}#{pr_number} returned an empty diff. "
            f"The PR may have no changed files."
        )

    if len(diff_text) > LARGE_DIFF_WARN_BYTES:
        logger.warning(
            "diff_fetcher: diff for %s #%d is %d bytes — may be truncated by GitHub",
            repo_full_name, pr_number, len(diff_text),
        )
    else:
        logger.info(
            "diff_fetcher: fetched %d bytes for %s #%d",
            len(diff_text), repo_full_name, pr_number,
        )

    return diff_text
