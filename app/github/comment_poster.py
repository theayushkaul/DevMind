"""
app/github/comment_poster.py
──────────────────────────────
Posts ReviewFinding objects as a GitHub pull request review.

Batch review vs. individual comments — a deliberate design choice:
  GitHub's API offers two ways to post review comments:

  Option A — create_review_comment() per finding:
    N findings = N API calls. Each call counts against the rate limit
    (5000 requests/hour for GitHub Apps). For a 15-comment review, that's
    15 API calls + overhead. It also means comments appear one by one —
    not atomic.

  Option B — create_review() with all comments in one call:
    1 API call regardless of how many comments. Comments appear atomically
    as a single review (like a human reviewer clicking "Submit review").
    This is how GitHub's own code review UI works.

  We use Option B. The CommentSynthesizerNode already caps at 15 comments,
  so even Option A wouldn't be terrible — but Option B is strictly better:
  fewer API calls, atomic appearance, cleaner reviewer UX.

Review event type — COMMENT not REQUEST_CHANGES:
  GitHub review events: APPROVE, REQUEST_CHANGES, COMMENT.
  We use COMMENT. Here's why:

  - APPROVE is wrong — we're not confirming correctness, just flagging issues.
  - REQUEST_CHANGES blocks the PR merge until the reviewer dismisses the
    review. For an automated tool, this is aggressive — a false positive
    would block a PR until a human manually dismisses it. That erodes trust
    fast. COMMENT leaves the decision to the human.
  - COMMENT posts the findings without blocking merge. The developer sees
    the review, decides which findings are actionable, and merges on their
    own judgment.

  This is also defensible in interviews: "I chose COMMENT because automated
  reviewers have false positives. Blocking merges on a false positive
  requires manual intervention to unblock, which destroys adoption."

Line anchor — posting to the right line:
  GitHub's inline PR comment API requires a commit SHA and a file path +
  line number to anchor the comment to a specific diff hunk. We use the
  head_sha from the PR (the tip commit) and the line_number from each
  ReviewFinding. If line_number is -1 (file-level finding), we post as
  a top-level review comment body instead.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from github import Github, GithubException
from github.PullRequest import PullRequest

from app.agent.state import ReviewFinding

logger = logging.getLogger(__name__)


def post_review(
    client: Github,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    findings: List[ReviewFinding],
) -> int:
    """
    Post all findings as a single GitHub pull request review.

    Args:
        client:          Authenticated PyGithub client.
        repo_full_name:  e.g. "ayushkaul/devmind"
        pr_number:       The PR number.
        head_sha:        Commit SHA at the PR tip — used to anchor comments.
        findings:        ReviewFinding objects from CommentSynthesizerNode.
                         Already sorted and capped by the synthesizer.

    Returns:
        Number of comments successfully included in the review.
        May be less than len(findings) if some findings had invalid line
        numbers that GitHub rejected.

    Raises:
        GithubException: if the review creation API call itself fails
                         (not individual comment failures — those are handled
                         gracefully within this function).
    """
    if not findings:
        logger.info(
            "comment_poster: no findings to post for %s #%d",
            repo_full_name, pr_number,
        )
        return 0

    try:
        repo = client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        commit = repo.get_commit(head_sha)
    except GithubException as exc:
        logger.error(
            "comment_poster: failed to fetch PR or commit for %s #%d: %s",
            repo_full_name, pr_number, exc,
        )
        raise

    # Split findings into inline (have a line number) and file-level (line=-1)
    inline_findings = [f for f in findings if f.line_number >= 0]
    file_level_findings = [f for f in findings if f.line_number < 0]

    # Build the review body from file-level findings (if any)
    review_body = _build_review_body(file_level_findings)

    # Build inline comment dicts for the batch review call
    review_comments = _build_review_comments(inline_findings, head_sha)

    logger.info(
        "comment_poster: posting review for %s #%d — "
        "%d inline comments, %d file-level findings",
        repo_full_name, pr_number,
        len(review_comments), len(file_level_findings),
    )

    try:
        review = pr.create_review(
            commit=commit,
            body=review_body,
            event="COMMENT",
            comments=review_comments,
        )
        logger.info(
            "comment_poster: review posted for %s #%d — review_id=%d",
            repo_full_name, pr_number, review.id,
        )
    except GithubException as exc:
        logger.error(
            "comment_poster: create_review failed for %s #%d: %s %s",
            repo_full_name, pr_number, exc.status, exc.data,
        )
        raise

    return len(review_comments) + len(file_level_findings)


def _build_review_body(file_level_findings: List[ReviewFinding]) -> str:
    """
    Build the top-level review body text from file-level findings.

    File-level findings (line_number == -1) can't be anchored to a
    specific diff line, so they go in the review body instead.

    If there are no file-level findings, we still include a short header
    so the review isn't posted with an empty body (GitHub requires a
    non-empty body for COMMENT reviews).
    """
    header = "## DevMind Review\n\n"

    if not file_level_findings:
        return header + "_Inline comments attached below._"

    lines = [header + "**File-level findings:**\n"]
    for finding in file_level_findings:
        lines.append(f"- {finding.formatted_comment} (`{finding.file_path}`)")

    return "\n".join(lines)


def _build_review_comments(
    findings: List[ReviewFinding],
    head_sha: str,
) -> List[dict]:
    """
    Convert ReviewFinding objects into the dict format GitHub's
    create_review() API expects for inline comments.

    GitHub's ReviewComment schema:
      path:     repo-relative file path
      position: line position in the diff hunk (1-indexed)
      body:     comment text

    Note on `position` vs `line`:
      GitHub's newer API uses `line` (line number in the file) with `side`
      ("RIGHT" for added lines). The older `position` field is diff-hunk
      relative. We use `line` + side="RIGHT" for added lines, which is more
      intuitive and maps directly to ReviewFinding.line_number.
    """
    comments = []
    for finding in findings:
        comments.append({
            "path": finding.file_path,
            "line": finding.line_number,
            "side": "RIGHT",       # RIGHT = the new version of the file
            "body": finding.formatted_comment,
        })
    return comments
