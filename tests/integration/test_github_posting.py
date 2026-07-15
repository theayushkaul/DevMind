"""
tests/integration/test_github_posting.py
──────────────────────────────────────────
Integration tests for the GitHub review posting flow.

Tests the full path from ReviewFinding objects → formatted comments → GitHub API call,
without hitting the real GitHub API. PyGithub is mocked at the client level.

What IS mocked:
  - The PyGithub Github client (no real API calls)
  - GithubIntegration (no real JWT signing)

What is NOT mocked:
  - comment_poster.post_review() — real orchestration logic
  - _build_review_body() — real body formatting
  - _build_review_comments() — real inline comment building
  - ReviewFinding.formatted_comment — real emoji + severity formatting
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from app.agent.state import ReviewFinding
from app.github.comment_poster import (
    _build_review_body,
    _build_review_comments,
    post_review,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _finding(
    file_path: str = "src/api.py",
    line_number: int = 10,
    category: str = "security",
    severity: str = "critical",
    comment: str = "SQL injection risk — use parameterized queries",
    confidence: float = 0.9,
) -> ReviewFinding:
    return ReviewFinding(
        file_path=file_path,
        line_number=line_number,
        category=category,      # type: ignore[arg-type]
        severity=severity,      # type: ignore[arg-type]
        comment=comment,
        confidence=confidence,
        source_node="security_checker",
    )


@pytest.fixture
def mock_github_client():
    client = MagicMock()
    repo = MagicMock()
    pr = MagicMock()
    commit = MagicMock()

    client.get_repo.return_value = repo
    repo.get_pull.return_value = pr
    repo.get_commit.return_value = commit
    pr.create_review.return_value = MagicMock()

    return client, repo, pr, commit


# ---------------------------------------------------------------------------
# Tests: _build_review_body()
# ---------------------------------------------------------------------------

class TestBuildReviewBody:

    def test_body_contains_devmind_header(self):
        body = _build_review_body([])
        assert "DevMind" in body

    def test_body_mentions_inline_comments_when_no_file_level(self):
        body = _build_review_body([])
        assert "inline" in body.lower() or "comment" in body.lower()

    def test_file_level_finding_appears_in_body(self):
        """line_number == -1 means file-level — goes in body, not inline."""
        f = _finding(line_number=-1, comment="Missing module docstring")
        body = _build_review_body([f])
        assert "Missing module docstring" in body

    def test_multiple_file_level_findings_all_appear(self):
        findings = [
            _finding(line_number=-1, comment="No type hints", file_path="src/a.py"),
            _finding(line_number=-1, comment="No docstring", file_path="src/b.py"),
        ]
        body = _build_review_body(findings)
        assert "No type hints" in body
        assert "No docstring" in body

    def test_critical_emoji_in_body(self):
        f = _finding(line_number=-1, severity="critical", comment="Critical issue")
        body = _build_review_body([f])
        assert "🚨" in body

    def test_warning_emoji_in_body(self):
        f = _finding(line_number=-1, severity="warning", comment="Warning issue")
        body = _build_review_body([f])
        assert "⚠️" in body

    def test_suggestion_emoji_in_body(self):
        f = _finding(line_number=-1, severity="suggestion", comment="Suggestion")
        body = _build_review_body([f])
        assert "💡" in body


# ---------------------------------------------------------------------------
# Tests: _build_review_comments()
# ---------------------------------------------------------------------------

class TestBuildReviewComments:

    def test_empty_findings_returns_empty_list(self):
        assert _build_review_comments([], "abc123") == []

    def test_each_finding_becomes_one_comment(self):
        findings = [_finding(line_number=10), _finding(line_number=20)]
        comments = _build_review_comments(findings, "abc123")
        assert len(comments) == 2

    def test_comment_has_required_fields(self):
        f = _finding(file_path="src/api.py", line_number=42)
        comments = _build_review_comments([f], "abc123")
        c = comments[0]
        assert "path" in c
        assert "line" in c
        assert "body" in c
        assert "side" in c

    def test_path_and_line_are_correct(self):
        f = _finding(file_path="src/api.py", line_number=42)
        comments = _build_review_comments([f], "abc123")
        assert comments[0]["path"] == "src/api.py"
        assert comments[0]["line"] == 42

    def test_side_is_right(self):
        """RIGHT = the new version of the file (where additions appear)."""
        f = _finding(line_number=10)
        comments = _build_review_comments([f], "abc123")
        assert comments[0]["side"] == "RIGHT"

    def test_body_contains_formatted_comment(self):
        f = _finding(severity="critical", comment="SQL injection here")
        comments = _build_review_comments([f], "abc123")
        assert "SQL injection here" in comments[0]["body"]
        assert "🚨" in comments[0]["body"]

    def test_file_level_findings_excluded(self):
        """
        _build_review_comments receives already-filtered inline findings
        (post_review splits them before calling this function).
        Verify only inline findings (line_number >= 0) produce comment dicts.
        """
        file_level = _finding(line_number=-1, comment="file-level issue")
        inline = _finding(line_number=10, comment="inline issue")
        # Simulate what post_review does: filter before passing to _build_review_comments
        inline_only = [f for f in [file_level, inline] if f.line_number >= 0]
        comments = _build_review_comments(inline_only, "abc123")
        assert len(comments) == 1
        assert "file-level issue" not in comments[0]["body"]


# ---------------------------------------------------------------------------
# Tests: post_review() — full orchestration
# ---------------------------------------------------------------------------

class TestPostReview:

    def test_returns_zero_for_empty_findings(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        count = post_review(client, "owner/repo", 1, "abc123", [])
        assert count == 0
        repo.get_pull.assert_not_called()

    def test_calls_create_review_once(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        findings = [_finding(line_number=10), _finding(line_number=20)]
        post_review(client, "owner/repo", 1, "abc123", findings)
        pr.create_review.assert_called_once()

    def test_create_review_uses_comment_event(self, mock_github_client):
        """
        Must use COMMENT event, not REQUEST_CHANGES.
        COMMENT posts without blocking the PR from being merged —
        REQUEST_CHANGES would block merge until the reviewer dismisses it.
        An AI reviewer blocking human-written code from merging is bad UX.
        """
        client, repo, pr, commit = mock_github_client
        post_review(client, "owner/repo", 1, "abc123", [_finding(line_number=10)])
        kwargs = pr.create_review.call_args.kwargs
        assert kwargs["event"] == "COMMENT"

    def test_inline_findings_passed_as_comments(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        findings = [_finding(line_number=10), _finding(line_number=20)]
        post_review(client, "owner/repo", 1, "abc123", findings)
        kwargs = pr.create_review.call_args.kwargs
        assert len(kwargs["comments"]) == 2

    def test_file_level_findings_in_body_not_comments(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        file_level = _finding(line_number=-1, comment="Missing docstring")
        inline = _finding(line_number=10, comment="SQL injection")
        post_review(client, "owner/repo", 1, "abc123", [file_level, inline])

        kwargs = pr.create_review.call_args.kwargs
        assert "Missing docstring" in kwargs["body"]
        assert len(kwargs["comments"]) == 1  # only the inline finding
        assert kwargs["comments"][0]["line"] == 10

    def test_returns_inline_comment_count(self, mock_github_client):
        """
        post_review returns len(inline_comments) + len(file_level_findings).
        Both inline and file-level findings are counted in the total.
        """
        client, repo, pr, commit = mock_github_client
        findings = [
            _finding(line_number=10),   # inline
            _finding(line_number=20),   # inline
            _finding(line_number=-1),   # file-level — counted in body, not inline
        ]
        count = post_review(client, "owner/repo", 1, "abc123", findings)
        # 2 inline + 1 file-level = 3 total
        assert count == 3

    def test_uses_correct_repo_and_pr_number(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        post_review(client, "owner/repo", 42, "abc123", [_finding(line_number=10)])
        client.get_repo.assert_called_with("owner/repo")
        repo.get_pull.assert_called_with(42)

    def test_raises_on_github_api_error(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        pr.create_review.side_effect = GithubException(
            status=422, data={"message": "Validation Failed"}, headers={}
        )
        with pytest.raises(GithubException):
            post_review(client, "owner/repo", 1, "abc123", [_finding(line_number=10)])

    def test_all_severity_types_post_correctly(self, mock_github_client):
        client, repo, pr, commit = mock_github_client
        findings = [
            _finding(severity="critical",   comment="Critical",   line_number=1),
            _finding(severity="warning",    comment="Warning",    line_number=2),
            _finding(severity="suggestion", comment="Suggestion", line_number=3),
        ]
        count = post_review(client, "owner/repo", 1, "abc123", findings)
        assert count == 3

        kwargs = pr.create_review.call_args.kwargs
        bodies = [c["body"] for c in kwargs["comments"]]
        assert any("🚨" in b for b in bodies)
        assert any("⚠️" in b for b in bodies)
        assert any("💡" in b for b in bodies)
