"""
tests/unit/test_github.py
──────────────────────────
Tests for app/github/client.py, diff_fetcher.py, and comment_poster.py.

Strategy: mock PyGithub objects at the boundary — we test OUR logic around
the PyGithub API (error handling, field extraction, comment formatting),
not PyGithub's own behaviour, which has its own test suite.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from github import GithubException

from app.github.comment_poster import (
    _build_review_body,
    _build_review_comments,
    post_review,
)
from app.github.diff_fetcher import fetch_pr_diff
from app.agent.state import ReviewFinding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _finding(
    file_path="src/api.py",
    line_number=10,
    category="security",
    severity="critical",
    comment="SQL injection risk",
    confidence=0.9,
) -> ReviewFinding:
    return ReviewFinding(
        file_path=file_path,
        line_number=line_number,
        category=category,  # type: ignore
        severity=severity,  # type: ignore
        comment=comment,
        confidence=confidence,
        source_node="security_checker",
    )


@pytest.fixture
def mock_github_client():
    return MagicMock()


@pytest.fixture
def mock_pr(mock_github_client):
    pr = MagicMock()
    pr.url = "https://api.github.com/repos/owner/repo/pulls/1"

    # Set up the requester to return a realistic diff
    diff_text = (
        "diff --git a/src/api.py b/src/api.py\n"
        "index abc..def 100644\n"
        "--- a/src/api.py\n"
        "+++ b/src/api.py\n"
        "@@ -1,3 +1,5 @@\n"
        " existing = True\n"
        "+new_line = 1\n"
        "+another = 2\n"
        " end = True\n"
    )
    pr._requester.requestBlobAndCheck.return_value = ({}, diff_text.encode())

    repo = MagicMock()
    repo.get_pull.return_value = pr
    repo.get_commit.return_value = MagicMock()
    mock_github_client.get_repo.return_value = repo
    return pr


# ---------------------------------------------------------------------------
# Tests: fetch_pr_diff()
# ---------------------------------------------------------------------------

class TestFetchPrDiff:

    def test_returns_diff_string(self, mock_github_client, mock_pr):
        result = fetch_pr_diff(mock_github_client, "owner/repo", 1)
        assert isinstance(result, str)
        assert "diff --git" in result

    def test_calls_correct_repo_and_pr(self, mock_github_client, mock_pr):
        fetch_pr_diff(mock_github_client, "owner/repo", 42)
        mock_github_client.get_repo.assert_called_once_with("owner/repo")
        mock_github_client.get_repo.return_value.get_pull.assert_called_once_with(42)

    def test_uses_diff_accept_header(self, mock_github_client, mock_pr):
        fetch_pr_diff(mock_github_client, "owner/repo", 1)
        call_kwargs = mock_pr._requester.requestBlobAndCheck.call_args
        headers = call_kwargs[1].get("headers") or call_kwargs[0][2]
        assert "application/vnd.github.v3.diff" in headers.get("Accept", "")

    def test_raises_on_empty_diff(self, mock_github_client, mock_pr):
        mock_pr._requester.requestBlobAndCheck.return_value = ({}, b"")
        with pytest.raises(ValueError, match="empty diff"):
            fetch_pr_diff(mock_github_client, "owner/repo", 1)

    def test_raises_on_github_exception(self, mock_github_client):
        mock_github_client.get_repo.side_effect = GithubException(
            status=404, data={"message": "Not Found"}, headers={}
        )
        with pytest.raises(GithubException):
            fetch_pr_diff(mock_github_client, "owner/nonexistent", 1)

    def test_decodes_bytes_response(self, mock_github_client, mock_pr):
        """If GitHub returns bytes (it always does), they should be decoded to str."""
        diff_bytes = b"diff --git a/f.py b/f.py\n@@ -1 +1 @@\n+x = 1\n"
        mock_pr._requester.requestBlobAndCheck.return_value = ({}, diff_bytes)
        result = fetch_pr_diff(mock_github_client, "owner/repo", 1)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: _build_review_body()
# ---------------------------------------------------------------------------

class TestBuildReviewBody:

    def test_no_file_level_findings_returns_header(self):
        body = _build_review_body([])
        assert "DevMind Review" in body
        assert "Inline comments" in body

    def test_file_level_findings_included_in_body(self):
        f = _finding(line_number=-1, comment="Missing module docstring")
        body = _build_review_body([f])
        assert "Missing module docstring" in body
        assert "src/api.py" in body

    def test_formatted_comment_used_in_body(self):
        f = _finding(severity="critical", comment="SQL injection", line_number=-1)
        body = _build_review_body([f])
        assert "🚨" in body  # formatted_comment includes severity emoji


# ---------------------------------------------------------------------------
# Tests: _build_review_comments()
# ---------------------------------------------------------------------------

class TestBuildReviewComments:

    def test_returns_list_of_dicts(self):
        findings = [_finding(line_number=10), _finding(line_number=20)]
        comments = _build_review_comments(findings, "abc123")
        assert isinstance(comments, list)
        assert len(comments) == 2
        assert all(isinstance(c, dict) for c in comments)

    def test_comment_contains_required_fields(self):
        f = _finding(file_path="src/api.py", line_number=42)
        comments = _build_review_comments([f], "abc123")
        c = comments[0]
        assert c["path"] == "src/api.py"
        assert c["line"] == 42
        assert "side" in c
        assert "body" in c

    def test_side_is_right(self):
        """RIGHT = the new version of the file (added lines)."""
        f = _finding(line_number=10)
        comments = _build_review_comments([f], "abc123")
        assert comments[0]["side"] == "RIGHT"

    def test_body_contains_formatted_comment(self):
        f = _finding(severity="warning", comment="Off-by-one error")
        comments = _build_review_comments([f], "abc123")
        assert "Off-by-one error" in comments[0]["body"]
        assert "⚠️" in comments[0]["body"]

    def test_empty_findings_returns_empty_list(self):
        assert _build_review_comments([], "abc123") == []


# ---------------------------------------------------------------------------
# Tests: post_review() — orchestration
# ---------------------------------------------------------------------------

class TestPostReview:

    def test_returns_zero_for_empty_findings(self, mock_github_client):
        count = post_review(mock_github_client, "owner/repo", 1, "abc", [])
        assert count == 0
        mock_github_client.get_repo.assert_not_called()

    def test_calls_create_review_once(self, mock_github_client, mock_pr):
        findings = [_finding(line_number=10), _finding(line_number=20)]
        post_review(mock_github_client, "owner/repo", 1, "abc123", findings)

        repo = mock_github_client.get_repo.return_value
        pr = repo.get_pull.return_value
        pr.create_review.assert_called_once()

    def test_create_review_event_is_comment(self, mock_github_client, mock_pr):
        """Must use COMMENT, not REQUEST_CHANGES — see comment_poster.py docstring."""
        findings = [_finding(line_number=10)]
        post_review(mock_github_client, "owner/repo", 1, "abc123", findings)

        repo = mock_github_client.get_repo.return_value
        pr = repo.get_pull.return_value
        call_kwargs = pr.create_review.call_args.kwargs
        assert call_kwargs["event"] == "COMMENT"

    def test_returns_count_of_posted_comments(self, mock_github_client, mock_pr):
        findings = [_finding(line_number=10), _finding(line_number=20)]
        count = post_review(mock_github_client, "owner/repo", 1, "abc123", findings)
        # 2 inline + 0 file-level = 2
        assert count == 2

    def test_file_level_findings_go_to_review_body_not_inline(
        self, mock_github_client, mock_pr
    ):
        """line_number == -1 means file-level — goes in body, not inline comments."""
        file_level = _finding(line_number=-1, comment="Missing docstring")
        inline = _finding(line_number=10, comment="SQL injection")

        post_review(mock_github_client, "owner/repo", 1, "abc123", [file_level, inline])

        repo = mock_github_client.get_repo.return_value
        pr = repo.get_pull.return_value
        call_kwargs = pr.create_review.call_args.kwargs

        # Body should mention the file-level finding
        assert "Missing docstring" in call_kwargs["body"]

        # Only the inline finding should be in the comments list
        assert len(call_kwargs["comments"]) == 1
        assert call_kwargs["comments"][0]["line"] == 10

    def test_raises_on_github_api_error(self, mock_github_client, mock_pr):
        repo = mock_github_client.get_repo.return_value
        pr = repo.get_pull.return_value
        pr.create_review.side_effect = GithubException(
            status=422, data={"message": "Validation Failed"}, headers={}
        )
        with pytest.raises(GithubException):
            post_review(
                mock_github_client, "owner/repo", 1, "abc123", [_finding()]
            )
