"""
tests/unit/test_comment_synthesizer.py
─────────────────────────────────────────
Unit tests for app/agent/nodes/comment_synthesizer.py.

No mocking needed — the synthesizer is pure Python with no external
dependencies. Tests are therefore fast, hermetic, and exhaustive.

Test structure:
- TestDeduplicate       — collision handling, confidence tiebreaking
- TestSortByPriority    — severity ordering, confidence as tiebreaker
- TestCapAtLimit        — edge cases around the cap
- TestRunNode           — full node orchestration via AgentState
- TestCrossCheckerDedup — verifies the reasoning for *when* to dedup vs. not
"""

import pytest

from app.agent.nodes.comment_synthesizer import (
    MAX_COMMENTS_PER_PR,
    cap_at_limit,
    deduplicate,
    run,
    sort_by_priority,
)
from app.agent.state import ReviewFinding, initial_state


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _finding(
    file_path: str = "src/api.py",
    line_number: int = 10,
    category: str = "security",
    severity: str = "warning",
    comment: str = "test comment",
    confidence: float = 0.8,
    source_node: str = "security_checker",
) -> ReviewFinding:
    return ReviewFinding(
        file_path=file_path,
        line_number=line_number,
        category=category,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        comment=comment,
        confidence=confidence,
        source_node=source_node,
    )


# ---------------------------------------------------------------------------
# Tests: deduplicate()
# ---------------------------------------------------------------------------

class TestDeduplicate:

    def test_empty_list_returns_empty(self):
        assert deduplicate([]) == []

    def test_single_finding_passes_through(self):
        f = _finding()
        assert deduplicate([f]) == [f]

    def test_no_duplicates_returns_all(self):
        findings = [
            _finding(line_number=1),
            _finding(line_number=2),
            _finding(line_number=3),
        ]
        assert len(deduplicate(findings)) == 3

    def test_exact_duplicate_keeps_one(self):
        f1 = _finding(line_number=10, category="security", confidence=0.8)
        f2 = _finding(line_number=10, category="security", confidence=0.8)
        result = deduplicate([f1, f2])
        assert len(result) == 1

    def test_keeps_higher_confidence_on_collision(self):
        low  = _finding(line_number=10, category="security", confidence=0.5, comment="low")
        high = _finding(line_number=10, category="security", confidence=0.9, comment="high")
        result = deduplicate([low, high])
        assert len(result) == 1
        assert result[0].comment == "high"

    def test_keeps_higher_confidence_regardless_of_input_order(self):
        """Input order must not determine the winner — confidence does."""
        high = _finding(line_number=10, category="security", confidence=0.9, comment="high")
        low  = _finding(line_number=10, category="security", confidence=0.5, comment="low")
        result = deduplicate([high, low])
        assert result[0].comment == "high"

    def test_same_line_different_category_both_survive(self):
        """
        A security finding and a bug finding on the same line are NOT
        duplicates — they flag different categories of issue.
        """
        security = _finding(line_number=42, category="security", comment="SQL injection")
        bug      = _finding(line_number=42, category="bug",      comment="null check missing")
        result = deduplicate([security, bug])
        assert len(result) == 2

    def test_same_line_same_category_different_files_both_survive(self):
        """dedup key includes file_path — same line in different files is not a duplicate."""
        f1 = _finding(file_path="src/a.py", line_number=10, category="security")
        f2 = _finding(file_path="src/b.py", line_number=10, category="security")
        result = deduplicate([f1, f2])
        assert len(result) == 2

    def test_three_way_collision_keeps_highest_confidence(self):
        f1 = _finding(line_number=10, category="security", confidence=0.5, comment="f1")
        f2 = _finding(line_number=10, category="security", confidence=0.95, comment="f2")
        f3 = _finding(line_number=10, category="security", confidence=0.7, comment="f3")
        result = deduplicate([f1, f2, f3])
        assert len(result) == 1
        assert result[0].comment == "f2"

    def test_confidence_tie_keeps_first_encountered(self):
        """When confidence is equal, input order determines the winner (stable)."""
        f1 = _finding(line_number=10, category="security", confidence=0.8, comment="first")
        f2 = _finding(line_number=10, category="security", confidence=0.8, comment="second")
        result = deduplicate([f1, f2])
        assert result[0].comment == "first"


# ---------------------------------------------------------------------------
# Tests: sort_by_priority()
# ---------------------------------------------------------------------------

class TestSortByPriority:

    def test_empty_list_returns_empty(self):
        assert sort_by_priority([]) == []

    def test_critical_before_warning_before_suggestion(self):
        suggestion = _finding(severity="suggestion")
        warning    = _finding(severity="warning")
        critical   = _finding(severity="critical")
        result = sort_by_priority([suggestion, warning, critical])
        assert [f.severity for f in result] == ["critical", "warning", "suggestion"]

    def test_confidence_tiebreaker_within_same_severity(self):
        """Higher confidence ranks before lower confidence at the same severity."""
        low  = _finding(severity="warning", confidence=0.5, comment="low")
        high = _finding(severity="warning", confidence=0.9, comment="high")
        result = sort_by_priority([low, high])
        assert result[0].comment == "high"

    def test_severity_beats_confidence_across_tiers(self):
        """A low-confidence critical outranks a high-confidence warning."""
        low_conf_critical  = _finding(severity="critical",   confidence=0.1, comment="crit")
        high_conf_warning  = _finding(severity="warning",    confidence=0.99, comment="warn")
        result = sort_by_priority([high_conf_warning, low_conf_critical])
        assert result[0].comment == "crit"

    def test_sort_is_stable_for_equal_keys(self):
        """Equal severity and confidence — original order preserved."""
        f1 = _finding(severity="warning", confidence=0.8, comment="first")
        f2 = _finding(severity="warning", confidence=0.8, comment="second")
        result = sort_by_priority([f1, f2])
        assert result[0].comment == "first"

    def test_all_three_categories_mixed(self):
        findings = [
            _finding(severity="suggestion", confidence=0.9, category="style"),
            _finding(severity="critical",   confidence=0.6, category="security"),
            _finding(severity="warning",    confidence=0.8, category="bug"),
            _finding(severity="critical",   confidence=0.9, category="security"),
            _finding(severity="warning",    confidence=0.5, category="style"),
        ]
        result = sort_by_priority(findings)
        severities = [f.severity for f in result]
        # All criticals must come before all warnings, which come before suggestions
        assert severities.index("warning") > severities.index("critical")
        assert severities.index("suggestion") > severities.index("warning")

    def test_does_not_mutate_input_list(self):
        findings = [
            _finding(severity="suggestion"),
            _finding(severity="critical"),
        ]
        original_order = [f.severity for f in findings]
        sort_by_priority(findings)
        assert [f.severity for f in findings] == original_order


# ---------------------------------------------------------------------------
# Tests: cap_at_limit()
# ---------------------------------------------------------------------------

class TestCapAtLimit:

    def test_empty_list_returns_empty(self):
        assert cap_at_limit([], limit=15) == []

    def test_fewer_than_limit_returns_all(self):
        findings = [_finding() for _ in range(5)]
        assert cap_at_limit(findings, limit=15) == findings

    def test_exact_limit_returns_all(self):
        findings = [_finding() for _ in range(15)]
        assert cap_at_limit(findings, limit=15) == findings

    def test_exceeds_limit_truncates_to_limit(self):
        findings = [_finding(comment=str(i)) for i in range(30)]
        result = cap_at_limit(findings, limit=15)
        assert len(result) == 15

    def test_truncates_from_the_end(self):
        """First `limit` items are kept — list must be sorted by priority before capping."""
        findings = [_finding(comment=str(i)) for i in range(20)]
        result = cap_at_limit(findings, limit=5)
        assert [f.comment for f in result] == ["0", "1", "2", "3", "4"]

    def test_zero_limit_returns_empty(self):
        findings = [_finding() for _ in range(10)]
        assert cap_at_limit(findings, limit=0) == []

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be >= 0"):
            cap_at_limit([_finding()], limit=-1)


# ---------------------------------------------------------------------------
# Tests: run() — full node orchestration
# ---------------------------------------------------------------------------

class TestRunNode:

    def test_returns_empty_when_all_input_lists_empty(self):
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        result = run(state)
        assert result["final_comments"] == []

    def test_merges_findings_from_all_three_nodes(self):
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        state["security_findings"] = [_finding(category="security", line_number=1)]
        state["bug_findings"]      = [_finding(category="bug",      line_number=2)]
        state["style_findings"]    = [_finding(category="style",    line_number=3)]

        result = run(state)
        assert len(result["final_comments"]) == 3

    def test_output_is_sorted_critical_first(self):
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        state["security_findings"] = [
            _finding(severity="suggestion", line_number=1, comment="s"),
            _finding(severity="critical",   line_number=2, comment="c"),
        ]
        state["bug_findings"]   = []
        state["style_findings"] = []

        result = run(state)
        assert result["final_comments"][0].severity == "critical"

    def test_deduplication_runs_before_cap(self):
        """
        If 20 findings exist but 10 are duplicates, the cap should see 10
        unique findings, not 20 — dedup must happen before cap.
        """
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")

        # 16 unique findings + 4 duplicates of line 1, category security
        unique = [_finding(line_number=i, category="security") for i in range(2, 18)]
        duplicates = [
            _finding(line_number=1, category="security", confidence=0.9, comment="winner"),
            _finding(line_number=1, category="security", confidence=0.5, comment="loser_a"),
            _finding(line_number=1, category="security", confidence=0.4, comment="loser_b"),
            _finding(line_number=1, category="security", confidence=0.3, comment="loser_c"),
        ]
        state["security_findings"] = unique + duplicates
        state["bug_findings"]      = []
        state["style_findings"]    = []

        result = run(state)
        # 16 unique + 1 winner from dedup = 17, then capped at 15
        assert len(result["final_comments"]) == 15

        # The winner from the dedup collision must be in the output
        comments = [f.comment for f in result["final_comments"]]
        assert "winner" in comments
        assert "loser_a" not in comments

    def test_cap_applied_at_max_comments_per_pr(self):
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        # 30 unique findings (no duplicates)
        state["security_findings"] = [
            _finding(line_number=i, category="security") for i in range(30)
        ]
        state["bug_findings"]   = []
        state["style_findings"] = []

        result = run(state)
        assert len(result["final_comments"]) == MAX_COMMENTS_PER_PR

    def test_cross_checker_dedup_same_category_same_line(self):
        """
        SecurityChecker and BugDetector both flag a security issue on the
        same line — only the higher-confidence one should survive.
        """
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        state["security_findings"] = [
            _finding(line_number=42, category="security", confidence=0.9, comment="from_security")
        ]
        state["bug_findings"] = [
            _finding(line_number=42, category="security", confidence=0.6, comment="from_bug")
        ]
        state["style_findings"] = []

        result = run(state)
        assert len(result["final_comments"]) == 1
        assert result["final_comments"][0].comment == "from_security"

    def test_cross_checker_different_category_same_line_both_survive(self):
        """
        SecurityChecker flags SQL injection (security) on line 42.
        BugDetector flags a logic error (bug) on the same line.
        Different categories — both are real issues and both must survive.
        """
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        state["security_findings"] = [
            _finding(line_number=42, category="security", comment="SQL injection")
        ]
        state["bug_findings"] = [
            _finding(line_number=42, category="bug", comment="null check missing")
        ]
        state["style_findings"] = []

        result = run(state)
        assert len(result["final_comments"]) == 2

    def test_run_never_raises(self):
        """Fuzz-style: no input state should cause an exception."""
        bad_states = [
            initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x"),
        ]
        for state in bad_states:
            try:
                result = run(state)
                assert "final_comments" in result
            except Exception as exc:
                pytest.fail(f"run() raised {type(exc).__name__}: {exc}")

    def test_final_comments_are_reviewfinding_instances(self):
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        state["security_findings"] = [_finding()]
        state["bug_findings"]      = []
        state["style_findings"]    = []

        result = run(state)
        for f in result["final_comments"]:
            assert isinstance(f, ReviewFinding)

    def test_style_checker_lower_confidence_ranked_below_security(self):
        """
        Verifies that the cross-checker confidence calibration from
        style_checker.py (critical=0.7) ranks below security_checker.py's
        critical (confidence=0.9) when both are present.
        Sort key is (severity_rank, -confidence), so same severity tier
        ranks by confidence descending.
        """
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        state["security_findings"] = [
            _finding(severity="critical", confidence=0.9,
                     line_number=1, category="security", comment="security_crit")
        ]
        state["bug_findings"] = []
        state["style_findings"] = [
            _finding(severity="critical", confidence=0.7,
                     line_number=2, category="style", comment="style_crit")
        ]

        result = run(state)
        assert result["final_comments"][0].comment == "security_crit"
        assert result["final_comments"][1].comment == "style_crit"
