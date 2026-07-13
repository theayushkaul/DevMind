"""
tests/unit/test_finding_parser.py
─────────────────────────────────────
Unit tests for app/agent/nodes/_finding_parser.py — the shared response
parsing logic used by SecurityCheckerNode, LogicBugDetectorNode, and
StyleCheckerNode.

These tests exercise the parsing logic ONCE, independent of any specific
checker node. Each node's own test file (test_security_checker.py,
test_bug_detector.py, test_style_checker.py) only needs to verify
node-level orchestration (per-chunk isolation, token accumulation, error
propagation) — not re-test JSON shape validation, since that's covered here.
"""

import pytest

from app.agent.nodes._finding_parser import (
    build_user_content,
    coerce_line_number,
    extract_findings,
)
from app.agent.state import DiffChunk

CONFIDENCE_BY_SEVERITY = {"critical": 0.9, "warning": 0.75, "suggestion": 0.6}


@pytest.fixture
def sample_chunk() -> DiffChunk:
    return DiffChunk(
        file_path="src/api/routes.py",
        start_line=10,
        end_line=20,
        content="+    user_id = await db.execute(f\"INSERT INTO users VALUES ({name})\")",
        language="python",
        chunk_index=0,
    )


def _extract(data, chunk, category="security", source_node="security_checker"):
    return extract_findings(
        data, chunk,
        category=category,
        source_node=source_node,
        confidence_by_severity=CONFIDENCE_BY_SEVERITY,
        node_label="TestNode",
    )


# ---------------------------------------------------------------------------
# Tests: extract_findings()
# ---------------------------------------------------------------------------

class TestExtractFindings:

    def test_extracts_valid_findings(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 5, "severity": "critical", "comment": "bad"}
        ]}
        findings = _extract(data, sample_chunk)
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert findings[0].category == "security"
        assert findings[0].source_node == "security_checker"

    def test_returns_empty_for_missing_findings_key(self, sample_chunk):
        assert _extract({}, sample_chunk) == []

    def test_returns_empty_for_non_list_findings(self, sample_chunk):
        assert _extract({"findings": "oops"}, sample_chunk) == []

    def test_returns_empty_for_non_dict_data(self, sample_chunk):
        assert _extract(["not", "a", "dict"], sample_chunk) == []
        assert _extract(None, sample_chunk) == []
        assert _extract("a string", sample_chunk) == []

    def test_skips_invalid_entries_keeps_valid_ones(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "critical", "comment": "ok"},
            {"file_path": "a.py", "severity": "made_up_severity", "comment": "bad severity"},
            None,
            42,
            "a string entry",
            {"file_path": "a.py", "line_number": 2, "severity": "warning", "comment": ""},
        ]}
        findings = _extract(data, sample_chunk)
        assert len(findings) == 1
        assert findings[0].comment == "ok"

    def test_empty_findings_list_returns_empty(self, sample_chunk):
        assert _extract({"findings": []}, sample_chunk) == []

    def test_category_is_stamped_correctly_per_caller(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "warning", "comment": "x"}
        ]}
        bug_findings = _extract(data, sample_chunk, category="bug", source_node="bug_detector")
        assert bug_findings[0].category == "bug"
        assert bug_findings[0].source_node == "bug_detector"

    def test_confidence_mapping_is_applied(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "suggestion", "comment": "x"}
        ]}
        findings = _extract(data, sample_chunk)
        assert findings[0].confidence == 0.6

    def test_caller_supplied_confidence_mapping_used_not_hardcoded(self, sample_chunk):
        """Different nodes may calibrate confidence differently — verify the
        caller's mapping is actually used, not some hardcoded default."""
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "critical", "comment": "x"}
        ]}
        custom_mapping = {"critical": 0.42, "warning": 0.3, "suggestion": 0.1}
        findings = extract_findings(
            data, sample_chunk,
            category="style", source_node="style_checker",
            confidence_by_severity=custom_mapping,
            node_label="StyleCheckerNode",
        )
        assert findings[0].confidence == 0.42


# ---------------------------------------------------------------------------
# Tests: single-finding validation (via extract_findings, the public surface)
# ---------------------------------------------------------------------------

class TestSingleFindingValidation:

    def test_falls_back_to_chunk_file_path_when_missing(self, sample_chunk):
        data = {"findings": [
            {"line_number": 10, "severity": "critical", "comment": "issue"}
        ]}
        findings = _extract(data, sample_chunk)
        assert findings[0].file_path == sample_chunk.file_path

    def test_falls_back_to_chunk_file_path_when_wrong_type(self, sample_chunk):
        data = {"findings": [
            {"file_path": 12345, "line_number": 10, "severity": "critical", "comment": "issue"}
        ]}
        findings = _extract(data, sample_chunk)
        assert findings[0].file_path == sample_chunk.file_path

    def test_falls_back_to_chunk_start_line_when_line_number_unusable(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": "not_a_number",
             "severity": "critical", "comment": "issue"}
        ]}
        findings = _extract(data, sample_chunk)
        assert findings[0].line_number == sample_chunk.start_line

    def test_rejects_invalid_severity(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "apocalyptic", "comment": "x"}
        ]}
        assert _extract(data, sample_chunk) == []

    def test_rejects_missing_severity(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "comment": "x"}
        ]}
        assert _extract(data, sample_chunk) == []

    def test_rejects_empty_comment(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "critical", "comment": "   "}
        ]}
        assert _extract(data, sample_chunk) == []

    def test_rejects_missing_comment(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "critical"}
        ]}
        assert _extract(data, sample_chunk) == []

    def test_strips_whitespace_from_comment(self, sample_chunk):
        data = {"findings": [
            {"file_path": "a.py", "line_number": 1, "severity": "critical",
             "comment": "  has padding  "}
        ]}
        findings = _extract(data, sample_chunk)
        assert findings[0].comment == "has padding"


# ---------------------------------------------------------------------------
# Tests: coerce_line_number()
# ---------------------------------------------------------------------------

class TestCoerceLineNumber:

    def test_int_passes_through(self):
        assert coerce_line_number(42, fallback=1) == 42

    def test_float_truncates_to_int(self):
        assert coerce_line_number(42.7, fallback=1) == 42

    def test_numeric_string_converts(self):
        assert coerce_line_number("42", fallback=1) == 42

    def test_numeric_string_with_whitespace_converts(self):
        assert coerce_line_number("  42  ", fallback=1) == 42

    def test_non_numeric_string_falls_back(self):
        assert coerce_line_number("not a number", fallback=99) == 99

    def test_none_falls_back(self):
        assert coerce_line_number(None, fallback=99) == 99

    def test_bool_falls_back(self):
        """bool is a subclass of int in Python — must not silently coerce to 0/1."""
        assert coerce_line_number(True, fallback=99) == 99
        assert coerce_line_number(False, fallback=99) == 99

    def test_list_falls_back(self):
        assert coerce_line_number([1, 2, 3], fallback=99) == 99

    def test_dict_falls_back(self):
        assert coerce_line_number({"line": 42}, fallback=99) == 99


# ---------------------------------------------------------------------------
# Tests: build_user_content()
# ---------------------------------------------------------------------------

class TestBuildUserContent:

    def test_includes_file_path_and_diff(self, sample_chunk):
        content = build_user_content(sample_chunk, {})
        assert "src/api/routes.py" in content
        assert "INSERT INTO users" in content

    def test_omits_related_code_section_when_no_context(self, sample_chunk):
        content = build_user_content(sample_chunk, {})
        assert "Related code" not in content

    def test_includes_related_snippets_when_present(self, sample_chunk):
        repo_context = {
            sample_chunk.dedup_key: ["def db_execute(query): ...", "class DB: ..."]
        }
        content = build_user_content(sample_chunk, repo_context)
        assert "Related code" in content
        assert "def db_execute" in content
        assert "class DB" in content

    def test_uses_dedup_key_to_look_up_context(self, sample_chunk):
        """Context for a DIFFERENT chunk's dedup_key must not leak in."""
        wrong_key_context = {"some/other/file.py:0": ["unrelated snippet"]}
        content = build_user_content(sample_chunk, wrong_key_context)
        assert "unrelated snippet" not in content
