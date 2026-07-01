"""
tests/unit/test_style_checker.py
───────────────────────────────────
Unit tests for app/agent/nodes/style_checker.py.

Scope: node-level orchestration only. See test_security_checker.py for
rationale. Also verifies the style-specific confidence calibration
(CONFIDENCE_BY_SEVERITY is deliberately lower than the other two checkers —
see module docstring in style_checker.py).
"""

from unittest.mock import patch

import pytest

from app.agent.nodes.style_checker import run
from app.agent.state import AgentState, DiffChunk, initial_state
from app.llm.client import LLMCallResult


@pytest.fixture
def sample_chunk() -> DiffChunk:
    return DiffChunk(
        file_path="src/utils.py",
        start_line=20,
        end_line=25,
        content="+    timeout = 86400  # magic number",
        language="python",
        chunk_index=0,
    )


def _success_result(data: dict, prompt_tokens=100, completion_tokens=50) -> LLMCallResult:
    return LLMCallResult(
        success=True, data=data, error=None, raw_response=str(data),
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


def _failure_result(error: str) -> LLMCallResult:
    return LLMCallResult(success=False, data=None, error=error, raw_response=None)


class TestRunNode:

    def test_returns_empty_findings_when_no_chunks(self):
        state = initial_state(raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x")
        result = run(state)
        assert result["style_findings"] == []

    def test_single_chunk_single_finding_has_correct_category(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        mock_result = _success_result({"findings": [{
            "file_path": "src/utils.py", "line_number": 23,
            "severity": "suggestion",
            "comment": "Magic number 86400 should be extracted to a constant.",
        }]})

        with patch("app.agent.nodes.style_checker.call_llm_json", return_value=mock_result):
            result = run(state)

        assert len(result["style_findings"]) == 1
        finding = result["style_findings"][0]
        assert finding.category == "style"
        assert finding.source_node == "style_checker"

    def test_confidence_calibration_is_lower_than_default(self, sample_chunk):
        """
        StyleChecker deliberately uses a lower confidence mapping than
        SecurityChecker/BugDetector for the same nominal severity — a
        'critical' style finding still shouldn't outrank a 'critical'
        security finding when the synthesizer caps comments later.
        """
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        mock_result = _success_result({"findings": [{
            "file_path": "src/utils.py", "line_number": 23,
            "severity": "critical", "comment": "x",
        }]})

        with patch("app.agent.nodes.style_checker.call_llm_json", return_value=mock_result):
            result = run(state)

        # style_checker.CONFIDENCE_BY_SEVERITY["critical"] == 0.7, vs 0.9
        # for security_checker/bug_detector on the same nominal severity.
        assert result["style_findings"][0].confidence == 0.7

    def test_one_failed_chunk_does_not_block_others(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/helpers.py", start_line=1, end_line=5,
            content="+ def doStuff(): pass",
            language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]

        results = [
            _failure_result("JSONDecodeError: bad output"),
            _success_result({"findings": [
                {"file_path": "src/helpers.py", "line_number": 1,
                 "severity": "suggestion", "comment": "camelCase mixed with snake_case"}
            ]}),
        ]

        with patch("app.agent.nodes.style_checker.call_llm_json", side_effect=results):
            result = run(state)

        assert len(result["style_findings"]) == 1
        assert result.get("error") is None

    def test_all_chunks_failing_sets_error(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        with patch(
            "app.agent.nodes.style_checker.call_llm_json",
            return_value=_failure_result("RuntimeError: GROQ_API_KEY not set"),
        ):
            result = run(state)

        assert result["style_findings"] == []
        assert result.get("error") is not None

    def test_accumulates_token_usage_across_chunks(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/helpers.py", start_line=1, end_line=5,
            content="+ x = 1", language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]
        state["tokens_used"] = 0

        results = [
            _success_result({"findings": []}, prompt_tokens=60, completion_tokens=5),
            _success_result({"findings": []}, prompt_tokens=70, completion_tokens=8),
        ]

        with patch("app.agent.nodes.style_checker.call_llm_json", side_effect=results):
            result = run(state)

        assert result["tokens_used"] == 143

    def test_run_never_raises_on_unexpected_llm_output(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        for payload in [{"findings": "nope"}, {}, {"findings": [None, 1, "x"]}]:
            with patch(
                "app.agent.nodes.style_checker.call_llm_json",
                return_value=_success_result(payload),
            ):
                try:
                    result = run(state)
                    assert "style_findings" in result
                except Exception as exc:
                    pytest.fail(f"run() raised on payload {payload}: {exc}")
