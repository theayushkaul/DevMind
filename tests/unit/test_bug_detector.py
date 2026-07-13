"""
tests/unit/test_bug_detector.py
─────────────────────────────────
Unit tests for app/agent/nodes/bug_detector.py.

Scope: node-level orchestration only — same rationale as
test_security_checker.py. Shared JSON parsing logic is tested once in
test_finding_parser.py.
"""

from unittest.mock import patch

import pytest

from app.agent.nodes.bug_detector import run
from app.agent.state import AgentState, DiffChunk, initial_state
from app.llm.client import LLMCallResult


@pytest.fixture
def sample_chunk() -> DiffChunk:
    return DiffChunk(
        file_path="src/processor.py",
        start_line=80,
        end_line=90,
        content="+    for i in range(len(items)):\n+        process(items[i])",
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
        assert result["bug_findings"] == []

    def test_single_chunk_single_finding_has_correct_category(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        mock_result = _success_result({"findings": [{
            "file_path": "src/processor.py", "line_number": 87,
            "severity": "warning",
            "comment": "Off-by-one: loop runs to len(items) but last element never processed.",
        }]})

        with patch("app.agent.nodes.bug_detector.call_llm_json", return_value=mock_result):
            result = run(state)

        assert len(result["bug_findings"]) == 1
        finding = result["bug_findings"][0]
        assert finding.category == "bug"
        assert finding.source_node == "bug_detector"
        assert finding.severity == "warning"

    def test_one_failed_chunk_does_not_block_others(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/handler.py", start_line=1, end_line=5,
            content="+ except Exception:\n+     pass",
            language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]

        results = [
            _failure_result("JSONDecodeError: bad output"),
            _success_result({"findings": [
                {"file_path": "src/handler.py", "line_number": 1,
                 "severity": "warning", "comment": "Silently swallowing all exceptions"}
            ]}),
        ]

        with patch("app.agent.nodes.bug_detector.call_llm_json", side_effect=results):
            result = run(state)

        assert len(result["bug_findings"]) == 1
        assert result.get("error") is None

    def test_all_chunks_failing_sets_error(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        with patch(
            "app.agent.nodes.bug_detector.call_llm_json",
            return_value=_failure_result("RuntimeError: GROQ_API_KEY not set"),
        ):
            result = run(state)

        assert result["bug_findings"] == []
        assert result.get("error") is not None

    def test_accumulates_token_usage_across_chunks(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/handler.py", start_line=1, end_line=5,
            content="+ x = 1", language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]
        state["tokens_used"] = 0

        results = [
            _success_result({"findings": []}, prompt_tokens=80, completion_tokens=10),
            _success_result({"findings": []}, prompt_tokens=90, completion_tokens=15),
        ]

        with patch("app.agent.nodes.bug_detector.call_llm_json", side_effect=results):
            result = run(state)

        assert result["tokens_used"] == 195

    def test_run_never_raises_on_unexpected_llm_output(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        for payload in [{"findings": "nope"}, {}, {"findings": [None, 1, "x"]}]:
            with patch(
                "app.agent.nodes.bug_detector.call_llm_json",
                return_value=_success_result(payload),
            ):
                try:
                    result = run(state)
                    assert "bug_findings" in result
                except Exception as exc:
                    pytest.fail(f"run() raised on payload {payload}: {exc}")
