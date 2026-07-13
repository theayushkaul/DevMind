"""
tests/unit/test_security_checker.py
─────────────────────────────────────
Unit tests for app/agent/nodes/security_checker.py.

Scope: ONLY node-level orchestration — per-chunk LLM call isolation, token
accumulation, partial vs total failure handling. JSON response parsing and
validation logic is shared across all three checker nodes and is tested
once in test_finding_parser.py, not re-tested here.

Mocking strategy: patch `call_llm_json` at the point security_checker.py
imports it. This node doesn't know Groq exists — it only knows about
LLMCallResult — so mocking at this boundary tests exactly this node's logic.
"""

from unittest.mock import patch

import pytest

from app.agent.nodes.security_checker import run
from app.agent.state import AgentState, DiffChunk, initial_state
from app.llm.client import LLMCallResult


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
        assert result["security_findings"] == []

    def test_single_chunk_single_finding(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        mock_result = _success_result({"findings": [{
            "file_path": "src/api/routes.py", "line_number": 15,
            "severity": "critical", "comment": "SQL injection: f-string interpolated.",
        }]})

        with patch("app.agent.nodes.security_checker.call_llm_json", return_value=mock_result):
            result = run(state)

        assert len(result["security_findings"]) == 1
        finding = result["security_findings"][0]
        assert finding.severity == "critical"
        assert finding.category == "security"
        assert finding.source_node == "security_checker"

    def test_multiple_chunks_aggregate_findings(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/auth.py", start_line=1, end_line=5,
            content="+ token = jwt.decode(t, verify=False)",
            language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]

        results = [
            _success_result({"findings": [
                {"file_path": "src/api/routes.py", "line_number": 15,
                 "severity": "critical", "comment": "SQL injection"}
            ]}),
            _success_result({"findings": [
                {"file_path": "src/auth.py", "line_number": 3,
                 "severity": "warning", "comment": "JWT signature not verified"}
            ]}),
        ]

        with patch("app.agent.nodes.security_checker.call_llm_json", side_effect=results):
            result = run(state)

        assert len(result["security_findings"]) == 2

    def test_one_failed_chunk_does_not_block_others(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/auth.py", start_line=1, end_line=5,
            content="+ token = jwt.decode(t, verify=False)",
            language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]

        results = [
            _failure_result("JSONDecodeError: bad output"),
            _success_result({"findings": [
                {"file_path": "src/auth.py", "line_number": 3,
                 "severity": "warning", "comment": "JWT signature not verified"}
            ]}),
        ]

        with patch("app.agent.nodes.security_checker.call_llm_json", side_effect=results):
            result = run(state)

        assert len(result["security_findings"]) == 1
        assert result["security_findings"][0].file_path == "src/auth.py"
        assert result.get("error") is None

    def test_all_chunks_failing_sets_error(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        with patch(
            "app.agent.nodes.security_checker.call_llm_json",
            return_value=_failure_result("RuntimeError: GROQ_API_KEY not set"),
        ):
            result = run(state)

        assert result["security_findings"] == []
        assert result.get("error") is not None
        assert "GROQ_API_KEY" in result["error"]

    def test_accumulates_token_usage_across_chunks(self, sample_chunk):
        chunk2 = DiffChunk(
            file_path="src/auth.py", start_line=1, end_line=5,
            content="+ token = jwt.decode(t)", language="python", chunk_index=0,
        )
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk, chunk2]
        state["tokens_used"] = 0

        results = [
            _success_result({"findings": []}, prompt_tokens=100, completion_tokens=20),
            _success_result({"findings": []}, prompt_tokens=150, completion_tokens=30),
        ]

        with patch("app.agent.nodes.security_checker.call_llm_json", side_effect=results):
            result = run(state)

        assert result["tokens_used"] == 300

    def test_run_never_raises_on_unexpected_llm_output(self, sample_chunk):
        state: AgentState = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [sample_chunk]

        weird_payloads = [
            {"findings": "not a list"},
            {"findings": [None, 123, "string", {}]},
            {"no_findings_key": True},
            {},
        ]

        for payload in weird_payloads:
            with patch(
                "app.agent.nodes.security_checker.call_llm_json",
                return_value=_success_result(payload),
            ):
                try:
                    result = run(state)
                    assert "security_findings" in result
                except Exception as exc:
                    pytest.fail(f"run() raised on payload {payload}: {exc}")
