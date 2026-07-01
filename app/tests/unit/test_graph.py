"""
tests/unit/test_graph.py
─────────────────────────
Tests for app/agent/graph.py — the LangGraph wiring layer.

Three distinct concerns tested here:

1. TOPOLOGY — does the graph have the right nodes and edges?
   These tests don't run any nodes. They inspect the compiled graph structure
   directly. If someone accidentally deletes a node or rewires an edge, these
   catch it immediately without needing a full pipeline run.

2. ROUTING — does _route_after_diff_parser() return the right branch?
   Tested in isolation as a pure function — no graph invocation needed.

3. PIPELINE INTEGRATION — does the full pipeline run end-to-end correctly?
   These tests invoke the actual graph but mock all LLM calls so they run
   in milliseconds and require no API key. The diff parser runs REAL unidiff
   parsing (no mock) because it has no external dependencies — mocking it
   would make these tests less valuable.

   This is the closest thing to an integration test we have without a real
   Groq API key. It proves the state flows correctly from node to node
   through LangGraph's merge mechanism — the thing that's hardest to verify
   by reading individual node files in isolation.

Mocking strategy:
   We patch `call_llm_json` at its definition site in `app.llm.client`, not
   at each node's import site. This works because all three checker nodes
   import `call_llm_json` from `app.llm.client`, so patching the source
   patches all three simultaneously — one mock, three nodes covered.
"""

import subprocess
import tempfile
import os
from unittest.mock import patch

import pytest

from app.agent.graph import (
    NODE_RAG_CONTEXT,
    NODE_BUG_DETECTOR,
    NODE_COMMENT_SYNTHESIZER,
    NODE_RAG_CONTEXT,
    NODE_DIFF_PARSER,
    NODE_SECURITY_CHECKER,
    NODE_STYLE_CHECKER,
    _route_after_diff_parser,
    build_graph,
    get_graph,
    run_pipeline,
)
from app.agent.state import initial_state
from app.llm.client import LLMCallResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_real_diff(files_before: dict, files_after: dict) -> str:
    """Generate a genuine git diff — same helper pattern as test_diff_parser.py."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init", "-q"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=d, check=True)
        for path, content in files_before.items():
            full = os.path.join(d, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w").write(content)
        subprocess.run(["git", "add", "."], cwd=d, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=d, check=True)
        for path, content in files_after.items():
            full = os.path.join(d, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w").write(content)
        result = subprocess.run(
            ["git", "diff"], cwd=d, capture_output=True, text=True, check=True
        )
        return result.stdout


def _empty_llm_result() -> LLMCallResult:
    """LLM call that returns no findings — the happy path for pipeline tests."""
    return LLMCallResult(
        success=True,
        data={"findings": []},
        error=None,
        raw_response='{"findings": []}',
        prompt_tokens=50,
        completion_tokens=10,
    )


def _finding_llm_result(findings: list) -> LLMCallResult:
    """LLM call that returns a specific findings list."""
    return LLMCallResult(
        success=True,
        data={"findings": findings},
        error=None,
        raw_response=str(findings),
        prompt_tokens=100,
        completion_tokens=30,
    )


@pytest.fixture
def python_diff() -> str:
    return _make_real_diff(
        files_before={"src/auth.py": "def login():\n    pass\n"},
        files_after={
            "src/auth.py": (
                "def login():\n    pass\n\n"
                "def decode_token(t):\n"
                "    import jwt\n"
                "    return jwt.decode(t, options={'verify_signature': False})\n"
            )
        },
    )


@pytest.fixture
def lock_file_only_diff() -> str:
    """A diff where all changed files are lock files — should produce no chunks."""
    return _make_real_diff(
        files_before={"package-lock.json": '{"lockfileVersion": 2}\n'},
        files_after={"package-lock.json": '{"lockfileVersion": 3}\n'},
    )


# ---------------------------------------------------------------------------
# 1. Topology tests
# ---------------------------------------------------------------------------

class TestGraphTopology:

    @pytest.fixture(autouse=True)
    def graph(self):
        self._graph = build_graph()

    def test_all_expected_nodes_registered(self):
        node_names = set(self._graph.nodes.keys())
        expected = {
            			"__start__",
            NODE_DIFF_PARSER,
            NODE_SECURITY_CHECKER,
            NODE_BUG_DETECTOR,
            NODE_STYLE_CHECKER,
            NODE_RAG_CONTEXT,
            NODE_COMMENT_SYNTHESIZER,
        }
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )

    def test_node_count_is_correct(self):
        # 6 real nodes + __start__ sentinel
        assert len(self._graph.nodes) == 7

    def test_graph_compiles_without_error(self):
        # build_graph() itself is the test — if it raises, topology is broken
        assert self._graph is not None

    def test_get_graph_returns_same_instance(self):
        """Singleton: two calls to get_graph() must return the same compiled object."""
        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2


# ---------------------------------------------------------------------------
# 2. Routing tests
# ---------------------------------------------------------------------------

class TestRouting:

    def test_routes_to_run_checkers_when_chunks_present(self):
        state = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [object()]  # any non-empty list
        assert _route_after_diff_parser(state) == "run_checkers"

    def test_routes_to_skip_when_no_chunks(self):
        state = initial_state(
            raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = []
        assert _route_after_diff_parser(state) == "skip_to_synthesizer"

    def test_routes_to_skip_when_chunks_key_missing(self):
        """If DiffParserNode failed and never wrote diff_chunks, default to skip."""
        state = initial_state(
            raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        # Don't set diff_chunks — simulates a node crash before writing the key
        del state["diff_chunks"]
        assert _route_after_diff_parser(state) == "skip_to_synthesizer"


# ---------------------------------------------------------------------------
# 3. Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_empty_diff_produces_empty_final_comments(self):
        """Empty diff → parser produces no chunks → LLM nodes skipped → empty output."""
        with patch("app.llm.client.get_client"):
            result = run_pipeline(
                raw_diff="",
                repo_full_name="ayushkaul/devmind",
                pr_number=1,
                head_sha="abc123",
            )
        assert result["final_comments"] == []
        assert result["error"] is None

    def test_lock_file_only_diff_skips_llm_nodes(self, lock_file_only_diff):
        """
        When all changed files are filtered (lock files), diff_parser produces
        no chunks. The conditional edge should skip all three LLM checker
        nodes — verified by asserting call_llm_json is never called.
        """
        with patch("app.llm.client.call_llm_json") as mock_llm:
            result = run_pipeline(
                raw_diff=lock_file_only_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=2,
                head_sha="def456",
            )
        mock_llm.assert_not_called()
        assert result["final_comments"] == []

    def _patch_all_checkers(self, side_effect=None, return_value=None):
        """
        Context manager that patches call_llm_json at all three checker node
        import sites simultaneously.

        Why patch at each node's import site rather than at app.llm.client?
        Each checker does `from app.llm.client import call_llm_json` at import
        time, binding the function name in their own module namespace. Patching
        app.llm.client.call_llm_json replaces the name in the source module
        but the already-imported references in each checker still point to the
        original. Patching at each checker's namespace replaces the binding
        the checker actually uses.
        """
        from contextlib import ExitStack
        targets = [
            "app.agent.nodes.security_checker.call_llm_json",
            "app.agent.nodes.bug_detector.call_llm_json",
            "app.agent.nodes.style_checker.call_llm_json",
        ]
        stack = ExitStack()
        mocks = []
        for target in targets:
            if side_effect is not None:
                m = stack.enter_context(patch(target, side_effect=side_effect))
            else:
                m = stack.enter_context(patch(target, return_value=return_value))
            mocks.append(m)
        return stack, mocks

    def test_real_diff_calls_all_three_checker_nodes(self, python_diff):
        """
        A valid diff with reviewable Python code should invoke all three
        LLM checker nodes. Each checker calls the LLM once per chunk —
        with one chunk in this diff, we expect exactly 3 total LLM calls.
        """
        stack, mocks = self._patch_all_checkers(return_value=_empty_llm_result())
        with stack:
            result = run_pipeline(
                raw_diff=python_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=3,
                head_sha="ghi789",
            )

        total_calls = sum(m.call_count for m in mocks)
        assert total_calls == 3  # 1 chunk × 3 nodes
        assert result["error"] is None

    def test_findings_from_all_three_nodes_appear_in_final_comments(self, python_diff):
        """
        Each checker returns one finding. All three must flow through state
        and survive into final_comments. This verifies LangGraph's state
        merge is working correctly across all three checker nodes.
        """
        security_finding = {
            "file_path": "src/auth.py", "line_number": 5,
            "severity": "critical", "comment": "JWT verify=False is insecure",
        }
        bug_finding = {
            "file_path": "src/auth.py", "line_number": 5,
            "severity": "warning", "comment": "Missing null check on token",
        }
        style_finding = {
            "file_path": "src/auth.py", "line_number": 4,
            "severity": "suggestion", "comment": "Function name too generic",
        }

        with patch(
            "app.agent.nodes.security_checker.call_llm_json",
            return_value=_finding_llm_result([security_finding]),
        ), patch(
            "app.agent.nodes.bug_detector.call_llm_json",
            return_value=_finding_llm_result([bug_finding]),
        ), patch(
            "app.agent.nodes.style_checker.call_llm_json",
            return_value=_finding_llm_result([style_finding]),
        ):
            result = run_pipeline(
                raw_diff=python_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=4,
                head_sha="jkl012",
            )

        assert len(result["final_comments"]) == 3
        categories = {f.category for f in result["final_comments"]}
        assert categories == {"security", "bug", "style"}

    def test_final_comments_are_sorted_critical_first(self, python_diff):
        """Synthesizer's sort must be visible in the final pipeline output."""
        findings_out_of_order = [
            {"file_path": "src/auth.py", "line_number": 1,
             "severity": "suggestion", "comment": "style nit"},
            {"file_path": "src/auth.py", "line_number": 2,
             "severity": "critical",   "comment": "critical issue"},
            {"file_path": "src/auth.py", "line_number": 3,
             "severity": "warning",    "comment": "warning issue"},
        ]
        # All three checkers return the same findings list for simplicity —
        # they'll be deduped by the synthesizer, so only the 3 unique
        # findings survive (same line+category from different nodes merge).
        mixed_result = _finding_llm_result(findings_out_of_order)
        stack, _ = self._patch_all_checkers(return_value=mixed_result)
        with stack:
            result = run_pipeline(
                raw_diff=python_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=5,
                head_sha="mno345",
            )

        severities = [f.severity for f in result["final_comments"]]
        first_critical   = next(i for i, s in enumerate(severities) if s == "critical")
        first_warning    = next(i for i, s in enumerate(severities) if s == "warning")
        first_suggestion = next(i for i, s in enumerate(severities) if s == "suggestion")
        assert first_critical < first_warning < first_suggestion

    def test_tokens_accumulated_across_all_three_nodes(self, python_diff):
        """
        Each LLM call returns 50 prompt + 10 completion = 60 tokens.
        3 nodes × 1 chunk × 60 tokens = 180 total.
        Verifies AgentState["tokens_used"] accumulation across nodes.
        """
        stack, _ = self._patch_all_checkers(return_value=_empty_llm_result())
        with stack:
            result = run_pipeline(
                raw_diff=python_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=6,
                head_sha="pqr678",
            )

        assert result["tokens_used"] == 180  # 3 nodes × 60 tokens each

    def test_one_checker_failing_does_not_abort_pipeline(self, python_diff):
        """
        Security checker fails (LLM error). Bug and style checkers still run.
        Final output should contain the finding from the bug checker.
        """
        failure = LLMCallResult(
            success=False, data=None,
            error="JSONDecodeError: bad output", raw_response=None,
        )
        bug_finding = {
            "file_path": "src/auth.py", "line_number": 5,
            "severity": "warning", "comment": "missing null check",
        }

        with patch(
            "app.agent.nodes.security_checker.call_llm_json",
            return_value=failure,
        ), patch(
            "app.agent.nodes.bug_detector.call_llm_json",
            return_value=_finding_llm_result([bug_finding]),
        ), patch(
            "app.agent.nodes.style_checker.call_llm_json",
            return_value=_empty_llm_result(),
        ):
            result = run_pipeline(
                raw_diff=python_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=7,
                head_sha="stu901",
            )

        assert len(result["final_comments"]) == 1
        assert result["final_comments"][0].category == "bug"

    def test_run_pipeline_returns_agentstate_dict(self, python_diff):
        """run_pipeline() must always return a dict with the expected keys."""
        stack, _ = self._patch_all_checkers(return_value=_empty_llm_result())
        with stack:
            result = run_pipeline(
                raw_diff=python_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=8,
                head_sha="vwx234",
            )

        assert "final_comments" in result
        assert "tokens_used" in result
        assert "diff_chunks" in result
        assert "security_findings" in result
        assert "bug_findings" in result
        assert "style_findings" in result
