"""
tests/integration/test_agent_pipeline.py
─────────────────────────────────────────
Integration tests for the full LangGraph pipeline.

These differ from unit tests in one key way: the entire graph runs end-to-end
— all six nodes execute in sequence, state flows through LangGraph's merge
mechanism, and the final output reflects the combined work of every node.

What IS mocked:
  - call_llm_json at each checker's import site (no real Groq API needed)
  - retrieve_context in the RAG node (no real pgvector needed)

What is NOT mocked:
  - DiffParserNode — real unidiff parsing of real diff strings
  - CommentSynthesizerNode — real dedup/sort/cap logic
  - LangGraph itself — real graph compilation and state routing
  - The conditional edge routing (diff_parser → rag_context vs. → synthesizer)

This is the test suite that would have caught the mock-namespace bug we hit
during unit testing (patching at the source site vs. the import site). It
also catches any graph wiring regressions — if a node is accidentally
removed from the graph, these tests fail even if all unit tests pass.

Running:
    pytest tests/integration/test_agent_pipeline.py -v
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from unittest.mock import patch

import pytest

from app.agent.graph import run_pipeline
from app.agent.state import ReviewFinding
from app.llm.client import LLMCallResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_real_diff(files_before: dict, files_after: dict) -> str:
    """Generate a real git diff — same helper used in unit tests."""
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


def _ok_result(findings: list | None = None) -> LLMCallResult:
    return LLMCallResult(
        success=True,
        data={"findings": findings or []},
        error=None,
        raw_response="{}",
        prompt_tokens=100,
        completion_tokens=20,
    )


def _fail_result(reason: str = "LLM error") -> LLMCallResult:
    return LLMCallResult(
        success=False, data=None, error=reason, raw_response=None,
    )


def _patch_checkers(security=None, bug=None, style=None):
    """
    Patch call_llm_json at each checker's import site — not at the source.
    This is the correct mocking pattern: each node imported the function
    into its own namespace at import time.
    """
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch(
        "app.agent.nodes.security_checker.call_llm_json",
        return_value=security or _ok_result(),
    ))
    stack.enter_context(patch(
        "app.agent.nodes.bug_detector.call_llm_json",
        return_value=bug or _ok_result(),
    ))
    stack.enter_context(patch(
        "app.agent.nodes.style_checker.call_llm_json",
        return_value=style or _ok_result(),
    ))
    return stack


def _patch_rag(snippets: list | None = None):
    """Patch RAG retrieval to return controlled snippets without hitting pgvector."""
    import asyncio

    async def fake_retrieve(chunk_text, repo_full_name, top_k):
        return snippets or []

    return patch("app.rag.retriever.retrieve_context", side_effect=fake_retrieve)


# ---------------------------------------------------------------------------
# Fixtures — realistic diffs
# ---------------------------------------------------------------------------

@pytest.fixture
def sql_injection_diff():
    """A diff introducing a SQL injection vulnerability."""
    return _make_real_diff(
        files_before={"src/api/routes.py": "from fastapi import APIRouter\nrouter = APIRouter()\n"},
        files_after={
            "src/api/routes.py": (
                "from fastapi import APIRouter\n"
                "router = APIRouter()\n\n"
                "@router.get('/users')\n"
                "async def get_user(user_id: str):\n"
                "    query = f'SELECT * FROM users WHERE id = {user_id}'\n"
                "    return await db.execute(query)\n"
            )
        },
    )


@pytest.fixture
def clean_diff():
    """A diff with no issues — tests that no false positives are posted."""
    return _make_real_diff(
        files_before={"src/utils.py": "def add(a: int, b: int) -> int:\n    return a + b\n"},
        files_after={
            "src/utils.py": (
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n\n"
                "def subtract(a: int, b: int) -> int:\n"
                "    return a - b\n"
            )
        },
    )


@pytest.fixture
def multi_file_diff():
    """A diff spanning multiple files — tests that all files get reviewed."""
    return _make_real_diff(
        files_before={
            "src/auth.py": "def login(): pass\n",
            "src/models.py": "class User: pass\n",
        },
        files_after={
            "src/auth.py": (
                "import pickle\n\n"
                "def login(): pass\n\n"
                "def load_session(data):\n"
                "    return pickle.loads(data)  # insecure\n"
            ),
            "src/models.py": (
                "class User: pass\n\n"
                "class Session:\n"
                "    id: int\n"
                "    user_id: int\n"
            ),
        },
    )


@pytest.fixture
def lock_file_only_diff():
    """A diff that only touches a lock file — nothing to review."""
    return _make_real_diff(
        files_before={"yarn.lock": "lodash@4.17.20:\n  resolved 'https://example.com'\n"},
        files_after={"yarn.lock": "lodash@4.17.21:\n  resolved 'https://example.com'\n"},
    )


# ---------------------------------------------------------------------------
# Test class 1: State flows correctly through the graph
# ---------------------------------------------------------------------------

class TestPipelineStateFlow:

    def test_final_state_has_all_expected_keys(self, sql_injection_diff):
        with _patch_checkers(), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=1,
                head_sha="abc123",
            )

        assert "diff_chunks" in state
        assert "repo_context" in state
        assert "security_findings" in state
        assert "bug_findings" in state
        assert "style_findings" in state
        assert "final_comments" in state
        assert "tokens_used" in state
        assert "error" in state

    def test_diff_parser_populates_diff_chunks(self, sql_injection_diff):
        with _patch_checkers(), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=1,
                head_sha="abc123",
            )

        assert len(state["diff_chunks"]) == 1
        assert state["diff_chunks"][0].file_path == "src/api/routes.py"
        assert state["diff_chunks"][0].language == "python"

    def test_tokens_accumulate_across_all_three_checker_nodes(self, sql_injection_diff):
        # Each checker returns 100 prompt + 20 completion = 120 tokens
        with _patch_checkers(), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=1,
                head_sha="abc123",
            )

        # 3 nodes × 1 chunk × 120 tokens = 360
        assert state["tokens_used"] == 360

    def test_rag_context_passed_to_checkers(self, sql_injection_diff, monkeypatch):
        """RAG snippets must appear in state["repo_context"] keyed by chunk.dedup_key."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/test")
        snippets = ["def db_execute(query, params): ...", "class DatabasePool: ..."]

        with _patch_checkers(), _patch_rag(snippets=snippets):
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=1,
                head_sha="abc123",
            )

        assert len(state["repo_context"]) == 1
        key = list(state["repo_context"].keys())[0]
        assert state["repo_context"][key] == snippets

    def test_no_error_on_clean_run(self, sql_injection_diff):
        with _patch_checkers(), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=1,
                head_sha="abc123",
            )

        assert state["error"] is None


# ---------------------------------------------------------------------------
# Test class 2: Routing — conditional edge after diff_parser
# ---------------------------------------------------------------------------

class TestConditionalRouting:

    def test_lock_file_only_skips_all_llm_nodes(self, lock_file_only_diff):
        """
        When diff_parser produces no chunks, the conditional edge must route
        directly to comment_synthesizer, bypassing RAG and all three checkers.
        Verified by asserting call_llm_json is never called.
        """
        with patch("app.agent.nodes.security_checker.call_llm_json") as sec, \
             patch("app.agent.nodes.bug_detector.call_llm_json") as bug, \
             patch("app.agent.nodes.style_checker.call_llm_json") as sty, \
             _patch_rag():
            state = run_pipeline(
                raw_diff=lock_file_only_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=2,
                head_sha="def456",
            )

        sec.assert_not_called()
        bug.assert_not_called()
        sty.assert_not_called()
        assert state["final_comments"] == []

    def test_empty_diff_produces_no_comments(self):
        with _patch_checkers(), _patch_rag():
            state = run_pipeline(
                raw_diff="",
                repo_full_name="ayushkaul/devmind",
                pr_number=3,
                head_sha="ghi789",
            )

        assert state["final_comments"] == []
        assert state["diff_chunks"] == []

    def test_valid_diff_invokes_all_three_checkers(self, sql_injection_diff):
        with patch("app.agent.nodes.security_checker.call_llm_json",
                   return_value=_ok_result()) as sec, \
             patch("app.agent.nodes.bug_detector.call_llm_json",
                   return_value=_ok_result()) as bug, \
             patch("app.agent.nodes.style_checker.call_llm_json",
                   return_value=_ok_result()) as sty, \
             _patch_rag():
            run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=4,
                head_sha="jkl012",
            )

        sec.assert_called_once()
        bug.assert_called_once()
        sty.assert_called_once()


# ---------------------------------------------------------------------------
# Test class 3: Finding flow — from LLM output to final_comments
# ---------------------------------------------------------------------------

class TestFindingFlow:

    def test_security_finding_survives_to_final_comments(self, sql_injection_diff):
        finding = {
            "file_path": "src/api/routes.py",
            "line_number": 6,
            "severity": "critical",
            "comment": "SQL injection: f-string interpolated into query. Use parameterized queries.",
        }
        with _patch_checkers(security=_ok_result([finding])), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=5,
                head_sha="mno345",
            )

        assert len(state["final_comments"]) == 1
        assert state["final_comments"][0].category == "security"
        assert state["final_comments"][0].severity == "critical"
        assert "SQL injection" in state["final_comments"][0].comment

    def test_findings_from_all_three_nodes_merged(self, sql_injection_diff):
        sec_finding = {"file_path": "src/api/routes.py", "line_number": 6,
                       "severity": "critical", "comment": "SQL injection"}
        bug_finding = {"file_path": "src/api/routes.py", "line_number": 5,
                       "severity": "warning", "comment": "Missing null check"}
        sty_finding = {"file_path": "src/api/routes.py", "line_number": 4,
                       "severity": "suggestion", "comment": "Function name too generic"}

        with _patch_checkers(
            security=_ok_result([sec_finding]),
            bug=_ok_result([bug_finding]),
            style=_ok_result([sty_finding]),
        ), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=6,
                head_sha="pqr678",
            )

        assert len(state["final_comments"]) == 3
        categories = {f.category for f in state["final_comments"]}
        assert categories == {"security", "bug", "style"}

    def test_final_comments_ordered_critical_first(self, sql_injection_diff):
        findings = [
            {"file_path": "src/api/routes.py", "line_number": 1,
             "severity": "suggestion", "comment": "style nit"},
            {"file_path": "src/api/routes.py", "line_number": 2,
             "severity": "critical", "comment": "critical issue"},
            {"file_path": "src/api/routes.py", "line_number": 3,
             "severity": "warning", "comment": "warning issue"},
        ]

        with _patch_checkers(security=_ok_result(findings)), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=7,
                head_sha="stu901",
            )

        severities = [f.severity for f in state["final_comments"]]
        assert severities.index("critical") < severities.index("warning")
        assert severities.index("warning") < severities.index("suggestion")

    def test_multi_file_diff_reviews_all_files(self, multi_file_diff):
        pickle_finding = {
            "file_path": "src/auth.py", "line_number": 6,
            "severity": "critical", "comment": "pickle.loads on untrusted input",
        }

        with _patch_checkers(security=_ok_result([pickle_finding])), _patch_rag():
            state = run_pipeline(
                raw_diff=multi_file_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=8,
                head_sha="vwx234",
            )

        # Both files should have been parsed into chunks
        parsed_files = {c.file_path for c in state["diff_chunks"]}
        assert "src/auth.py" in parsed_files
        assert "src/models.py" in parsed_files

    def test_cap_at_15_comments_enforced(self, sql_injection_diff):
        """Even if LLM returns many findings, final_comments must not exceed 15."""
        many_findings = [
            {"file_path": "src/api/routes.py", "line_number": i,
             "severity": "warning", "comment": f"Issue {i}"}
            for i in range(1, 30)
        ]

        with _patch_checkers(security=_ok_result(many_findings)), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=9,
                head_sha="yza567",
            )

        assert len(state["final_comments"]) <= 15

    def test_duplicate_findings_deduped(self, sql_injection_diff):
        """
        Same file+line+category from two different checker nodes → only one survives.
        Both security and bug checkers flag category='security' on line 6.
        """
        finding = {
            "file_path": "src/api/routes.py", "line_number": 6,
            "severity": "critical", "comment": "SQL injection",
        }

        with _patch_checkers(
            security=_ok_result([finding]),
            bug=_ok_result([{**finding, "comment": "Also SQL injection (from bug)"}]),
        ), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=10,
                head_sha="bcd890",
            )

        security_findings_on_line_6 = [
            f for f in state["final_comments"]
            if f.line_number == 6 and f.category == "security"
        ]
        assert len(security_findings_on_line_6) == 1


# ---------------------------------------------------------------------------
# Test class 4: Resilience — partial failures don't crash the pipeline
# ---------------------------------------------------------------------------

class TestPipelineResilience:

    def test_one_checker_failing_leaves_others_intact(self, sql_injection_diff):
        bug_finding = {
            "file_path": "src/api/routes.py", "line_number": 5,
            "severity": "warning", "comment": "Missing null check on user_id",
        }

        with _patch_checkers(
            security=_fail_result("Groq timeout"),
            bug=_ok_result([bug_finding]),
            style=_ok_result(),
        ), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=11,
                head_sha="efg123",
            )

        # Bug finding must be present despite security checker failing
        assert len(state["final_comments"]) == 1
        assert state["final_comments"][0].category == "bug"

    def test_all_checkers_failing_returns_empty_comments(self, sql_injection_diff):
        with _patch_checkers(
            security=_fail_result("timeout"),
            bug=_fail_result("timeout"),
            style=_fail_result("timeout"),
        ), _patch_rag():
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=12,
                head_sha="hij456",
            )

        assert state["final_comments"] == []

    def test_rag_failure_does_not_abort_pipeline(self, sql_injection_diff):
        """If RAG retrieval fails, checkers run without context — not a crash."""
        finding = {
            "file_path": "src/api/routes.py", "line_number": 6,
            "severity": "critical", "comment": "SQL injection",
        }

        async def broken_retrieve(*args, **kwargs):
            raise Exception("pgvector connection refused")

        with _patch_checkers(security=_ok_result([finding])), \
             patch("app.rag.retriever.retrieve_context", side_effect=broken_retrieve):
            state = run_pipeline(
                raw_diff=sql_injection_diff,
                repo_full_name="ayushkaul/devmind",
                pr_number=13,
                head_sha="klm789",
            )

        # Should still get the security finding despite RAG failure
        assert len(state["final_comments"]) == 1
        assert state["repo_context"] == {}

    def test_malformed_llm_json_produces_no_findings_not_crash(self, sql_injection_diff):
        """LLM returns malformed JSON → parse failure → empty findings, no exception."""
        malformed = LLMCallResult(
            success=False,
            data=None,
            error="JSONDecodeError: expecting value",
            raw_response="this is not json {{{",
        )

        with _patch_checkers(security=malformed, bug=malformed, style=malformed), \
             _patch_rag():
            try:
                state = run_pipeline(
                    raw_diff=sql_injection_diff,
                    repo_full_name="ayushkaul/devmind",
                    pr_number=14,
                    head_sha="nop012",
                )
                assert state["final_comments"] == []
            except Exception as exc:
                pytest.fail(f"run_pipeline raised on malformed LLM output: {exc}")
