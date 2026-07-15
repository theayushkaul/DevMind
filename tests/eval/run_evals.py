"""
tests/eval/run_evals.py
────────────────────────
Evaluation framework for DevMind.

Runs the full agent pipeline against every sample in eval_dataset/samples.json,
compares findings to ground truth labels, and computes precision/recall metrics.
Results are written to eval_results.json.

Usage:
    # Requires real GROQ_API_KEY — this hits the actual LLM
    cd devmind
    export GROQ_API_KEY=gsk_...
    python tests/eval/run_evals.py

    # Run on a subset:
    python tests/eval/run_evals.py --sample-ids sql_injection_fstring hardcoded_api_key

    # Dry run (no real LLM calls — use mocked results):
    python tests/eval/run_evals.py --dry-run

Metrics computed:
    Precision:  Of all findings DevMind posts, what % match a ground truth issue?
    Recall:     Of all real issues, what % did DevMind catch?
    FP rate:    Of all findings, what % have no corresponding ground truth?
    Latency:    p50/p95 of pipeline execution time in seconds.

A "match" is defined as: same category AND line_number within ±3 lines
(LLMs sometimes off-by-one on line numbers) AND same file_path.

Results are written to tests/eval/eval_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.graph import run_pipeline
from app.agent.state import ReviewFinding

DATASET_PATH = Path(__file__).parent / "eval_dataset" / "samples.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

LINE_MATCH_TOLERANCE = 3  # findings within ±3 lines count as a match


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _is_match(finding: ReviewFinding, ground_truth: dict) -> bool:
    """
    A finding matches a ground truth label if:
      1. file_path matches exactly
      2. category matches exactly (security/bug/style)
      3. line_number is within ±LINE_MATCH_TOLERANCE lines

    We don't match on comment text — LLM phrasing varies, but the location
    and category are stable signals.
    """
    if finding.file_path != ground_truth["file_path"]:
        return False
    if finding.category != ground_truth["category"]:
        return False
    gt_line = ground_truth.get("line_number", -1)
    if gt_line >= 0 and abs(finding.line_number - gt_line) > LINE_MATCH_TOLERANCE:
        return False
    return True


def _evaluate_sample(
    sample: dict,
    findings: list[ReviewFinding],
) -> dict:
    """
    Compare findings against ground truth for one sample.

    Returns a per-sample result dict with:
      - true_positives:  findings that match a ground truth issue
      - false_positives: findings with no ground truth match
      - false_negatives: ground truth issues DevMind missed
      - precision:       TP / (TP + FP)
      - recall:          TP / (TP + FN)
    """
    ground_truths = sample.get("ground_truth", [])

    true_positives = 0
    false_positives = 0
    matched_gt_indices = set()

    for finding in findings:
        matched = False
        for i, gt in enumerate(ground_truths):
            if i not in matched_gt_indices and _is_match(finding, gt):
                true_positives += 1
                matched_gt_indices.add(i)
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(ground_truths) - len(matched_gt_indices)

    precision = true_positives / (true_positives + false_positives) if findings else 1.0
    recall = true_positives / len(ground_truths) if ground_truths else 1.0

    return {
        "sample_id": sample["id"],
        "description": sample["description"],
        "ground_truth_count": len(ground_truths),
        "findings_count": len(findings),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "findings": [
            {
                "file_path": f.file_path,
                "line_number": f.line_number,
                "category": f.category,
                "severity": f.severity,
                "comment": f.comment,
                "confidence": f.confidence,
            }
            for f in findings
        ],
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _compute_aggregate_metrics(results: list[dict], latencies: list[float]) -> dict:
    """Compute precision, recall, FP rate, and latency across all samples."""
    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    total_findings = sum(r["findings_count"] for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    fp_rate = total_fp / total_findings if total_findings > 0 else 0.0

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)] if latencies_sorted else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_positive_rate": round(fp_rate, 3),
        "latency_p50_seconds": round(p50, 2),
        "latency_p95_seconds": round(p95, 2),
        "total_samples": len(results),
        "total_findings": total_findings,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }


# ---------------------------------------------------------------------------
# Dry run mode — no real LLM calls
# ---------------------------------------------------------------------------

def _make_dry_run_result(severity: str, comment: str) -> Any:
    """Return a mock LLM result for dry-run mode."""
    from app.llm.client import LLMCallResult
    return LLMCallResult(
        success=True,
        data={"findings": [
            {
                "file_path": "src/placeholder.py",
                "line_number": 1,
                "severity": severity,
                "comment": f"[DRY RUN] {comment}",
            }
        ]},
        error=None,
        raw_response="{}",
        prompt_tokens=50,
        completion_tokens=10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evals(
    sample_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the evaluation suite.

    Args:
        sample_ids: If provided, only run these sample IDs. Otherwise run all.
        dry_run:    If True, mock LLM calls (no GROQ_API_KEY needed).

    Returns:
        Full results dict (also written to eval_results.json).
    """
    dataset = json.loads(DATASET_PATH.read_text())
    samples = dataset["samples"]

    if sample_ids:
        samples = [s for s in samples if s["id"] in sample_ids]
        if not samples:
            print(f"No samples found for IDs: {sample_ids}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DevMind Evaluation — {len(samples)} samples")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    per_sample_results = []
    latencies = []

    for i, sample in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] {sample['id']}...", end=" ", flush=True)

        start = time.monotonic()

        if dry_run:
            # Mock all LLM calls — tests infrastructure without spending tokens
            dry_result = _make_dry_run_result("warning", "dry run placeholder")
            with patch("app.agent.nodes.security_checker.call_llm_json", return_value=dry_result), \
                 patch("app.agent.nodes.bug_detector.call_llm_json", return_value=dry_result), \
                 patch("app.agent.nodes.style_checker.call_llm_json", return_value=dry_result), \
                 patch("app.rag.retriever.retrieve_context", return_value=[]):
                state = run_pipeline(
                    raw_diff=sample["diff"],
                    repo_full_name="eval/repo",
                    pr_number=i,
                    head_sha=f"eval{i:04d}",
                )
        else:
            # Real LLM calls — requires GROQ_API_KEY
            with patch("app.rag.retriever.retrieve_context", return_value=[]):
                state = run_pipeline(
                    raw_diff=sample["diff"],
                    repo_full_name="eval/repo",
                    pr_number=i,
                    head_sha=f"eval{i:04d}",
                )

        latency = time.monotonic() - start
        latencies.append(latency)

        findings = state.get("final_comments", [])
        result = _evaluate_sample(sample, findings)
        result["latency_seconds"] = round(latency, 2)
        result["tokens_used"] = state.get("tokens_used", 0)
        per_sample_results.append(result)

        status = "✅" if result["false_positives"] == 0 and result["false_negatives"] == 0 else "⚠️"
        print(
            f"{status} {latency:.1f}s | "
            f"TP={result['true_positives']} "
            f"FP={result['false_positives']} "
            f"FN={result['false_negatives']}"
        )

    # Aggregate
    aggregate = _compute_aggregate_metrics(per_sample_results, latencies)

    # Check against targets
    targets = dataset.get("metrics_target", {})
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    _print_metric("Precision",       aggregate["precision"],             targets.get("security_precision", 0.70), higher_is_better=True)
    _print_metric("Recall",          aggregate["recall"],                targets.get("bug_recall", 0.50), higher_is_better=True)
    _print_metric("FP rate",         aggregate["false_positive_rate"],   targets.get("false_positive_rate", 0.20), higher_is_better=False)
    _print_metric("Latency p50 (s)", aggregate["latency_p50_seconds"],   targets.get("latency_p50_seconds", 30), higher_is_better=False)
    print(f"  Latency p95:         {aggregate['latency_p95_seconds']:.1f}s")
    print(f"  Total findings:      {aggregate['total_findings']}")
    print(f"  Total TP/FP/FN:      {aggregate['total_true_positives']}/{aggregate['total_false_positives']}/{aggregate['total_false_negatives']}")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": dry_run,
        "aggregate": aggregate,
        "per_sample": per_sample_results,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to: {RESULTS_PATH}")

    return results


def _print_metric(name: str, value: float, target: float, higher_is_better: bool) -> None:
    met = (value >= target) if higher_is_better else (value <= target)
    symbol = "✅" if met else "❌"
    direction = "≥" if higher_is_better else "≤"
    print(f"  {name:<20} {value:.3f}  {symbol} (target: {direction}{target})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DevMind evaluation suite")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        metavar="ID",
        help="Run only specific sample IDs (e.g. sql_injection_fstring hardcoded_api_key)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock LLM calls — tests infra without spending Groq tokens",
    )
    args = parser.parse_args()

    run_evals(sample_ids=args.sample_ids, dry_run=args.dry_run)
