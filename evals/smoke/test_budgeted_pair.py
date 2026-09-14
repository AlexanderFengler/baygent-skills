"""Controller checks use local dummy processes, never a model service."""

import json
import os
import sys

from run_hssm_handoff_pair import run_session, summarize_results


def test_success_preserves_usage_without_counting_cached_input_twice(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "prompt.txt").write_text("literal fixture")
    event = {
        "type": "turn.completed",
        "usage": {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10},
    }
    script = "import json; print(" + repr(json.dumps(event)) + ")"
    result = run_session(
        [sys.executable, "-c", script],
        workspace,
        os.environ.copy(),
        tmp_path / "evidence",
        5,
    )
    assert result["status"] == "completed"
    assert result["usage"] == event["usage"]
    assert result["model_turn_completed"]


def test_timeout_preserves_partial_evidence_without_fabricating_usage(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "prompt.txt").write_text("literal fixture")
    script = "import time; print('partial evidence', flush=True); time.sleep(10)"
    evidence = tmp_path / "evidence"
    result = run_session(
        [sys.executable, "-c", script], workspace, os.environ.copy(), evidence, 0.3
    )
    assert result["status"] == "timeout"
    assert result["usage"] is None
    assert not result["model_turn_completed"]
    assert result["elapsed_seconds"] < 3
    assert (evidence / "events.jsonl").read_text() == "partial evidence\n"


def test_failed_process_is_not_a_completed_agent_turn(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "prompt.txt").write_text("literal fixture")
    result = run_session(
        [sys.executable, "-c", "raise SystemExit(7)"],
        workspace,
        os.environ.copy(),
        tmp_path / "evidence",
        5,
    )
    assert result["returncode"] == 7
    assert result["status"] == "failed"
    assert result["usage"] is None


def test_native_budget_error_is_classified_without_reporting_zero_tokens(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "prompt.txt").write_text("literal fixture")
    event = {"type": "error", "message": "shared rollout token budget exhausted"}
    script = "print(" + repr(json.dumps(event)) + "); raise SystemExit(1)"
    result = run_session(
        [sys.executable, "-c", script],
        workspace,
        os.environ.copy(),
        tmp_path / "evidence",
        5,
    )
    assert result["status"] == "budget_exhausted"
    assert result["usage"] is None
    summary = summarize_results([result])
    assert summary["reported_input_plus_output_tokens"] is None
    assert not summary["usage_complete"]


def test_partial_usage_is_not_misreported_as_total():
    summary = summarize_results(
        [
            {
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 10,
                }
            },
            {"usage": None},
        ]
    )
    assert summary["known_reported_tokens_subtotal"] == 110
    assert summary["reported_input_plus_output_tokens"] is None
