"""Prepare or execute the authorized two-session, no-fit HSSM handoff pilot.

Preparation is local only. Execution uses existing Codex authentication, preserves
execution rules, and launches at most two sessions without retries. Token targets
use under-development native tracking; only the wall timeout is controller-enforced.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_session(argv, workspace, env, evidence, timeout):
    """Preserve partial output, terminate the process group on timeout, never retry."""
    evidence.mkdir()
    started = time.monotonic()
    result = {"command": argv, "status": "running", "usage": None}
    write_json(evidence / "execution.json", result)
    with (
        (evidence / "events.jsonl").open("w") as stdout,
        (evidence / "stderr.txt").open("w") as stderr,
        (workspace / "prompt.txt").open() as stdin,
    ):
        proc = subprocess.Popen(
            argv,
            cwd=workspace,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
            result["status"] = "completed" if proc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    result.update(
        returncode=proc.returncode, elapsed_seconds=time.monotonic() - started
    )
    events = []
    for line in (evidence / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Retain partial/raw output; never infer missing usage.
    completed = [e for e in events if e.get("type") == "turn.completed"]
    if completed:
        result["usage"] = completed[-1].get("usage")
    errors = [e.get("message") for e in events if e.get("type") == "error"]
    result["errors"] = errors
    if "shared rollout token budget exhausted" in errors:
        result["status"] = "budget_exhausted"
    result["thread_ids"] = [
        e["thread_id"] for e in events if e.get("type") == "thread.started"
    ]
    result["model_turn_completed"] = bool(completed)
    write_json(evidence / "execution.json", result)
    return result


def summarize_results(results):
    complete = all(r["usage"] is not None for r in results)
    known_total = sum(
        r["usage"]["input_tokens"] + r["usage"]["output_tokens"]
        for r in results
        if r["usage"] is not None
    )
    return {
        "runs": results,
        "reported_input_plus_output_tokens": known_total if complete else None,
        "known_reported_tokens_subtotal": known_total,
        "usage_complete": complete,
    }


def prepare(repo, root, env_bin):
    original = json.loads((repo / "evals/agent-evaluation-round-1.json").read_text())
    scenario = next(s for s in original["scenarios"] if s["id"] == "hssm-misalignment")
    codex = Path(original["cli"]["candidate_argv"][0])
    assert sha(codex) == original["cli"]["binary_sha256"], "Codex binary changed"
    expected_env = original["environments"]["hssm"]
    requirement = expected_env["requirements_source"]
    assert sha(repo / requirement["source"]) == requirement["sha256"]
    check_script = (
        "import importlib.metadata,json,sys; "
        "r={k:importlib.metadata.version(k) for k in "
        + repr([k for k in expected_env["runtime_versions"] if k != "python"])
        + "}; r['python']='.'.join(map(str,sys.version_info[:3])); print(json.dumps(r))"
    )
    checked = subprocess.run(
        [str(env_bin / "python"), "-c", check_script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(checked.stdout) == expected_env["runtime_versions"]
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    for number, arm in ((1, "baseline"), (2, "treatment")):
        run_id = f"run-{number:02d}"
        workspace = root / run_id
        workspace.mkdir()
        (workspace / "results").mkdir()
        skill_names = ["bayesian-workflow"]
        if arm == "treatment":
            skill_names.append("hssm-workflow")
        files = list(scenario["inputs"])
        for name in skill_names:
            files.extend(original["skill_payloads"][name])
        hashes = {}
        for item in files:
            source, destination = repo / item["source"], workspace / item["destination"]
            assert sha(source) == item["sha256"], item["source"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            hashes[item["destination"]] = sha(destination)
        write_json(workspace / "environment.json", expected_env["runtime_versions"])
        assert (
            sha(workspace / "environment.json")
            == expected_env["environment_json_sha256"]
        )
        prompt = (
            original["shared_prompt_prefix"]
            + "\n\n"
            + scenario["prompt"]
            + "\n\n"
            + scenario["pilot_settings_text"]
            + "\n\nBudget for this review: 25,000 total input/output tokens as a target, "
            "10 minutes wall time, no retries. Read only the relevant installed skill "
            "sections; keep shell outputs and review.md concise. Save the requested "
            "review under results/review.md. Finish with a brief status.\n"
        )
        (workspace / "prompt.txt").write_text(prompt)
        hashes.update(
            {p: sha(workspace / p) for p in ("environment.json", "prompt.txt")}
        )
        argv = list(original["cli"]["candidate_argv"])
        argv = [a.replace("<RUN_DIR>", str(workspace)) for a in argv]
        argv[argv.index('model_reasoning_effort="ultra"')] = (
            'model_reasoning_effort="medium"'
        )
        argv[-1:-1] = [
            "--enable",
            "skip_host_skill_discovery",
            "-c",
            "features.rollout_budget.enabled=true",
            "-c",
            "features.rollout_budget.limit_tokens=25000",
            "-c",
            "features.rollout_budget.reminder_at_remaining_tokens=[10000,2500]",
            "-c",
            "tool_output_token_limit=2000",
        ]
        runs.append(
            {
                "id": run_id,
                "arm": arm,
                "workspace": str(workspace),
                "argv": argv,
                "input_sha256": hashes,
            }
        )
    manifest = {
        "kind": "authorized-reduced-hssm-handoff-pair",
        "authorization": "User: ok let's run this then; accepted two sessions, medium reasoning, 25k target each, 10-minute timeout each; no automatic expansion.",
        "source_manifest_sha256": sha(repo / "evals/agent-evaluation-round-1.json"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "codex_sha256": sha(codex),
        "environment_versions": expected_env["runtime_versions"],
        "environment_bin": str(env_bin),
        "maximum_sessions": 2,
        "automatic_retries": 0,
        "per_session_timeout_seconds": 600,
        "per_session_token_target": 25000,
        "combined_token_target": 50000,
        "model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "limit_caveat": "Native rollout tracking is under development; configuration parsing was verified, hard token enforcement was not. Stop after the first session if usage is absent or already exceeds the combined target. Timeout termination cannot recall a dispatched service request.",
        "isolation": "Fresh workspaces outside Spine, allowlisted copies, ignore user config, skip host skill discovery, disabled optional tools, preserved execpolicy. Actual model-visible catalog and hard host-read isolation are not proven; audit recorded actions for contamination.",
        "runs": runs,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def execute(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["runs"]) == 2 and manifest["maximum_sessions"] == 2
    assert [run["id"] for run in manifest["runs"]] == ["run-01", "run-02"]
    # Exclusive launch marker prevents an accidental second launch of this pair.
    with (root / "launched.txt").open("x") as marker:
        marker.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    results = []
    total = 0
    for run in manifest["runs"]:
        workspace = Path(run["workspace"])
        for path, expected in run["input_sha256"].items():
            assert sha(workspace / path) == expected, path
        env = {
            k: os.environ[k]
            for k in ("HOME", "USER", "LOGNAME", "TMPDIR", "CODEX_HOME")
            if k in os.environ
        }
        env.update(
            {
                "PATH": manifest["environment_bin"]
                + ":/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "LANG": "en_US.UTF-8",
                "MPLBACKEND": "Agg",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
        cache = root / (run["id"] + "-cache")
        cache.mkdir()
        env.update(
            MPLCONFIGDIR=str(cache / "mpl"),
            XDG_CACHE_HOME=str(cache),
            PYTENSOR_FLAGS="base_compiledir=" + str(cache / "pytensor"),
        )
        print("Starting " + run["id"], flush=True)
        result = run_session(
            run["argv"], workspace, env, evidence_root / run["id"], 600
        )
        result["run_id"] = run["id"]
        results.append(result)
        usage = result["usage"]
        if usage is not None:
            total += usage["input_tokens"] + usage["output_tokens"]
        write_json(
            root / "results.json",
            summarize_results(results),
        )
        print(
            json.dumps(
                {
                    "run": run["id"],
                    "status": result["status"],
                    "seconds": result["elapsed_seconds"],
                    "usage": usage,
                }
            ),
            flush=True,
        )
        if (
            result["status"] != "completed"
            or usage is None
            or total >= 50000
            or usage["input_tokens"] + usage["output_tokens"] > 25000
        ):
            print(
                "Stopping pair: failure, unavailable usage, or token target reached/exceeded.",
                flush=True,
            )
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-bin", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute(args.root.resolve())
    else:
        if args.env_bin is None:
            parser.error("--env-bin is required for local preparation")
        manifest = prepare(
            args.repo.resolve(), args.root.resolve(), args.env_bin.resolve()
        )
        print(
            json.dumps({"prepared": str(args.root), "sessions": len(manifest["runs"])})
        )


if __name__ == "__main__":
    main()
