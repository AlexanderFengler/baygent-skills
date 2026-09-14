"""Deterministic report regressions using the real shared rating implementation.

Evidence below is literal test data. No model construction, simulation, sampling,
or diagnostic estimation runs; report rendering and helper process failures are
tested separately from those expensive operations.
"""

import copy
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reporting():
    return load_module(
        "hssm_reporting_under_test",
        REPO / "examples/hssm-workflow/hssm_example_support.py",
    )


@pytest.fixture(scope="module")
def checker():
    return load_module(
        "shared_bayesian_checker",
        REPO / "bayesian-workflow/scripts/check_diagnostics.py",
    )


def diagnostics_fixture(convergence="excellent", loo="excellent"):
    convergence_data = {
        "all_ok": convergence == "excellent",
        "rhat": {"ok": True, "max": 1.001, "problematic_params": []},
        "ess_bulk": {"ok": True, "min": 830, "problematic_params": []},
        "ess_tail": {"ok": True, "min": 720, "problematic_params": []},
        "divergences": {"ok": True, "count": 0, "pct": 0.0},
    }
    if convergence in {"fair", "poor"}:
        convergence_data["rhat"] = {
            "ok": False,
            "max": 1.02 if convergence == "fair" else 1.2,
            "problematic_params": ["v"],
        }
    if loo == "not computed":
        loo_data = {
            "computed": False,
            "error": "Pointwise likelihood unavailable in fixture.",
        }
    else:
        loo_data = {
            "computed": True,
            "pareto_k": {
                "max": {"excellent": 0.2, "fair": 0.6, "poor": 0.9}[loo],
                "n_bad": int(loo == "poor"),
                "n_nonfinite": 0,
            },
        }
    return {
        "convergence": convergence_data,
        "loo": loo_data,
        "posterior_predictive": {"available": True},
    }


def calibration_fixture(rating):
    if rating == "not computed":
        return {}
    return {
        "assessment": {
            "well_calibrated": rating == "excellent",
            "mean_coverage_deviation": {"excellent": 0.0, "fair": 0.03, "poor": 0.2}[
                rating
            ],
            "calibration_diagnosis": "well-calibrated"
            if rating == "excellent"
            else "under-confident",
        }
    }


@pytest.fixture
def evidence(checker):
    parameters = ("v", "a", "z", "t")
    model = SimpleNamespace(
        data=pd.DataFrame({"rt": [0.4, 0.6], "response": [-1, 1]}),
        params={
            name: SimpleNamespace(prior=f"Declared {name} prior") for name in parameters
        },
        pymc_model=SimpleNamespace(
            free_RVs=[SimpleNamespace(name=name) for name in parameters]
        ),
    )
    rows = []
    for stage in ("prior_predictive", "posterior_predictive"):
        for choice in (-1, 1):
            rows.append(
                {
                    "stage": stage,
                    "choice": choice,
                    "quantity": "choice proportion",
                    "observed": 0.5,
                    "predicted_mean": 0.6,
                    "lower_94": 0.2,
                    "upper_94": 0.8,
                    "replicates_with_quantity": 4,
                    "total_replicates": 4,
                }
            )
    diagnostics = diagnostics_fixture()
    psense = pd.DataFrame(
        {
            "prior": [0.01] * 4,
            "likelihood": [0.02] * 4,
            "diagnosis": ["Fixture evidence"] * 4,
        },
        index=parameters,
    )
    checks = checker.check_diagnostics(
        diagnostics, psense=psense.to_dict(orient="index")
    )
    margins = {
        name: checker.check_diagnostics(diagnostics, calibration_fixture("excellent"))
        for name in ("rt", "choice")
    }
    return {
        "model": model,
        "details": {
            "title": "Literal DDM reporting fixture",
            "question": "Compare RT and choice evidence.",
            "generative_story": "A flat four-parameter DDM.",
            "prior_policy": "Declared natural-scale priors.",
            "prior_assessment": "Observed choice fraction 0.5; replicate interval [0.2, 0.8].",
            "prior_density_verified": True,
            "sampler_settings": {"draws": "literal fixture, no sampling"},
            "interpretation": "A reporting fixture; no fitted result is claimed.",
            "limitations": ["Literal fixture: no inference was performed."],
            "seed": "No random generation",
            "predictive_settings": "Literal paired observations",
        },
        "domain_checks": pd.DataFrame(rows),
        "summary": pd.DataFrame(
            {
                "mean": [0.7, 1.2, 0.5, 0.25],
                "hdi_3%": [0.1, 0.8, 0.3, 0.1],
                "hdi_97%": [1.3, 1.6, 0.7, 0.3],
            },
            index=parameters,
        ),
        "psense": psense,
        "diagnostics": diagnostics,
        "checks": checks,
        "margins": margins,
        "versions": {"hssm": "0.5.0", "fixture": "No execution evidence"},
        "manifest": {
            "limitations": [
                "Marginal checks do not establish joint or held-out calibration."
            ]
        },
    }


@pytest.mark.parametrize("rating", ["excellent", "good", "fair", "poor"])
def test_convergence_uses_real_shared_rating_and_ddm_actions(
    reporting, checker, evidence, rating
):
    diagnostics = diagnostics_fixture(convergence=rating)
    evidence["diagnostics"] = diagnostics
    evidence["checks"] = checker.check_diagnostics(
        diagnostics, psense=evidence["psense"].to_dict(orient="index")
    )
    before = copy.deepcopy(evidence["checks"])
    report, actions = reporting.assemble_report(**evidence)
    assert evidence["checks"]["convergence"]["rating"] == rating
    assert evidence["checks"]["summary"]["convergence"] in report
    assert evidence["checks"] == before
    sampling_actions = [action for action in actions if "HSSM.sample" in action]
    assert bool(sampling_actions) == (rating in {"fair", "poor"})
    if sampling_actions:
        assert all(
            term in sampling_actions[0]
            for term in ("RT units", "response coding", "t support")
        )
        assert "v" in evidence["checks"]["convergence"]["problematic_params"]


@pytest.mark.parametrize("rating", ["excellent", "fair", "poor", "not computed"])
def test_joint_loo_failures_and_unavailability_remain_visible(
    reporting, checker, evidence, rating
):
    evidence["diagnostics"] = diagnostics_fixture(loo=rating)
    evidence["checks"] = checker.check_diagnostics(
        evidence["diagnostics"], psense=evidence["psense"].to_dict(orient="index")
    )
    report, actions = reporting.assemble_report(**evidence)
    assert evidence["checks"]["loo"]["rating"] == rating
    assert evidence["checks"]["summary"]["loo"] in report
    action_text = " ".join(actions)
    if rating in {"fair", "poor"}:
        assert "Pareto-k" in action_text and "joint RT/choice" in action_text
    elif rating == "not computed":
        assert evidence["diagnostics"]["loo"]["error"] in report
        assert "unavailable joint-trial LOO" in action_text
    else:
        assert "Pareto-k" not in action_text
    assert "StudentT" not in action_text and "NegBinomial" not in action_text


@pytest.mark.parametrize("margin", ["rt", "choice"])
@pytest.mark.parametrize("rating", ["fair", "poor", "not computed"])
def test_one_failing_margin_cannot_be_hidden_by_the_other(
    reporting, checker, evidence, margin, rating
):
    evidence["margins"][margin] = checker.check_diagnostics(
        evidence["diagnostics"], calibration_fixture(rating)
    )
    report, actions = reporting.assemble_report(**evidence)
    for name, assessment in evidence["margins"].items():
        assert assessment["summary"]["calibration"] in report
        assert assessment["calibration"]["rating"] == (
            rating if name == margin else "excellent"
        )
    targeted = [action for action in actions if f"{margin} marginal" in action]
    assert len(targeted) == 1
    other = "choice" if margin == "rt" else "rt"
    assert not any(f"{other} marginal" in action for action in actions)
    assert (
        "not establish joint RT/choice, conditional, or held-out calibration" in report
    )
    assert "StudentT" not in " ".join(actions)


def test_strong_sensitivity_preserves_parameter_and_support_caveat(
    reporting, checker, evidence
):
    evidence["psense"].loc["v", "prior"] = 0.2
    evidence["checks"] = checker.check_diagnostics(
        evidence["diagnostics"], psense=evidence["psense"].to_dict(orient="index")
    )
    report, actions = reporting.assemble_report(**evidence)
    assert evidence["checks"]["psense"] == {
        "rating": "strong sensitivity",
        "flagged_params": ["v"],
    }
    assert evidence["checks"]["summary"]["psense"] in report
    assert any(
        "alternative priors" in action and "different support" in action
        for action in actions
    )


def test_report_retains_canonical_structure_and_both_marginal_figures(
    reporting, evidence
):
    report, _ = reporting.assemble_report(**evidence)
    template = (
        (REPO / "bayesian-workflow/references/reporting.md")
        .read_text()
        .split("````markdown\n", 1)[1]
        .split("````", 1)[0]
    )
    expected_headings = [
        heading.removeprefix("[IF SENSITIVITY] ")
        for heading in re.findall(r"^## (.+)$", template, re.MULTILINE)
        if not heading.startswith("[IF MODEL_COMPARISON]")
    ]
    assert re.findall(r"^## (.+)$", report, re.MULTILINE) == expected_headings
    assert not re.search(r"<[^\n]+>|\[IF ", report)
    figures = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report))
    assert {
        "calibration/rt/pit_ecdf.png",
        "calibration/rt/pit_coverage.png",
        "calibration/choice/pit_ecdf.png",
        "calibration/choice/pit_coverage.png",
        "quantile_probability.png",
    } <= figures
    assert "| v | 0.7 | 0.1 | 1.3 |" in report
    assert evidence["details"]["prior_assessment"] in report


@pytest.mark.parametrize(
    "defect",
    [
        "no_convergence",
        "missing_rhat",
        "contradictory",
        "no_loo_reason",
        "missing_margin",
        "missing_rating",
        "bad_summaries",
        "missing_domain",
    ],
)
def test_malformed_evidence_cannot_render_a_success_report(reporting, evidence, defect):
    if defect == "no_convergence":
        evidence["diagnostics"].pop("convergence")
    elif defect == "missing_rhat":
        evidence["diagnostics"]["convergence"].pop("rhat")
    elif defect == "contradictory":
        evidence["diagnostics"]["convergence"]["rhat"]["ok"] = False
    elif defect == "no_loo_reason":
        evidence["diagnostics"]["loo"] = {"computed": False}
    elif defect == "missing_margin":
        evidence["margins"].pop("choice")
    elif defect == "missing_rating":
        evidence["checks"]["convergence"].pop("rating")
    elif defect == "bad_summaries":
        evidence["checks"]["summary"] = None
    elif defect == "missing_domain":
        evidence["domain_checks"] = evidence["domain_checks"].iloc[:0]
    with pytest.raises((ValueError, TypeError)):
        reporting.assemble_report(**evidence)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_figure",
        "unverified_density",
        "incomplete_density",
        "nonfinite_density",
        "wrong_likelihood_shape",
        "nonfinite_likelihood",
    ],
)
def test_failed_write_invalidates_old_report_but_preserves_checkpoint(
    reporting, evidence, tmp_path, failure
):
    for name in (
        "model_graph",
        "prior_predictive",
        "posterior_predictive",
        "quantile_probability",
    ):
        (tmp_path / f"{name}.png").write_bytes(b"literal figure fixture")
    (tmp_path / "report.md").write_text("Previous successful report")
    checkpoint = tmp_path / "posterior.nc"
    checkpoint.write_bytes(b"previous checkpoint must survive")
    posterior = xr.DataTree.from_dict(
        {
            "log_prior": xr.Dataset(
                {
                    name: (("chain", "draw"), [[-0.7, -0.8]])
                    for name in ("v", "a", "z", "t")
                }
            ),
            "log_likelihood": xr.Dataset(
                {
                    "rt,response": (
                        ("chain", "draw", "__obs__"),
                        [[[-0.4, -0.5], [-0.6, -0.7]]],
                    )
                }
            ),
        }
    )
    if failure == "missing_figure":
        (tmp_path / "prior_predictive.png").unlink()
    elif failure == "unverified_density":
        evidence["details"]["prior_density_verified"] = False
    elif failure == "incomplete_density":
        posterior["log_prior"] = posterior["log_prior"].to_dataset().drop_vars("t")
    elif failure == "nonfinite_density":
        posterior["log_prior"]["a"].values[0, 0] = np.nan
    elif failure == "wrong_likelihood_shape":
        posterior["log_likelihood"] = xr.Dataset(
            {"rt,response": ("draw", [-0.4, -0.5])}
        )
    elif failure == "nonfinite_likelihood":
        posterior["log_likelihood"]["rt,response"].values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        reporting.write_report(
            evidence["model"],
            None,
            posterior,
            tmp_path,
            details=evidence["details"],
            domain_checks=evidence["domain_checks"],
        )
    assert not (tmp_path / "report.md").exists()
    assert checkpoint.read_bytes() == b"previous checkpoint must survive"


def test_new_run_invalidates_report_before_any_computation(
    reporting, tmp_path, monkeypatch
):
    monkeypatch.setenv("BAYGENT_OUTPUT_ROOT", str(tmp_path))
    directory = tmp_path / "analysis"
    directory.mkdir()
    (directory / "report.md").write_text("Stale success")
    (directory / "posterior.nc").write_bytes(b"checkpoint")
    assert reporting.result_directory("analysis") == directory
    assert not (directory / "report.md").exists()
    assert (directory / "posterior.nc").read_bytes() == b"checkpoint"


def test_missing_late_figure_cannot_publish_or_reuse_report(
    reporting, evidence, tmp_path
):
    report, _ = reporting.assemble_report(**evidence)
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        path = tmp_path / link
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"literal figure fixture")
    reporting._save_report(report, tmp_path)
    assert (tmp_path / "report.md").read_text() == report
    (tmp_path / "calibration/choice/pit_coverage.png").unlink()
    with pytest.raises(ValueError, match="calibration/choice/pit_coverage.png"):
        reporting._save_report(report, tmp_path)
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize(
    "body",
    [
        "raise SystemExit(7)",
        "pass",
        "print('no output file')",
        "from pathlib import Path\nPath('evidence.json').write_text('not JSON')",
        "from pathlib import Path\nPath('evidence.json').write_text('[]')",
        "from pathlib import Path\nPath('evidence.json').write_text('{\"error\": \"missing samples\"}')",
    ],
)
def test_helper_failure_or_missing_output_cannot_reuse_stale_evidence(
    reporting, tmp_path, monkeypatch, body
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "fixture.py").write_text(body + "\n")
    monkeypatch.setattr(reporting, "BAYESIAN", tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "evidence.json"
    output.write_text(json.dumps({"convergence": "old healthy result"}))
    with pytest.raises((RuntimeError, ValueError)):
        reporting._run_helper("fixture.py", [], output)
    if output.exists():
        assert "old healthy result" not in output.read_text()


def test_real_shared_cli_accepts_each_margin_with_joint_diagnostics(
    reporting, evidence, checker, tmp_path
):
    diagnostic_path = tmp_path / "diagnostics.json"
    diagnostic_path.write_text(json.dumps(evidence["diagnostics"]))
    for name, rating in (("rt", "poor"), ("choice", "excellent")):
        calibration = calibration_fixture(rating)
        calibration.update(variable=name, pit_method="ppc_pit")
        directory = tmp_path / name
        directory.mkdir()
        cal_path = directory / "calibration.json"
        cal_path.write_text(json.dumps(calibration))
        result = reporting._assess_saved_margin(diagnostic_path, cal_path, name)
        expected = checker.check_diagnostics(evidence["diagnostics"], calibration)
        assert result["calibration"] == expected["calibration"]
        assert result["summary"] == expected["summary"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_assessment",
        "missing_deviation",
        "nonfinite",
        "wrong_quantity",
        "joint_loo_pit",
    ],
)
def test_malformed_margin_cannot_receive_a_default_healthy_rating(
    reporting, evidence, tmp_path, defect
):
    diagnostic_path = tmp_path / "diagnostics.json"
    diagnostic_path.write_text(json.dumps(evidence["diagnostics"]))
    calibration = calibration_fixture("excellent")
    calibration.update(variable="rt", pit_method="ppc_pit")
    if defect == "missing_assessment":
        calibration["assessment"] = {}
    elif defect == "missing_deviation":
        calibration["assessment"].pop("mean_coverage_deviation")
    elif defect == "nonfinite":
        calibration["assessment"]["mean_coverage_deviation"] = float("nan")
    elif defect == "wrong_quantity":
        calibration["variable"] = "choice"
    elif defect == "joint_loo_pit":
        calibration["pit_method"] = "loo_pit"
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(calibration))
    (tmp_path / "check_report.json").write_text('{"calibration": "stale success"}')
    with pytest.raises(ValueError, match="marginal PPC-PIT evidence for rt"):
        reporting._assess_saved_margin(diagnostic_path, cal_path, "rt")
    assert not (tmp_path / "check_report.json").exists()
