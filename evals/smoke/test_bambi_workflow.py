"""Execute both marimo examples and verify Bambi's statistical artifact handoff.

Run in the Bambi 0.21 / PyMC 6 analysis environment:
    python -m pytest evals/smoke/test_bambi_workflow.py

Each notebook runs once with BAYGENT_SMOKE=1 and a temporary output root.
These checks validate API behavior, prediction semantics and the reporting
contract. They deliberately do not require healthy convergence or calibration
from the small sampling budget.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import arviz as az
import arviz_stats as azs
import bambi as bmb
import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "bambi-workflow"
NOTEBOOKS = ("gaussian_regression", "hierarchical_bernoulli")


@pytest.fixture(scope="module")
def analyses(tmp_path_factory):
    """Run real models once; retain their live definitions and saved artifacts."""
    root = tmp_path_factory.mktemp("bambi-workflow")
    original_path = sys.path.copy()
    completed = {}
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("BAYGENT_SMOKE", "1")
            patch.setenv("BAYGENT_OUTPUT_ROOT", str(root))
            patch.setenv("MPLBACKEND", "Agg")
            for name in NOTEBOOKS:
                spec = importlib.util.spec_from_file_location(
                    name, EXAMPLES / f"{name}.py"
                )
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                patch.setitem(sys.modules, name, module)
                spec.loader.exec_module(module)
                _, definitions = module.app.run()
                completed[name] = definitions
    finally:
        sys.path[:] = original_path
    return completed


def _strict_json(path):
    def reject_nonfinite(value):
        raise ValueError(f"Non-finite JSON number in {path}: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def _observation_dims(array):
    return tuple(dim for dim in array.dims if dim not in {"chain", "draw"})


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_saved_artifact_preserves_original_rows_after_prediction_grids(analyses, name):
    analysis = analyses[name]
    original = analysis["idata"]
    posterior = analysis["posterior"]
    model = analysis["model"]
    outcome = model.response_term.name
    required = {
        "posterior",
        "sample_stats",
        "observed_data",
        "posterior_predictive",
        "log_likelihood",
        "log_prior",
    }
    assert required <= set(posterior.children)
    xr.testing.assert_identical(
        posterior["observed_data"].ds, original["observed_data"].ds
    )
    observed = posterior["observed_data"][outcome]
    np.testing.assert_array_equal(observed.values, model.response_term.data)
    predictive = posterior["posterior_predictive"][outcome]
    log_likelihood = posterior["log_likelihood"][outcome]
    assert _observation_dims(predictive) == observed.dims
    assert _observation_dims(log_likelihood) == observed.dims
    xr.align(observed, predictive, log_likelihood, join="exact")
    assert np.isfinite(log_likelihood.values).all()
    assert predictive.sizes[observed.dims[0]] == len(analysis["data"])

    # Grid draws describe another dataset; they must not replace the original
    # observations or reduce the .nc artifact to the few prediction rows.
    grid_draws = analysis["result"].draws["posterior"][model.family.likelihood.parent]
    assert grid_draws.sizes[_observation_dims(grid_draws)[0]] != observed.size
    directory = analysis["output_dir"]
    with xr.open_datatree(directory / "inference_data.nc") as restored:
        assert required <= set(restored.children)
        xr.testing.assert_equal(
            restored["observed_data"].ds, posterior["observed_data"].ds
        )
        xr.testing.assert_allclose(
            restored["log_likelihood"].ds, posterior["log_likelihood"].ds
        )
    with xr.open_datatree(directory / "posterior.nc") as checkpoint:
        xr.testing.assert_allclose(checkpoint["posterior"].ds, original["posterior"].ds)
        assert "log_prior" not in checkpoint.children


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_native_log_likelihood_matches_outcome_distribution(analyses, name):
    analysis = analyses[name]
    model = analysis["model"]
    posterior = analysis["posterior"]
    outcome = model.response_term.name
    observed = posterior["observed_data"][outcome]
    draws = posterior["posterior"]
    if model.family.name == "gaussian":
        standardized = (observed - draws["mu"]) / draws["sigma"]
        expected = (
            -0.5 * np.log(2 * np.pi) - np.log(draws["sigma"]) - 0.5 * standardized**2
        )
    else:
        assert model.family.name == "bernoulli"
        expected = xr.where(observed == 1, np.log(draws["p"]), np.log1p(-draws["p"]))
    actual = posterior["log_likelihood"][outcome]
    xr.testing.assert_allclose(actual, expected.transpose(*actual.dims))


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_complete_log_prior_matches_declared_densities(analyses, name):
    """Catch skipped offsets and evaluation of an uncentered intercept.

    Merely writing a finite log_prior group is insufficient: Bambi 0.21
    filters out absent free RVs and evaluates the saved intercept without
    reversing its default post-fit uncentering. These examples retain all
    free RVs and disable that centering before fitting.
    """
    analysis = analyses[name]
    model = analysis["model"]
    posterior = analysis["posterior"]
    assert model.center_predictors is False
    free_names = {variable.name for variable in model.backend.model.free_RVs}
    assert free_names <= set(posterior["posterior"].data_vars)
    assert set(posterior["log_prior"].data_vars) == free_names
    if name == "gaussian_regression":
        normal_priors = {"Intercept": (10, 5), "x": (0, 3)}
        halfnormal_priors = {"sigma": 3}
    else:
        offsets = {variable for variable in free_names if variable.endswith("_offset")}
        assert offsets, "The fixture must exercise non-centered group offsets."
        normal_priors = {
            "Intercept": (0, 1),
            "age_scaled": (0, 0.5),
            "income_bracket": (0, 0.5),
            "urban": (0, 0.5),
            **{offset: (0, 1) for offset in offsets},
        }
        halfnormal_priors = {"1|region_sigma": 0.6}
    assert set(normal_priors) | set(halfnormal_priors) == free_names
    for variable in free_names:
        samples = posterior["posterior"][variable]
        actual = posterior["log_prior"][variable]
        assert np.isfinite(actual.values).all()
        if variable in normal_priors:
            mean, sigma = normal_priors[variable]
            expected = (
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma)
                - 0.5 * ((samples - mean) / sigma) ** 2
            )
        else:
            sigma = halfnormal_priors[variable]
            assert (samples >= 0).all()
            expected = (
                0.5 * np.log(2 / np.pi) - np.log(sigma) - 0.5 * (samples / sigma) ** 2
            )
        xr.testing.assert_allclose(actual, expected.transpose(*actual.dims))


def test_gaussian_contrast_matches_posterior_coefficient_draws(analyses):
    analysis = analyses["gaussian_regression"]
    # For y ~ x with an identity link, yhat(+0.5)-yhat(-0.5) is exactly
    # the x coefficient in every draw; aggregation of these identical
    # row contrasts must not create extra precision or reverse direction.
    contrast_draws = analysis["posterior"]["posterior"]["x"]
    summary = analysis["result"].summary
    assert len(summary) == 1
    np.testing.assert_allclose(
        summary["estimate"], float(contrast_draws.mean()), rtol=1e-9
    )
    interval = az.hdi(contrast_draws, prob=0.94)
    for prefix, bound in (("lower_", "lower"), ("upper_", "upper")):
        columns = [column for column in summary if column.startswith(prefix)]
        assert len(columns) == 1
        np.testing.assert_allclose(
            summary[columns[0]], float(interval.sel(ci_bound=bound)), rtol=1e-9
        )


def test_known_region_probability_uses_correct_income_and_group_draws(analyses):
    analysis = analyses["hierarchical_bernoulli"]
    model = analysis["model"]
    data = analysis["data"]
    posterior = analysis["posterior"]["posterior"]
    result = analysis["result"]
    assert model.response_term.success == 1
    np.testing.assert_array_equal(model.response_term.data, data.support)
    np.testing.assert_allclose(data.age_scaled, (data.age - 40) / 10)
    assert data.groupby("region", observed=True).size().nunique() > 1
    income_term = model.parameters["p"].common_terms["income_bracket"]
    assert list(income_term.levels) == ["medium", "high"]
    for level, columns in {"low": [0, 0], "medium": [1, 0], "high": [0, 1]}.items():
        actual = income_term.data[data.income_bracket == level]
        np.testing.assert_array_equal(actual, np.tile(columns, (len(actual), 1)))

    # Independently reconstruct the precise requested logit in each posterior
    # draw. Using the population intercept, the wrong income reference, or
    # averaging over regions would give a different probability distribution.
    income_draws = posterior["income_bracket"]
    (income_dim,) = _observation_dims(income_draws)
    group_draws = posterior["1|region"]
    (group_dim,) = _observation_dims(group_draws)
    assert set(group_draws[group_dim].values) == set(data.region.cat.categories)
    linear_predictor = (
        posterior["Intercept"]
        + income_draws.sel({income_dim: "medium"}, drop=True)
        + posterior["urban"]
        + group_draws.sel({group_dim: "north"}, drop=True)
    )
    expected_probability = 1 / (1 + np.exp(-linear_predictor))
    probabilities = result.draws["posterior"]["p"]
    (observation_dim,) = _observation_dims(probabilities)
    assert probabilities.sizes[observation_dim] == 1
    probabilities = probabilities.isel({observation_dim: 0}, drop=True)
    np.testing.assert_allclose(
        probabilities.values,
        expected_probability.transpose(*probabilities.dims).values,
        rtol=1e-9,
    )
    assert ((probabilities > 0) & (probabilities < 1)).all()
    assert len(result.summary) == 1
    row = result.summary.iloc[0]
    assert row["region"] == "north" and row["income_bracket"] == "medium"
    assert row["age_scaled"] == 0 and row["urban"] == 1
    np.testing.assert_allclose(
        row["estimate"], float(expected_probability.mean()), rtol=1e-9
    )
    interval = az.hdi(expected_probability, prob=0.94)
    for prefix, bound in (("lower_", "lower"), ("upper_", "upper")):
        (column,) = [column for column in result.summary if column.startswith(prefix)]
        np.testing.assert_allclose(
            row[column], float(interval.sel(ci_bound=bound)), rtol=1e-9
        )


def test_explicit_low_income_reference_matches_design_rows():
    """Check the encoding taught by the skill independently of any fit."""
    labels = ["low", "medium", "high"] * 3
    data = pd.DataFrame(
        {
            "support": [0, 1, 1, 1, 0, 1, 0, 1, 0],
            "income_bracket": pd.Categorical(
                labels, categories=["low", "medium", "high"], ordered=True
            ),
        }
    )
    model = bmb.Model("support ~ income_bracket", data, family="bernoulli")
    term = model.parameters["p"].common_terms["income_bracket"]
    assert list(term.levels) == ["medium", "high"]
    np.testing.assert_array_equal(term.data, np.tile([[0, 0], [1, 0], [0, 1]], (3, 1)))
    assert model.response_term.success == 1
    np.testing.assert_array_equal(model.response_term.data, data.support)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_report_preserves_shared_assessments_and_resolves_artifacts(analyses, name):
    directory = analyses[name]["output_dir"]
    report = (directory / "report.md").read_text()
    checks = _strict_json(directory / "check_report.json")
    analysis_details = _strict_json(directory / "analysis.json")
    assert analysis_details["smoke"] is True
    for section in ("convergence", "calibration", "psense"):
        assert checks["summary"][section] in report
    for recommendation in checks["next_steps"]:
        assert recommendation in report
    assert not re.search(r"<[^<>\n]+>", report), (
        "Unfilled template or metadata placeholder."
    )

    # Check the canonical section contract without freezing explanatory prose.
    source = (REPO / "bayesian-workflow" / "references" / "reporting.md").read_text()
    template = source.split("````markdown\n", 1)[1].split("````", 1)[0]
    expected_sections = [
        heading.removeprefix("[IF SENSITIVITY] ")
        for heading in re.findall(r"(?m)^## (.+)$", template)
        if not heading.startswith("[IF MODEL_COMPARISON]")
    ]
    assert re.findall(r"(?m)^## (.+)$", report) == expected_sections
    local_links = []
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report):
        parsed = urlparse(target)
        if not parsed.scheme and parsed.path:
            artifact = directory / unquote(parsed.path)
            assert artifact.is_file(), f"Missing report artifact: {artifact}"
            assert artifact.stat().st_size > 0
            local_links.append(artifact)
    assert local_links, "The report must link its diagnostic evidence."
    calibration = _strict_json(directory / "calibration.json")
    assert "error" not in calibration
    for path in calibration.get("plots", {}).values():
        assert (directory / path).is_file()
    for filename in (
        "inference_data.nc",
        "prior_data.nc",
        "summary.csv",
        "predictions.csv",
    ):
        assert (directory / filename).stat().st_size > 0
    assert _strict_json(directory / "versions.json")["bambi"] == "0.21.0"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_sensitivity_json_maps_parameters_to_actual_values(analyses, name):
    analysis = analyses[name]
    directory = analysis["output_dir"]
    payload = _strict_json(directory / "psense.json")
    summary_labels = pd.read_csv(directory / "summary.csv", index_col=0).index.tolist()
    parameter_names = list(
        dict.fromkeys(label.split("[", 1)[0] for label in summary_labels)
    )
    recomputed = azs.psense_summary(analysis["posterior"], var_names=parameter_names)
    assert set(payload) == set(recomputed.index)
    for parameter, row in recomputed.iterrows():
        assert {"prior", "likelihood"} <= payload[parameter].keys()
        for component in ("prior", "likelihood"):
            np.testing.assert_allclose(payload[parameter][component], row[component])
