"""Verify a completed analytical-DDM export without repeating its fit.

Set BAYGENT_HSSM_ARTIFACT_DIR to the analysis folder containing inference_data.nc.
These checks consume saved synthetic results; none constructs or samples a model.
"""

import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hssm_numerical_reference import wiener_log_density
from scipy import stats

RESPONSE = "rt,response"
OBS = "__obs__"


@pytest.fixture(scope="module")
def saved():
    folder = os.environ.get("BAYGENT_HSSM_ARTIFACT_DIR")
    if not folder:
        pytest.skip("Set BAYGENT_HSSM_ARTIFACT_DIR to verify an existing full export.")
    directory = Path(folder).resolve()
    with ExitStack() as stack:
        trees = {
            name: stack.enter_context(xr.open_datatree(directory / filename))
            for name, filename in {
                "joint": "inference_data.nc",
                "checkpoint": "posterior.nc",
                "prior": "prior_data.nc",
                "rt": "calibration/rt/inference_data.nc",
                "choice": "calibration/choice/inference_data.nc",
            }.items()
        }
        yield {
            **trees,
            "directory": directory,
            "details": json.loads((directory / "analysis.json").read_text()),
            "data": pd.read_csv(directory / "data.csv", float_precision="round_trip"),
        }


def _components(tree, group):
    array = tree[group][RESPONSE]
    event = next(dim for dim in array.dims if dim.startswith(RESPONSE))
    np.testing.assert_array_equal(array[event], [0, 1])
    return array.sel({event: 0}, drop=True), array.sel({event: 1}, drop=True)


def test_saved_checkpoint_and_joint_samples_preserve_original_data(saved):
    joint, checkpoint = saved["joint"], saved["checkpoint"]
    xr.testing.assert_equal(joint["posterior"].ds, checkpoint["posterior"].ds)
    xr.testing.assert_equal(joint["observed_data"].ds, checkpoint["observed_data"].ds)
    assert "log_prior" not in checkpoint.children
    rt, choice = _components(joint, "observed_data")
    np.testing.assert_array_equal(rt, saved["data"].rt)
    np.testing.assert_array_equal(choice, saved["data"].response)
    assert bool((rt > 0).all()) and set(np.unique(choice)) == {-1, 1}
    posterior, predictions = joint["posterior"], joint["posterior_predictive"]
    settings = saved["details"]["sampler_settings"]
    for dim, expected in (("chain", settings["chains"]), ("draw", settings["draws"])):
        assert posterior.sizes[dim] == expected
        np.testing.assert_array_equal(posterior[dim], predictions[dim])
    np.testing.assert_array_equal(rt[OBS], predictions[OBS])
    likelihood = joint["log_likelihood"][RESPONSE]
    assert likelihood.dims == ("chain", "draw", OBS)
    for dim, reference in (("chain", posterior), ("draw", posterior), (OBS, rt)):
        assert likelihood.get_index(dim).equals(reference[dim].get_index(dim))
    assert set(joint["log_likelihood"].data_vars) == {RESPONSE}
    assert np.isfinite(likelihood).all()


def test_saved_joint_likelihood_matches_compiled_wiener_reference(saved):
    tree = saved["joint"]
    posterior = tree["posterior"]
    observed_rt, choice = _components(tree, "observed_data")
    # Whole trial vectors at several draws expose ordering, sign and boundary mistakes.
    for chain in (0, posterior.sizes["chain"] - 1):
        for draw in (0, posterior.sizes["draw"] // 2, posterior.sizes["draw"] - 1):
            parameters = {
                name: float(posterior[name].isel(chain=chain, draw=draw))
                for name in ("v", "a", "z", "t")
            }
            expected = wiener_log_density(
                observed_rt.to_numpy(), choice.to_numpy(), **parameters
            )
            actual = tree["log_likelihood"][RESPONSE].isel(chain=chain, draw=draw)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_saved_priors_match_scipy_and_observed_time_support(saved):
    tree = saved["joint"]
    samples = tree["posterior"]
    upper = min(0.5, 0.95 * float(saved["data"].rt.min()))
    distributions = {
        "v": stats.norm(loc=0, scale=1.5),
        "a": stats.lognorm(s=0.35, scale=1.2),
        "z": stats.beta(a=2, b=2),
        "t": stats.uniform(loc=0, scale=upper),
    }
    assert set(tree["log_prior"].data_vars) == set(distributions)
    assert bool(((samples["t"] > 0) & (samples["t"] < upper)).all())
    assert bool((samples["a"] > 0).all())
    assert bool(((samples["z"] > 0) & (samples["z"] < 1)).all())
    for name, distribution in distributions.items():
        assert np.isfinite(samples[name]).all()
        expected = distribution.logpdf(samples[name])
        actual = tree["log_prior"][name]
        assert actual.dims == ("chain", "draw")
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        for dim in ("chain", "draw"):
            np.testing.assert_array_equal(actual[dim], samples[dim])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_scalar_artifacts_are_exact_margins_without_joint_densities(saved):
    for group in ("observed_data", "posterior_predictive"):
        rt, response = _components(saved["joint"], group)
        for quantity, expected in (
            ("rt", rt),
            ("choice", (response == 1).astype("int8")),
        ):
            marginal = saved[quantity]
            assert set(marginal.children) == {"observed_data", "posterior_predictive"}
            assert set(marginal[group].data_vars) == {quantity}
            xr.testing.assert_equal(
                marginal[group][quantity], expected.rename(quantity)
            )


def test_saved_domain_summaries_use_replicate_choice_and_conditional_rt(saved):
    table = pd.read_csv(saved["directory"] / "domain_checks.csv")
    observed = saved["data"]
    for group, tree in (
        ("prior_predictive", saved["prior"]),
        ("posterior_predictive", saved["joint"]),
    ):
        rt, response = _components(tree, group)
        rt = rt.transpose("chain", "draw", OBS).to_numpy().reshape(-1, len(observed))
        response = response.transpose("chain", "draw", OBS).to_numpy().reshape(rt.shape)
        for choice in (-1, 1):
            for quantity in (
                "choice proportion",
                "RT q=0.1 (s)",
                "RT q=0.5 (s)",
                "RT q=0.9 (s)",
            ):
                rows = table.loc[
                    (table.stage == group)
                    & (table.choice == choice)
                    & (table.quantity == quantity)
                ]
                assert len(rows) == 1
                row = rows.iloc[0]
                if quantity == "choice proportion":
                    values = (response == choice).mean(axis=1)
                    target = float((observed.response == choice).mean())
                else:
                    probability = float(quantity.split("=")[1].split()[0])
                    values = np.array(
                        [
                            np.quantile(trials[labels == choice], probability)
                            for trials, labels in zip(rt, response, strict=True)
                            if np.any(labels == choice)
                        ]
                    )
                    target = np.quantile(
                        observed.loc[observed.response == choice, "rt"], probability
                    )
                assert row.total_replicates == rt.shape[0]
                assert row.replicates_with_quantity == len(values)
                np.testing.assert_allclose(
                    [row.observed, row.predicted_mean, row.lower_94, row.upper_94],
                    [target, values.mean(), *np.quantile(values, [0.03, 0.97])],
                    rtol=0,
                    atol=1e-10,
                )


def test_saved_report_preserves_shared_assessments_and_real_figure_links(saved):
    directory = saved["directory"]
    report = (directory / "report.md").read_text()
    checks = json.loads((directory / "check_report.json").read_text())
    for summary in checks["summary"].values():
        assert summary in report
    for name in ("rt", "choice"):
        margin = json.loads(
            (directory / "calibration" / name / "check_report.json").read_text()
        )
        assert margin["summary"]["calibration"] in report
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert len(links) >= 9
    for target in links:
        assert (directory / target).is_file(), target
    assert "does not establish joint" in report or "do not establish joint" in report
    assert not re.search(r"<[^\n]+>", report)


def test_report_tables_match_saved_numbers_and_underlying_samples(saved):
    directory = saved["directory"]
    report = (directory / "report.md").read_text()
    summary = pd.read_csv(
        directory / "summary.csv", index_col=0, float_precision="round_trip"
    )
    psense = pd.DataFrame.from_dict(
        json.loads((directory / "psense.json").read_text()), orient="index"
    )
    domain = pd.read_csv(directory / "domain_checks.csv", float_precision="round_trip")
    expected_tables = {
        "Posterior": summary.reset_index(names="Parameter"),
        "Prior Sensitivity": psense.reset_index(names="Parameter"),
        "Prior Predictive Check": domain.loc[domain.stage == "prior_predictive"],
        "Posterior Predictive Check": domain.loc[
            domain.stage == "posterior_predictive"
        ],
    }
    for heading, expected in expected_tables.items():
        section = report.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
        lines = [line for line in section.splitlines() if line.startswith("|")]
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]
        ]
        assert headers == list(expected.columns)
        assert len(rows) == len(expected)
        for actual_row, expected_row in zip(
            rows, expected.itertuples(index=False, name=None), strict=True
        ):
            for actual, value in zip(actual_row, expected_row, strict=True):
                if isinstance(value, (float, np.floating)):
                    # Published tables use four significant digits; compare that explicit precision.
                    assert actual == f"{value:.4g}"
                else:
                    assert actual == str(value)
    for name in ("v", "a", "z", "t"):
        mean = float(saved["joint"]["posterior"][name].mean())
        np.testing.assert_allclose(summary.loc[name, "mean"], mean, rtol=0, atol=1e-12)
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    assert diagnostics["convergence"]["divergences"]["count"] == int(
        saved["joint"]["sample_stats"]["diverging"].sum()
    )
