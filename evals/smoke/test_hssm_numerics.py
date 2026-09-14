"""Notebook prior and joint-likelihood handoffs, without simulation or sampling.

Run in the separately resolved HSSM environment. The model constructor and
data-informed t bound are read from the notebook itself, without importing or
running its marimo app. Expected priors come from SciPy distributions; expected
joint likelihoods come directly from the separate Cython Wiener implementation.

Float64 tolerances are declared before running comparisons: 1e-6 absolute log
density error for the independently approximated Wiener series; 1e-10 for
closed-form priors.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hssm_numerical_reference import wiener_log_density
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "examples/hssm-workflow/analytical_ddm.py"
RESPONSE = "rt,response"
LOG_DENSITY_ATOL = 1e-6
PRIOR_ATOL = 1e-10


def _notebook_expression(target: str):
    """Read the unique notebook assignment, so specification drift is tested."""
    tree = ast.parse(NOTEBOOK.read_text())
    matches = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(name, ast.Name) and name.id == target for name in node.targets
        )
    ]
    assert len(matches) == 1, f"Expected exactly one notebook assignment to {target}."
    return compile(ast.Expression(matches[0]), str(NOTEBOOK), "eval")


@pytest.fixture(scope="module")
def hssm_module():
    import hssm

    return hssm


@pytest.fixture(autouse=True)
def forbid_sampling(monkeypatch, hssm_module):
    """A numerical regression must not quietly become a simulation or fit."""

    def forbidden(*args, **kwargs):
        pytest.fail("Fixed-point numerical tests must not simulate or sample.")

    monkeypatch.setattr(hssm_module, "simulate_data", forbidden)
    for name in ("sample", "sample_prior_predictive", "sample_posterior_predictive"):
        monkeypatch.setattr(hssm_module.HSSM, name, forbidden)


@pytest.fixture(scope="module")
def teaching_model(request, hssm_module):
    minimum_rt = getattr(request, "param", 0.4)
    data = pd.DataFrame(
        {
            "rt": minimum_rt + np.array([0, 0.07, 0.16, 0.31, 0.55, 0.93]),
            "response": [-1, 1, -1, 1, -1, 1],
        }
    )
    namespace = {"data": data, "np": np, "hssm": hssm_module}
    t_upper = eval(_notebook_expression("t_upper"), namespace)
    namespace["t_upper"] = t_upper
    model = eval(_notebook_expression("model"), namespace)
    return model, data, t_upper


def _fixed_posterior(points: list[dict[str, float]]) -> xr.DataTree:
    """Each point is explicitly labeled; these are evaluations, not draws."""
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    name: (("chain", "draw"), [[point[name] for point in points]])
                    for name in ("v", "a", "z", "t")
                },
                coords={"chain": [0], "draw": np.arange(len(points))},
            )
        }
    )


@pytest.mark.parametrize(
    "teaching_model", [0.4, 0.8], indirect=True, ids=["data-limited-t", "capped-t"]
)
def test_notebook_priors_have_declared_density_and_support(
    teaching_model, record_property
):
    import pymc as pm

    model, data, t_upper = teaching_model
    assert t_upper == (0.38 if data.rt.min() == 0.4 else 0.5)
    assert t_upper < data.rt.min()
    assert {rv.name for rv in model.pymc_model.free_RVs} == {"v", "a", "z", "t"}
    assert {rv.name for rv in model.pymc_model.observed_RVs} == {RESPONSE}
    # Include points beyond typical DDM fitting boxes to detect hidden bounds.
    values = {
        "v": [-5.0, 0.4, 5.0],
        "a": [-0.1, 0.0, 0.7, 1.2, 8.0],
        "z": [-0.01, 0.0, 0.23, 0.87, 1.0, 1.01],
        "t": [-0.01, 0.0, t_upper / 2, t_upper, t_upper + 0.01],
    }
    base = {"v": 0.4, "a": 1.2, "z": 0.45, "t": t_upper / 2}
    points = [
        {**base, name: value} for name, probes in values.items() for value in probes
    ]
    dt = _fixed_posterior(points)
    pm.compute_log_prior(dt, model=model.pymc_model, progressbar=False)
    assert set(dt["log_prior"].data_vars) == set(base)
    distributions = {
        "v": stats.norm(loc=0, scale=1.5),
        "a": stats.lognorm(s=0.35, scale=1.2),
        "z": stats.beta(a=2, b=2),
        "t": stats.uniform(loc=0, scale=t_upper),
    }
    for name, distribution in distributions.items():
        actual = dt["log_prior"][name]
        expected = distribution.logpdf(dt["posterior"][name].values)
        assert actual.dims == ("chain", "draw")
        xr.testing.assert_identical(
            actual.coords.to_dataset(), dt["posterior"].coords.to_dataset()
        )
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        assert not np.isnan(actual).any()
        finite = np.isfinite(expected)
        error = float(np.max(np.abs(actual.values[finite] - expected[finite])))
        record_property(f"prior_{name}_max_absolute_log_error", error)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=PRIOR_ATOL)


def test_model_joint_log_likelihood_matches_independent_wiener(
    teaching_model, record_property
):
    model, data, t_upper = teaching_model
    # Contrasting drift/bias and distinct trial RTs expose response-sign,
    # boundary-scale and sample/trial ordering mistakes in the native handoff.
    points = [
        {"v": 0.8, "a": 0.7, "z": 0.67, "t": t_upper / 2},
        {"v": -1.1, "a": 1.3, "z": 0.31, "t": t_upper / 2},
    ]
    dt = _fixed_posterior(points)
    model.log_likelihood(dt=dt, inplace=True)
    actual = dt["log_likelihood"][RESPONSE]
    assert actual.dims == ("chain", "draw", "__obs__")
    np.testing.assert_array_equal(actual.chain, [0])
    np.testing.assert_array_equal(actual.draw, np.arange(len(points)))
    np.testing.assert_array_equal(actual["__obs__"], np.arange(len(data)))
    assert set(dt["log_likelihood"].data_vars) == {RESPONSE}
    expected = np.stack(
        [wiener_log_density(data.rt, data.response, **point) for point in points]
    )[None, ...]
    assert np.isfinite(expected).all()
    errors = np.abs(actual.values - expected)
    for index in range(len(points)):
        record_property(
            f"point_{index}_max_absolute_log_error", float(errors[0, index].max())
        )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=LOG_DENSITY_ATOL)
