"""Fixed-point HSSM 0.5 density checks: no simulation or posterior sampling.

Run in the separately resolved HSSM environment. The model constructor and
data-informed t bound are read from the notebook itself, without importing or
running its marimo app. Expected priors come from SciPy distributions; expected
joint likelihoods come directly from the separate Cython Wiener implementation.

Float64 tolerances are declared before running comparisons: 1e-6 absolute log
density error for the independently approximated Wiener series; 1e-10 for
closed-form priors; 2e-8 for bounded choice-mass quadrature (requested error
2e-10 plus an explicitly checked independent survival-tail bound).
"""

import ast
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hssm_numerical_reference import (
    survival_tail_bound,
    upper_choice_probability,
    wiener_log_density,
)
from scipy import integrate, stats

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "examples/hssm-workflow/analytical_ddm.py"
RESPONSE = "rt,response"
LOG_DENSITY_ATOL = 1e-6
PRIOR_ATOL = 1e-10
MASS_ATOL = 2e-8


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

    assert version("hssm") == "0.5.0", (
        "This numerical gate targets the HSSM 0.5 release."
    )
    return hssm


@pytest.fixture(autouse=True)
def forbid_sampling(monkeypatch, hssm_module):
    """A numerical regression must not quietly become a simulation or fit."""

    def forbidden(*args, **kwargs):
        pytest.fail("Fixed-point numerical tests must not simulate or sample.")

    monkeypatch.setattr(hssm_module, "simulate_data", forbidden)
    for name in ("sample", "sample_prior_predictive", "sample_posterior_predictive"):
        monkeypatch.setattr(hssm_module.HSSM, name, forbidden)


@pytest.fixture(scope="module", params=[0.4, 0.8], ids=["data-limited-t", "capped-t"])
def teaching_model(request, hssm_module):
    minimum_rt = request.param
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


def test_native_priors_have_declared_natural_density_and_effective_support(
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
        "v": [-5.0, -0.7, 0.0, 0.9, 5.0],
        "a": [-0.1, 0.0, 0.05, 0.7, 1.2, 3.0, 8.0],
        "z": [-0.01, 0.0, 0.01, 0.23, 0.5, 0.87, 0.99, 1.0, 1.01],
        "t": [-0.01, 0.0, 0.01, t_upper / 2, t_upper, t_upper + 0.01],
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
    points = [
        {"v": 0.0, "a": 0.7, "z": 0.5, "t": 0.0},
        {"v": 0.0, "a": 1.2, "z": 0.23, "t": t_upper / 3},
        {"v": 0.8, "a": 0.7, "z": 0.67, "t": t_upper / 2},
        {"v": -1.1, "a": 1.3, "z": 0.31, "t": t_upper / 2},
        {"v": 1.3, "a": 1.1, "z": 0.71, "t": t_upper},
        {"v": -0.6, "a": 0.8, "z": 0.27, "t": t_upper},
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="PyMC 6.1 compute_log_prior resets noncanonical chain/draw coordinates.",
)
def test_native_prior_export_preserves_noncanonical_sample_coordinates(teaching_model):
    import pymc as pm

    model, _, t_upper = teaching_model
    dt = _fixed_posterior([{"v": 0.4, "a": 1.2, "z": 0.45, "t": t_upper / 2}])
    dt["posterior"] = dt["posterior"].ds.assign_coords(chain=[7], draw=[11])
    pm.compute_log_prior(dt, model=model.pymc_model, progressbar=False)
    xr.testing.assert_identical(
        dt["log_prior"].coords.to_dataset(), dt["posterior"].coords.to_dataset()
    )


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="HSSM 0.5 log_likelihood uses sample coordinate labels as array positions.",
)
def test_native_likelihood_accepts_noncanonical_sample_coordinates(teaching_model):
    model, data, t_upper = teaching_model
    point = {"v": 0.4, "a": 1.2, "z": 0.45, "t": t_upper / 2}
    dt = _fixed_posterior([point])
    dt["posterior"] = dt["posterior"].ds.assign_coords(chain=[7], draw=[11])
    model.log_likelihood(dt=dt, inplace=True)
    actual = dt["log_likelihood"][RESPONSE]
    np.testing.assert_array_equal(actual.chain, [7])
    np.testing.assert_array_equal(actual.draw, [11])
    np.testing.assert_allclose(
        actual.values[0, 0],
        wiener_log_density(data.rt, data.response, **point),
        rtol=0,
        atol=LOG_DENSITY_ATOL,
    )


@pytest.fixture(scope="module")
def native_log_density(hssm_module):
    import pytensor
    import pytensor.tensor as pt
    from hssm.likelihoods.analytical import logp_ddm

    pairs = pt.dmatrix("pairs")
    v, a, z, t = (pt.dscalar(name) for name in ("v", "a", "z", "t"))
    function = pytensor.function([pairs, v, a, z, t], logp_ddm(pairs, v, a, z, t))

    def evaluate(rt, response, v, a, z, t):
        rt, response = np.broadcast_arrays(rt, response)
        pairs = np.column_stack((rt.reshape(-1), response.reshape(-1)))
        return function(pairs, v, a, z, t).reshape(rt.shape)

    return evaluate


@pytest.mark.parametrize(
    "v,a,z", [(0.0, 0.8, 0.5), (0.0, 1.1, 0.27), (0.8, 0.7, 0.67), (-0.9, 1.2, 0.31)]
)
def test_both_density_implementations_obey_choice_reflection(
    native_log_density, v, a, z
):
    times = 0.2 + np.array([0.08, 0.2, 0.5, 1.5, 4.0])
    for density in (wiener_log_density, native_log_density):
        upper = density(times, 1, v, a, z, 0.2)
        reflected = density(times, -1, -v, a, 1 - z, 0.2)
        np.testing.assert_allclose(upper, reflected, rtol=0, atol=LOG_DENSITY_ATOL)


@pytest.mark.parametrize(
    "v,a,z", [(0.0, 0.8, 0.5), (0.0, 1.1, 0.27), (0.8, 0.7, 0.67), (-0.9, 1.2, 0.31)]
)
def test_choice_masses_match_closed_form_with_bounded_tail(
    native_log_density, v, a, z, record_property
):
    cutoff = 8 * (2 * a) ** 2
    tail_bound = survival_tail_bound(cutoff, v, a)
    assert tail_bound < 1e-12
    upper_mass = upper_choice_probability(v, a, z)
    for label, density in (
        ("cython", wiener_log_density),
        ("native", native_log_density),
    ):
        masses = []
        for response, expected in ((-1, 1 - upper_mass), (1, upper_mass)):
            mass, quadrature_error = integrate.quad(
                lambda decision_time, density=density, response=response: float(
                    np.exp(density(0.2 + decision_time, response, v, a, z, 0.2))
                ),
                0,
                cutoff,
                epsabs=2e-10,
                epsrel=2e-10,
                limit=150,
            )
            assert quadrature_error < 2e-9
            record_property(
                f"{label}_choice_{response}_mass_error", abs(mass - expected)
            )
            assert abs(mass - expected) < MASS_ATOL + tail_bound
            masses.append(mass)
        assert abs(sum(masses) - 1) < 2 * MASS_ATOL + tail_bound
    record_property("independent_survival_tail_bound", tail_bound)


@pytest.mark.parametrize("response", [-1, 1])
def test_impossible_rt_uses_native_floor_but_has_zero_reference_mass(
    native_log_density, response
):
    # Test the release floor separately: a finite log value is not valid support.
    rt = np.array([0.1, 0.2])
    native = native_log_density(rt, response, 0.7, 1.2, 0.4, 0.2)
    reference = wiener_log_density(rt, response, 0.7, 1.2, 0.4, 0.2)
    np.testing.assert_array_equal(native, [-66.1, -66.1])
    assert np.isneginf(reference).all()
