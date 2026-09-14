"""HSSM adapter contracts and an explicitly gated future integration run.

These tests are authored but execution is deferred. Adapter fixtures are literal
arrays, not simulator output. Run them later with pytest in the modern analysis
environment. The full notebook test additionally requires BAYGENT_TEST_HSSM=1;
that opt-in permits its simulation and declared full sampling budget. Collection
does not import HSSM or run the notebook. Passing these tests would establish
software behavior, not parameter recovery or healthy inference.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
RESPONSE = "rt,response"
EVENT = "rt,response_dim"
OBS_EVENT = "rt,response_extra_dim_0"
OBS = "__obs__"


@pytest.fixture
def adapter():
    path = REPO / "hssm-workflow" / "scripts" / "prepare_rt_choice.py"
    spec = importlib.util.spec_from_file_location("hssm_response_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paired():
    """Distinct coordinates and values expose axis loss or accidental alignment."""
    predicted = np.array(
        [
            [
                [[0.2, -1], [0.7, 1], [1.1, -1]],
                [[0.3, 1], [0.8, 1], [1.2, -1]],
                [[0.4, -1], [0.9, -1], [1.3, 1]],
            ],
            [
                [[0.5, 1], [1.0, -1], [1.4, 1]],
                [[0.6, -1], [1.1, 1], [1.5, 1]],
                [[0.7, 1], [1.2, -1], [1.6, -1]],
            ],
        ]
    )
    sample_coords = {"chain": [2, 7], "draw": [11, 17, 23], OBS: [101, 108, 113]}
    return xr.DataTree.from_dict(
        {
            "observed_data": xr.Dataset(
                {RESPONSE: ((OBS, OBS_EVENT), [[0.4, -1], [0.8, 1], [1.2, -1]])},
                coords={OBS: sample_coords[OBS], OBS_EVENT: [0, 1]},
            ),
            "posterior_predictive": xr.Dataset(
                {RESPONSE: (("chain", "draw", OBS, EVENT), predicted)},
                coords={**sample_coords, EVENT: [0, 1]},
            ),
            "posterior": xr.Dataset(
                {"v": (("chain", "draw"), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])},
                coords={key: sample_coords[key] for key in ("chain", "draw")},
            ),
            "log_likelihood": xr.Dataset(
                {RESPONSE: (("chain", "draw", OBS), -np.ones((2, 3, 3)))},
                coords=sample_coords,
            ),
        }
    )


def test_scalar_views_preserve_exact_values_axes_and_source(adapter, paired):
    original = paired.copy(deep=True)
    views = adapter.scalar_ppc_views(paired)
    assert set(views) == {"rt", "choice"}
    expected_rt = np.array(
        [
            [[0.2, 0.7, 1.1], [0.3, 0.8, 1.2], [0.4, 0.9, 1.3]],
            [[0.5, 1.0, 1.4], [0.6, 1.1, 1.5], [0.7, 1.2, 1.6]],
        ]
    )
    expected_choice = np.array(
        [[[0, 1, 0], [1, 1, 0], [0, 0, 1]], [[1, 0, 1], [0, 1, 1], [1, 0, 0]]]
    )
    for name, expected, observed in (
        ("rt", expected_rt, [0.4, 0.8, 1.2]),
        ("choice", expected_choice, [0, 1, 0]),
    ):
        view = views[name]
        assert set(view.children) == {"observed_data", "posterior_predictive"}
        assert set(view["posterior_predictive"].data_vars) == {name}
        assert set(view["observed_data"].data_vars) == {name}
        predictions = view["posterior_predictive"][name]
        assert predictions.dims == ("chain", "draw", OBS)
        assert view["observed_data"][name].dims == (OBS,)
        np.testing.assert_array_equal(predictions.values, expected)
        np.testing.assert_array_equal(view["observed_data"][name].values, observed)
        for dim, labels in {
            "chain": [2, 7],
            "draw": [11, 17, 23],
            OBS: [101, 108, 113],
        }.items():
            np.testing.assert_array_equal(predictions[dim].values, labels)
        assert view.attrs["source_response"] == RESPONSE
        assert "not joint" in view.attrs["applicability"]
    assert views["choice"]["posterior_predictive"]["choice"].dtype == np.dtype("int8")
    xr.testing.assert_identical(paired, original)

    # Source safety includes isolation after the caller edits either view.
    views["rt"]["observed_data"]["rt"].values[0] = 99
    views["rt"]["posterior_predictive"]["rt"].values[0, 0, 0] = 99
    views["choice"]["posterior_predictive"]["choice"].values[0, 0, 0] = 1
    xr.testing.assert_identical(paired, original)


@pytest.mark.parametrize("predictive", [False, True])
def test_component_selection_uses_labels_and_restores_named_axis_order(
    adapter, paired, predictive
):
    group = "posterior_predictive" if predictive else "observed_data"
    array = paired[group][RESPONSE]
    rt, response = adapter.response_components(
        array.transpose(*reversed(array.dims)), predictive=predictive
    )
    expected_dims = ("chain", "draw", OBS) if predictive else (OBS,)
    assert rt.dims == response.dims == expected_dims
    event = EVENT if predictive else OBS_EVENT
    xr.testing.assert_identical(rt, array.sel({event: 0}, drop=True).rename("rt"))
    xr.testing.assert_identical(
        response, array.sel({event: 1}, drop=True).rename("response")
    )


@pytest.mark.parametrize("predictive", [False, True])
@pytest.mark.parametrize(
    "defect", ["unknown", "reversed", "short", "unlabeled", "duplicate"]
)
def test_rejects_ambiguous_or_malformed_component_axis(
    adapter, paired, predictive, defect
):
    group = "posterior_predictive" if predictive else "observed_data"
    event = EVENT if predictive else OBS_EVENT
    array = paired[group][RESPONSE].copy(deep=True)
    if defect == "unknown":
        array = array.rename({event: "unknown_component"})
    elif defect == "reversed":
        array = array.assign_coords({event: [1, 0]})
    elif defect == "short":
        array = array.isel({event: slice(0, 1)})
    elif defect == "unlabeled":
        array = array.drop_vars(event)
    else:
        array = array.assign_coords({event: [0, 0]})
    with pytest.raises(ValueError):
        adapter.response_components(array, predictive=predictive)


@pytest.mark.parametrize("predictive", [False, True])
@pytest.mark.parametrize(
    ("component", "invalid"),
    [
        (0, np.nan),
        (0, np.inf),
        (0, 0.0),
        (0, -999.0),
        (1, 0.0),
        (1, 2.0),
        (1, 0.5),
        (1, np.nan),
    ],
)
def test_rejects_invalid_rt_or_choice_at_any_sample(
    adapter, paired, predictive, component, invalid
):
    group = "posterior_predictive" if predictive else "observed_data"
    array = paired[group][RESPONSE].copy(deep=True)
    # Corrupt one trial in the final sample, not just the first inspected row.
    index = (-1,) * (array.ndim - 1) + (component,)
    array.values[index] = invalid
    with pytest.raises(ValueError):
        adapter.response_components(array, predictive=predictive)


@pytest.mark.parametrize("dim", ["chain", "draw", OBS])
def test_rejects_empty_sample_or_trial_axes(adapter, paired, dim):
    array = paired["posterior_predictive"][RESPONSE].isel({dim: slice(0, 0)})
    with pytest.raises(ValueError, match="Empty or unlabeled"):
        adapter.response_components(array, predictive=True)


@pytest.mark.parametrize(
    "defect", ["missing_draw", "unlabeled_trials", "duplicate_draws"]
)
def test_rejects_missing_or_ambiguous_sample_coordinates(adapter, paired, defect):
    array = paired["posterior_predictive"][RESPONSE]
    if defect == "missing_draw":
        array = array.isel(draw=0, drop=True)
    elif defect == "unlabeled_trials":
        array = array.drop_vars(OBS)
    else:
        array = array.assign_coords(draw=[11, 11, 23])
    with pytest.raises(ValueError):
        adapter.response_components(array, predictive=True)


@pytest.mark.parametrize("labels", [[108, 101, 113], [101, 108, 999], [101, 108]])
def test_rejects_prediction_rows_that_do_not_match_observations(
    adapter, paired, labels
):
    prediction = paired["posterior_predictive"].ds.isel({OBS: slice(0, len(labels))})
    paired["posterior_predictive"] = prediction.assign_coords({OBS: labels})
    with pytest.raises(ValueError, match="trial coordinates must match exactly"):
        adapter.scalar_ppc_views(paired)


@pytest.mark.parametrize("group", ["observed_data", "posterior_predictive"])
@pytest.mark.parametrize("defect", ["group", "variable"])
def test_rejects_missing_paired_evidence(adapter, paired, group, defect):
    if defect == "group":
        del paired[group]
    else:
        paired[group] = paired[group].ds.rename({RESPONSE: "unrelated"})
    with pytest.raises(ValueError, match="Missing"):
        adapter.scalar_ppc_views(paired)


def test_saved_scalar_artifacts_match_values_and_document_scope(
    adapter, paired, tmp_path
):
    original = paired.copy(deep=True)
    expected = adapter.scalar_ppc_views(paired)
    manifest = adapter.save_scalar_views(paired, tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["component_order"] == ["rt_seconds", "response"]
    assert manifest["choices"] == [-1, 1]
    assert manifest["n_observations"] == 3
    assert set(manifest["views"]) == {"rt", "choice"}
    assert any("joint" in item for item in manifest["limitations"])
    assert any("LOO-PIT" in item for item in manifest["limitations"])
    for name, entry in manifest["views"].items():
        with xr.open_datatree(tmp_path / entry["artifact"]) as restored:
            xr.testing.assert_equal(restored, expected[name])
            assert set(restored.children) == {"observed_data", "posterior_predictive"}
    xr.testing.assert_identical(paired, original)


@pytest.mark.skipif(
    os.environ.get("BAYGENT_TEST_HSSM") != "1",
    reason="HSSM simulation and full notebook integration require explicit opt-in; execution is deferred.",
)
def test_authorized_ddm_notebook_preserves_joint_and_marginal_evidence(
    tmp_path, monkeypatch
):
    """Future full integration; diagnostic problems remain valid reported outcomes."""
    notebook = REPO / "examples" / "hssm-workflow" / "analytical_ddm.py"
    original_path = sys.path.copy()
    monkeypatch.setenv("BAYGENT_RUN_HSSM", "1")
    monkeypatch.setenv("BAYGENT_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delitem(sys.modules, "hssm_example_support", raising=False)
    try:
        spec = importlib.util.spec_from_file_location("hssm_ddm_integration", notebook)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        _, analysis = module.app.run()
    finally:
        sys.path[:] = original_path

    model, posterior = analysis["model"], analysis["posterior"]
    observed = posterior["observed_data"][RESPONSE]
    joint = posterior["log_likelihood"][RESPONSE]
    assert joint.dims == ("chain", "draw", OBS)
    assert joint.sizes[OBS] == len(analysis["data"])
    np.testing.assert_array_equal(joint[OBS], observed[OBS])
    assert np.isfinite(joint.values).all()
    assert set(posterior["log_likelihood"].data_vars) == {RESPONSE}
    xr.testing.assert_identical(
        posterior["observed_data"].ds, analysis["idata"]["observed_data"].ds
    )
    free_names = {variable.name for variable in model.pymc_model.free_RVs}
    assert free_names == {"v", "a", "z", "t"}
    assert set(posterior["log_prior"].data_vars) == free_names

    # Independently check the declared flat priors, including lognormal's
    # Jacobian. Completeness and finiteness alone do not prove correct density.
    samples = posterior["posterior"]
    expected_prior = {
        "v": -0.5 * np.log(2 * np.pi) - np.log(1.5) - 0.5 * (samples["v"] / 1.5) ** 2,
        "a": -0.5 * np.log(2 * np.pi)
        - np.log(0.35)
        - np.log(samples["a"])
        - 0.5 * ((np.log(samples["a"]) - np.log(1.2)) / 0.35) ** 2,
        "z": np.log(6) + np.log(samples["z"]) + np.log1p(-samples["z"]),
        "t": xr.zeros_like(samples["t"]) - np.log(analysis["t_upper"]),
    }
    assert bool(((samples["t"] > 0) & (samples["t"] < analysis["t_upper"])).all())
    for name, expected in expected_prior.items():
        actual = posterior["log_prior"][name]
        assert np.isfinite(actual.values).all()
        xr.testing.assert_allclose(actual, expected.transpose(*actual.dims))

    directory = analysis["output_dir"]
    report = (directory / "report.md").read_text()
    with xr.open_datatree(directory / "inference_data.nc") as restored:
        xr.testing.assert_allclose(
            restored["log_likelihood"].ds, posterior["log_likelihood"].ds
        )
    with xr.open_datatree(directory / "posterior.nc") as checkpoint:
        assert "log_prior" not in checkpoint.children
        xr.testing.assert_allclose(
            checkpoint["posterior"].ds, analysis["idata"]["posterior"].ds
        )
    for quantity in ("rt", "choice"):
        scalar_dir = directory / "calibration" / quantity
        with xr.open_datatree(scalar_dir / "inference_data.nc") as marginal:
            assert set(marginal.children) == {"observed_data", "posterior_predictive"}
            assert marginal["posterior_predictive"][quantity].dims == (
                "chain",
                "draw",
                OBS,
            )
        calibration = json.loads((scalar_dir / "calibration.json").read_text())
        checks = json.loads((scalar_dir / "check_report.json").read_text())
        assert "error" not in calibration
        assert isinstance(calibration["assessment"]["well_calibrated"], bool)
        assert checks["calibration"]["rating"] in {
            "excellent",
            "fair",
            "poor",
            "not computed",
        }
        assert checks["summary"]["calibration"] in report
        assert isinstance(checks["next_steps"], list)
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        assert (directory / target).is_file(), target
    assert "marginal" in report.lower() and "joint" in report.lower()
