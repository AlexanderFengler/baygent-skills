"""Fixed-array calibration contracts; no model construction or fitting."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "bayesian-workflow/scripts/calibration_check.py"


@pytest.fixture(scope="module")
def calibration():
    spec = importlib.util.spec_from_file_location("calibration_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pit():
    # Nearly uniform except for a central atom. The mean coverage displacement
    # is small, while pot_c detects the exact-zero transformed PIT.
    values = np.linspace(0.003, 0.997, 300)
    values[0], values[150] = 0.0001, 0.5
    return xr.Dataset({"y": ("trial", values)}, coords={"trial": np.arange(300) + 50})


@pytest.fixture
def predictive():
    return xr.DataTree.from_dict(
        {
            "observed_data": xr.Dataset(
                {"y": ("trial", [0, 1, 0, 1, 1, 0])},
                coords={"trial": [11, 13, 17, 19, 23, 29]},
            ),
            "posterior_predictive": xr.Dataset(
                {
                    "y": (
                        ("chain", "draw", "trial"),
                        [
                            [
                                [0, 0, 1, 1, 0, 0],
                                [1, 1, 0, 1, 1, 1],
                                [0, 1, 0, 0, 1, 0],
                            ],
                            [
                                [1, 1, 0, 0, 1, 0],
                                [0, 0, 1, 1, 0, 1],
                                [0, 1, 0, 1, 1, 0],
                            ],
                        ],
                    )
                },
                coords={
                    "chain": [2, 4],
                    "draw": [7, 8, 9],
                    "trial": [11, 13, 17, 19, 23, 29],
                },
            ),
        }
    )


def test_default_method_is_capability_explicit(calibration):
    expected = "pot_c" if calibration._has_modern_pit() else "envelope"
    assert calibration._resolve_method("auto") == expected
    with pytest.raises(ValueError, match="Unknown uniformity method"):
        calibration._resolve_method("looks-uniform")


def test_unavailable_modern_method_fails_visibly(calibration, monkeypatch):
    monkeypatch.setattr(calibration, "_has_modern_pit", lambda: False)
    with pytest.raises(ValueError, match="requires modern ArviZ"):
        calibration._resolve_method("pot_c")


def test_legacy_discrete_pit_matches_native_tie_convention(calibration, predictive):
    pit = calibration.prepare_pit_values(predictive, "y", method="envelope")
    # Fixed expected fractions from the literal predictions above; exact native
    # seed/orientation makes a silent randomized-rank convention change visible.
    lower = np.array([0, 2 / 6, 0, 2 / 6, 2 / 6, 0])
    upper = np.array([4 / 6, 1, 4 / 6, 1, 1, 4 / 6])
    uniforms = np.random.default_rng(214).uniform(size=6)
    np.testing.assert_allclose(pit.y.values, uniforms * lower + (1 - uniforms) * upper)
    xr.testing.assert_identical(pit.trial, predictive["observed_data"].trial)
    assert pit.y.dims == ("trial",)


def test_modern_native_pit_is_used_without_coverage_transform(calibration, predictive):
    if not calibration._has_modern_pit():
        pytest.skip("pot_c is unavailable on the legacy stack")
    original = predictive.copy(deep=True)
    result = calibration.prepare_pit_values(predictive, "y", method="pot_c")
    expected = calibration.get_ppc_pit(
        predictive["posterior_predictive"].ds,
        predictive["observed_data"].ds,
        ["chain", "draw"],
        coverage=False,
        method="pot_c",
    )["ecdf_pit"].ds
    xr.testing.assert_identical(result, expected)
    xr.testing.assert_identical(predictive, original)


@pytest.mark.parametrize("group", ["observed_data", "posterior_predictive"])
@pytest.mark.parametrize("method", ["envelope", "auto"])
def test_nonfinite_input_cannot_become_a_finite_pit(
    calibration, predictive, group, method
):
    bad = predictive.copy(deep=True)
    bad[group]["y"] = bad[group]["y"].astype(float) * np.nan
    with pytest.raises(ValueError, match="finite"):
        calibration.prepare_pit_values(bad, "y", method=method)


@pytest.mark.parametrize("method", ["envelope", "auto"])
def test_empty_predictive_draws_rejected(calibration, predictive, method):
    predictive["posterior_predictive"] = predictive["posterior_predictive"].isel(
        draw=slice(0, 0)
    )
    with pytest.raises(ValueError, match="nonempty"):
        calibration.prepare_pit_values(predictive, "y", method=method)


@pytest.mark.parametrize("method", ["envelope", "auto"])
def test_row_coordinate_mismatch_rejected(calibration, predictive, method):
    predictive["observed_data"] = xr.DataTree(
        predictive["observed_data"].ds.assign_coords(trial=np.arange(6))
    )
    with pytest.raises(ValueError, match="align|index|label"):
        calibration.prepare_pit_values(predictive, "y", method=method)


def test_envelope_preserves_native_interior_test_and_reports_real_zero_atom(
    calibration, pit
):
    result = calibration.assess_calibration(
        None, "y", False, method="envelope", pit_values=pit
    )
    assert result["well_calibrated"] is True
    assert result["pit_ecdf_inside_bands"] is True
    assert result["coverage_ecdf_inside_bands"] is True
    assert result["pit_p_value"] is None and result["coverage_p_value"] is None
    coverage = 2 * np.abs(pit - 0.5)
    evidence = calibration._evaluate_pit(coverage, "y", "envelope", 0.99)
    assert evidence["delta"][0] == 1 / 300
    assert np.isnan(evidence["lower"][0])  # Native endpoint padding is not a test.
    x, native_ecdf, lo, hi = calibration.ecdf_pit(
        coverage.y.values, 0.99, n_simulations=1000
    )
    assert evidence["passed"] == bool(((native_ecdf >= lo) & (native_ecdf <= hi)).all())
    np.testing.assert_allclose(evidence["x"], x)
    np.testing.assert_allclose(evidence["delta"][1:-1], (native_ecdf - x)[1:-1])


def test_modern_shape_failure_is_not_called_well_calibrated(calibration, pit):
    if not calibration._has_modern_pit():
        pytest.skip("pot_c is unavailable on the legacy stack")
    result = calibration.assess_calibration(
        None, "y", False, method="pot_c", pit_values=pit
    )
    assert result["pit_uniformity_passed"] is True
    assert result["coverage_uniformity_passed"] is False
    assert result["well_calibrated"] is False
    assert abs(result["mean_coverage_deviation"]) < 0.02
    assert (
        result["calibration_diagnosis"]
        == "non-uniform predictive PIT; no dominant mean coverage direction"
    )
    assert result["pit_ecdf_inside_bands"] is None
    assert result["coverage_ecdf_inside_bands"] is None
    p_raw, _, _ = pit.azstats.uniformity_test(dim=["trial"], method="pot_c")
    p_cov, _, _ = (2 * abs(pit - 0.5)).azstats.uniformity_test(
        dim=["trial"], method="pot_c"
    )
    assert result["pit_p_value"] == p_raw.y.item()
    assert result["coverage_p_value"] == p_cov.y.item()
    assert result["coverage_p_value"] < result["significance_level"]


@pytest.mark.parametrize("coverage", [False, True])
@pytest.mark.parametrize("method", ["envelope", "auto"])
def test_plot_curve_matches_tested_values_without_axis_rescaling(
    calibration, pit, tmp_path, monkeypatch, coverage, method
):
    prepared = pit.copy(deep=True)
    expected_values = 2 * abs(pit - 0.5) if coverage else pit
    resolved = calibration._resolve_method(method)
    evidence = calibration._evaluate_pit(expected_values, "y", resolved, 0.99)
    captured = {}
    savefig = matplotlib.figure.Figure.savefig

    def capture(self, path, *args, **kwargs):
        ax = self.axes[0]
        captured["curve"] = ax.lines[1].get_data()
        captured["texts"] = [text.get_text() for text in ax.texts]
        captured["xlabel"] = ax.get_xlabel()
        captured["xlim"] = ax.get_xlim()
        return savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    monkeypatch.setattr(
        calibration,
        "prepare_pit_values",
        lambda *a, **k: pytest.fail("PIT regenerated"),
    )
    output = tmp_path / "pit.png"
    calibration.save_pit_plot(
        None, "y", output, coverage=coverage, method=method, pit_values=prepared
    )
    assert output.stat().st_size > 1000
    xr.testing.assert_identical(pit, prepared)
    np.testing.assert_allclose(captured["curve"][0], evidence["x"])
    np.testing.assert_allclose(captured["curve"][1], evidence["delta"])
    assert captured["xlim"] == (0, 1)
    if resolved == "pot_c":
        assert captured["texts"] == [
            f"pot_c: p={evidence['p_value']:.3g}, α={1 - 0.99:.3g}"
        ]
    else:
        assert captured["texts"] == [
            f"envelope: {'pass' if evidence['passed'] else 'fail'}"
        ]
    if coverage:
        assert "coverage (%)" in captured["xlabel"]
        assert captured["curve"][1][0] == 1 / 300  # Exactly one transform.


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_precomputed_pit_rejected(calibration, pit, bad):
    pit.y.values[0] = bad
    with pytest.raises(ValueError, match="PIT values"):
        calibration.assess_calibration(
            None, "y", False, method="envelope", pit_values=pit
        )


@pytest.mark.parametrize("probability", [0, 1, np.nan])
def test_invalid_significance_rejected(calibration, pit, probability):
    with pytest.raises(ValueError, match="ci_prob"):
        calibration.assess_calibration(
            None, "y", False, probability, method="envelope", pit_values=pit
        )


def test_cli_records_explicit_method_and_plots(calibration, predictive, tmp_path):
    data_path, report_path = tmp_path / "input.nc", tmp_path / "calibration.json"
    predictive.to_netcdf(data_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--idata",
            str(data_path),
            "--output",
            str(report_path),
            "--uniformity-method",
            "envelope",
            "--save-plots",
            "--plot-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(
        report_path.read_text(), parse_constant=lambda value: pytest.fail(value)
    )
    assert (
        report["uniformity_method"]
        == report["assessment"]["uniformity_method"]
        == "envelope"
    )
    assert report["coverage_transform"].endswith("applied once")
    assert report["assessment"]["significance_level"] == pytest.approx(0.01)
    assert all(Path(path).is_file() for path in report["plots"].values())


@pytest.mark.parametrize("dtype", ["bool", "uint8", "int8"])
def test_binary_storage_dtype_does_not_change_randomized_pit(
    calibration, predictive, dtype
):
    expected = calibration.prepare_pit_values(predictive, "y", method="envelope")
    converted = predictive.copy(deep=True)
    for group in ("observed_data", "posterior_predictive"):
        converted[group]["y"] = converted[group]["y"].astype(dtype)
    actual = calibration.prepare_pit_values(converted, "y", method="envelope")
    xr.testing.assert_identical(actual, expected)
