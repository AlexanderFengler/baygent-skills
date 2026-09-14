"""Literal-array domain summaries and fail-closed notebook execution gates.

These tests never call a simulator, construct an HSSM model, or run MCMC.
Named marimo cells keep the tested calculations in the readable example;
all external execution boundaries use explicit sentinels or literal fixtures.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "examples" / "hssm-workflow" / "analytical_ddm.py"
RESPONSE = "rt,response"
EVENT = "rt,response_dim"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def notebook(monkeypatch):
    # Loading the source registers cells; app.run/Cell.run controls execution.
    module = _load(NOTEBOOK, "hssm_notebook_contract_source")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original_path = sys.path.copy()
    yield module
    sys.path[:] = original_path
    plt.close("all")


@pytest.fixture
def adapter():
    return _load(
        REPO / "hssm-workflow" / "scripts" / "prepare_rt_choice.py",
        "hssm_notebook_response_adapter",
    )


@pytest.fixture
def observed():
    return pd.DataFrame({"rt": [1.0, 3.0, 2.0, 4.0], "response": [-1, -1, 1, 1]})


def _tree(values, group="posterior_predictive"):
    values = np.asarray(values, dtype=float)
    return xr.DataTree.from_dict(
        {
            group: xr.Dataset(
                {RESPONSE: (("chain", "draw", "__obs__", EVENT), values)},
                coords={
                    "chain": np.arange(values.shape[0]) + 5,
                    "draw": np.arange(values.shape[1]) + 11,
                    "__obs__": np.arange(values.shape[2]) + 100,
                    EVENT: [0, 1],
                },
            )
        }
    )


@pytest.fixture
def predictive():
    # Four replicates intentionally have unequal numbers of each choice:
    # +1 proportions 1/2, 1/4, 0, 1. The missing-choice replicates must
    # remain in these proportions but not acquire fabricated RT quantiles.
    return _tree(
        [
            [
                [[1, -1], [3, -1], [2, 1], [4, 1]],
                [[6, -1], [8, -1], [10, -1], [20, 1]],
            ],
            [
                [[2, -1], [4, -1], [6, -1], [8, -1]],
                [[10, 1], [20, 1], [30, 1], [40, 1]],
            ],
        ]
    )


def _summarizer(notebook, adapter, data):
    _, definitions = notebook.define_domain_checks.run(
        data=data, np=np, pd=pd, response_components=adapter.response_components
    )
    return definitions["choice_rt_checks"]


def test_domain_quantities_match_hand_calculated_replicates(
    notebook, adapter, observed, predictive
):
    original = predictive.copy(deep=True)
    original_data = observed.copy(deep=True)
    result = _summarizer(notebook, adapter, observed)(
        predictive, "posterior_predictive"
    )
    # Columns are observed, replicate mean, central 94% endpoints. These
    # constants are calculated from the literal per-replicate values above,
    # without calling the implementation's percentile routine for expected values.
    expected = {
        (-1, "choice proportion"): [0.5, 0.5625, 0.045, 0.9775],
        (1, "choice proportion"): [0.5, 0.4375, 0.0225, 0.955],
        (-1, "RT q=0.1 (s)"): [1.2, 3.4, 1.284, 6.172],
        (-1, "RT q=0.5 (s)"): [2.0, 5.0, 2.18, 7.82],
        (-1, "RT q=0.9 (s)"): [2.8, 6.6, 3.076, 9.468],
        (1, "RT q=0.1 (s)"): [2.2, 35.2 / 3, 2.848, 19.58],
        (1, "RT q=0.5 (s)"): [3.0, 16.0, 4.02, 24.7],
        (1, "RT q=0.9 (s)"): [3.8, 60.8 / 3, 4.772, 35.98],
    }
    indexed = result.set_index(["choice", "quantity"])
    assert set(indexed.index) == set(expected)
    for key, values in expected.items():
        row = indexed.loc[key]
        np.testing.assert_allclose(
            row[["observed", "predicted_mean", "lower_94", "upper_94"]].to_numpy(
                dtype=float
            ),
            values,
            rtol=1e-12,
            atol=1e-12,
        )
        assert row.total_replicates == 4
        assert row.replicates_with_quantity == (
            4 if key[1] == "choice proportion" else 3
        )
        assert row.stage == "posterior_predictive"
    # Pooling the seven +1 RTs would instead produce median 20. The mean
    # of the three replicate medians is 16, with three equally weighted draws.
    assert indexed.loc[(1, "RT q=0.5 (s)"), "predicted_mean"] != 20.0
    xr.testing.assert_identical(predictive, original)
    pd.testing.assert_frame_equal(observed, original_data)


def test_domain_summary_preserves_named_axes_and_group_identity(
    notebook, adapter, observed, predictive
):
    summarize = _summarizer(notebook, adapter, observed)
    expected = summarize(predictive, "posterior_predictive")
    transposed = _tree(
        predictive["posterior_predictive"][RESPONSE].values, "prior_predictive"
    )
    transposed["prior_predictive"][RESPONSE] = transposed["prior_predictive"][
        RESPONSE
    ].transpose(EVENT, "__obs__", "draw", "chain")
    result = summarize(transposed, "prior_predictive")
    pd.testing.assert_frame_equal(
        result.drop(columns="stage"), expected.drop(columns="stage")
    )
    assert result.stage.eq("prior_predictive").all()


def test_one_replicate_keeps_literal_quantiles_and_degenerate_interval(
    notebook, adapter, observed
):
    result = _summarizer(notebook, adapter, observed)(
        _tree([[[[1, -1], [3, -1], [2, 1], [4, 1]]]]), "posterior_predictive"
    )
    np.testing.assert_allclose(result.predicted_mean, result.observed)
    np.testing.assert_allclose(result.lower_94, result.observed)
    np.testing.assert_allclose(result.upper_94, result.observed)
    assert result.replicates_with_quantity.eq(1).all()
    assert result.total_replicates.eq(1).all()


@pytest.mark.parametrize("absent", [-1, 1])
def test_all_absent_predictive_choice_fails_explicitly(
    notebook, adapter, observed, predictive, absent
):
    predictive["posterior_predictive"][RESPONSE].loc[{EVENT: 1}] = -absent
    with pytest.raises(
        ValueError, match=f"No predictive replicate contains choice {absent}"
    ):
        _summarizer(notebook, adapter, observed)(predictive, "posterior_predictive")


@pytest.mark.parametrize("absent", [-1, 1])
def test_missing_observed_choice_has_an_explicit_failure(
    notebook, adapter, observed, predictive, absent
):
    observed["response"] = -absent
    with pytest.raises(ValueError, match=f"No observed trials contain choice {absent}"):
        _summarizer(notebook, adapter, observed)(predictive, "posterior_predictive")


def test_predictive_trial_count_must_match_data(
    notebook, adapter, observed, predictive
):
    predictive["posterior_predictive"] = predictive["posterior_predictive"].isel(
        __obs__=[0, 1, 2]
    )
    with pytest.raises(ValueError, match="one predicted response per observed trial"):
        _summarizer(notebook, adapter, observed)(predictive, "posterior_predictive")


@pytest.mark.parametrize("invalid", [0.0, -999.0, float("nan")])
def test_domain_summary_cannot_hide_invalid_rt(
    notebook, adapter, observed, predictive, invalid
):
    predictive["posterior_predictive"][RESPONSE].values[0, 0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite positive RTs"):
        _summarizer(notebook, adapter, observed)(predictive, "posterior_predictive")


def _forbidden(name):
    return Mock(side_effect=AssertionError(f"Forbidden execution boundary: {name}"))


@pytest.mark.parametrize("dimension", ["chain", "draw"])
def test_nondefault_sample_labels_fail_before_native_recomputation(
    notebook, tmp_path, dimension
):
    coords = {"chain": [0], "draw": [0]}
    coords[dimension] = [7]
    idata = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"v": (("chain", "draw"), [[0.2]])}, coords=coords)}
    )
    original = idata.copy(deep=True)
    model = SimpleNamespace(
        sample_posterior_predictive=_forbidden("posterior prediction"),
        log_likelihood=_forbidden("joint density"),
    )
    pm = SimpleNamespace(compute_log_prior=_forbidden("prior density"))
    with pytest.raises(
        ValueError, match=f"zero-based consecutive {dimension} coordinates"
    ):
        notebook.prepare_posterior_evidence.run(
            choice_rt_checks=_forbidden("domain checks"),
            idata=idata,
            model=model,
            np=np,
            output_dir=tmp_path,
            pd=pd,
            pm=pm,
            prior_checks=pd.DataFrame(),
            t_upper=0.2,
        )
    model.sample_posterior_predictive.assert_not_called()
    model.log_likelihood.assert_not_called()
    pm.compute_log_prior.assert_not_called()
    xr.testing.assert_identical(idata, original)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_default_preview_runs_no_modeling_and_creates_no_results(
    notebook, monkeypatch, tmp_path, flag
):
    if flag is None:
        monkeypatch.delenv("BAYGENT_RUN_HSSM", raising=False)
    else:
        monkeypatch.setenv("BAYGENT_RUN_HSSM", flag)
    output_root = tmp_path / "results-that-must-not-exist"
    monkeypatch.setenv("BAYGENT_OUTPUT_ROOT", str(output_root))
    fake_hssm = ModuleType("hssm")
    fake_hssm.simulate_data = _forbidden("simulation")
    fake_hssm.HSSM = _forbidden("model construction")
    fake_hssm.Param = _forbidden("parameter construction")
    fake_hssm.Prior = _forbidden("prior construction")
    fake_helper = ModuleType("hssm_example_support")
    fake_helper.result_directory = _forbidden("output directory creation")
    fake_helper.write_report = _forbidden("report writing")
    monkeypatch.setitem(sys.modules, "hssm", fake_hssm)
    monkeypatch.setitem(sys.modules, "hssm_example_support", fake_helper)
    _, definitions = notebook.app.run()
    assert not {"data", "model", "prior", "idata", "posterior", "output_dir"} & set(
        definitions
    )
    assert not output_root.exists()
    for boundary in (
        fake_hssm.simulate_data,
        fake_hssm.HSSM,
        fake_hssm.Param,
        fake_hssm.Prior,
        fake_helper.result_directory,
        fake_helper.write_report,
    ):
        boundary.assert_not_called()


def test_starting_an_opted_in_run_initializes_outputs_before_simulation(
    notebook, tmp_path
):
    calls = []

    def start_output(slug):
        calls.append(("output", slug))
        return tmp_path

    def stop_at_simulation(**kwargs):
        calls.append(("simulation", kwargs["model"]))
        raise AssertionError("Reached the simulation boundary without running it")

    with pytest.raises(AssertionError, match="simulation boundary"):
        notebook.generate_teaching_data.run(
            hssm=SimpleNamespace(simulate_data=stop_at_simulation),
            mo=mo,
            np=np,
            os=SimpleNamespace(environ={"BAYGENT_RUN_HSSM": "1"}),
            pd=pd,
            result_directory=start_output,
            run_workflow=SimpleNamespace(value=False),
            version=lambda name: "0.5.0" if name == "hssm" else "test-fixture",
        )
    assert calls == [("output", "analytical-ddm"), ("simulation", "ddm")]


def test_prior_only_activation_still_blocks_fit(notebook, adapter, observed, tmp_path):
    literal_data = pd.concat([observed] * 3, ignore_index=True)
    simulation_boundary = Mock(return_value=literal_data.copy(deep=True))
    _, generation = notebook.generate_teaching_data.run(
        hssm=SimpleNamespace(simulate_data=simulation_boundary),
        mo=mo,
        np=np,
        os=SimpleNamespace(environ={}),
        pd=pd,
        result_directory=lambda slug: tmp_path,
        run_workflow=SimpleNamespace(value=True),
        version=lambda name: "0.5.0" if name == "hssm" else "test-fixture",
    )
    simulation_boundary.assert_called_once()  # Returns literal rows, never a simulator.
    assert (tmp_path / "data.csv").exists()
    data = generation["data"]
    summarize = _summarizer(notebook, adapter, data)
    prior_literal = _tree([[data[["rt", "response"]].to_numpy()]], "prior_predictive")
    model = SimpleNamespace(
        sample_prior_predictive=Mock(return_value=prior_literal),
        sample=_forbidden("MCMC"),
        save_model=_forbidden("checkpoint"),
    )
    plot_boundary = Mock()
    _, predictions = notebook.prepare_prior_predictions.run(
        RANDOM_SEED=generation["RANDOM_SEED"],
        choice_rt_checks=summarize,
        data=data,
        hssm=SimpleNamespace(plotting=SimpleNamespace(plot_predictive=plot_boundary)),
        mo=mo,
        model=model,
        np=np,
        plt=plt,
    )
    model.sample_prior_predictive.assert_called_once()
    plot_boundary.assert_called_once()
    assert len(predictions["prior_checks"]) == 8
    assert "descriptive comparisons" in predictions["prior_assessment"]
    assert predictions["prior_review"].value is False
    _, fit = notebook.fit_ddm.run(
        RANDOM_SEED=generation["RANDOM_SEED"],
        mo=mo,
        model=model,
        os=SimpleNamespace(environ={}),
        output_dir=tmp_path,
        prior_review=predictions["prior_review"],
    )
    assert "idata" not in fit
    model.sample.assert_not_called()
    model.save_model.assert_not_called()
    assert not (tmp_path / "posterior.nc").exists()
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize("button, flag", [(True, None), (False, "1")])
def test_sampling_requires_its_own_explicit_activation(
    notebook, tmp_path, button, flag
):
    model = SimpleNamespace(sample=_forbidden("authorized sampler entry"))
    with pytest.raises(AssertionError, match="authorized sampler entry"):
        notebook.fit_ddm.run(
            RANDOM_SEED=1,
            mo=mo,
            model=model,
            os=SimpleNamespace(
                environ={} if flag is None else {"BAYGENT_RUN_HSSM": flag}
            ),
            output_dir=tmp_path,
            prior_review=SimpleNamespace(value=button),
        )
    model.sample.assert_called_once()
    assert model.sample.call_args.kwargs["draws"] == 1000
    assert model.sample.call_args.kwargs["chains"] == 2
    assert not (tmp_path / "posterior.nc").exists()
