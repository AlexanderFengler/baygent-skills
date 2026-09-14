# Marimo generates explicit cell returns and uses a final expression for display.
# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from importlib.metadata import version
    from pathlib import Path

    import hssm
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hssm_example_support import result_directory, write_report

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "hssm-workflow" / "scripts")
    )
    from prepare_rt_choice import response_components

    return (
        hssm,
        mo,
        np,
        os,
        pd,
        plt,
        pm,
        response_components,
        result_directory,
        version,
        write_report,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Analytical DDM — priors, choice/RT evidence and diagnostics

    This notebook implements a flat drift-diffusion workflow against the
    HSSM 0.5.0 baseline. Run it to generate synthetic data, fit the model and
    inspect the resulting evidence. Opening the preview does not imply that
    a fit has run; the repository's validation record states which checks
    have been completed on its recorded dependency stack.

    The teaching question is whether a constant-drift, constant-boundary DDM
    can describe the joint choices and reaction times of a simple two-choice
    task. The generated data have no subject effects, missing times, deadlines,
    or lapses. A drift parameter describes evidence accumulation under this
    model; it is not a model-free measure of ability or a causal effect.

    The button below starts synthetic data generation and prior predictions.
    In an interactive session, a second button starts fitting after you
    review those predictions. Merely opening this notebook does not simulate or sample.
    For a later authorized command-line export, `BAYGENT_RUN_HSSM=1` is an
    equivalent explicit opt-in. It is off by default. A single teaching fit
    does not establish general parameter recovery or skill-agent performance.
    """)
    return


@app.cell
def _(mo):
    run_workflow = mo.ui.run_button(
        label="Generate teaching data and inspect prior predictions"
    )
    run_workflow
    return (run_workflow,)


@app.cell
def generate_teaching_data(
    hssm, mo, np, os, pd, result_directory, run_workflow, version
):
    _run_explicitly = run_workflow.value or os.environ.get("BAYGENT_RUN_HSSM") == "1"
    mo.stop(
        not _run_explicitly, mo.md("Preview only: no data or fit has been generated.")
    )
    # Starting an explicitly requested run invalidates any earlier success
    # report before simulation can fail. Default preview never reaches here.
    output_dir = result_directory("analytical-ddm")
    RANDOM_SEED = sum(map(ord, "analytical-ddm"))
    source_baseline = "fefed57d2142637af503b0e92cefe799715c0f46"
    runtime_versions = {
        name: version(name)
        for name in (
            "hssm",
            "ssm-simulators",
            "bambi",
            "pymc",
            "arviz",
            "arviz-stats",
            "marimo",
        )
    }
    if runtime_versions["hssm"] != "0.5.0":
        raise RuntimeError(
            "This teaching example targets HSSM 0.5.0; review another version explicitly."
        )
    generating_parameters = {"v": 0.7, "a": 1.2, "z": 0.5, "t": 0.25}
    data = hssm.simulate_data(
        model="ddm",
        theta=generating_parameters,
        size=300,
        random_state=RANDOM_SEED,
        output_df=True,
    )
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected simulator output_df=True to return a DataFrame.")
    if not np.isfinite(data[["rt", "response"]].to_numpy()).all():
        raise ValueError(
            "This example requires finite, complete RT/choice observations."
        )
    if not (data.rt > 0).all() or not data.response.isin([-1, 1]).all():
        raise ValueError(
            "Expected positive RTs and response labels exactly -1 or 1; no censoring is supported."
        )
    data["response"] = data.response.astype(int)
    data["condition"] = "teaching"  # A display label; not a parameter regression.
    observed_counts = data.response.value_counts().reindex([-1, 1], fill_value=0)
    if (observed_counts < 5).any():
        raise ValueError(
            "Too few observations of one choice for this conditional RT demonstration."
        )
    # Reaction times are in seconds because the simulator uses seconds.
    t_upper = min(0.5, 0.95 * float(data.rt.min()))
    data.to_csv(output_dir / "data.csv", index=False)
    return (
        RANDOM_SEED,
        data,
        generating_parameters,
        output_dir,
        runtime_versions,
        source_baseline,
        t_upper,
    )


@app.cell
def _(data, mo, t_upper):
    mo.md(f"""
    ## State the model and prior policy

    The generated observations contain {len(data)} positive reaction times
    in seconds and choices labelled −1 and +1. These labels mean lower- and
    upper-boundary responses; neither is called “correct” without a task
    definition. No observations have been excluded or recoded except an
    integer cast of the already valid choice labels.

    The likelihood is explicitly **analytical DDM**. We use Normal(0,1.5)
    for drift `v`, LogNormal(log(1.2),0.35) for positive boundary parameter
    `a`, Beta(2,2) for relative starting point `z`, and a Uniform prior on
    non-decision time `t`. Its upper limit is `min(0.5, 0.95 × min(rt))`,
    currently {t_upper:.4f} seconds. This is an explicit data-informed
    teaching restriction, not an independently elicited prior. Real data
    require a measurement-informed policy and assessment of sensitivity.

    All four parameters are scalar. No formula or hierarchical
    parameterization is needed. Their natural supports, prior distributions,
    and a regression link would be different objects; this example uses no
    regression links. We explicitly disable the outlier mixture and missing/
    deadline modes. Those choices would require reassessment for real data.
    """)
    return


@app.cell
def _(data, hssm, mo, np, output_dir, t_upper):
    model = hssm.HSSM(
        data=data,
        model="ddm",
        loglik_kind="analytical",
        choices=[-1, 1],
        p_outlier=None,
        missing_data=False,
        deadline=False,
        include=[
            hssm.Param(name="v", prior=hssm.Prior("Normal", mu=0, sigma=1.5)),
            hssm.Param(
                name="a", prior=hssm.Prior("LogNormal", mu=np.log(1.2), sigma=0.35)
            ),
            hssm.Param(name="z", prior=hssm.Prior("Beta", alpha=2, beta=2)),
            hssm.Param(
                name="t",
                prior=hssm.Prior("Uniform", lower=0, upper=t_upper),
                bounds=(0, t_upper),
            ),
        ],
    )
    # HSSM constructs its own model; Bambi's build/fit workflow does not apply.
    model_graph = model.graph()
    model_graph.render(
        filename=str(output_dir / "model_graph"), format="png", cleanup=True
    )
    mo.vstack([mo.md(f"```text\n{model}\n```"), model_graph])
    return (model,)


@app.cell
def define_domain_checks(data, np, pd, response_components):
    def choice_rt_checks(tree, group):
        """Compare replicate choice rates and conditional RT quantiles without flattening axes."""
        # The installed adapter validates labeled component coordinates [0,1]
        # as [RT, response] and preserves chain/draw/trial axes.
        rt_array, choice_array = response_components(
            tree[group]["rt,response"], predictive=True
        )
        rts, choices = rt_array.to_numpy(), choice_array.to_numpy()
        if rts.shape[-1] != len(data):
            raise ValueError("Expected one predicted response per observed trial.")
        # Combine only chain/draw into a replicate axis, retaining each replicate's trials.
        replicate_rts = rts.reshape(-1, rts.shape[-1])
        replicate_choices = choices.reshape(-1, choices.shape[-1])
        rows = []
        for choice in [-1, 1]:
            observed_rt = data.loc[data.response == choice, "rt"].to_numpy()
            if observed_rt.size == 0:
                raise ValueError(
                    f"No observed trials contain choice {choice}; conditional RT checks are unavailable."
                )
            proportions = (replicate_choices == choice).mean(axis=1)
            low, high = np.quantile(proportions, [0.03, 0.97])
            rows.append(
                {
                    "stage": group,
                    "choice": choice,
                    "quantity": "choice proportion",
                    "observed": float((data.response == choice).mean()),
                    "predicted_mean": float(proportions.mean()),
                    "lower_94": float(low),
                    "upper_94": float(high),
                    "replicates_with_quantity": len(proportions),
                    "total_replicates": len(proportions),
                }
            )
            for q in [0.1, 0.5, 0.9]:
                # A replicate with no such choice has no conditional RT quantile.
                quantiles = np.asarray(
                    [
                        np.quantile(rt[response == choice], q)
                        for rt, response in zip(
                            replicate_rts, replicate_choices, strict=True
                        )
                        if np.any(response == choice)
                    ]
                )
                if len(quantiles) == 0:
                    raise ValueError(
                        f"No predictive replicate contains choice {choice}; conditional RT checks are unavailable."
                    )
                low, high = np.quantile(quantiles, [0.03, 0.97])
                rows.append(
                    {
                        "stage": group,
                        "choice": choice,
                        "quantity": f"RT q={q:.1f} (s)",
                        "observed": float(np.quantile(observed_rt, q)),
                        "predicted_mean": float(quantiles.mean()),
                        "lower_94": float(low),
                        "upper_94": float(high),
                        "replicates_with_quantity": len(quantiles),
                        "total_replicates": len(proportions),
                    }
                )
        return pd.DataFrame(rows)

    return (choice_rt_checks,)


@app.cell
def prepare_prior_predictions(
    RANDOM_SEED, choice_rt_checks, data, hssm, mo, model, np, plt
):
    prior = model.sample_prior_predictive(
        draws=200, random_seed=np.random.default_rng(RANDOM_SEED + 1)
    )
    prior_checks = choice_rt_checks(prior, "prior_predictive")
    _upper_choice = prior_checks.loc[
        (prior_checks.choice == 1) & (prior_checks.quantity == "choice proportion")
    ].iloc[0]
    prior_assessment = (
        f"The observed +1 choice fraction is {_upper_choice.observed:.3f}; "
        "the central 94% prior-predictive interval for that fraction is "
        f"[{_upper_choice.lower_94:.3f}, {_upper_choice.upper_94:.3f}]. "
        "The table also compares observed RT quantiles separately for each choice. "
        "These are descriptive comparisons, not a calibration test or an automatic "
        "prior-adequacy verdict. The Normal drift, positive LogNormal boundary, "
        "Beta starting-point and restricted Uniform non-decision-time priors express "
        "plausible teaching assumptions; their suitability still depends on the task "
        "and RT measurement process. Inspect excessively slow tails, implausibly fast "
        "RTs, and extreme choice imbalance before fitting. Revisit the explicit "
        "data-informed t upper bound when applying this workflow to real observations."
    )
    prior_review = mo.ui.run_button(
        label="I reviewed the prior predictions; fit the DDM"
    )
    prior_figure, _ax = plt.subplots(figsize=(8, 4))
    hssm.plotting.plot_predictive(
        model,
        dt=prior,
        data=data,
        predictive_group="prior_predictive",
        n_samples=None,
        uncertainty="band",
        hdi=0.94,
        ax=_ax,
        title="Prior predictions: signed RT retains choice",
        xlabel="Signed RT (seconds)",
    )
    mo.vstack(
        [
            mo.md(
                "## Prior predictions before fitting\nInspect RT scale and tails separately for each choice, as well as choice frequencies. Negative positions on the signed plot encode choice −1, not negative observed RTs."
            ),
            prior_figure,
            prior_checks,
            mo.md(prior_assessment),
            prior_review,
        ]
    )
    return prior, prior_assessment, prior_checks, prior_figure, prior_review


@app.cell
def fit_ddm(RANDOM_SEED, mo, model, os, output_dir, prior_review):
    mo.stop(
        not (prior_review.value or os.environ.get("BAYGENT_RUN_HSSM") == "1"),
        mo.md(
            "Inspect the prior RT/choice evidence above, then use the prior-review button to start fitting."
        ),
    )
    sampler_settings = {
        "sampler": "pymc",
        "draws": 1000,
        "tune": 1000,
        "chains": 2,
        "cores": 1,
        "random_seed": RANDOM_SEED + 2,
        "target_accept": 0.9,
    }
    idata = model.sample(**sampler_settings)
    # Checkpoint before posterior prediction, plotting, or report calculations.
    idata.to_netcdf(output_dir / "posterior.nc")
    model.save_model(
        model_name="checkpoint",
        base_path=output_dir,
        allow_absolute_base_path=True,
        save_traces_only=True,
    )
    return idata, sampler_settings


@app.cell
def prepare_posterior_evidence(
    choice_rt_checks,
    idata,
    model,
    np,
    output_dir,
    pd,
    pm,
    prior_checks,
    t_upper,
):
    # These native release APIs index samples by position or rebuild their
    # labels. Refuse arbitrary labels before recomputation; scalar adapter
    # views independently preserve arbitrary valid labels without this limit.
    for _dim in ("chain", "draw"):
        _samples = idata["posterior"]
        if _dim not in _samples.coords or not np.array_equal(
            _samples[_dim].values, np.arange(_samples.sizes[_dim])
        ):
            raise ValueError(
                "HSSM 0.5.0 / PyMC 6.1 native density recomputation requires "
                f"zero-based consecutive {_dim} coordinates; the artifact was not relabeled."
            )
    posterior = model.sample_posterior_predictive(
        dt=idata, kind="response", draws=None, inplace=False, safe_mode=True
    )
    if posterior is None:
        raise RuntimeError("Expected inplace=False to return predictive draws.")
    # Keep the joint density of one (RT, choice) trial, not two marginal densities.
    model.log_likelihood(dt=posterior, inplace=True)
    _loglik = posterior["log_likelihood"]["rt,response"]
    if set(_loglik.dims) != {"chain", "draw", "__obs__"}:
        raise ValueError(
            "Joint pointwise likelihood must have one value per chain/draw/trial."
        )
    # Flat parameters have no centered regression intercept or omitted offsets.
    # HSSM has no public log-prior wrapper; this PyMC bridge is source-checked
    # for this case only and verified below against analytical prior densities.
    _free_names = {rv.name for rv in model.pymc_model.free_RVs}
    if _free_names != {"v", "a", "z", "t"}:
        raise ValueError(
            "The density verification is specific to the four scalar DDM parameters."
        )
    if not _free_names.issubset(set(posterior["posterior"].data_vars)):
        raise ValueError(
            "The saved posterior is missing free variables needed for prior density."
        )
    pm.compute_log_prior(posterior, model=model.pymc_model, progressbar=False)
    if _free_names != set(posterior["log_prior"].data_vars):
        raise ValueError(
            "Prior density must cover exactly the flat model's free random variables."
        )
    # Names alone cannot prove that effective densities match the declared
    # priors. Check their normalized analytic densities draw by draw before
    # allowing power sensitivity or report assessment to consume them.
    _v, _a, _z, _t = (posterior["posterior"][name] for name in ("v", "a", "z", "t"))
    if any(not np.isfinite(value).all().item() for value in (_v, _a, _z, _t)):
        raise ValueError("Density verification requires finite parameter draws.")
    if not (
        (_a > 0).all() & ((_z > 0) & (_z < 1)).all() & ((_t > 0) & (_t < t_upper)).all()
    ).item():
        raise ValueError("Draws must respect positive a, 0<z<1, and 0<t<t_upper.")
    _normal_constant = 0.5 * np.log(2 * np.pi)
    _analytic_log_prior = {
        "v": -0.5 * (_v / 1.5) ** 2 - np.log(1.5) - _normal_constant,
        "a": -0.5 * ((np.log(_a) - np.log(1.2)) / 0.35) ** 2
        - np.log(_a)
        - np.log(0.35)
        - _normal_constant,
        "z": np.log(6.0) + np.log(_z) + np.log1p(-_z),
        "t": 0 * _t - np.log(t_upper),
    }
    for _name, _expected in _analytic_log_prior.items():
        _actual = posterior["log_prior"][_name]
        if set(_expected.dims) != {"chain", "draw"} or set(_actual.dims) != {
            "chain",
            "draw",
        }:
            raise ValueError(
                "Each scalar parameter prior must preserve only chain and draw dimensions."
            )
        for _dim in ("chain", "draw"):
            if not _actual.get_index(_dim).equals(_expected.get_index(_dim)):
                raise ValueError(
                    "Native and analytic prior sample coordinates must match exactly."
                )
        np.testing.assert_allclose(
            _actual.transpose("chain", "draw").to_numpy(),
            _expected.transpose("chain", "draw").to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Effective prior density differs for {_name}",
        )
    prior_density_verified = True
    posterior.to_netcdf(output_dir / "posterior_predictive.nc")
    posterior_checks = choice_rt_checks(posterior, "posterior_predictive")
    domain_checks = pd.concat([prior_checks, posterior_checks], ignore_index=True)
    domain_checks.to_csv(output_dir / "choice_rt_checks.csv", index=False)
    return domain_checks, posterior, posterior_checks, prior_density_verified


@app.cell
def plot_posterior_evidence(
    data,
    hssm,
    mo,
    model,
    output_dir,
    plt,
    posterior,
    posterior_checks,
    prior_figure,
):
    posterior_figure, _ax = plt.subplots(figsize=(8, 4))
    hssm.plotting.plot_predictive(
        model,
        dt=posterior,
        data=data,
        n_samples=None,
        uncertainty="band",
        hdi=0.94,
        ax=_ax,
        title="Posterior predictions: joint choice and RT",
        xlabel="Signed RT (seconds)",
    )
    quantile_figure, _qp_ax = plt.subplots(figsize=(8, 4))
    # HSSM already computes quantiles separately by response sign. quantile_by
    # adds other grouping variables; passing "response" duplicates its column.
    # Use the current axes: HSSM 0.5 forwards an explicit ax twice to seaborn.
    hssm.plotting.plot_quantile_probability(
        model,
        cond="condition",
        data=data,
        dt=posterior,
        n_samples=None,
        q=[0.1, 0.5, 0.9],
    )
    prior_figure.savefig(
        output_dir / "prior_predictive.png", dpi=160, bbox_inches="tight"
    )
    posterior_figure.savefig(
        output_dir / "posterior_predictive.png", dpi=160, bbox_inches="tight"
    )
    quantile_figure.savefig(
        output_dir / "quantile_probability.png", dpi=160, bbox_inches="tight"
    )
    mo.vstack(
        [
            mo.md(
                "## Inspect joint predictions\nThe table reports replicate intervals, not a calibrated pass/fail threshold. Conditional RT quantiles exclude replicates without that choice and show their count. Check choice proportions alongside these quantiles; agreement in pooled RT alone can hide a choice-specific mismatch."
            ),
            posterior_figure,
            quantile_figure,
            posterior_checks,
        ]
    )
    figures_written = True
    return figures_written, quantile_figure


@app.cell
def _(
    RANDOM_SEED,
    domain_checks,
    figures_written,
    generating_parameters,
    mo,
    model,
    output_dir,
    posterior,
    prior,
    prior_assessment,
    prior_density_verified,
    runtime_versions,
    sampler_settings,
    source_baseline,
    t_upper,
    write_report,
):
    if not figures_written:
        raise RuntimeError("The report requires the actual predictive figures.")
    report = write_report(
        model,
        prior,
        posterior,
        output_dir,
        domain_checks=domain_checks,
        details={
            "title": "Flat analytical DDM",
            "model": "ddm",
            "likelihood": "analytical",
            "source_baseline": source_baseline,
            "runtime_versions": runtime_versions,
            "seed": RANDOM_SEED,
            "sampler_settings": sampler_settings,
            "generating_parameters": generating_parameters,
            "choices": [-1, 1],
            "units": "seconds",
            "question": "Does a flat DDM describe the joint RT and choices in the generated two-choice task?",
            "generative_story": "Constant-drift, constant-boundary DDM with scalar v, a, z and t; no lapses, deadlines or missing responses.",
            "prior_policy": f"v Normal(0,1.5); a LogNormal(log(1.2),0.35); z Beta(2,2); t Uniform(0,{t_upper}), with an explicitly data-informed upper bound.",
            "prior_assessment": prior_assessment,
            "prior_density_verified": prior_density_verified,
            "predictive_settings": {
                "prior_draws": 200,
                "prior_seed": RANDOM_SEED + 1,
                "posterior_draws_per_chain": "All retained posterior draws (draws=None); preserve complete sample coordinates.",
                "safe_mode": True,
                "posterior_random_seed": "HSSM 0.5.0's public posterior predictive API has no random_seed argument; no seeded repeatability claim is made for those draws.",
            },
            "interpretation": "Inspect choice proportions and conditional RT quantiles together. Posterior parameters describe this DDM, not causal or model-free cognitive effects.",
            "limitations": "This single synthetic teaching run cannot establish general parameter recovery. Fitted-data marginal RT/choice PPC-PIT does not establish joint, conditional, or held-out calibration. The PyMC log-prior bridge is verified here only for this flat model with default sample coordinates; power sensitivity does not assess changes to the data-informed t support policy.",
        },
    )
    mo.vstack(
        [
            mo.md(
                "## Shared assessment and domain evidence\nFollow diagnostic next steps. Separate marginal calibration, joint RT/choice checks, and parameter interpretation; none substitutes for the others."
            ),
            report["checks"]["summary"],
            {"next_steps": report["next_steps"]},
            {
                name: {
                    "calibration": assessment["summary"]["calibration"],
                }
                for name, assessment in report["marginal_checks"].items()
            },
        ]
    )
    return


@app.cell
def _(mo):
    displayed_choice = mo.ui.dropdown(
        {"Lower boundary (−1)": -1, "Upper boundary (+1)": 1},
        value="Upper boundary (+1)",
        label="Display choice",
    )
    mo.vstack(
        [
            mo.md(
                "## Inspect one choice\nThis control only filters an existing table. It never changes the data, model, or samples."
            ),
            displayed_choice,
        ]
    )
    return (displayed_choice,)


@app.cell
def _(displayed_choice, posterior_checks):
    posterior_checks.loc[posterior_checks.choice == displayed_choice.value]
    return


if __name__ == "__main__":
    app.run()
