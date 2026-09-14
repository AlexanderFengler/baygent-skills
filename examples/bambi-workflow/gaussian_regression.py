# Marimo generates explicit cell returns and uses a final expression for display.
# ruff: noqa: B018, PLR1711
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import bambi as bmb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from example_support import result_directory, write_report

    return bmb, mo, np, os, pd, plt, result_directory, write_report


@app.cell
def _(mo):
    mo.md(r"""
    # Gaussian regression: from a question to a checked contrast

    Does increasing a standardized exposure from −0.5 to +0.5 change the
    expected outcome? We generate data with a known linear relationship,
    inspect explicit priors, fit with Bambi, and report the response-scale
    difference with uncertainty. The shared Bayesian workflow supplies the
    diagnostic verdict; a completed fit is not itself evidence of reliability.

    This example targets Bambi 0.21 and PyMC 6. Run normally for 1,000 tuning
    steps and 1,000 retained draws per chain, letting the sampler choose the
    chain count. `BAYGENT_SMOKE=1` uses two chains with 150 tuning steps and
    150 retained draws to check execution; inspect its diagnostics before
    drawing any conclusions. The final slider changes predictions only.
    """)
    return


@app.cell
def _(np, os, pd, result_directory):
    RANDOM_SEED = sum(map(ord, "gaussian-regression"))
    smoke = os.environ.get("BAYGENT_SMOKE") == "1"
    sampling = (
        {"draws": 150, "tune": 150, "chains": 2}
        if smoke
        else {"draws": 1000, "tune": 1000}
    )
    output_dir = result_directory("gaussian-regression")
    _rng = np.random.default_rng(RANDOM_SEED)
    _x = _rng.uniform(-1.5, 1.5, 120)
    data = pd.DataFrame({"x": _x, "y": 10 + 2 * _x + _rng.normal(0, 1.5, 120)})
    assert data.notna().all().all() and np.isfinite(data.to_numpy()).all()
    return RANDOM_SEED, data, output_dir, sampling, smoke


@app.cell
def _(data, mo):
    mo.md(f"""
    ## Specify the data-generating story and priors

    Each row is an independent measurement. Conditional on exposure `x`,
    `y` is Normal with mean `Intercept + x_coefficient × x` and a common
    residual standard deviation. Exposure is already centered in meaningful
    units; the intercept describes exposure zero. There are no categorical
    predictors or reference levels to infer. All {len(data)} rows are finite.

    Before seeing these synthetic outcomes, we allow a baseline around 10
    with standard deviation 5, a slope around zero with standard deviation
    3, and a positive residual scale with HalfNormal(3). These are explicit
    domain assumptions rather than a judgment based on coefficient scale
    alone. A homoscedastic Gaussian likelihood is deliberate here.

    Our explicitly scaled design defines the intercept prior. For the
    Bambi 0.21 log-prior reporting path, we set `center_predictors=False`
    so the saved intercept and the log-prior graph use the same scale.
    We also retain any offsets with `omit_offsets=False` when fitting,
    allowing the full prior density to be evaluated from the saved draws.
    These settings address this release's reporting behavior; they are
    not universal statistical requirements.
    """)
    return


@app.cell
def _(bmb, data, mo):
    model = bmb.Model(
        "y ~ x",
        data,
        family="gaussian",
        link="identity",
        center_predictors=False,
        priors={
            "Intercept": bmb.Prior("Normal", mu=10, sigma=5),
            "x": bmb.Prior("Normal", mu=0, sigma=3),
            "sigma": bmb.Prior("HalfNormal", sigma=3),
        },
    )
    model.build()
    mo.md(f"```text\n{model}\n```")
    return (model,)


@app.cell
def _(RANDOM_SEED, data, mo, model, np, plt, smoke):
    prior = model.prior_predictive(draws=150 if smoke else 500, random_seed=RANDOM_SEED)
    _draws = prior["prior_predictive"]["y"].to_numpy().ravel()
    _low, _high = np.quantile(_draws, [0.03, 0.97])
    prior_assessment = (
        f"The pooled 94% prior-predictive interval is [{_low:.2f}, {_high:.2f}]. "
        "This broad range covers the intended measurement scale; tails outside "
        "a plausible domain would require a prior or likelihood revision."
    )
    prior_review = {"finite": bool(np.isfinite(_draws).all())}
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(_draws, bins=60, density=True, alpha=0.5, label="Prior predictive")
    _ax.hist(data["y"], bins=20, density=True, histtype="step", label="Observed")
    _ax.set(
        xlabel="Outcome y",
        ylabel="Density",
        title="Inspect plausibility before fitting",
    )
    _ax.legend()
    mo.vstack([mo.md("## Prior predictive check"), _fig, mo.md(prior_assessment)])
    return prior, prior_assessment, prior_review


@app.cell
def _(RANDOM_SEED, model, output_dir, prior_review, sampling):
    if not prior_review["finite"]:
        raise ValueError(
            "The prior predictive contains non-finite outcomes; revise the model."
        )
    idata = model.fit(
        inference_method="nutpie",
        omit_offsets=False,
        random_seed=RANDOM_SEED,
        progressbar=False,
        **sampling,
    )
    idata.to_netcdf(output_dir / "posterior.nc")
    return (idata,)


@app.cell
def _(RANDOM_SEED, data, idata, mo, model, np, plt):
    posterior = idata.copy(deep=True)
    model.predict(posterior, kind="response", random_seed=RANDOM_SEED + 1)
    model.compute_log_likelihood(posterior)
    model.compute_log_prior(posterior)
    _pred = posterior["posterior_predictive"]["y"].to_numpy()
    _mean_interval = np.quantile(_pred.mean(axis=-1), [0.03, 0.97])
    ppc_assessment = (
        f"Observed mean {data.y.mean():.2f}; 94% interval for replicated means "
        f"[{_mean_interval[0]:.2f}, {_mean_interval[1]:.2f}]. "
        "Agreement of the pooled distribution is only one check: inspect residual "
        "shape and changes in spread across exposure before trusting constant variance."
    )
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(
        _pred.ravel(), bins=40, density=True, alpha=0.5, label="Posterior predictive"
    )
    _ax.hist(data["y"], bins=20, density=True, histtype="step", label="Observed")
    _ax.set(xlabel="Outcome y", ylabel="Density")
    _ax.legend()
    mo.vstack([mo.md("## Posterior predictive check"), _fig, mo.md(ppc_assessment)])
    return posterior, ppc_assessment


@app.cell
def _(bmb, mo, model, posterior):
    result = bmb.interpret.comparisons(
        model,
        posterior,
        contrast={"x": [-0.5, 0.5]},
        average_by="all",
        prob=0.94,
        use_hdi=True,
    )
    mo.vstack(
        [
            mo.md(
                "## Response-scale answer\nThe contrast is the expected outcome at +0.5 minus −0.5. In this linear identity-link model the contrast is identical for every row; averaging collapses those duplicate summaries. This simplification does not transfer to nonlinear contrasts."
            ),
            result.summary,
        ]
    )
    return (result,)


@app.cell
def _(
    RANDOM_SEED,
    mo,
    model,
    output_dir,
    posterior,
    ppc_assessment,
    prior,
    prior_assessment,
    result,
    sampling,
    smoke,
    write_report,
):
    report = write_report(
        model,
        prior,
        posterior,
        result,
        output_dir,
        outcome="y",
        var_names=["Intercept", "x", "sigma"],
        details={
            "title": "Gaussian regression",
            "formula": "y ~ x",
            "family": "gaussian",
            "link": "identity",
            "seed": RANDOM_SEED,
            "smoke": smoke,
            "sampling": sampling,
            "question": "How does expected y change when x increases from -0.5 to +0.5?",
            "generative_story": "Independent Normal outcomes with a linear exposure mean and common residual scale.",
            "prior_rationale": "Baseline Normal(10,5), slope Normal(0,3), and residual HalfNormal(3) express the measurement scale.",
            "encoding": "Numeric centered exposure; no categorical predictors; rows checked finite.",
            "prediction_target": "Expected outcome difference: x=+0.5 minus x=-0.5, with a 94% HDI.",
            "interpretation": "The contrast is a conditional association on the outcome scale, not a causal effect.",
            "prior_assessment": prior_assessment,
            "ppc_assessment": ppc_assessment,
            "limitations": "Synthetic linear data cannot validate a real-data likelihood; pooled PPCs do not establish constant variance.",
        },
    )
    mo.vstack(
        [
            mo.md(
                "## Shared diagnostic report\nRead the checks and their next steps before using the estimate."
            ),
            mo.md("\n\n".join(report["checks"]["summary"].values())),
            mo.md(
                "### Next steps\n"
                + "\n".join(f"- {step}" for step in report["checks"]["next_steps"])
            ),
            mo.md(
                "The complete report, figures, summaries and saved fit are in this notebook's results folder."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    prediction_extent = mo.ui.slider(
        0.5, 2.5, step=0.1, value=1.5, label="Prediction range: ± exposure"
    )
    mo.vstack(
        [
            mo.md(
                "## Explore the fitted mean\nChanging this range only recomputes predictions. Values beyond ±1.5 extrapolate beyond the design."
            ),
            prediction_extent,
        ]
    )
    return (prediction_extent,)


@app.cell
def _(bmb, data, model, np, plt, posterior, prediction_extent):
    _grid = np.linspace(-prediction_extent.value, prediction_extent.value, 50)
    _prediction = bmb.interpret.predictions(
        model,
        posterior.copy(deep=True),
        conditional={"x": _grid},
        prob=0.94,
        use_hdi=True,
    )
    _draws = _prediction.draws["posterior"][model.family.likelihood.parent]
    _mean = _draws.mean(dim=("chain", "draw")).to_numpy().ravel()
    _interval = (
        _draws.quantile([0.03, 0.97], dim=("chain", "draw")).to_numpy().reshape(2, -1)
    )
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.scatter(data.x, data.y, s=15, alpha=0.45, label="Observed")
    _ax.plot(_grid, _mean, label="Expected outcome")
    _ax.fill_between(_grid, *_interval, alpha=0.2, label="94% equal-tailed interval")
    _ax.set(xlabel="Standardized exposure x", ylabel="Outcome y")
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
