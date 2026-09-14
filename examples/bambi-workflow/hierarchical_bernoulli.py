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
    # Hierarchical Bernoulli regression: probabilities in known regions

    What is support probability for a 40-year-old urban respondent with
    medium income in a known region? We fit a varying-intercept logistic
    survey model and report probabilities with uncertainty. Group effects partially pool;
    they do not make a conditional prediction into a population-wide claim.

    This example targets Bambi 0.21 and PyMC 6. The full run uses 1,000 tuning
    steps and 1,000 retained draws per chain and lets the sampler choose its
    chain count. `BAYGENT_SMOKE=1` uses two chains with 150 tuning steps and
    150 retained draws as an execution check. Diagnostic results, rather
    than successful execution, determine whether either fit supports use.
    The final controls change prediction inputs without refitting.
    """)
    return


@app.cell
def _(np, os, pd, result_directory):
    RANDOM_SEED = sum(map(ord, "hierarchical-bernoulli"))
    smoke = os.environ.get("BAYGENT_SMOKE") == "1"
    sampling = (
        {"draws": 150, "tune": 150, "chains": 2}
        if smoke
        else {"draws": 1000, "tune": 1000}
    )
    output_dir = result_directory("hierarchical-bernoulli")
    regions = ["north", "south", "east", "west", "central", "coast", "valley", "hill"]
    _rng = np.random.default_rng(RANDOM_SEED)
    _region_id = np.repeat(np.arange(len(regions)), [25, 35, 45, 55, 60, 65, 70, 85])
    _age = _rng.uniform(20, 70, len(_region_id))
    _income = _rng.choice(["low", "medium", "high"], len(_region_id), p=[0.4, 0.4, 0.2])
    _urban = _rng.binomial(1, 0.6, len(_region_id))
    _offsets = _rng.normal(0, 0.55, len(regions))
    _log_odds = (
        -0.4
        + 0.2 * ((_age - 40) / 10)
        + 0.35 * (_income == "medium")
        + 0.6 * (_income == "high")
        + 0.45 * _urban
        + _offsets[_region_id]
    )
    _probability = 1 / (1 + np.exp(-_log_odds))
    data = pd.DataFrame(
        {
            "age": _age,
            "age_scaled": (_age - 40) / 10,
            "income_bracket": pd.Categorical(
                _income, categories=["low", "medium", "high"], ordered=True
            ),
            "urban": _urban,
            "region": pd.Categorical(
                np.asarray(regions)[_region_id], categories=regions
            ),
            "support": _rng.binomial(1, _probability),
        }
    )
    assert data.notna().all().all() and set(data.support.unique()) == {0, 1}
    assert set(data.urban.unique()) == {0, 1}
    assert list(data.region.cat.categories) == regions
    return RANDOM_SEED, data, output_dir, regions, sampling, smoke


@app.cell
def _(data, mo):
    mo.md(f"""
    ## Verify encoding and choose a hierarchical model

    The {len(data)} observations have `support=1` for support and `0` otherwise.
    `urban` is a 0/1 indicator. Age is centered at 40 and divided by 10, so
    its coefficient describes one decade. The income categories are ordered
    `low`, `medium`, `high`: `ordered=True` preserves this explicit order for
    Formulae, which creates treatment indicators with **low as reference**.
    This does not turn income into a numeric score; verify the design below.

    Eight regions have unequal sample sizes. Each receives a varying
    intercept, with no omitted reference region. The intercept describes
    age 40, low income, rural (`urban=0`) at the population location.

    Conditional on region intercept, age, income, and urban status, outcomes are independent
    Bernoulli draws. We use a logit link, a Normal(0,1) common intercept,
    Normal(0,0.5) common coefficients, and zero-centered Normal region
    deviations with HalfNormal(0.6) scale. These priors live on the log-odds
    scale. The common intercept owns the population location and group
    deviations are non-centered. Check prior predictions on probabilities
    and binary outcomes rather than comparing slope units to outcome SD.

    This explicitly scaled design defines our intercept prior. For Bambi
    0.21's log-prior reporting path, `center_predictors=False` keeps the
    saved coefficients on the same scale as the log-prior graph. Retaining
    the sampled group offsets with `omit_offsets=False` permits evaluating
    the full prior density from the saved draws. These are release-specific
    reporting settings, not a general rule against predictor centering.
    """)
    return


@app.cell
def _(bmb, data, mo, np):
    model = bmb.Model(
        "support ~ age_scaled + income_bracket + urban + (1|region)",
        data,
        family="bernoulli",
        link="logit",
        center_predictors=False,
        noncentered=True,
        priors={
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "age_scaled": bmb.Prior("Normal", mu=0, sigma=0.5),
            "income_bracket": bmb.Prior("Normal", mu=0, sigma=0.5),
            "urban": bmb.Prior("Normal", mu=0, sigma=0.5),
            "1|region": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.6)
            ),
        },
    )
    assert model.response_term.success == 1
    _income_term = model.parameters[model.family.likelihood.parent].common_terms[
        "income_bracket"
    ]
    assert _income_term.levels == ["medium", "high"]
    for _level, _expected in {"low": [0, 0], "medium": [1, 0], "high": [0, 1]}.items():
        assert np.all(_income_term.data[data.income_bracket == _level] == _expected)
    model.build()
    mo.md(f"```text\n{model}\n```")
    return (model,)


@app.cell
def _(RANDOM_SEED, data, mo, model, np, plt, smoke):
    prior = model.prior_predictive(draws=150 if smoke else 500, random_seed=RANDOM_SEED)
    _replicates = prior["prior_predictive"]["support"].to_numpy()
    _rates = _replicates.mean(axis=-1).ravel()
    _low, _high = np.quantile(_rates, [0.03, 0.97])
    prior_assessment = (
        f"The pooled 94% interval for prior-predictive support rates is [{_low:.2f}, {_high:.2f}]. "
        "The prior permits substantial uncertainty without assuming all regions are identical. "
        "This pooled check must be complemented by domain judgments about regional variation."
    )
    prior_review = {"binary": bool(np.isin(_replicates, [0, 1]).all())}
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(_rates, bins=25, alpha=0.6, density=True, label="Replicated support rates")
    _ax.axvline(data.support.mean(), color="black", label="Observed support rate")
    _ax.set(xlabel="Support proportion", ylabel="Density", xlim=(0, 1))
    _ax.legend()
    mo.vstack([mo.md("## Prior predictive check"), _fig, mo.md(prior_assessment)])
    return prior, prior_assessment, prior_review


@app.cell
def _(RANDOM_SEED, model, output_dir, prior_review, sampling):
    if not prior_review["binary"]:
        raise ValueError(
            "Prior predictions do not respect the binary outcome encoding."
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
    _pred = posterior["posterior_predictive"]["support"].to_numpy()
    _interval = np.quantile(_pred.mean(axis=-1), [0.03, 0.97])
    _by_region = (
        data.assign(predicted=_pred.mean(axis=(0, 1)))
        .groupby("region", observed=True)[["support", "predicted"]]
        .mean()
    )
    ppc_assessment = (
        f"Observed support proportion {data.support.mean():.2f}; replicated pooled 94% interval "
        f"[{_interval[0]:.2f}, {_interval[1]:.2f}]. Compare regional observed and predicted "
        "proportions as well: pooled agreement can hide a region-specific failure. "
        "A binary histogram alone is not a calibration check."
    )
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_by_region.index.astype(str), _by_region.support, "o", label="Observed")
    _ax.plot(
        _by_region.index.astype(str),
        _by_region.predicted,
        "x",
        label="Posterior predictive mean",
    )
    _ax.set(ylabel="Support proportion", ylim=(0, 1))
    _ax.legend()
    mo.vstack(
        [mo.md("## Posterior predictive check by region"), _fig, mo.md(ppc_assessment)]
    )
    return posterior, ppc_assessment


@app.cell
def _(bmb, mo, model, posterior, regions):
    assert set(regions).issubset(set(model.data.region.cat.categories))
    result = bmb.interpret.predictions(
        model,
        posterior,
        conditional={
            "age_scaled": [0.0],
            "income_bracket": ["medium"],
            "urban": [1],
            "region": ["north"],
        },
        prob=0.94,
        use_hdi=True,
    )
    mo.vstack(
        [
            mo.md(
                "## A specified known-region probability\nExpected support probability at age 40, medium income, urban=1, in the observed north region, with a 94% HDI. This is conditional on its fitted region effect."
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
        outcome="support",
        var_names=[
            "Intercept",
            "age_scaled",
            "income_bracket",
            "urban",
            "1|region_sigma",
        ],
        details={
            "title": "Hierarchical Bernoulli regression",
            "formula": "support ~ age_scaled + income_bracket + urban + (1|region)",
            "family": "bernoulli",
            "link": "logit",
            "seed": RANDOM_SEED,
            "smoke": smoke,
            "sampling": sampling,
            "question": "What is support probability at age 40, medium income, urban=1 in the known north region?",
            "generative_story": "Conditionally independent Bernoulli outcomes with common age, income and urban effects and partially pooled region intercepts.",
            "prior_rationale": "Log-odds intercept Normal(0,1), common slopes Normal(0,0.5), zero-mean Normal group deviations with HalfNormal(0.6) scale.",
            "encoding": "Support and urban are exactly 0/1; age_scaled=(age-40)/10. Ordered low/medium/high income treatment design is checked: low=[0,0], medium=[1,0], high=[0,1]. Eight regions have varying intercepts.",
            "prediction_target": "Conditional support probability at age 40, medium income, urban=1 in known north, with a 94% HDI; new-group sampling is not requested.",
            "interpretation": "Region-specific probabilities describe modeled associations; they do not marginalize over unseen regions or identify a causal effect.",
            "prior_assessment": prior_assessment,
            "ppc_assessment": ppc_assessment,
            "limitations": "Synthetic unequal region samples; no survey weighting or poststratification. This known-region conditional probability is not an overall population estimate or a causal effect.",
        },
    )
    mo.vstack(
        [
            mo.md(
                "## Shared diagnostic report\nFollow diagnostic and calibration next steps before using these probabilities."
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
def _(mo, regions):
    selected_region = mo.ui.dropdown(regions, value="north", label="Known region")
    prediction_extent = mo.ui.slider(45, 80, step=5, value=70, label="Maximum age")
    mo.vstack(
        [
            mo.md(
                "## Explore a known region\nThe controls recompute probabilities for medium-income urban respondents only. Age above 70 extrapolates outside the design."
            ),
            mo.hstack([selected_region, prediction_extent]),
        ]
    )
    return prediction_extent, selected_region


@app.cell
def _(bmb, model, np, plt, posterior, prediction_extent, selected_region):
    _grid = np.linspace(20, prediction_extent.value, 50)
    _prediction = bmb.interpret.predictions(
        model,
        posterior.copy(deep=True),
        conditional={
            "age_scaled": (_grid - 40) / 10,
            "income_bracket": ["medium"],
            "urban": [1],
            "region": [selected_region.value],
        },
        prob=0.94,
        use_hdi=True,
    )
    _draws = _prediction.draws["posterior"][model.family.likelihood.parent]
    _mean = _draws.mean(dim=("chain", "draw")).to_numpy().ravel()
    _interval = (
        _draws.quantile([0.03, 0.97], dim=("chain", "draw")).to_numpy().reshape(2, -1)
    )
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_grid, _mean, label=f"Expected probability: {selected_region.value}")
    _ax.fill_between(_grid, *_interval, alpha=0.2, label="94% equal-tailed interval")
    _ax.set(xlabel="Age (years)", ylabel="Pr(support=1)", ylim=(0, 1))
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
