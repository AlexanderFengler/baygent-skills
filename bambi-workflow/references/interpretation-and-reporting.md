# Interpretation and Reporting

## Fit, save and prepare the diagnostic artifact

Following the prior checks, fit with a supported Bambi backend and save before
post-processing. The snippets below continue with the built `model` using
`center_predictors=False`, `results_dir`, and descriptive `RANDOM_SEED` from
the prior reference. Retain offsets so the prior artifact can include the full
joint density of hierarchical models.

```python
idata = model.fit(
    inference_method="nutpie", omit_offsets=False, random_seed=RANDOM_SEED
)
idata.to_netcdf(results_dir / "inference_data.nc")

# All three operate on the original data and mutate idata by default.
model.predict(idata, kind="response", random_seed=RANDOM_SEED)
model.compute_log_likelihood(idata)
model.compute_log_prior(idata)
print(idata)
idata.to_netcdf(results_dir / "inference_data.nc")
```

Use `inference_method="pymc"` if nutpie is unavailable and record the choice.
Keep normal chain selection unless a resource constraint or controlled smoke
test justifies explicit chains. Small draws/tuning budgets test the API path;
they do not demonstrate adequate inference.

Check that `posterior`, `sample_stats`, `observed_data`, `posterior_predictive`,
`log_likelihood`, and `log_prior` have the expected contents. Verify the outcome
name and corresponding observation coordinates before running the shared
utilities. Bernoulli probabilities are stored as parent parameter `p` in
`posterior`; the observed/predictive variable retains the response name.

## Define the prediction target

Specify the outcome quantity, predictor values, group, and averaging target.
The initial workflow predicts for an observed group at stated covariates.
Averaging over observed groups, omitting group effects, and integrating over
a new group's effect answer different questions.

Use `kind="response_params"` for parameter draws (`mu` or `p`);
use `kind="response"` for replicated outcomes. For a new grid, use
`inplace=False` to preserve the original-data diagnostic artifact:

```python
grid = pd.DataFrame({
    "age_centered": [0.0],  # age 40 under the recorded transformation
    "income_bracket": pd.Categorical(
        ["medium"], categories=df["income_bracket"].cat.categories,
        ordered=df["income_bracket"].cat.ordered,
    ),
    "urban": [1],
    "region": pd.Categorical(
        [observed_region], categories=df["region"].cat.categories,
        ordered=df["region"].cat.ordered,
    ),
})
assert grid.notna().all().all()
predicted = model.predict(idata, kind="response_params", data=grid, inplace=False)
probability_draws = predicted["posterior"]["p"]
posterior_mean = probability_draws.mean(dim=["chain", "draw"])
```

Here `pd` is pandas and `observed_region` is a specified label present in the
training data. Check it explicitly. The mean over posterior probability draws
answers the predictive-probability question; their median is not that mean.

## Native interpretation results

In Bambi 0.21, `predictions` and `comparisons` return a `Result` named tuple with
`.summary` (a DataFrame) and `.draws` (associated DataTree draws). Set interval
width explicitly: ArviZ's default need not be a 94% HDI.

```python
prediction = bmb.interpret.predictions(
    model,
    idata,
    conditional={
        "age_centered": [0.0],
        "income_bracket": ["medium"],
        "urban": [1],
        "region": [observed_region],
    },
    target="mean",
    prob=0.94,
    use_hdi=True,
)
prediction.summary.to_csv(results_dir / "predictions.csv", index=False)
```

For Bernoulli, `target="mean"` selects `p`; for Gaussian it selects `mu`.
Report the summary's posterior mean and HDI with the full conditioning target.
For a fitted Gaussian `y ~ x + z` model, a native outcome-scale comparison
between `x=0` and `x=1` at `z=0` is:

```python
contrast = bmb.interpret.comparisons(
    gaussian_model,
    gaussian_idata,
    contrast={"x": [0.0, 1.0]},
    conditional={"z": [0.0]},
    comparison="diff",
    target="mean",
    prob=0.94,
    use_hdi=True,
)
print(contrast.summary)
```

Inspect the comparison label and direction before describing the effect. Do
not pass the retired draft's `value=` argument. State whether the interval
describes an expected response or a future outcome; coefficient summaries can
complement these substantive answers. Without `conditional`, comparisons use
the training rows with substituted contrast values; repeated rows are not
additional independent evidence. A nonempty grid makes the target explicit.

Population-average intervals need separate care. This release's `average_by`
aggregates row summaries: averaging interval endpoints is not generally the
HDI of a posterior average. Define weights and average predictions within each
posterior draw before computing an interval for that estimand. Such averaging
and unseen-group workflows are outside this initial known-group example.

## Shared diagnostics and calibration

Resolve `BAYESIAN_SKILL_DIR` as described in [../SKILL.md](../SKILL.md).
Use its `diagnose_model.py`, `calibration_check.py` and `check_diagnostics.py`
in that order on the saved original-data artifact. Pass the actual outcome
name to calibration. Ordinary PPC-PIT is the default; `--loo-pit` requests
LOO-PIT when the aligned log-likelihood group is available. Consult the installed
`references/model-criticism.md` for interpretation.

Before prior sensitivity, verify the complete-density and intercept-coordinate
contract in [priors-in-bambi.md](priors-in-bambi.md#prior-sensitivity-after-fitting).
Then consult the installed Bayesian `references/sensitivity.md` and run
`arviz_stats.psense_summary` on selected parameters after Bambi computes the
log-density groups. Save the table, and orient the JSON by parameter for the
shared interpreter:

```python
from arviz_stats import psense_summary

sensitivity = psense_summary(idata, var_names=["age_centered", "urban"])
sensitivity.to_csv(results_dir / "psense.csv")
sensitivity.to_json(results_dir / "psense.json", orient="index")
```

Adjust `var_names` to the fitted model. Supply this file to shared
`check_diagnostics.py --psense ...`; the JSON must map parameter names to
`prior`/`likelihood` values. A default column-oriented JSON silently changes
that meaning. Inspect warnings and returned assessments; a filename alone does
not establish a successful sensitivity check.

## Canonical report

Read `BAYESIAN_SKILL_DIR/references/reporting.md` and fill its canonical
`<slug>/report.md`. Keep the fixed explanatory text and standard figure names.
Use `check_report.json` for the relevant Assessment lines and Suggested Next
Steps, adding problem-specific context without replacing missing results with
healthy ratings.

Add the following within the template's existing sections:

- **Data and Question:** response event, category reference, analysis rows and
  exclusions, predictor transformations, grouping and sample sizes.
- **Model Specification:** Bambi formula, family/link, resolved priors with
  justification, and recorded group parameterization.
- **Prior Predictive Check:** the actual plausibility judgment and any revisions.
- **Posterior:** mean-response prediction or contrast, its complete grid/group
  definition, scale, posterior mean and uncertainty interval.
- **Limitations and Threats:** failed or unavailable checks, sparse groups,
  separation, predictive mismatch, and restrictions on generalization.
- **Appendix:** seed, exact package versions, sampler/budget, saved artifact and
  code locations, and prediction/contrast tables.

Save `summary.csv`, the sampled artifact and existing Bayesian diagnostic
outputs under the same results folder. If a required plot cannot be generated,
state the reason where its evidence belongs and remove the broken image link.
Keep successful output even when a later stage fails, and label partial runs
explicitly.

Sources: Bambi 0.21.0
[prediction and log-density APIs](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/models.py),
[interpretation API](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/interpret/effects.py),
and [summary aggregation](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/interpret/utils.py).
