# Priors in Bambi

## Inspect resolved priors

Bambi assembles and scales default priors during model construction. They are
available before `model.build()`, which builds the PyMC graph. Scaling depends
on the family, link and predictor scales; it is not a universal multiple of
the raw response standard deviation.

Start with `print(model)`. For structured inspection, include the intercept,
common terms, group terms and constant likelihood parameters:

```python
print(model)
for parameter, component in model.conditional_parameters.items():
    for name, term in component.terms.items():
        print(parameter, name, term.prior)
for parameter, component in model.marginal_parameters.items():
    print(parameter, component.prior)
```

These properties are the 0.21 API. The older `model.components` name is
deprecated. Preserve the table or model description with the analysis, along
with why each default was retained or overridden. The Bambi-specific behavior
in this reference is not a prior policy for other packages that reuse formulas.

## Override on the relevant scale

Use `bmb.Prior` values keyed by actual model terms. An explicit prior overrides
the auto-scaled default. Inspect the resulting model so misspelled keys or
unintended defaults are visible. Term-specific entries are often easier to
audit than broad `common`/`group_specific` aliases.

For example, with the survey's age measured in years and centered at 40:

```python
priors = {
    "age_centered": bmb.Prior("Normal", mu=0, sigma=0.05),
    "1|region": bmb.Prior(
        "Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)
    ),
}
model = bmb.Model(
    "support ~ age_centered + income_bracket + urban + (1|region)",
    df,
    family="bernoulli",
    link="logit",
    center_predictors=False,
    priors=priors,
)
```

The age prior expresses modest log-odds changes per year; the nested prior
governs region heterogeneity in log odds. These are illustrative choices,
not recommended defaults for all surveys. Justify them against domain knowledge
and inspect their induced probability-scale predictions. Standardizing age
would change the appropriate coefficient scale.

Bambi defaults to non-centered group-specific terms. Record that choice and
inspect difficult posterior geometry with the Bayesian workflow; do not add a
second manual reparameterization merely because the model is hierarchical.

## Prior predictive checks

Build before drawing prior predictions. Use the native Bambi API and the
modern ArviZ plotting surface:

```python
from pathlib import Path
import arviz_plots as azp

RANDOM_SEED = sum(map(ord, "survey-support"))
results_dir = Path("survey-support")
results_dir.mkdir(parents=True, exist_ok=True)
model.build()
prior = model.prior_predictive(draws=500, random_seed=RANDOM_SEED)
prior.to_netcdf(results_dir / "prior_data.nc")
pc = azp.plot_ppc_dist(prior, group="prior_predictive")
pc.savefig(str(results_dir / "prior_predictive.png"))
```

For a Gaussian response, assess plausible locations, spreads and extremes in
the original units. For Bernoulli data, individual replicated values are always
0/1, so also examine variation in replicated support rates across observations
and groups, and how often predictions concentrate near all-zero/all-one data.
State the scientific implications; a saved plot alone is not a pass criterion.

Revise priors if implausible patterns conflict with the scientific assumptions,
then rebuild and repeat. Avoid “suspect prior” ratings from generic comparisons
of coefficient scales to response SD: those mix predictor units and link scales.

## Prior sensitivity after fitting

**Bambi 0.21 has two pitfalls in this handoff.** Fitting with its default
`center_predictors=True` writes an uncentered `Intercept` into the posterior;
`compute_log_prior` evaluates those saved values directly without restoring
the prior's centered coordinates. Its default `omit_offsets=True` also removes
non-centered group offset RVs, and `compute_log_prior` skips those missing
variables rather than reconstructing their densities. Finite output is not
evidence that the complete intended joint prior was evaluated.

For this workflow's explicitly scaled predictors, use
`center_predictors=False` when constructing the model and `omit_offsets=False`
when calling `fit`. Then compute Bambi's `model.compute_log_likelihood(idata)`
and `model.compute_log_prior(idata)`. Check completeness:

```python
free_names = {rv.name for rv in model.backend.model.free_RVs}
assert free_names <= set(idata["posterior"].data_vars)
assert free_names == set(idata["log_prior"].data_vars)
```

Also verify representative densities against the declared priors, particularly
the intercept and standard-Normal offsets of non-centered groups. For example,
a retained offset `u` has log density `-0.5 * log(2*pi) - 0.5*u**2`; applying
the group-scale prior to `u` would be a different parameterization.

This is a release-scoped workaround, not a general prohibition on predictor
centering or omitted plotting variables. Preserve an existing fit; if it used
different settings, establish a valid coordinate correction or refit before
reporting its sensitivity. Do not toggle model settings afterward and assume
the stored draws changed with them.

With that density contract verified, use the installed Bayesian workflow's
`references/sensitivity.md` and choose interpretable quantities for the check.

Record failures, instability or unavailable groups. A warning can identify a
deliberately informative prior rather than an automatic modeling error; explain
the prior's justification and how conclusions change under reasonable choices.

Source: Bambi 0.21.0
[prior assembly and log-prior bridge](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/models.py)
[component priors](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/model_components.py),
and [posterior intercept adjustment](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/backend/pymc.py).
