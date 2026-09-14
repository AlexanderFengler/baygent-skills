# Predictions and Reporting

## Execution boundary

The commands below are analysis recipes, not evidence of a completed run.
Honor the user's requested execution scope. If sampling or simulation is
deferred, leave the corresponding steps pending and provide code/input
requirements without generating synthetic observations, predictive draws or
agent benchmark results.

HSSM plotting can generate predictions when the supplied DataTree lacks them.
Check the predictive group before plotting; otherwise a nominal plotting call
can start computation that was meant to remain deferred.

## Prior predictions

With the validated flat `model` from the prior reference, generate prior draws
only when authorized by the analysis task:

```python
from pathlib import Path
import numpy as np

RANDOM_SEED = sum(map(ord, "analytical-ddm"))
results_dir = Path("analytical-ddm")
results_dir.mkdir(parents=True, exist_ok=True)
prior = model.sample_prior_predictive(
    draws=500, random_seed=np.random.default_rng(RANDOM_SEED)
)
prior.to_netcdf(results_dir / "prior_data.nc")
assert "prior_predictive" in prior.children
model.plot_predictive(dt=prior, predictive_group="prior_predictive", n_samples=None)
```

Save the returned plot using its actual Matplotlib/Seaborn surface. For this
unfaceted call the return is an Axes, whose figure can be saved with
`.figure.savefig(...)`. With facets, inspect the return type rather than
assuming the same method. Name the prior evidence `prior_predictive.png` for
the shared report template.

Judge plausible RT ranges, mass in each choice, and extreme or highly
asymmetric predictions before fitting. The signed-RT display in HSSM's
two-choice plot is a visualization of separate choices; it does not change
the positive-RT input convention.

## Fit and save

```python
idata = model.sample(sampler="pymc", random_seed=RANDOM_SEED)
idata.to_netcdf(results_dir / "posterior.nc")
```

HSSM's analytical path supports this sampler selection. Record the actual
package versions, seed, draws/tuning and chain configuration. Choose budgets
for the analysis; a small execution smoke run would not establish adequate
posterior inference. Do not inherit a blanket “always nutpie” requirement from
a generic raw-PyMC example without verifying HSSM's applicable backend.

`sample` normally computes log likelihood; inspect the returned group rather
than assuming that every sampling configuration does. To request it through
the owning API, use `model.log_likelihood(dt=idata, inplace=True)`. Establish
that it contains one joint choice/RT log density per original trial, with
matching coordinates, before using it for LOO. Do not treat RT and response as
two independent likelihood factors merely because they occupy two columns.

On the tested HSSM 0.5 / PyMC 6.1 stack, native density recomputation requires
default zero-based consecutive `chain` and `draw` coordinates. HSSM's
`log_likelihood` treats those labels as positional indices; nondefault labels
can raise `IndexError`. PyMC's `compute_log_prior` can reset nondefault sample
labels to `0..N`. Verify the original labels before recomputation and compare
the density coordinates afterward. Do not silently relabel or align an artifact
to hide a mismatch. The scalar PPC adapter can preserve arbitrary unique labels;
this restriction belongs to the release's native density APIs.

## Posterior predictive evidence

```python
predictive = model.sample_posterior_predictive(
    dt=idata, inplace=False, draws=None
)
assert "posterior_predictive" in predictive.children
predictive.to_netcdf(results_dir / "inference_data.nc")
model.plot_predictive(dt=predictive, n_samples=None)
```

For HSSM 0.5.0, `sample_posterior_predictive` has **no `random_seed` argument**.
Do not invent that keyword or claim deterministic PPC reproduction from the
MCMC seed alone. Preserve the generated artifact and record any supported
RNG controls actually used. The prior-predictive API does accept a seed.

Check that predictions refer to the original rows and that `rt` and `response`
are still distinct components of each trial. `inplace=False` preserves the
saved fit while producing a separate predictive artifact. Keep this combined
artifact as the source for domain plots even if an adapter produces additional
scalar views for shared diagnostics.

Assess both:

- Choice proportions: observed rates versus replicated rates for each boundary.
- Conditional RT distributions: central quantiles and tails for each response,
  with sparse or absent choices explicitly flagged.

Pooled RT agreement can hide choice-specific failures. Conversely, getting the
choice rate right does not establish the latency distribution. Interpret
posterior parameters jointly and report their uncertainty; a point estimate
or a converged chain alone cannot establish cognitive validity.

`model.plot_quantile_probability(cond=...)` requires a real condition column.
For a flat single-condition analysis, an explicitly labeled common plotting
condition may be supplied in the plotting data; it is not a modeled covariate.
Document its meaning and retain the original row alignment. Do not invent a
condition effect solely to obtain that plot.

HSSM 0.5 already computes these quantiles separately by response sign.
`quantile_by` adds other groups whose quantiles are averaged; passing
`quantile_by="response"` duplicates a mandatory column and fails. The same
release forwards an explicit `ax` twice to seaborn in an unfaceted QP call.
Create the desired current axes and omit that keyword, then save the returned
Axes' `.figure`. Native plotting joins condition labels by zero-based observation
position; verify row alignment before plotting an externally relabeled artifact.

## Adaptation to shared diagnostics

Resolve the installed directories as described in [../SKILL.md](../SKILL.md).
Set `RESULTS` to the analysis directory containing the saved combined
`inference_data.nc`. The installed
[prepare_rt_choice.py](../scripts/prepare_rt_choice.py) accepts the released
`rt,response` variable, `__obs__` trial coordinates and an explicitly labeled
two-component axis `[0, 1]` meaning `[RT, response]`. It rejects unsupported
dimensions, invalid RT/choice values and mismatched observed/predicted rows.

```bash
python "$BAYESIAN_SKILL_DIR/scripts/diagnose_model.py" \
  --idata "$RESULTS/inference_data.nc" --output "$RESULTS/diagnostics.json"
python "$HSSM_SKILL_DIR/scripts/prepare_rt_choice.py" \
  --idata "$RESULTS/inference_data.nc" --output-dir "$RESULTS/calibration"

for quantity in rt choice; do
  python "$BAYESIAN_SKILL_DIR/scripts/calibration_check.py" \
    --idata "$RESULTS/calibration/$quantity/inference_data.nc" \
    --var-name "$quantity" --save-plots \
    --plot-dir "$RESULTS/calibration/$quantity" \
    --output "$RESULTS/calibration/$quantity/calibration.json"
  python "$BAYESIAN_SKILL_DIR/scripts/check_diagnostics.py" \
    --diagnostics "$RESULTS/diagnostics.json" \
    --calibration "$RESULTS/calibration/$quantity/calibration.json" \
    --output "$RESULTS/calibration/$quantity/assessment.json"
done
```

The adapter preserves the original paired artifact. Its two scalar DataTrees
contain only observed and posterior-predictive groups: `calibration/rt/` holds
RT in seconds and `calibration/choice/` holds the binary event `response == +1`.
`calibration/manifest.json` records this mapping and its limitations. The Python
interface `scalar_ppc_views(dt)` returns the corresponding `{"rt": ..., "choice":
...}` mapping without changing `dt`.

Shared convergence and joint-trial LOO use the original artifact. The scalar
views intentionally contain **no log likelihood**: never pass `--loo-pit` for
these views or duplicate a joint likelihood under marginal variable names.
Do not flatten the original component axis into observations.

Interpret each scalar check as **fitted-data marginal PPC-PIT**. Even agreement
of both marginals does not establish joint, conditional or held-out calibration.
The choice-proportion and response-conditional RT checks remain necessary domain
evidence. Keep the two shared assessment outputs distinct, with these scope
limits next to their ratings. Missing groups, invalid schema and failed checks
must remain visible. Literal-array adapter tests and one complete analytical-DDM
run verified this handoff on the recorded HSSM 0.5 stack. These software checks
do not extend the statistical meaning of the marginal assessments.

## Prior sensitivity

Read the release-scoped prior-density bridge in
[priors-and-links.md](priors-and-links.md) before computing sensitivity. When
its completeness and numerical density checks have passed, use the original
artifact with both `log_prior` and joint `log_likelihood` groups:

```python
from arviz_stats import psense_summary

sensitivity = psense_summary(idata, var_names=["v", "a", "z", "t"])
sensitivity.to_csv(results_dir / "psense.csv")
sensitivity.to_json(results_dir / "psense.json", orient="index")
```

Add `--psense "$RESULTS/psense.json"` to the shared assessment calls once this
file exists. The JSON must map each parameter to its `prior`, `likelihood` and
`diagnosis` fields; the default DataFrame column orientation is incompatible
with the shared reader. Omit fixed parameters from `var_names`. Interpret the
shared result with domain context, without inventing HSSM-specific thresholds.
Until the density checks run, keep sensitivity explicitly unassessed.

## Canonical report

Read `BAYESIAN_SKILL_DIR/references/reporting.md` and fill `<slug>/report.md`.
Retain the canonical report structure and explanatory text where its stated
quantity applies. Use shared assessments verbatim with domain context; mark
unsupported sections unassessed with the reason rather than inventing a rating.
When no run has occurred, provide the analysis plan instead of a result report.

Add within the existing sections:

- **Data and Question:** task, RT units, choice/accuracy mapping, original and
  retained trial counts, exclusions and the intended flat-model assumptions.
- **Model Specification:** `ddm`, analytical likelihood, `p_outlier=None`, the
  effective parameter priors/bounds, configured `t` limits and their justification.
- **Prior Predictive Check:** plausible RT range, choice-proportion patterns,
  tails and any prior revisions.
- **Posterior:** parameter summaries with explicit intervals and meaning under
  the declared boundary/starting-position convention.
- **Posterior Predictive Check:** separate choice-rate and conditional-RT
  evidence, not only a pooled RT plot. Name the HSSM plot
  `posterior_predictive.png` and link any additional domain tables/figures.
- **Calibration / Prior Sensitivity:** the precise assessed quantity, available
  evidence and omitted or unsupported checks. Do not label a marginal-RT
  calibration result as calibration of the complete DDM.
- **Limitations and Threats:** weak identification, sparse choices, predictive
  mismatches, flat-model assumptions, omitted contaminants and deferred checks.
- **Appendix:** exact versions, seed/RNG limitations, sampler/budget, saved
  artifacts, data provenance and analysis code locations.

Remove links to figures that were not produced and state why the evidence is
missing. Keep failed-run diagnostics alongside successful checkpoints. Suggested
next steps should follow the actual failed or unassessed checks, with domain
model revisions stated as hypotheses to investigate.

Sources: HSSM 0.5.0
[sampling, likelihood and predictive APIs](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/base.py),
[predictive plotting](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/plotting/predictive.py),
and [quantile-probability plotting](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/plotting/quantile_probability.py).
