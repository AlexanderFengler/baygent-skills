---
name: bambi-workflow
description: >
  Fit and interpret Bayesian regressions with Bambi. Use when the user asks
  for Bambi, bmb.Model, formula-based GLMs or GLMMs, partial pooling,
  Bambi prior inspection, or response-scale predictions and contrasts.
  Guides formula and outcome encoding, family/link choice, prior predictive
  checks, and native Bambi interpretation; delegates shared diagnostics
  and reporting to bayesian-workflow. Preserve explicit requests to use
  raw PyMC, HSSM sequential-sampling models, or BayesFlow training.
license: MIT
metadata:
  author: "[Alexander Fengler](https://www.alexanderfengler.com)"
  version: "0.1"
---

# Bambi Workflow

Build the requested regression with Bambi and answer the scientific question
with predictions, uncertainty, and a reproducible diagnostic report.

## Dependencies

Requires **bayesian-workflow** for shared diagnostics, calibration, prior
sensitivity, and the canonical report. Locate it in the agent's available
skill catalog. If necessary, check the directory beside this installed skill
for `bayesian-workflow/SKILL.md`; installations may use another location.
Read its `SKILL.md` and resolve its absolute directory before invoking its
scripts or references. Below, `BAYESIAN_SKILL_DIR` denotes that verified path.

If absent, explain the missing dependency and install both skill folders into
the user's selected skill location when installation is authorized. The source
is [baygent-skills](https://github.com/Learning-Bayesian-Statistics/baygent-skills).
Do not silently replace missing scripts with invented ratings.

Follow the shared workflow's statistical checks and reporting conventions.
Bambi owns model construction, fitting, prediction, and log-density computation:
use the Bambi calls below when the dependency shows raw PyMC equivalents.

## Supported scope and stack

This initial skill covers Gaussian regression and Bernoulli regression with
a varying intercept, including prediction for an observed group. Target
**Bambi 0.21.0, PyMC 6, ArviZ 1.x, and Python 3.12**. Check installed versions
before adapting examples; modern sampler results are xarray `DataTree` objects.
The shared Bayesian skill's legacy-stack support does not establish compatibility
with older Bambi releases.

For distributional models, counts, ordinal outcomes, splines/HSGP, custom
families, or unseen-group predictions, state that the request extends this
initial scope and consult the applicable release documentation. Preserve the
user's chosen model and distinguish unverified extensions from supported paths.
For causal claims, use `causal-inference` for identification and assumptions.

## Workflow overview

For an existing fitted model or a focused question, enter at the relevant step.
A full analysis follows this sequence.

1. **Frame the question.** Identify the outcome, observational unit, grouping,
   and prediction or contrast to report. Use choices already supplied; resolve
   missing scientific assumptions before treating them as facts.
2. **Check data and formula.** Verify missingness, response event, predictor
   scales, categorical reference, and group support. Inspect the encoded terms.
   See [references/formula-syntax.md](references/formula-syntax.md).
3. **Choose family/link and priors.** Specify the intended likelihood explicitly;
   the default is Gaussian. Inspect resolved priors and justify overrides on the
   relevant scale. For this release's explicitly scaled examples, construct with
   `center_predictors=False` so saved intercepts retain the prior's coordinates.
   See [references/families-and-links.md](references/families-and-links.md)
   and [references/priors-in-bambi.md](references/priors-in-bambi.md).
4. **Build and check prior predictions.** Call `model.build()`, then
   `model.prior_predictive(draws=500, random_seed=RANDOM_SEED)`. Assess whether
   replicated outcomes and group variation are scientifically plausible before
   fitting; drawing or plotting alone is not the assessment.
5. **Fit and save.** Prefer `model.fit(inference_method="nutpie", omit_offsets=False, ...)`
   when available for the supported models. Retain offsets for complete prior
   densities. Use the available `"pymc"` backend if needed and record the choice.
   Save the returned artifact immediately with
   `idata.to_netcdf(...)`, before prediction or diagnostics.
6. **Prepare diagnostic inputs.** Use `model.predict(idata, kind="response")`,
   `model.compute_log_likelihood(idata)`, and `model.compute_log_prior(idata)`.
   These mutate `idata` by default. Verify group names, outcome names and
   observation coordinates, then save the enriched artifact.
7. **Diagnose and criticize.** Run the shared diagnostics and calibration
   pipeline, inspect predictive checks, and assess prior sensitivity using its
   reference. Investigate failed checks before interpreting results; preserve
   missing or unstable checks as limitations.
8. **Answer on the requested scale.** Use `bmb.interpret.predictions` or
   `comparisons`, specifying the target grid and interval explicitly. Report
   posterior-mean probabilities and distinguish an observed-group prediction
   from an average over groups or a new group's prediction.
9. **Report.** Fill `<slug>/report.md` from the installed Bayesian report template,
   using shared assessments and next steps. Include the Bambi formula,
   family/link, priors, event/reference coding and prediction target in its
   existing sections. See
   [references/interpretation-and-reporting.md](references/interpretation-and-reporting.md).

## Critical rules

- **Inspect the actual encoding and priors.** Bambi constructs/scales priors
  during model construction. Build before graph or prior-predictive operations;
  do not claim priors are absent before `build()`. Use `model.parameters`,
  `conditional_parameters`, and `marginal_parameters` rather than deprecated
  `model.components`.
- **Choose a scientifically meaningful reference.** Object columns can already
  be categorical. `C()` or `categorical=` alone does not select a reference
  category; an unordered category list may be alphabetized by Formulae. Use
  explicit reference/order handling, verify encoded columns, and preserve the
  coding for predictions.
- **Keep the original-data diagnostic artifact intact.** New-data predictions
  can overwrite likelihood-parameter and predictive arrays. Use `inplace=False`
  for a new grid; run calibration and LOO on aligned original observations.
- **Verify complete prior densities before sensitivity checks.** In Bambi 0.21,
  `compute_log_prior` skips missing offset RVs and does not undo the intercept
  uncentering performed after a fit with `center_predictors=True`. Use
  `center_predictors=False` for these explicitly scaled models and
  `omit_offsets=False` in `fit`; check every free RV's density. Existing fits
  using other settings need a verified coordinate correction or refit before
  sensitivity can be interpreted. See the prior reference for the limitation.
- **Use the native interpretation result correctly.** In 0.21, interpretation
  calls return a `Result` with `.summary` and `.draws`. `comparisons` takes
  `contrast={...}`; pass `prob=0.94, use_hdi=True` for a 94% HDI instead of
  relying on the ArviZ default. A mean-response interval is not a predictive
  interval for a future outcome.
- **Keep evidence and assessments together.** Use shared script outputs for
  diagnostic ratings. An error JSON, absent calibration plot, or tiny smoke
  run cannot establish that an analysis is healthy. Never invent an assessment
  or link a figure that was not produced.

## Shared utility pipeline

Set `BAYESIAN_SKILL_DIR` to the dependency directory resolved above, `RESULTS_DIR`
to the analysis folder, and `OUTCOME` to its saved observed-variable name. Run
these in the analysis environment:

```bash
python "$BAYESIAN_SKILL_DIR/scripts/diagnose_model.py" \
  --idata "$RESULTS_DIR/inference_data.nc" --output "$RESULTS_DIR/diagnostics.json"
python "$BAYESIAN_SKILL_DIR/scripts/calibration_check.py" \
  --idata "$RESULTS_DIR/inference_data.nc" --var-name "$OUTCOME" \
  --save-plots --plot-dir "$RESULTS_DIR" --output "$RESULTS_DIR/calibration.json"
python "$BAYESIAN_SKILL_DIR/scripts/check_diagnostics.py" \
  --diagnostics "$RESULTS_DIR/diagnostics.json" \
  --calibration "$RESULTS_DIR/calibration.json" --output "$RESULTS_DIR/check_report.json"
```

Read the outputs, including errors and missing checks, before using their ratings.
The reporting reference explains optional LOO-PIT and sensitivity inputs.

## When things go wrong

| Symptom | Check and next action |
|---|---|
| Binary outcome fitted as continuous | Verify `family="bernoulli"` and the encoded success event. |
| Prior predictions are implausible | Inspect resolved priors, predictor units and the link; revise and rerun before fitting. |
| Divergences or poor mixing | Follow Bayesian diagnostics; inspect separation, sparse groups and prior scales. Record Bambi's existing non-centered default. |
| `predict` returns `None` | It mutated the supplied artifact; use `inplace=False` when a separate return value is needed. |
| Calibration variables or coordinates differ | Recompute predictions/log likelihood on the original rows and inspect names and dimensions. |
| Interpretation output has no DataFrame methods | Use the returned `.summary`; `.draws` holds the associated draws. |
