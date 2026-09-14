# Bambi M1 validation — 2026-09-14

The skill and two examples are implemented and locally executed. **Paired agent
and trigger evaluations remain pending**; `benchmark.json` contains null scores.
The [prepared agent pilot](../../agent-evaluation-round-1.md) requires separate
transmission authorization. Local runtime checks do not substitute for agent
behavior scores.

## Current reports

The reports below reassess the original saved draws after the shared calibration
fix. No new fit or predictive generation was performed during that reassessment.

| Example | Response-scale answer (posterior mean, 94% HDI) | Current shared assessment |
|---|---|---|
| [Gaussian regression](runtime/calibration-reassessment/gaussian-regression/report.md) | Difference at x=+0.5 minus −0.5: 1.847 [1.501, 2.161] | Convergence and both fitted-data PPC-PIT checks pass; low prior sensitivity. |
| [Hierarchical survey](runtime/calibration-reassessment/hierarchical-bernoulli/report.md) | Support probability at age 40, medium income, urban=1, north: 0.540 [0.386, 0.681] | Convergence and both fitted-data PPC-PIT checks pass; strong sensitivity for region SD. |

Both are synthetic teaching analyses. The original full fits used four chains
with 1,000 tuning and 1,000 retained draws per chain on Bambi 0.21.0 / PyMC 6.3.2 /
ArviZ 1.3.0. The survey's next modeling step remains a justified alternate
region-scale prior and comparison of the requested probability. Fitted-data
PPC-PIT does not establish held-out calibration or causal identification.

The [reassessment record](runtime/calibration-reassessment/README.md) provides
exact p-values, provenance, corrected plots and validation XML. Native `pot_c`
and legacy envelope checks are distinguished in the
[shared calibration evidence](../../calibration-method-consistency/README.md).
The updated convergence tables use available unrounded summary extrema without
changing shared diagnostic statuses.

The original [Gaussian HTML](runtime/gaussian-regression/notebook.html) and
[survey HTML](runtime/hierarchical-bernoulli/notebook.html) are **historical
pre-calibration-fix exports**. Their embedded calibration output is superseded
by the current reports linked above; replacement executed HTML is not claimed.

## Runtime provenance and checks

[Original execution.json](runtime/execution.json) records the full-run commands,
seeds, settings and NetCDF hashes. Exact packages are in
[requirements-resolved.txt](runtime/requirements-resolved.txt); the uv/pip path
from the [example README](../../../examples/bambi-workflow/README.md) was used.
Large NetCDF artifacts remain in the gitignored example results. Committed
reports, figures, diagnostic JSON and synthetic CSVs preserve reviewable output.

- The integration suite passed **13 tests in 27.420 s** after the shared
  calibration fix. These separate smoke runs use two chains with 150 tuning and
  150 retained draws, without requiring healthy convergence at that small budget.
- Independent formulas verify native Gaussian/Bernoulli likelihoods, all
  free-variable prior densities, the Gaussian contrast and the conditional survey
  probability and HDI. Tests also check treatment coding, original observations,
  artifact roundtrips, shared psense values and report links.
- Five additional literal report regressions verify missing-extrema fallback,
  retained native values and unchanged shared pass/flag statuses without fitting.
- Original full exports, installed-skill link checks, reference-snippet parsing,
  Ruff and marimo checks passed. Paired agent/trigger scores remain unmeasured.

## Retained implementation findings

Formulae 0.6.2 can alphabetize unordered categories. Explicit ordered metadata
and design assertions establish low income as the reference. Bambi interpretation
returns `Result.summary` and `Result.draws`; prediction-grid draws must not
replace the fitted-data diagnostic artifact. Nonlinear population-average
intervals require draw-wise averaging.

Bambi 0.21 directly evaluates saved intercept values in `compute_log_prior` and
skips missing latent offsets. Scaled designs with `center_predictors=False` and
`fit(omit_offsets=False)` preserve complete native prior-density evaluation;
changing those settings after a fit is not a correction. The report helper
retains the shared canonical structure and ratings. The survey uses support
proportions and regional intervals rather than pooled binary density plots.
