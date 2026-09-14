# Bambi runtime validation — 2026-09-14

The skill and two examples were locally executed on the development branch.
**Paired agent and trigger evaluations remain unexecuted**; [benchmark.json](benchmark.json)
contains null scores. Runtime checks do not measure agent behavior. The source
hashes and results below are historical evidence, not a fresh run of this PR.

## Recorded results

The archived reports reassess the original saved draws after the shared
calibration fix, without a new fit or predictive generation.

| Example | Response-scale answer (posterior mean, 94% HDI) | Shared assessment |
|---|---|---|
| [Gaussian regression](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/gaussian-regression/report.md) | Difference at x=+0.5 minus −0.5: 1.847 [1.501, 2.161] | Convergence and both fitted-data PPC-PIT checks pass; low prior sensitivity. |
| [Hierarchical survey](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/hierarchical-bernoulli/report.md) | Support probability at age 40, medium income, urban=1, north: 0.540 [0.386, 0.681] | Convergence and both fitted-data PPC-PIT checks pass; strong sensitivity for region SD. |

Both are synthetic teaching analyses. The original full fits used four chains
with 1,000 tuning and 1,000 retained draws per chain on Bambi 0.21.0 / PyMC 6.3.2 /
ArviZ 1.3.0. The survey's next modeling step remains a justified alternate
region-scale prior and comparison of the requested probability. Fitted-data
PPC-PIT does not establish held-out calibration or causal identification.

The [reassessment record](runtime/calibration-reassessment/README.md) provides
exact p-values, provenance and immutable links to the corrected report bundles
and test records. Native `pot_c` and legacy envelope checks are distinguished in
the [shared calibration evidence](../../calibration-method-consistency/README.md).

The original [Gaussian HTML](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/gaussian-regression/notebook.html)
and [survey HTML](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/hierarchical-bernoulli/notebook.html)
are historical exports from before the calibration fix. Their embedded
calibration output is superseded by the reassessed reports above; replacement
executed HTML is not claimed.

## Runtime provenance and checks

[Original execution.json](runtime/execution.json) records full-run commands,
seeds, settings, development source hashes and local NetCDF hashes. Exact packages
are in [requirements-resolved.txt](runtime/requirements-resolved.txt); the uv/pip
path in the [example README](../../../examples/bambi-workflow/README.md) was used.
Reports, figures, diagnostic JSON and synthetic CSVs are archived at the linked
fork commit. Large NetCDF artifacts were never committed and are not available
at those links. Maintained smoke tests generate temporary outputs and do not
need the archived report bundles.

- The integration suite passed **13 tests in 27.420 s** after the shared
  calibration fix. These separate smoke runs used two chains with 150 tuning and
  150 retained draws, without requiring healthy convergence at that small budget.
- Independent formulas verified native Gaussian/Bernoulli likelihoods, all
  free-variable prior densities, the Gaussian contrast and the conditional survey
  probability and HDI. Tests also checked treatment coding, original observations,
  artifact roundtrips, shared psense values and report links.
- Five additional literal report regressions verified missing-extrema fallback,
  retained native values and unchanged shared pass/flag statuses without fitting.
- Original full exports, installed-skill link checks, reference-snippet parsing,
  Ruff and marimo checks passed on the recorded development sources.

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
