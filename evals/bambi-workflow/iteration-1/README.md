# Bambi M1 validation — 2026-09-14

The skill and two examples are implemented and executed. **The paired agent
and trigger evaluations are pending**, so M1's complete behavioral acceptance
claim remains open. `benchmark.json` has null results, not invented scores.

Independent evaluation requires authorization to transmit the installed skill
and synthetic prompts to the model service. That gate is pending; local runtime
checks are separate evidence and do not substitute for agent behavior scores.

## Executed examples

| Example | Response-scale answer (posterior mean, 94% HDI) | Shared assessment |
|---|---|---|
| [Gaussian regression](runtime/gaussian-regression/report.md) · [HTML](runtime/gaussian-regression/notebook.html) | Difference at x=+0.5 minus −0.5: 1.847 [1.501, 2.161] | Convergence and fitted-data PPC-PIT pass; low prior sensitivity. |
| [Hierarchical survey](runtime/hierarchical-bernoulli/report.md) · [HTML](runtime/hierarchical-bernoulli/notebook.html) | Support probability at age 40, medium income, urban=1, north: 0.540 [0.386, 0.681] | Convergence and fitted-data PPC-PIT pass; strong sensitivity for region SD. |

Both are synthetic teaching analyses. Full exports use four chains with 1,000
tuning and 1,000 retained draws per chain on the resolved Bambi 0.21.0 / PyMC
6.3.2 / ArviZ 1.3.0 environment. Sampler warnings and sensitivity flags remain
visible; these are not empirical research findings. The survey's next modeling
step is a justified alternate region-scale prior and comparison of the requested
probability, not suppression of the sensitivity warning.

[execution.json](runtime/execution.json) records commands, seeds, sampling
settings, source hashes, numerical results and local NetCDF hashes. Exact pip
packages are in [requirements-resolved.txt](runtime/requirements-resolved.txt).
The conda recipe was updated but conda resolution was not tested; the uv/pip
path in the [example README](../../../examples/bambi-workflow/README.md) was used.

Reports, figures, summary/diagnostic JSON, synthetic CSVs and static HTML are
committed here. Large NetCDF artifacts remain in the gitignored example results
folder and are reproducible by running the notebooks. HTML is a static executed
view; reactive controls require opening the `.py` notebook in marimo.

## Checks performed

- `python -m pytest -q evals/smoke/test_bambi_workflow.py`: **13 passed**, 10.63 s
  on the recorded run. Each notebook executes once at 150 tuning/150 retained
  draws and two chains. Smoke budgets intentionally do not assert healthy
  convergence. The 128 warnings are netCDF4/NumPy 2.5 shape deprecations during
  file roundtrips; no warnings were suppressed.
- Independent formulas verify both native likelihoods, every free-variable prior
  density, the Gaussian contrast and the exact conditional survey probability
  and HDI. Other checks verify treatment coding, preserved original observations,
  saved/restored artifacts, shared psense values and canonical report links.
- Both full notebooks exported to HTML successfully; prior/PPC and sensitivity
  figures were visually inspected, including label spacing and binary plots.
- Ruff check/format and strict marimo checks passed for both notebooks and their
  helper/test surface. Skill frontmatter validation passed (491-character
  description). Twelve Python reference snippets parse.
- Copied only `bambi-workflow` and `bayesian-workflow` into a clean temporary
  installation: all nine local Bambi links resolve and dependency scripts/report
  resources are present. No repo plans, examples or evaluation files are required
  by the installed skill.
- Independent read-only review found and confirmed the Bambi intercept/offset
  density pitfalls and legacy report prose issues; implementation now addresses
  them. Shared Bayesian diagnostic scripts are unchanged, so this does not claim
  a new legacy PyMC 5 compatibility run.

## Findings that shaped the implementation

Formulae 0.6.2 can alphabetize an unordered pandas category list. Explicit ordered
metadata and assertions establish low income as reference. Bambi interpretation
returns `Result.summary` plus `Result.draws`; the latter describes prediction
rows and must not replace fitted-data diagnostics. Its `average_by` aggregates
summaries, so the Gaussian example uses it only for the row-invariant identity-link
contrast; nonlinear population-average intervals need draw-wise averaging.

Bambi 0.21 evaluates saved intercept values directly in `compute_log_prior` and
skips missing latent offsets. Explicitly scaled designs with
`center_predictors=False` and `fit(omit_offsets=False)` align the saved posterior
and complete joint prior; numerical density tests guard this release-specific
workaround. Changing these settings on an existing fit is not a correction.

The report helper retains the shared canonical structure and assessment outputs,
while correcting two legacy explanations: modern `plot_trace` shows draw order,
and PIT-ECDF location deviations are not simple under/overconfidence labels.
The survey report shows support proportions and regional intervals instead of
uninformative pooled binary densities. No diagnostic rating is reimplemented.
