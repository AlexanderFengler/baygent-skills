# Bambi examples

Two narrated **marimo notebooks** exercise the initial skill from model design
through a response-scale answer and the shared Bayesian report:

- [Gaussian regression](gaussian_regression.py): explicit priors and the expected
  outcome difference between exposure +0.5 and −0.5. A final slider explores
  predictions without refitting.
- [Hierarchical Bernoulli survey](hierarchical_bernoulli.py): unequal region sizes,
  verified low-income reference coding, and a support probability for a 40-year-old
  urban respondent with medium income in an observed region. Final age/region
  controls recompute predictions from a copy of the fit.

Both generate deterministic synthetic data; neither supplies evidence about a
real population. Modeling and sampling remain visible in notebook cells.
`example_support.py` only assembles plots, artifacts and the shared report. It
uses the existing `bayesian-workflow/scripts` and report template in this checkout;
it is **not** a dependency of the independently installed Bambi skill.

## Environment

Use Python 3.12 and the modern [environment-pymc6.yml](../../environment-pymc6.yml):

```bash
mamba env create -f environment-pymc6.yml
conda activate baygent6
```

A pip/uv alternative was executed for this contribution. From the repository root,
choose an environment path (the Spine workspace uses `_local/venvs/`):

```bash
uv venv --python 3.12 /path/to/baygent-bambi-env
uv pip install --python /path/to/baygent-bambi-env/bin/python \
  'bambi==0.21.0' 'pymc==6.3.2' 'arviz==1.3.0' \
  'arviz-stats==1.3.2' 'arviz-plots==1.3.1' 'nutpie==0.16.11' \
  'marimo==0.24.2' netcdf4 h5netcdf pymc-extras pytest
source /path/to/baygent-bambi-env/bin/activate
```

Graphviz's `dot` executable is needed for the model graph. It is included in the
conda recipe; for the uv path install Graphviz with your system package manager
(e.g. `brew install graphviz` on macOS). Each run writes `versions.json`; see the
[validation record](../../evals/bambi-workflow/iteration-1/README.md) for the exact
executed environment and results. Conda resolution was not exercised here.

## Run and inspect

```bash
marimo edit examples/bambi-workflow/gaussian_regression.py
marimo edit examples/bambi-workflow/hierarchical_bernoulli.py
```

A normal run uses 1,000 tuning and 1,000 retained draws per chain, with Bambi's
normal chain selection. Each notebook saves immediately after sampling as
`results/<analysis>/posterior.nc`, then saves the completed original-data
predictive/log-density artifact as `inference_data.nc`. The report and figures
live beside it. `BAYGENT_OUTPUT_ROOT=/another/folder` changes the results root;
rerunning a notebook overwrites its own outputs.

To run without the editor and export a static HTML view:

```bash
mkdir -p examples/bambi-workflow/results
marimo export html examples/bambi-workflow/gaussian_regression.py \
  -o examples/bambi-workflow/results/gaussian_regression.html
marimo export html examples/bambi-workflow/hierarchical_bernoulli.py \
  -o examples/bambi-workflow/results/hierarchical_bernoulli.html
```

Use `--force` to replace an existing export. Static HTML preserves the executed
view; use the live notebook for reactive controls. Generated result folders are
ignored by Git. Selected full-run reports, plots, summaries and HTML snapshots
are retained under the validation record; raw NetCDF files remain local.

## Validation and limits

```bash
python -m pytest evals/smoke/test_bambi_workflow.py
marimo check --strict examples/bambi-workflow/gaussian_regression.py \
  examples/bambi-workflow/hierarchical_bernoulli.py
```

The smoke suite executes both notebooks with `BAYGENT_SMOKE=1` (150 tuning and
150 retained draws, two chains) in a temporary output folder. It checks numerical
prediction and density semantics, category encoding, saved coordinates, shared
assessments and report artifacts. It deliberately does not demand healthy
convergence from that small budget.

Bambi 0.21's default saved intercept coordinates and omitted group offsets can
make `compute_log_prior` incomplete or inconsistent. These examples explicitly
scale predictors, set `center_predictors=False`, and retain offsets with
`omit_offsets=False`. The suite verifies every free-variable prior density
against its declared Normal/HalfNormal density. See the skill's
[prior reference](../../bambi-workflow/references/priors-in-bambi.md).

Calibration uses fitted-data PPC-PIT, not held-out validation. Sensitivity flags
are reported even when sampling succeeds; a synthetic example does not justify
its priors for real survey data. The survey report plots replicated support proportions and regional uncertainty
instead of pooled binary density curves. The example report preserves the canonical
section/artifact structure and assessments, with two narrow explanatory-text
corrections for modern iteration traces and PIT-versus-coverage interpretation.
