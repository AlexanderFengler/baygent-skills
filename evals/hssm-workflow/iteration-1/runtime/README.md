# Analytical DDM runtime evidence

This record covers one flat HSSM analytical-DDM teaching workflow on Python
3.12.13/macOS arm64. It separates executable software checks from scientific
interpretation and from the still-pending skill-agent benchmarks.

**Runtime validated; RT marginal calibration flagged.** The final export at
`986a9db` completed in 40.55 seconds with the corrected shared calibration
pipeline. The combined suite passed **171 tests** (139 HSSM contracts/numerical
checks and 32 shared calibration regressions), with one intentional skip and
four strict expected upstream coordinate failures.

Convergence and joint-trial LOO are excellent; prior sensitivity is low.
The RT assessment is **fair**: its coverage-transformed PIT test rejects
uniformity (`p=3.1641e-15`, significance level 0.01), although the ordinary
RT PIT test does not (`p=0.1076`). Both choice tests pass. These are fitted-data
marginal checks, not a general calibration or recovery result. The report
preserves the RT flag and recommends examining conditional RT tails and the
model's assumptions before interpretation.

The [shared calibration correction](../../../calibration-method-consistency/README.md)
aligns assessment and plotting methods, applies coverage transformation once,
and preserves the raw PIT coordinate scale. It passed independent mathematical
checks and legacy/modern compatibility checks. The earlier inconsistent
assessment is superseded; its export attempt and artifacts remain preserved.

## Evidence and reproduction

- [Executed notebook](analytical-ddm/notebook.html) and
  [analysis report](analytical-ddm/report.md), with linked native figures.
- [Execution record](execution.json): exact source hashes, commands, budgets,
  timings, previous attempts and hashes of the retained local NetCDF files.
  Portable logs retain warnings and failures; the record documents path/whitespace
  normalization and original hashes. Exporter-only blank-line whitespace was
  removed from the published HTML, with both hashes recorded.
- [Environment record](environment.json) and
  [resolved requirements](requirements-resolved.txt). The pip pins were installed
  with uv and Graphviz 15.1 was supplied by the host. The conda recipe itself was
  not executed, and this is not a platform support matrix.
- [Test results](test-results.json) and [JUnit evidence](pytest-results.xml).
  The adapter has 65/66 executable lines covered in the parent pytest process;
  subprocess tests separately exercise its installed CLI.
- [Fixed-point numerical comparisons](fixed-point-numerics.json) and
  [saved-draw density comparisons](saved-density-verification.json).

From the repository root, with a Python 3.12 environment activated and Graphviz
on PATH:

```bash
uv pip install -r evals/hssm-workflow/iteration-1/runtime/requirements-resolved.txt
uv pip check
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/baygent-hssm-mpl
export XDG_CACHE_HOME=/tmp/baygent-hssm-cache
export PYTENSOR_FLAGS=base_compiledir=/tmp/baygent-hssm-numerics

# Deterministic contracts and fixed-point numerical checks; no fit.
env -u BAYGENT_TEST_HSSM -u BAYGENT_RUN_HSSM -u BAYGENT_HSSM_ARTIFACT_DIR \
  python -m pytest -q evals/smoke/test_hssm*.py

# Full synthetic teaching run, bypassing the notebook's interactive buttons.
mkdir -p examples/hssm-workflow/results/round-1
BAYGENT_RUN_HSSM=1 \
  BAYGENT_OUTPUT_ROOT=examples/hssm-workflow/results/round-1 \
  marimo export html examples/hssm-workflow/analytical_ddm.py \
  -o examples/hssm-workflow/results/round-1/analytical_ddm.html

# Verify that completed export without a second fit.
env -u BAYGENT_TEST_HSSM -u BAYGENT_RUN_HSSM \
  BAYGENT_HSSM_ARTIFACT_DIR=examples/hssm-workflow/results/round-1/analytical-ddm \
  python -m pytest -q evals/smoke/test_hssm*.py \
  evals/smoke/test_calibration_consistency.py
```

Before repeating an export, preserve the existing results directory and HTML.
The recorded attempts used identical input CSV bytes and unchanged declared
sampling settings. The retained earlier attempts document an IPC restriction,
native quantile plotting defects, a marimo display defect, and the shared
calibration defects; they were not
discarded to select favorable diagnostics. The workflow records descriptive
simulation/prior/sampler seeds but does not promise bitwise reproduction.
HSSM's public posterior-predictive API has no seed argument.

## What is checked

The exported budget is 300 trials, 200 prior draws, two chains, 1,000 tuning and
1,000 retained draws per chain, with one sampling process. Saved-artifact checks
verify the original posterior and observed rows, all predictive sample/trial
coordinates, complete natural-scale priors, selected joint likelihood vectors,
exact scalar margins, independent domain summaries, report tables and figure
links. The HTML session must contain both posterior figures and the completed
shared assessment; a successful CLI exit alone is insufficient.

The separately compiled `hddm-wfpt==0.1.7` reference is called directly through
`from hddm_wfpt import wfpt`, using signed RT, boundary separation `2*a`, and
zero inter-trial variability/outlier mixture. This differs from HSSM's native
PyTensor likelihood execution path. Normalization, choice-mass and reflection
checks further constrain the reference because implementations can share
mathematical ancestry. Numerical tolerances were declared before comparison:
`1e-6` absolute log-density error, `1e-10` for fixed-point prior densities, and
`2e-8` for bounded choice-mass quadrature. The tail allowance is independently
bounded; HSSM's positive numerical floor is never integrated to infinity.

Four strict expected failures retain two upstream coordinate limitations across
both non-decision-time support policies: HSSM 0.5 interprets sample labels as
positions, and PyMC 6.1 can reset them during prior computation. The notebook
rejects nondefault chain/draw labels before these native calls. The installed
scalar adapter independently preserves arbitrary valid labels. The separate
opt-in full-fit test is intentionally skipped when the full export supplies
the real fitted artifact for the offline checks.

All saved PNGs and the complete report are reviewed. Direct inspection of the
HTML's serialized outputs verifies content, including its visible plot images.
Browser layout inspection was blocked by the browser's local-file URL policy;
no alternate browsing route was used. The HTML requires its pinned marimo CDN
frontend to render; the report and standalone figures remain directly readable.

## Limits and retained warnings

One synthetic dataset does not establish general parameter recovery, model
identification, or joint/conditional/held-out calibration. The prior for `t`
explicitly uses the observed minimum RT; power sensitivity does not test a
different support policy. The scalar PPC-PIT artifacts exclude joint likelihood;
joint-trial LOO remains in the original paired artifact.

The posterior mean of `a` is 1.304, with 94% HDI [1.224, 1.388]; its generating
value was 1.2. This is a descriptive discrepancy from one finite dataset, not
a repeated-simulation recovery test. Observed choice proportions and reported
conditional RT quantiles lie within their predictive intervals, while the
more detailed RT coverage-transformed PIT check remains flagged.

ArviZ-stats 1.2 can raise a native error for sufficiently uniform LOO-PIT
values; the helper propagates it visibly. The HSSM scalar views use PPC-PIT,
not LOO-PIT, and this run does not establish general LOO-PIT reliability.

The run retains PyTensor/Numba object-mode warnings from native predictive
simulation, the PyMC root `compute_log_prior` deprecation, and a netCDF4
`numpy.ndarray size changed` import warning. No new warning filters were added
to pass this gate; the actual serialization and graph rendering checks passed.
The PyMC import path needs updating before the first PyMC release of 2027.

Large posterior/prior NetCDF files remain in ignored example results, with
hashes in the execution record. Committed CSV, JSON, HTML and PNG files are
compact review evidence. Independent model-service skill evaluations and
trigger trials have not run; their benchmark fields remain null.
