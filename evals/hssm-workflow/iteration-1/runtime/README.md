# Analytical DDM runtime validation

This record describes the historical full export at source `986a9db`, on
Python 3.12.13/macOS arm64, using HSSM 0.5.0, Bambi 0.19, PyMC 6.1,
ArviZ 1.2 and NumPy 2.4.6. It completed in 40.55 seconds. The combined suite
passed **171 tests** (139 HSSM plus 32 shared calibration regressions), with
one intentional duplicate-fit skip and four strict expected failures.

## Evidence

- [Execution provenance](execution.json), [environment](environment.json),
  and [resolved requirements](requirements-resolved.txt).
- [Test summary](test-results.json), [fixed-point numerical checks](fixed-point-numerics.json),
  and [saved-draw density checks](saved-density-verification.json).
- [Archived report](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/analytical-ddm/report.md),
  [figures and scalar assessments](https://github.com/AlexanderFengler/baygent-skills/tree/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/analytical-ddm), and
  [HTML export](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/analytical-ddm/notebook.html).

The reports and raw execution history are preserved on the fork at an immutable
commit. This contribution keeps the executable sources and compact records.
Historical hashes refer to their recorded source/artifact bytes, not a later PR
commit. Large posterior/prior NetCDFs were local generated files and were never
committed; reproduce an export to run saved-artifact tests.

## Reproduction

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


Use a fresh output directory when repeating the export. The example records
simulation/prior/sampler seeds, but HSSM 0.5's public posterior-predictive API
has no seed argument, so bitwise PPC reproduction is not promised. The pip
requirements were installed with uv and system Graphviz 15.1. The alternative
conda recipe was not executed; this is not a platform support matrix.

## Numerical and artifact checks

The full budget was 300 trials, 200 prior draws, two chains, 1,000 tuning and
1,000 retained draws per chain, with one sampling process. Saved-artifact checks
verify input/sample/trial coordinates, joint likelihood vectors, complete
natural-scale priors, exact scalar margins, domain summaries, report tables,
figure links and the HTML's serialized plot outputs. Standalone figures and
report text were inspected; browser layout was not verified.

The independent `hddm-wfpt==0.1.7` Cython reference uses signed RT, boundary
separation `2*a`, and no inter-trial variability or outlier mixture. It shares
the Wiener model's mathematical ancestry with HSSM; normalization, choice-mass
and reflection checks further constrain it. Declared tolerances were `1e-6`
absolute log-density error, `1e-10` for fixed-point prior densities and `2e-8`
for bounded choice-mass quadrature. Saved joint-density error was at most
`1.15e-14`; all saved prior-density errors were below `4e-16`.

## Retained limitations

- Convergence and joint-trial LOO were excellent, and prior sensitivity was low.
  RT marginal calibration was **fair**: coverage-transformed PIT rejected
  uniformity (`p=3.1641e-15`), while ordinary RT PIT (`p=0.1076`) and both choice
  tests passed. The [shared correction](../../../calibration-method-consistency/README.md)
  ensures the assessment and figures use the same method and one transformation.
- These are fitted-data marginal checks, not recovery or joint/conditional/
  held-out calibration. The posterior mean of `a` was 1.304 with 94% HDI
  [1.224, 1.388]; generating `a=1.2` lay outside. The `t` support explicitly
  uses minimum observed RT; power sensitivity does not test that support policy.
- Four strict expected failures document two native sample-coordinate defects
  across two `t` support policies. The notebook requires default sample labels
  before native density recomputation; the installed adapter preserves arbitrary
  valid labels.
- ArviZ-stats 1.2 can raise on sufficiently uniform LOO-PIT; the helper propagates
  that error. The HSSM scalar views use PPC-PIT. Native Numba object-mode,
  `compute_log_prior` deprecation and netCDF4 import warnings were retained;
  serialization and graph rendering checks passed.
- Earlier attempts exposed IPC, native plotting, marimo display and shared
  calibration defects. Their [archived records](https://github.com/AlexanderFengler/baygent-skills/tree/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/logs) remain available;
  the dataset and sampling budget were not changed to obtain favorable flags.
- No with/without-skill comparison or trigger evaluation has completed. The
  [evaluation status](../README.md) records the incomplete budgeted baseline.
