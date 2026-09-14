# Calibration method consistency and compatibility evidence

The corrected shared calibration script passes the legacy and modern reporting
harnesses and preserves strict cross-environment agreement when both environments
use the same calibration method. This record contains local software and
scientific correctness checks, not model-service or agent-performance evaluations.
Evidence was collected on macOS arm64 with Python 3.12.13; publication only copied,
redacted, and compressed existing artifacts.

## What changed

The independent review reproduced two upstream plotting problems:

1. The native ECDF plot's `pit=True` path divides probability coordinates by their
   maximum. For PIT values `[.1, .2, .3, .4]`, it displays
   `[.25, .5, .75, 1]`; the resulting displayed difference curve disagrees with
   `ECDF(x) - x` by as much as **0.5**.
2. The native PPC wrapper applies the coverage transformation during PIT
   preparation and forwards `coverage=True` to a plotting function that applies
   it again.

The [shared script](../../bayesian-workflow/scripts/calibration_check.py) now
prepares one validated PIT array, applies `2 * abs(PIT - .5)` exactly once for
coverage, and renders Matplotlib curves from the same evidence used for the
assessment. Native uniformity tests remain in use. Automatic method selection
uses `pot_c` on the modern stacks and simultaneous `envelope` bands on the legacy
stack. These are different tests; equal default verdicts are not promised.
Modern outputs report p-values and null band flags; legacy outputs report band
flags and null p-values.

The [independent review](review/summary.json) checked **10 saved curves**, including
raw PIT and coverage on all three stacks, against independent `searchsorted`
counts: maximum absolute plotting error was **0**. It also checked p-value
annotations, endpoint handling, binary dtype invariance, malformed-input
rejection, and coverage direction. Detailed probes are retained for
[legacy](review/probe-legacy.json), [modern 1.2](review/probe-hssm.json), and
[modern 1.3](review/probe-bambi.json), alongside the [actual plot files](review/figures/).

## Executed checks

| Environment | PyMC | ArviZ umbrella / base / stats / plots | NumPy | Calibration cases |
|---|---|---|---|---|---|
| Legacy | 5.28.0 | 0.23.0 / 1.0.0 / 1.0.0 / 1.0.0 | 2.4.6 | 29 passed, 3 skipped |
| HSSM modern | 6.1.0 | 1.2.0 / 1.2.0 / 1.2.0 / 1.2.0 | 2.4.6 | 32 passed |
| Bambi modern | 6.3.2 | 1.3.0 / 1.3.0 / 1.3.2 / 1.3.1 | 2.5.3 | 32 passed |

Modern coverage comprises an initial 29-case run plus the three subsequently
added regressions, as recorded in the review. The final full
[legacy run](compatibility/calibration-legacy-tests.log.txt) records 29 passes and
three skips for modern-only capabilities. These checks cover both default-method
correctness and explicit common-method behavior.

The unchanged upstream 16-check reporting harness passed **16/16** on
[legacy](compatibility/harness-legacy.log.txt) and
[modern 1.2](compatibility/harness-modern.log.txt). Each successful run used the
harness's real 400-draw, 400-tune, two-chain fit. The legacy serialization failure
and retry are retained below; this harness was not run on modern 1.3.

The [cross-environment result](compatibility/equivalence.json) compares identical
frozen healthy and pathological fixtures using `method="envelope"` in both
environments. All strict fields and original numerical tolerances remain
unchanged: convergence/calibration ratings, structural flags, `loo_computed`,
summary text, and non-LOO next steps agree. Both environments compute LOO; healthy
convergence and calibration are excellent, while the pathological fixture fails
both. The historical gate still reports rather than gates the LOO Pareto-k rating
because upstream PSIS estimators differ.

The [fixture description](compatibility/fixtures.json) records four chains,
400 draws, and 80 observations built from deterministic Gaussian quantile grids.
The pathology injects chain shifts, 100 divergence flags, and a predictive shift
of +8 while retaining the common likelihood. These are diagnostic fixtures,
**not coherent fitted posteriors**. No fitting was used to create them. With no
conda CLI available, explicit uv interpreters ran the upstream `--emit-payload`
worker and unchanged `_diff` comparison. The conda orchestrator and its fitting
`--build-fixture` mode were not used. Modern payloads from the first attempt were
reused only after verifying unchanged source hashes.

## Limits and preserved failures

ArviZ-stats 1.2's native `pot_c` test raises on the deterministic healthy LOO PIT:
`Cannot compute truncated Cauchy combination test. No p-values below 0.5 found.`
The prepared PIT exactly matches the public API. The error propagates without a
passing verdict or silent fallback; stats 1.3.2 accepts this fixture. Explicit
envelope checks pass on all three stacks. This known native limitation remains
visible in the [LOO probes](review/summary.json).

The corrected check flags the saved HSSM example's **RT coverage mismatch**:
coverage p-value `3.0531133177191805e-15` at alpha `0.01`, despite a small mean
coverage deviation of `-0.0102`. Its raw RT PIT passes, and both choice tests pass.
A small mean deviation does not override failed shape evidence. These are
in-sample marginal checks; they establish neither held-out nor joint calibration.

The isolated legacy environment needed Matplotlib 3.10.8 because plots 1.0 failed
to import with Matplotlib 3.11.2 (`matplotlib.style.core` missing). The original
traceback was observed interactively but not retained as a standalone log; the
[corrective install log](setup/baygent-arviz10-matplotlib-install.log.txt) is saved.
The first harness fit then failed during serialization because the h5netcdf
backend lacked h5py. After installing h5py 3.16.0, an InferenceData roundtrip and
the same harness passed. The [failed first attempt](compatibility/attempt-1/),
[backend install log](compatibility/h5py-install.log.txt), intermediate test run,
and visible dependency warnings are preserved. No pass was substituted for a
failed attempt. Resolved requirements are saved for
[legacy](compatibility/requirements-legacy.txt) and
[modern 1.2](compatibility/requirements-modern.txt).

## Artifact provenance and reproduction

[provenance.json](provenance.json) maps all copied artifacts to original and
published SHA256 hashes. Local paths use documented placeholders. Hashes already
inside source JSON identify original bytes; they need not match redacted copies.
NetCDF fixtures are losslessly gzip-compressed: decompression recovers the
original hashes in `fixtures.json`. PNG bytes are unchanged. Logs use `.log.txt`
to avoid the repository's log ignore rule.

The [reproduction snapshots](reproduction/) preserve the exact historical fixture
generator, uv orchestration, and independent reviewer with path redactions. They
are text records requiring path substitution, not newly tested portable runners.
To repeat the maintained checks in a prepared environment, run from the repo root:

```bash
python -m pytest -q evals/smoke/test_calibration_consistency.py
python evals/smoke/test_reporting_harness.py
```

The second command performs the small fit described above. To repeat only the
frozen-fixture comparison, decompress `compatibility/healthy.nc.gz` and
`compatibility/pathological.nc.gz`, invoke
`python evals/smoke/cross_env_equivalence.py --emit-payload --idata PATH`
with each interpreter, and compare the payloads using the script's unchanged
`_diff`, as recorded in the orchestration snapshot. No modeling is needed for
that comparison.
