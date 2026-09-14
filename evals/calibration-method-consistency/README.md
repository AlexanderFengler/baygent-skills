# Calibration method consistency and compatibility evidence

These records preserve software and scientific correctness checks performed on
the development revisions identified by their source hashes, before extraction
of this contribution. They are not test results for the extracted branch or
agent-performance evaluations. The runs used macOS arm64 and Python 3.12.13.
The complete historical evidence is [archived at an immutable revision][archive].

## Calibration contract

The independent review reproduced two native plotting problems:

1. The ECDF plot's `pit=True` path divides probability coordinates by their
   maximum. For PIT values `[.1, .2, .3, .4]`, it displays `[.25, .5, .75, 1]`;
   the displayed difference curve disagrees with `ECDF(x) - x` by up to **0.5**.
2. The PPC wrapper applies the coverage transformation during PIT preparation
   and forwards `coverage=True` to a plot function that applies it again.

The [shared script](../../bayesian-workflow/scripts/calibration_check.py)
prepares one validated PIT array, applies `2 * abs(PIT - .5)` once for coverage,
and draws curves from the evidence used for assessment. Native uniformity tests
remain in use. Automatic selection uses `pot_c` on the tested modern stacks
and simultaneous `envelope` bands on the legacy stack. Equal automatic-default
verdicts are not promised: modern outputs report p-values and null band flags;
legacy outputs report band flags and null p-values.

The [review summary](review/summary.json) records **10 saved curves** checked
against independent `searchsorted` counts, with maximum absolute plotting error
**0**. It also covers p-value annotations, endpoints, binary dtype invariance,
malformed-input rejection and coverage direction. Retained probes cover
[legacy](review/probe-legacy.json), [modern stats 1.2](review/probe-hssm.json)
and [modern stats 1.3](review/probe-bambi.json); their [plot files are archived][figures].
The historical environment labels `hssm` and `bambi` identify the interpreters
used for these shared-script checks, not skill dependencies.

## Recorded execution results

| Environment | PyMC | ArviZ umbrella / base / stats / plots | NumPy | Calibration cases |
|---|---|---|---|---|
| Legacy | 5.28.0 | 0.23.0 / 1.0.0 / 1.0.0 / 1.0.0 | 2.4.6 | 29 passed, 3 skipped |
| Modern stats 1.2 | 6.1.0 | 1.2.0 / 1.2.0 / 1.2.0 / 1.2.0 | 2.4.6 | 29 initial + 3 follow-up passed |
| Modern stats 1.3 | 6.3.2 | 1.3.0 / 1.3.0 / 1.3.2 / 1.3.1 | 2.5.3 | 29 initial + 3 follow-up passed |

Modern results combine the initial suite and three subsequently added
regressions; they are not a single 32-case run. The final full
[legacy run](compatibility/calibration-legacy-tests.log.txt) records 29 passes
and three modern-only skips. The unchanged reporting harness passed **16/16**
on [legacy](compatibility/harness-legacy.log.txt) and
[modern stats 1.2](compatibility/harness-modern.log.txt), using its real
400-draw, 400-tune, two-chain fit. It was not run on modern stats 1.3.

The [cross-environment result](compatibility/equivalence.json) compares identical
healthy and pathological fixtures with `method="envelope"` on both stacks.
All strict fields and original numerical tolerances were preserved: convergence
and calibration ratings, structural flags, `loo_computed`, summaries and
non-LOO next steps agreed. Both stacks computed LOO. Healthy convergence and
calibration were excellent; the pathological fixture failed both. The gate
reports, but does not gate, the LOO Pareto-k rating because the upstream PSIS
estimators differ.

The [fixture description](compatibility/fixtures.json) records four chains,
400 draws and 80 observations from deterministic Gaussian quantile grids.
Pathology injects chain shifts, 100 divergence flags and a predictive shift of
+8 while keeping the likelihood common. These are diagnostic fixtures,
**not coherent fitted posteriors**. Their construction used no sampling.
Explicit uv interpreters ran the maintained `--emit-payload` worker and `_diff`
comparison. Neither the conda orchestrator nor its sampling `--build-fixture`
mode ran. Modern payloads from the first attempt were reused after checking
unchanged source hashes.

## Limits and failures

ArviZ-stats 1.2's native `pot_c` test raises on the deterministic healthy LOO PIT:
`Cannot compute truncated Cauchy combination test. No p-values below 0.5 found.`
The prepared PIT exactly matches the public API. The error propagates without
a passing verdict or fallback; stats 1.3.2 accepts this fixture. Explicit
envelope checks passed on all three stacks. See the [LOO probes](review/summary.json).

The probes also preserve **external historical HSSM evidence**: the saved
example's RT coverage test failed (`p = 3.0531133177191805e-15`, alpha `0.01`)
despite a small mean coverage deviation (`-0.0102`). Raw RT PIT and both choice
tests passed. A small mean deviation does not override failed shape evidence.
These in-sample marginal checks establish neither held-out nor joint calibration.
The original HSSM NetCDF inputs were ignored runtime files and are not included
in this contribution or the Git archive; their historical paths and hashes are
retained in the probes. This observation is not needed for the standalone
common-envelope comparison.

Legacy ArviZ plots 1.0 failed to import with Matplotlib 3.11.2
(`matplotlib.style.core` missing); pinning Matplotlib 3.10.8 resolved it.
The interactive traceback was not retained; the [corrective install log is
archived][matplotlib-install]. The first harness fit failed on serialization
because h5netcdf lacked h5py. After installing h5py 3.16.0, an InferenceData
roundtrip and the same harness passed. The [failed first attempt][attempt-1]
and [backend install log][h5py-install] remain archived. Resolved requirements
are retained for [legacy](compatibility/requirements-legacy.txt) and
[modern stats 1.2](compatibility/requirements-modern.txt).

## Provenance and repeating checks

[provenance.json](provenance.json) separates the files retained here from
historical artifacts available only in the immutable archive. Original source
hashes and archived publication hashes remain intact. New publication hashes
identify curated local bytes, not a rerun of the measurements. Path placeholders
identify historical locations and do not imply those inputs exist locally.

The retained [fixture-generator.py.txt](reproduction/fixture-generator.py.txt)
is historical text. To use it, copy it outside this evidence directory to a
`.py` file and replace `<COMPAT_WORKDIR>` with a new output directory whose
parent exists. It writes `healthy.nc`, `pathological.nc` and `fixtures.json`;
keep those new outputs separate from these records. Install its NumPy, SciPy,
xarray and NetCDF dependencies in a prepared environment. Regenerated NetCDF
byte hashes are **not** promised to match the historical files.

The archived [equivalence runner][equivalence-runner] depends on first-attempt
JSON payloads. The archived [independent reviewer][independent-review] also
depends on the ignored HSSM NetCDF inputs. Neither snapshot is a standalone
reproduction runner, and neither was tested during extraction.

For the maintained checks, run from the repository root in each prepared
environment:

```bash
python -m pytest -q evals/smoke/test_calibration_consistency.py
python evals/smoke/test_reporting_harness.py
```

The second command performs the small fit above. For a frozen-fixture
comparison without fitting, use the [archived healthy fixture][healthy] and
[archived pathological fixture][pathological], decompress them, and invoke
`python evals/smoke/cross_env_equivalence.py --emit-payload --idata PATH`
with each interpreter. Compare payloads with that module's `_diff` function.
The original uncompressed byte hashes are in `compatibility/fixtures.json`.
This is a procedure for new validation, not a claim that it ran on this branch.

[archive]: https://github.com/AlexanderFengler/baygent-skills/tree/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/
[figures]: https://github.com/AlexanderFengler/baygent-skills/tree/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/review/figures/
[matplotlib-install]: https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/setup/baygent-arviz10-matplotlib-install.log.txt
[attempt-1]: https://github.com/AlexanderFengler/baygent-skills/tree/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/compatibility/attempt-1/
[h5py-install]: https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/compatibility/h5py-install.log.txt
[equivalence-runner]: https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/reproduction/equivalence-runner.py.txt
[independent-review]: https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/reproduction/independent-review.py.txt
[healthy]: https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/compatibility/healthy.nc.gz
[pathological]: https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/calibration-method-consistency/compatibility/pathological.nc.gz
