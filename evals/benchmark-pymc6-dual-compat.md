# Benchmark — bayesian-workflow PyMC 5/6 dual-compatibility

## Current calibration contract (2026-09-14)

The cross-environment gate explicitly selects `method="envelope"` on both
stacks. It compares convergence, calibration ratings, structural flags,
`loo_computed`, summaries and non-LOO next steps exactly, with the existing
numeric tolerances unchanged. Each payload records the selected method.

Automatic calibration uses `pot_c` where the modern API is available and a
simultaneous-envelope fallback otherwise. Those methods are not promised to
produce identical ratings. The modern default and agreement between plotted
evidence and its assessment are verified separately from common-method
compatibility. The earlier blanket equivalence wording below describes the
historical run, not an automatic-default guarantee across later ArviZ releases.

The current local verification uses isolated uv environments, because conda
is unavailable. The legacy environment pins PyMC 5.28.0, ArviZ 0.23.0 and
ArviZ base/stats/plots 1.0.0; modern verification uses PyMC 6.1.0 and ArviZ
base/stats/plots 1.2.0. Both use Python 3.12.13 and NumPy 2.4.6 on macOS arm64.
ArviZ plots 1.0 could not import against Matplotlib 3.11.2, so only the new
legacy environment pins Matplotlib 3.10.8. HDF5 serialization also required
`h5py==3.16.0`, which `h5netcdf` did not install automatically. After adding
that backend, a legacy InferenceData read/write roundtrip and `uv pip check`
passed (57 packages). Existing environments were left unchanged.

The current runs retain the unchanged 16-check reporting harness and its
400-draw/tune, two-chain model in each environment. Common-method comparison
uses the upstream `--emit-payload` and `_diff` functions with explicit Python
interpreters and frozen deterministic healthy/pathological fixtures, without
running the conda orchestrator or sampling its additional fixture models.
The pathological fixture explicitly injects incompatible chain locations,
divergence flags and biased predictions; it is diagnostic test data, not a
claimed fitted posterior.

### Current executed results

| Check | Legacy PyMC 5 / ArviZ stats 1.0 | Modern PyMC 6 / ArviZ stats 1.2 |
|---|---|---|
| Unchanged reporting harness | 16/16 passed | 16/16 passed |
| Common-envelope healthy fixture | Convergence excellent; calibration excellent | Same strict payload; numeric differences within existing tolerances |
| Common-envelope pathological fixture | Convergence poor; calibration poor; 100 injected divergences | Same strict payload; numeric differences within existing tolerances |
| Focused calibration regression suite | 29 passed, three modern-only tests skipped | Modern-default evidence recorded separately |

All existing strict comparisons and numeric tolerances were preserved. Both
fixtures retained `loo_computed=true`. The shared fix was committed as
`29175ee`; payloads record source hashes and `calibration_method="envelope"`.
The [HSSM runtime record](hssm-workflow/iteration-1/runtime/README.md) preserves
compatibility logs, resolved requirements and the separate modern-default
calibration findings. This run did not repeat the historical taught-API sweep,
conda environment resolution or its regression/eight-schools fixture fits.

The initial legacy harness fit completed but failed on serialization because
`h5py` was missing. That failed log is retained; the failed legacy run was
repeated with its original seed/budget after the backend and roundtrip check
passed. The modern harness was not repeated. Common-method modern payloads
were reused only after asserting identical script hashes, while legacy payloads
were rerun on the unchanged frozen fixture files. No additional cross-comparison
MCMC was performed.

## Historical baseline

The following environment versions and outcomes belong to the earlier
upstream transition benchmark. They were not re-created by the current focused
compatibility run, and its taught-API sweep was not repeated.

Goal: make the **bayesian-workflow** skill teach the latest PyMC 6 / ArviZ 1.x idioms
while staying runnable on PyMC 5.x for the transition. Method: build both envs, run the
harness on each, and let breakage reveal the real divergences (not a changelog).

### Historical: Environments

| Env | PyMC | ArviZ umbrella | arviz-stats/plots | nutpie | pymc-extras | sampler output |
|-----|------|----------------|-------------------|--------|-------------|----------------|
| `baygent`  | 5.28.1 | 0.23.4 (classic) | 1.0.0 | 0.16.7  | 0.10.0 | `InferenceData` |
| `baygent6` | 6.0.1  | 1.1.0 (umbrella) | 1.1.0 | 0.16.10 | 0.12.0 | `DataTree` |

`environment-pymc6.yml` resolves PyMC 6 + the full ArviZ 1.x stack + nutpie + pymc-extras
cleanly. It also pins `netcdf4` + `h5netcdf`: a pip-only ArviZ-1 install pulls no
group-aware netcdf engine, so without them `az.from_netcdf` / `convert_to_datatree(path)`
can't read a `.nc` (the scripts' first step).

### Historical: Divergences found by running on both stacks, and the fixes

| Divergence | Symptom on PyMC 6 / ArviZ 1.x | Fix |
|---|---|---|
| `difference_ecdf_pit` relocated (arviz_plots 1.0 → arviz_stats 1.1) | `calibration_check.py` failed to **import** | import from the stable `arviz_stats.ecdf_utils` home (both stacks), with a fallback |
| `InferenceData.groups()` (method) vs `DataTree.groups` (property of paths) | LOO **silently dropped** — `"'tuple' object is not callable"`, swallowed by check_loo's except | `_group_names()` normalizes both to bare names |
| ELPDData field renames (`elpd_loo→elpd`, `p_loo→p`) | `check_loo` would `AttributeError` after the groups fix | `_loo_field()` tries each name |
| `Dataset.to_array()` removed in modern xarray | R-hat "max" perpetually `null` (both envs, latent) | reduce per-variable; empty (all-converged) → `None` |
| arviz 1.x returns **NaN** Pareto-k for degenerate points; 0.23 smoothed | a bare `NaN` would leak into the JSON report (invalid JSON) | take the max over finite k only; count non-finite as bad |
| no netcdf engine in a fresh pip ArviZ-1 env | `.nc` read/write fails entirely | add `netcdf4` + `h5netcdf` to the env |

`arviz_stats.diagnose` (the primary convergence call), `az.loo`, `az.summary`,
`az.from_netcdf`, `convert_to_datatree`, and the diagnose→check schema are unchanged.

### Historical: Cross-env equivalence gate (`evals/smoke/cross_env_equivalence.py`)

One shared idata is fed to **both** envs; the comparison is partitioned by who owns each
difference: `strict` (exact) for the safety-critical verdict, `numeric` (tolerance) for
quantities arviz estimates differently, `info` (reported, non-gating) for the
threshold-sensitive LOO Pareto-k rating.

| Fixture | convergence | calibration | result |
|---|---|---|---|
| healthy (well-behaved regression) | excellent = excellent | excellent = excellent | **identical** |
| pathological (centered eight-schools funnel) | poor = poor (divergences, same flagged params) | excellent = excellent | **strict identical** |

Documented limitation (surfaced by the pathological fixture, not hidden): arviz 0.23 and
1.x use different PSIS tail estimators, so on a divergent fit the **finite** Pareto-k
values match but 1.x marks one degenerate point `NaN` where 0.23 smoothed it — flipping
the qualitative LOO rating (excellent ↔ poor). This is upstream, not our bug; the
convergence verdict (the "don't interpret this posterior" guidance) agrees exactly, and
LOO is not trustworthy on a non-converged model anyway.

### Historical: Taught-API resolution sweep (checkpoint: every call resolves on both)

Verified identical on both envs: `arviz_stats.diagnose`, `psense_summary`,
`plot_psense_dist` / `plot_psense_quantities`, `pm.compute_log_likelihood` /
`compute_log_prior`, `pmx.marginalize` / `MarginalModel` / `fit`, `preliz`, and the
distribution families (`Censored`, `Truncated`, `OrderedLogistic`, `ZeroInflatedPoisson`,
`HurdlePoisson`, `NegativeBinomial`). The only APIs that genuinely differ are the handful
documented in SKILL.md → "Stack compatibility" (`az.plot_ppc` → `arviz_plots.plot_ppc_dist`;
`plot_trace(kind="rank_vlines")` → `plot_trace`/`plot_rank`; `summary` interval kwargs;
`sample_prior_predictive` `samples=`→`draws=`; `az.compare` `elpd_loo`→`elpd`).

### Historical: Scope

- **amortized-workflow** imports only `bayesflow` / `keras` / `numpy` — no pymc/arviz
  coupling, so it is independent of the PyMC major (its own backend env, unaffected).
- **causal-inference** stays on PyMC 5 (`baygent`): CausalPy caps `pymc<6`. Migrate when
  CausalPy ships PyMC-6 support.

### Historical: Verdict

The bayesian-workflow diagnostics scripts run on both PyMC 5 and PyMC 6 and agree on the
**safety-critical verdict** (convergence + calibration ratings, structural flags, and
`loo_computed`) for the same idata. The one place they can differ — the LOO Pareto-k
qualitative rating on a degenerate fit — is an upstream PSIS difference (arviz 1.x marks a
point's k non-finite where 0.23 smoothed it), is reported honestly on each stack, and is
immaterial because LOO is untrustworthy on a non-converged model (the convergence verdict,
which agrees, already says "don't interpret"). Gates green on both envs:
`test_reporting_harness.py` (16/16 each) and `cross_env_equivalence.py` (healthy +
pathological). Every API the skill teaches resolves on both — with the version-specific
forms (the `az.summary` interval kwargs; `var_names=` on `plot_trace`/`plot_rank` to stay
under ArviZ 1.x's subplot cap) spelled out in SKILL.md → "Stack compatibility".
