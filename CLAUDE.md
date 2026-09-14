# CLAUDE.md

## Project overview

**baygent-skills** is a collection of Agent Skills for Bayesian modeling, causal inference, and probabilistic thinking. Each skill is a self-contained subfolder following the [Agent Skills spec](https://agentskills.io/specification).

## Repo structure

```
baygent-skills/
├── bayesian-workflow/          # Shipped skill (v1.5, dual PyMC 5/6)
│   ├── SKILL.md                # Main workflow instructions
│   ├── references/             # Detailed reference docs (priors, diagnostics, sensitivity, reporting)
│   └── scripts/                # diagnose_model.py, calibration_check.py, check_diagnostics.py
├── bambi-workflow/             # Bambi 0.21 / PyMC 6; depends directly on bayesian-workflow
│   ├── SKILL.md
│   └── references/             # Formulas, families, priors, interpretation/reporting
├── examples/bambi-workflow/    # Two marimo notebooks; example-only report assembly
├── hssm-workflow/              # Flat analytical DDM; HSSM 0.5 runtime validated
│   ├── SKILL.md                # Direct bayesian-workflow dependency; Bambi skill optional
│   ├── references/             # Data, parameter support, native predictions and reporting
│   └── scripts/                # Strict paired RT/choice to scalar marginal PPC adapter
├── examples/hssm-workflow/     # Gated marimo notebook; full run and report recorded
├── causal-inference/           # Shipped skill (v1.2)
│   ├── SKILL.md                # Main workflow instructions (depends on bayesian-workflow)
│   ├── references/             # DAGs, quasi-experiments, structural models, refutation, reporting
│   └── scripts/                # check_refutation.py (calibrated causal language harness)
├── amortized-workflow/         # Shipped skill (v2.0, co-authored with Stefan Radev)
│   ├── SKILL.md                # Amortized Bayesian workflow with BayesFlow
│   ├── references/             # Adapters, conditioning logic, model sizes, reporting
│   └── scripts/                # check_diagnostics.py, inspect_training.py
├── evals/                         # Eval scenarios and benchmarks
│   ├── bayesian-workflow/         # 6 scenarios, 3 iterations
│   ├── bambi-workflow/            # 2 scenarios + trigger set + runtime evidence
│   ├── hssm-workflow/             # Runtime evidence; no completed agent/trigger benchmarks
│   ├── causal-inference/          # 6 scenarios
│   ├── amortized-workflow/        # 6 scenarios + trigger set + benchmark results
│   └── smoke/                     # Reporting-harness smoke test + cross-env (PyMC 5/6) equivalence gate
├── environment.yml             # Mamba/conda env (env name: baygent, PyMC 5)
├── environment-pymc6.yml       # Mamba/conda env (env name: baygent6, PyMC 6 / ArviZ 1.x)
├── environment-hssm.yml        # Pip requirements tested via uv; conda recipe itself untested
├── LICENSE                     # MIT
└── CLAUDE.md                   # This file
```

## Development conventions

### Python environment
- **Two envs during the PyMC 5 → 6 transition:**
  - `baygent` (PyMC 5.28 / arviz 0.23 + arviz-stats/plots 1.0) — `environment.yml`. The **causal-inference** skill is pinned here (CausalPy caps `pymc<6`).
  - `baygent6` (PyMC 6.0.1 / arviz 1.x + pymc-extras 0.12) — `environment-pymc6.yml`. The **bayesian-workflow** scripts run on **both**; that dual run is the compatibility guarantee.
- Run `conda run -n baygent python <script>` (or `-n baygent6`). Recreate with `mamba env create -f environment.yml` / `mamba env create -f environment-pymc6.yml`
- The Bambi workflow is tested separately on Bambi 0.21.0 / PyMC 6.3.2 / ArviZ 1.3.0. Its [example README](examples/bambi-workflow/README.md) includes a tested uv alternative and exact validation artifacts. This does not extend the Bambi skill to PyMC 5.
- HSSM's flat analytical-DDM workflow passed local runtime validation on HSSM 0.5.0 / Bambi 0.19 / PyMC 6.1 / ArviZ 1.2 / NumPy 2.4.6 with Python 3.12.13/macOS arm64. The pins in `environment-hssm.yml` were installed via uv with system Graphviz; the conda recipe itself was not tested. See the [runtime evidence](evals/hssm-workflow/iteration-1/runtime/README.md) for exact resolved requirements. Do not infer compatibility from The Bambi workflow's environment.
- Never use system Python

### Skill structure
Every skill follows the Agent Skills spec:
- `SKILL.md` with YAML frontmatter (name, description, license, metadata)
- `references/` for detailed docs (plural, not `reference/`)
- `scripts/` for utility scripts
- Description must be agent-neutral (no "Claude"-specific language)

### Skill authoring heuristics
- **The `description` is the only thing the agent sees when deciding to load the skill.** Lead with what it does, then an explicit "Use when …" trigger sentence listing concrete keywords/situations. This is the single highest-leverage field.
- **Keep `SKILL.md` lean; push depth to `references/`.** The main file is the always-loaded budget — progressive disclosure, link out for detail.
- **Prefer a script over generated code for deterministic, repeated, or error-prone operations.** Scripts save tokens and run consistently; reserve inline code for one-off, model-specific logic.

### Code style
- PyMC 5+ syntax with coords and dims; the bayesian-workflow scripts are dual-compatible with PyMC 5 and 6 via capability detection (`hasattr` / try-import / field-name fallbacks), not version branches
- nutpie sampler by default; the HSSM analytical-DDM example explicitly selects HSSM's `pymc` sampler for its own scoped baseline
- Descriptive seeds: `RANDOM_SEED = sum(map(ord, "analysis-name"))`
- xarray-first for InferenceData operations

### Testing
- All evals live in `evals/`, grouped by skill; authored metadata is distinct from executed evidence
- Benchmark target: 100% with skill vs ~90% without
- Each eval has: `eval_metadata.json` (prompt + assertions), `with_skill/` and `without_skill/` outputs + grading
- **Reporting harness smoke test** (`evals/smoke/test_reporting_harness.py`): runs the bayesian diagnostics pipeline (`diagnose_model → calibration_check → check_diagnostics`) end-to-end on a tiny model and the causal `check_refutation` harness on fixtures. Run after any change to the `scripts/` of either skill — on **both** envs: `conda run -n baygent python evals/smoke/test_reporting_harness.py` and `conda run -n baygent6 python evals/smoke/test_reporting_harness.py`. Guards JSON-serializability, the diagnose→check schema contract, and refutation metric direction.
- **Cross-env equivalence gate** (`evals/smoke/cross_env_equivalence.py`): feeds the same fixture to PyMC 5 and PyMC 6 and explicitly selects `method="envelope"` on both stacks. Strict convergence/calibration verdicts, structural flags, summaries and non-LOO actions must agree; numeric tolerances are unchanged. Automatic modern calibration uses `pot_c` when available, so automatic-default ratings are tested separately and are not promised to match a legacy envelope. Run: `python evals/smoke/cross_env_equivalence.py` (needs conda on PATH; skips loudly if `baygent6` is absent, fails with `--require-both`). With uv environments, call the worker and comparison functions using their explicit interpreters. See the [compatibility record](evals/benchmark-pymc6-dual-compat.md) for actual environments, fixture provenance and results. Run after changes to the Bayesian scripts. The 2026-09-14 focused check passed the unchanged 16-check harness in both PyMC 5/ArviZ stats 1.0 and PyMC 6/ArviZ stats 1.2 environments, plus common-envelope healthy/pathological equivalence. It did not repeat the historical taught-API sweep or promise equal automatic-default ratings.

- **Bambi integration smoke:** in the modern environment, install `pytest` and run `python -m pytest evals/smoke/test_bambi_workflow.py`. Both marimo notebooks execute with small budgets and exercise the unchanged shared diagnostics/report pipeline. Numerical health is not a smoke assertion. Use `marimo check --strict examples/bambi-workflow/gaussian_regression.py examples/bambi-workflow/hierarchical_bernoulli.py` for notebook structure.
- **HSSM workflow checks:** run `python -m pytest evals/smoke/test_hssm*.py` in the separate HSSM environment. Tests cover the installed adapter/CLI, notebook domain summaries and execution gates, report failure handling, and compact checks of the example’s priors/support and joint-likelihood handoff. `BAYGENT_TEST_HSSM=1` opts into the full-fit integration; ordinary checks do not fit. The broader dependency-certification suite and saved-export audit remain on the development fork, with their historical results linked from the [runtime record](evals/hssm-workflow/iteration-1/runtime/README.md). Native coordinate guards remain tested locally. Behavioral and trigger evaluations are reserved for a follow-up PR; one teaching run does not establish recovery or joint calibration.
