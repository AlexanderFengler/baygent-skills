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
│   ├── hssm-workflow/             # Runtime evidence; agent/trigger scenarios still unexecuted
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
- Bambi M1 is tested separately on Bambi 0.21.0 / PyMC 6.3.2 / ArviZ 1.3.0. Its [example README](examples/bambi-workflow/README.md) includes a tested uv alternative and exact validation artifacts. This does not extend the Bambi skill to PyMC 5.
- HSSM's flat analytical-DDM workflow passed local runtime validation on HSSM 0.5.0 / Bambi 0.19 / PyMC 6.1 / ArviZ 1.2 / NumPy 2.4.6 with Python 3.12.13/macOS arm64. The pins in `environment-hssm.yml` were installed via uv with system Graphviz; the conda recipe itself was not tested. See the [runtime evidence](evals/hssm-workflow/iteration-1/runtime/README.md) for exact resolved requirements. Do not infer compatibility from Bambi M1's environment.
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
- **Cross-env equivalence gate** (`evals/smoke/cross_env_equivalence.py`): feeds one shared idata to both `baygent` (PyMC 5) and `baygent6` (PyMC 6) and asserts identical user-facing diagnostics/ratings across a healthy and a pathological fixture — this is the dual-compat guarantee. Run: `python evals/smoke/cross_env_equivalence.py` (needs conda on PATH; skips loudly if `baygent6` is absent, fails with `--require-both`). Run after any change to the bayesian-workflow `scripts/`.

- **Bambi integration smoke:** in the modern environment, install `pytest` and run `python -m pytest evals/smoke/test_bambi_workflow.py`. Both marimo notebooks execute with small budgets and exercise the unchanged shared diagnostics/report pipeline. Numerical health is not a smoke assertion. Use `marimo check --strict examples/bambi-workflow/gaussian_regression.py examples/bambi-workflow/hierarchical_bernoulli.py` for notebook structure.
- **HSSM runtime validated; skill behavior pending:** deterministic adapter/CLI, notebook-domain/gate, reporting and independent numerical-density suites passed, followed by a full notebook export and saved-artifact checks. Exact commands/results and native release limitations are in the [runtime record](evals/hssm-workflow/iteration-1/runtime/README.md). The duplicate full-fit integration in `evals/smoke/test_hssm_workflow.py` retains its `BAYGENT_TEST_HSSM=1` opt-in; the recorded validation checked the export's saved artifacts instead. Opening the notebook defaults to preview, with separate simulation/prior and fitting buttons. Paired model-service and trigger evaluations remain unexecuted; one teaching run does not establish recovery, joint calibration or broader platform support.
