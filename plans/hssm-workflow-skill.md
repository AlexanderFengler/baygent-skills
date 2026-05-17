# `hssm-workflow` — Skill Architecture Plan

**Status:** iteration 1 — architecture only. Reference, script, eval content
is drafted in iteration 2; SKILL.md content is finalised in iteration 3.

## Purpose

A workflow skill for fitting hierarchical sequential-sampling models (DDM,
LBA, race, RL-SSM) to choice + reaction-time data using HSSM. Enforces
SSM-specific guardrails: data-shape validation, likelihood-backend selection
with LAN domain-coverage check, physically bounded priors, hierarchical RT
non-centered parameterisations, SSM-specific PPCs (quantile-probability
plots) — not raw `az.plot_ppc`.

## Scope

**In scope**

- Built-in `model=` strings (DDM family + angle / levy / ornstein / weibull
  + race + LBA + RL-SSM variants).
- Choice of likelihood backend: `analytical` (DDM family only),
  `approx_differentiable` (LAN / ONNX, default), `blackbox`.
- Parameter spec via `include=[Param(name=..., formula=..., prior=...,
  bounds=..., link=...)]` — using Bambi formulas under the hood.
- HSSM-specific helpers: `link_settings="log_logit"`, `prior_settings="safe"`,
  `p_outlier` + `lapse`, `global_formula`, `missing_data` (-999.0 sentinel),
  `deadline`, `extra_namespace`.
- Sampling via `model.sample(sampler="numpyro" | "nutpie" | "pymc" |
  "blackjax" | "laplace")`.
- RT / choice PPC via `hssm.plotting.plot_predictive`,
  `plot_quantile_probability`, `plot_model_cartoon`.
- RL-SSM (RLSSM) — same skill, with its own reference file for the RL
  likelihood machinery. (D3 — see "Decisions carried in this skill".)
- The `do_operator` for counterfactual sampling — light coverage in a
  reference, not the main path.

**Out of scope**

- Training your own LAN / NLE surrogate (LANfactory, ssm-simulators). That
  belongs in a future skill (a likelihood-surrogate companion to
  `amortized-workflow`, aligned with the ecosystem focus on neural likelihood
  surrogates). (D4.)
- Registering a fully custom non-SSM cognitive model. `register_model` is
  mentioned in references but not the central workflow.

## Triggers (frontmatter `description:` keywords)

HSSM, drift-diffusion model, DDM, sequential sampling model, SSM, response-
time modelling, RT modelling, choice-RT, two-alternative forced choice, 2AFC,
race model, leaky competing accumulator, LBA, accumulator model, full DDM,
collapsing bounds, angle model, levy model, ornstein, weibull SSM,
reinforcement-learning SSM, RL-SSM, RLSSM, ssm-simulators, LAN, likelihood
approximator network, stimulus coding, lapse rate, `p_outlier`,
`model.sample(...)`, `hssm.plotting.plot_predictive`, quantile-probability
plot.

## Composition with other skills

**Depends on `bambi-workflow` via *scoped reference reuse*** — not via
blanket "load the whole skill" delegation. The dependency is engineered to
avoid inheriting *noise* — `bambi-workflow` rules that would mislead the
agent inside an HSSM context (`family=`, `model.fit`,
`model.predict(kind=)`, `bmb.interpret.*`, `model.prior_predictive`).
Transitively, `bayesian-workflow` is brought in the same way.

### The robust dependency pattern

1. **Scoped pointer, not blanket delegation.** The `## Dependencies` block
   in `SKILL.md` names `bambi-workflow` specifically for the *formula-DSL
   reference* and *prior-spec patterns* — not for sampling, prediction, PPC,
   or interpret. Exact phrasing:

   > Depends on `bambi-workflow` for formula syntax (`y ~ x + (1|g)`) and
   > prior-spec dict patterns. When writing a `Param(formula=...)` entry,
   > consult `bambi-workflow/references/formula-syntax.md`. **Do not apply
   > other bambi-workflow rules to HSSM models — HSSM owns its own sampling,
   > prediction, prior-predictive, and interpretation surfaces under
   > different method names.** See the Hard Rules section for the explicit
   > carve-outs.

2. **Reference-level reuse.**
   `hssm-workflow/references/hierarchical-rt-models.md` links specifically to
   `bambi-workflow/references/formula-syntax.md` and
   `bambi-workflow/references/priors-in-bambi.md` (DSL-level content). It
   does *not* link to `families-and-links.md` (HSSM bypasses Bambi families)
   or `interpret-marginal-effects.md` (HSSM's PPC story is its own).
3. **Carve-outs in `hssm-workflow` Hard Rules.** Explicit `**NEVER**` rules
   enumerate the `bambi-workflow` patterns that *do not transfer*: `family=`
   kwarg, `model.fit(...)`, `model.predict(idata, kind=...)`, raw
   `model.prior_predictive(...)`, `bmb.interpret.*`. Each rule names the
   correct HSSM alternative (`model.sample(...)`,
   `model.sample_posterior_predictive(...)`, HSSM-specific simulator-backed
   prior pred, `hssm.plotting.*`).
4. **Structural support in `bambi-workflow`.** `bambi-workflow`'s
   `references/` is built (iteration 2) with the DSL-reusable vs
   Bambi-API-specific split deliberately in mind — see
   [`bambi-workflow-skill.md`](./bambi-workflow-skill.md). `hssm-workflow`
   leans on the DSL-reusable files only.
5. **Install-time decoupling.** The install instructions present
   `bambi-workflow` as *recommended but not strictly required* — the formula
   DSL is widespread enough that a competent agent will get most of it right
   from its prior. The hard-rule carve-outs are what actually prevent silent
   failure; the reference reuse is a quality boost.

Net effect: the agent loading `hssm-workflow` plus, optionally, the *named
files* from `bambi-workflow` gets the formula DSL precisely once, with no
contradictory Bambi-API rules in scope.

### Ownership split

- `bayesian-workflow` continues to own: convergence diagnostics
  (`azs.diagnose`), LOO + Pareto-k, calibration (`plot_ppc_pit`), prior
  sensitivity (`psense_summary`), descriptive-seed convention, 94% HDI
  default, save-to-disk discipline, xarray-first idiom, report template.
- `bambi-workflow` continues to own: formula-DSL syntax (reusable),
  prior-spec dict patterns (reusable), family/link decision (Bambi-only),
  auto-prior inspection workflow (Bambi-only), `model.fit` / `predict` /
  `prior_predictive` semantics (Bambi-only), `bmb.interpret.*` patterns
  (Bambi-only).
- `hssm-workflow` adds: SSM model-family selection, likelihood-backend
  selection with LAN-coverage check, SSM-specific priors and bounds,
  hierarchical RT non-centered defaults, SSM-specific PPCs (quantile-
  probability), HSSM-specific sampling and PPC surfaces, RL-SSM likelihood
  compilation, lapse / missing-data / deadline modes.

## Workflow overview (numbered, ~10 steps)

1. **Formulate** — Which cognitive model fits the task? See
   `references/model-zoo.md`. ⚠️ **ASK USER TO CONFIRM** model choice
   before coding (similar to `causal-inference`'s confirmation checkpoints).
2. **Validate data shape** — Required columns: `rt` (>0), `response` (∈
   declared `choices`). Standardise response coding before constructing the
   model.
3. **Pick the likelihood backend** — `analytical` (DDM family only),
   `approx_differentiable` (LAN, default; **must check domain coverage**),
   `blackbox` (last resort). See `references/likelihood-backends.md`.
4. **Define per-parameter spec** — Build `include=[Param(...)]` list with
   formula + prior + bounds. Defer formula syntax to `bambi-workflow`'s
   `formula-syntax.md`. See `references/priors-and-bounds.md`.
5. **Construct model** — `hssm.HSSM(data=..., model=..., choices=...,
   include=..., loglik_kind=..., p_outlier=..., link_settings=...,
   prior_settings="safe")`. Inspect with `model.graph()` /
   `hssm.show_defaults`.
6. **Prior predictive check** — Use HSSM's simulator-backed prior predictive
   so PPCs are RT/choice-aware (not raw `pm.sample_prior_predictive` which
   only checks the linear predictor).
7. **Sample** — `model.sample(sampler=...)` — default `"numpyro"` for LAN
   models, `"nutpie"` available for `approx_differentiable`, fall back to
   `"pymc"` for `blackbox`. Save trace immediately.
8. **Diagnose** — Delegate convergence / ESS / divergences / calibration to
   `bayesian-workflow`. Add HSSM-specific LAN domain-coverage post-check
   (was the posterior support inside the LAN training domain?).
9. **SSM-specific posterior predictive checks** —
   `model.sample_posterior_predictive(idata)` then
   `hssm.plotting.plot_predictive` and `plot_quantile_probability`. Raw
   `az.plot_ppc` alone is insufficient.
10. **Report** — Reuse `bayesian-workflow` report template + an SSM-specific
    addendum (model string, likelihood backend, bounds table, quantile-prob
    plot, per-subject parameter table).

## Hard rules (categories; iteration 2 fills in rationales + examples)

Style: `**MUST**` / `**NEVER**` prefixes (matches `amortized-workflow`; see
D5).

**SSM-specific guardrails**

- **MUST validate `response` coding before constructing the model.** Default
  `choices=[-1, 1]` for 2-choice; `{0, 1}` data silently mis-fits.
- **MUST pick `loglik_kind` deliberately.** Default is silently
  `approx_differentiable` for non-analytical models — this hides the
  LAN-domain-coverage trap.
- **MUST run a LAN domain-coverage check before trusting any LAN-backed
  fit.** Verify that prior support (and posterior support, post hoc) lies
  inside the LAN training domain declared in `model_config.bounds`.
- **MUST use bounded priors for `a`, `t`, `z`, `p_outlier`.** Unbounded
  Normals on threshold / non-decision-time produce silent failures and NaN
  logps.
- **MUST use non-centered parameterisations for hierarchical SSM models.**
  Subject-level variance in `v` / `a` / `t` is large and centered
  parameterisations diverge. HSSM's `prior_settings="safe"` and
  `link_settings="log_logit"` help but do not replace this rule.
- **MUST run SSM-specific PPCs.** `hssm.plotting.plot_predictive` and
  `plot_quantile_probability` are non-negotiable. Raw `az.plot_ppc` does not
  see the joint (choice, RT) structure.
- **MUST save `model.traces` immediately after sampling.** HSSM fits are
  10–60× slower than typical PyMC; checkpoint after every successful sample.
- **For RL-SSM, MUST check balanced-panel structure**
  (`hssm.check_data_for_rl`) before constructing the model.
- **For missing-data / deadline modes, MUST declare them explicitly.** The
  `-999.0` sentinel is reserved and silent if `missing_data=False`.

**Scoped-dependency carve-outs (from the `bambi-workflow` dependency)**

These rules exist to prevent the agent from importing `bambi-workflow`
patterns that do not transfer to HSSM:

- **NEVER pass `family=` to `hssm.HSSM(...)`.** HSSM owns the likelihood
  spec via `model=` and `loglik_kind=`. The `family=` kwarg is meaningless
  in HSSM context.
- **NEVER call `model.fit(...)` on an HSSM model.** Use `model.sample(...)`.
- **NEVER call `model.predict(idata, kind=...)` on an HSSM model.** Use
  `model.sample_posterior_predictive(idata)` and the HSSM plotting API.
- **NEVER call raw `model.prior_predictive(...)` for SSM models.** Use
  HSSM's simulator-backed prior-pred flow so PPCs are RT/choice-aware.
- **NEVER apply `bmb.interpret.*` directly to an HSSM model.** Marginal
  effects on RT/choice quantities are not what `bmb.interpret` is designed
  for; use HSSM's PPC + quantile-prob tooling and report per-condition
  posterior summaries from `idata`.

## References (6 files; one-liner each)

- `model-zoo.md` — Which built-in `model=` string for which task. Table:
  task type (DDM-like, race, LBA, with collapsing bounds, with RL update) →
  model string → likelihood backends available → typical use case. The
  `do_operator` (counterfactual) flow lives here as one section.
- `likelihood-backends.md` — `analytical` vs `approx_differentiable` vs
  `blackbox`. Speed / accuracy / differentiability / sampler-compatibility
  trade-offs. LAN training-domain bounds and how to inspect them
  (`hssm.show_defaults(model, kind)`).
- `priors-and-bounds.md` — Physical bounds per SSM parameter (`v`, `a`,
  `t`, `z`, `p_outlier`, `theta` / `alpha` for RL). Default priors. When to
  override.
- `hierarchical-rt-models.md` — Subject-level regression formulas via the
  `Param.formula` + `prior` pattern. Why non-centered is mandatory. Common
  task-design encodings (stim-coding, trial-wise covariates). Links to
  `bambi-workflow/references/formula-syntax.md` for DSL details (scoped
  reuse — see Composition section above).
- `ssm-specific-diagnostics.md` — Beyond standard diagnostics: LAN coverage
  post-check, quantile-probability plots, defective CDFs by choice,
  response-proportion calibration, choice-by-RT-quantile tables.
- `rl-ssm.md` — RL-SSM specifics: data structure (balanced panel),
  parameters (learning rate `alpha`, inverse-temperature `theta`),
  likelihood builders (`make_rl_logp_func` / `make_rl_logp_op`), reference
  RL-DDM formulation.

## Scripts (2 files; one-liner each)

- `check_lan_coverage.py` — Loads `idata` + the HSSM model config, checks
  that every parameter's posterior support is inside the LAN training-
  domain bounds. Outputs JSON:
  `{param: {in_bounds: bool, frac_in: float, recommendation: str}}` with a
  `next_steps` list when violations are found (e.g. "switch to `blackbox`",
  "widen LAN training domain via custom config").
- `ssm_ppc_report.py` — Runs the SSM-specific PPC battery
  (`plot_predictive`, `plot_quantile_probability`, response-proportion
  calibration) and emits standardised figure files + a structured markdown
  table for the report.

## Evals

### Design notes — four-layer anatomy

baygent-skills evals decompose into four checkable layers — *trigger*,
*workflow*, *hard rules*, *scripts* — each measured separately. Applied
to `hssm-workflow`:

| Layer | Artifact | This skill's specifics |
|---|---|---|
| Trigger | `description:` keywords + `trigger_eval_set.json` | Fires on RT / DDM / SSM / RL-SSM asks; does NOT fire on raw-PyMC RT models or pure Bambi GLMMs |
| Workflow | Numbered steps + `references/` | Scenario assertions cite which workflow steps executed (data validation, backend choice, LAN coverage check, SSM PPCs) |
| Hard rules | `MUST` / `NEVER` bullets | One scenario assertion per rule; carve-outs become explicit `NEVER` assertions (see below) |
| Scripts | `check_lan_coverage.py`, `ssm_ppc_report.py` | Assertions check the script was *called*, not just present |

`hssm-workflow` is primarily a **silent-failure prevention** layer (in
contrast with `bambi-workflow`'s quality-and-ergonomics emphasis). The
high-stakes rules — LAN domain coverage, response coding, hierarchical
non-centered parameterisation, SSM-specific PPCs — protect against
results that *look* converged but are quantitatively wrong if missed. At
least one scenario (`angle-lan-coverage-violation`) is designed as an
adversarial pressure test that fires only when those guards are
intentionally circumvented by a default workflow.

### Carve-out enforcement assertions

The scoped-reference-reuse dependency on `bambi-workflow` (see
"Composition with other skills") is enforced by **explicit `NEVER`
assertions** in every scenario where the agent could be tempted to import
a Bambi pattern that does not transfer to HSSM:

- *"Does NOT pass `family=` to `hssm.HSSM(...)`."*
- *"Calls `model.sample(...)` (not `model.fit(...)`)."*
- *"Uses `model.sample_posterior_predictive(idata)` (not
  `model.predict(idata, kind=...)`)."*
- *"Does NOT call `bmb.interpret.predictions` / `comparisons` / `slopes`
  on the HSSM model."*
- *"For prior-predictive checks, uses HSSM's simulator-backed prior
  predictive (not raw `model.prior_predictive(...)`)."*

These are scenario assertions, not separate trigger evals — they fire
inside every scenario that exercises the model-construction path. The
`ddm-hierarchical-condition` scenario is the primary carrier because
that's the most natural place an agent would reach for Bambi APIs (a
GLMM-flavoured formula with hierarchical structure).

### Runner / grader externality

The eval runner and LLM grader live **outside this repo**. Only authored
fixtures (`eval_metadata.json`, `trigger_eval_set.json`) and committed
results (`grading.json`, `timing.json`, `outputs/`, `benchmark.json`) are
versioned. Consequence: every assertion MUST be checkable from the static
output artifact — code, comments, generated markdown — *without re-running
the code*. Assertions like *"recovers DDM parameters within 5%"* are out;
assertions like *"Calls `check_lan_coverage.py` (or
`hssm.show_defaults(model, kind)`) before sampling"* are in.

### Per-scenario file layout

```text
evals/hssm-workflow/iteration-N/<scenario>/
  eval_metadata.json
  with_skill/{grading.json, timing.json, outputs/}
  without_skill/{grading.json, timing.json, outputs/}
```

### Benchmark schema target

Match the **iteration-3** `benchmark.json` shape (the most recent in the
repo, used by `review.html`):

```jsonc
{
  "skill_name": "hssm-workflow",
  "iteration": 1,
  "configs": [
    { "name": "with_skill", "evals": [ {
        "eval_name": "...", "pass_rate": 0.93, "passed": 14, "total": 15,
        "expectations": [
          { "text": "...", "passed": true, "evidence": "Line N: ..." }
        ]
    } ] },
    { "name": "without_skill", "evals": [ /* same shape */ ] }
  ],
  "run_summary": {
    "with_skill":    { "pass_rate": { "mean": ..., "stddev": ..., "min": ..., "max": ... } },
    "without_skill": { "pass_rate": { "mean": ..., "stddev": ..., "min": ..., "max": ... } }
  },
  "notes": [ "carry-over baseline caveats, executor permission artefact warnings, ..." ]
}
```

### Scenario seeds (6; assertion lists drafted in iteration 2)

Each scenario will carry **8–12 assertions** spread across the four-layer
anatomy. Annotations below identify which layers each scenario stresses
hardest.

1. `ddm-flat` — Non-hierarchical DDM on synthetic 2AFC data. *Baseline
   case; assertions cover backend selection, response coding,
   SSM-specific PPC. Light carve-out coverage (sampling + PPC API).*
2. `ddm-hierarchical-condition` — `v ~ condition + (1|subj_id)`,
   `a ~ 1 + (1|subj_id)`. ***Primary carrier of the carve-out enforcement
   assertions*** (formula DSL meets sampling/PPC/interpret API
   boundaries). Also tests non-centered parameterisation.
3. `angle-lan-coverage-violation` — Angle model with prior support
   outside the LAN training domain. ***Adversarial silent-failure
   pressure test:*** agent must run `check_lan_coverage.py` (or
   equivalent) and propose a fix (switch to `blackbox`, tighten priors,
   or document the override).
4. `lba-race` — LBA / racing-accumulator model. *Tests multi-alternative
   choice handling (`choices=[0, 1, 2, 3]`) and 3+ choice coding
   validation.*
5. `rl-ssm-balanced` — RL-DDM on balanced-panel bandit data. *Tests
   `check_data_for_rl` invocation and RL-likelihood compilation
   (`make_rl_logp_op`).*
6. `lapse-stim-coding` — Stim-coded DDM with `p_outlier` regression and
   `link_settings="log_logit"`. *Tests stim-coding `gen_logit` link,
   `p_outlier` regression formula, and the lapse-rate report section.*

### Trigger-eval seed queries (~10 true / ~15 false)

**True triggers (samples)**

- *"Fit a hierarchical DDM to my 2AFC data — I have RTs, choices,
  condition labels, subject IDs."*
- *"Build a racing accumulator model for a 4-alternative perceptual
  decision task."*
- *"I want to fit an RL-DDM to bandit data with learning rate and inverse
  temperature."*
- *"The angle model is sampling slowly and I'm getting weird divergences."*

**False triggers — adjacent-domain adversaries**

Drawn from carve-out boundaries — these are the prompts where the trigger
`description:` must distinguish HSSM from neighbour skills:

- *"Build a hierarchical Bayesian regression on my RT data — y =
  log(RT), x = condition, group = subject."* → `bambi-workflow`, not
  HSSM. (Closest adversary: looks like RT data but no SSM structure;
  the user wants a log-Normal GLMM.)
- *"Write a PyMC model with a custom likelihood for response times — I
  have my own functional form."* → `bayesian-workflow`, not HSSM.
- *"Train a neural likelihood approximator for my DDM simulator."* →
  future surrogate-training skill (or `amortized-workflow` as closest
  fit today).
- *"I want to estimate the causal effect of a stimulant on reaction
  times."* → `causal-inference`.
- *"Help me plot the reaction-time distribution and compute quantiles
  from my CSV."* → out of scope (descriptive stats, not modelling).
- (~10 more drafted in iteration 2)

## Decisions carried in this skill

- **D1 (resolved): scoped reference reuse, not blanket delegation.** See
  the "Composition with other skills" section above for the full pattern.
- **D3 (resolved):** RL-SSM lives inside `hssm-workflow` as one reference
  file (`rl-ssm.md`) plus one eval scenario.
- **D4 (resolved):** LANfactory / NLE surrogate training is out of scope.
  Obvious next skill in the amortized family after these two ship.
- **D5 (resolved):** Hard rules use `**MUST**` / `**NEVER**` prefixes.

## Iteration-2 follow-ups

- Flesh out each Hard Rule with a one-paragraph rationale + example.
- Draft each reference file's TOC + first prose paragraph.
- Draft each script's docstring + argparse signature.
- Write `eval_metadata.json` for at least 1 scenario — strong candidate is
  `angle-lan-coverage-violation`, which uniquely exercises the LAN
  domain-coverage rule.
- Sketch the SKILL.md frontmatter `description:` paragraph in full.
- Confirm `bambi-workflow`'s DSL-vs-API references split has landed so the
  scoped-reuse pointers resolve.
- Consider an extra eval that explicitly tests the scoped-dependency
  carve-outs (an agent that tries `model.fit` on an HSSM model and is
  expected to course-correct).
- Update `environment.yml` with HSSM-stack pins (`hssm>=…`,
  `ssm-simulators`) and any HuggingFace LAN download dependencies.
