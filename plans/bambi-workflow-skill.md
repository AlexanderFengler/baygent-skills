# `bambi-workflow` — Skill Architecture Plan

**Status:** iteration 1 — architecture only. Reference, script, eval content
is drafted in iteration 2; SKILL.md content is finalised in iteration 3.

## Purpose

A workflow skill that takes the user from "I want to fit a GLM/GLMM" to
"I have a fitted Bambi model with valid diagnostics, calibrated posteriors,
and interpretable marginal effects." It does **not** re-derive PyMC mechanics
— those are delegated upstream to `bayesian-workflow`.

## Scope

**In scope**

- Wilkinson-formula model specification (`y ~ x + (1|g)`, group-specific terms,
  interactions, `C()`, offsets, `0 + x`).
- Built-in families (29 univariate + Multinomial / DirichletMultinomial) and
  custom families via `bmb.Family(name, likelihood, link_dict)`.
- Auto-priors: when to trust them, when to inspect via `model.build()` →
  `model.components[...].common_terms[...].prior`, when to override.
- Distributional models (`Formula("y ~ x", "sigma ~ z")` etc.) — when needed
  and the parameter-name-must-match-family gotcha.
- HSGP and splines (`hsgp(x, m=20, c=1.5)`, `bs(x)`, `poly(x)`) — light
  coverage, one reference file.
- Inference backends: `inference_method="pymc"` (default), `"nutpie"`,
  `"nuts_numpyro"`, `"nuts_blackjax"`, `"vi"`, `"laplace"`.
- `predict(idata, kind="response_params"|"response")` — semantics and when
  to use each.
- `bmb.interpret.*` (`predictions`, `comparisons`, `slopes` + plotting
  variants) as the canonical interpretation surface. (D2 — see "Decisions
  carried in this skill".)

**Out of scope**

- Causal interpretation — let `causal-inference` handle.
- Time-series / state-space models — Bambi is not the right tool.
- BART / mixtures / structured time-series priors — out of Bambi's scope.

## Triggers (frontmatter `description:` keywords)

Bambi, `bmb.Model`, Wilkinson formula, `y ~ x`, `(1|group)`, group-specific
terms, GLM, GLMM, hierarchical regression, multilevel regression,
mixed-effects, distributional model, location-scale model, HSGP, spline
regression, `bs()`, `hsgp()`, marginal effects, average marginal effect,
conditional effects, `bmb.interpret`, `predict`, prior auto-scaling, Bambi
families, Bambi priors.

## Composition with other skills

**Depends on `bayesian-workflow`** — declared in three places per repo
convention:

1. `SKILL.md` `## Dependencies` block (with detection + install bash).
2. Workflow cross-references where applicable — steps that produce/use
   `InferenceData`, run diagnostics, run calibration, run sensitivity, write
   the report are delegated upstream.
3. `README.md` "depends on bayesian-workflow — install both" note.

**Ownership split**

- `bayesian-workflow` continues to own: convergence diagnostics
  (`azs.diagnose`), LOO + Pareto-k, calibration (`plot_ppc_pit` / LOO-PIT),
  prior sensitivity (`psense_summary`), report template, descriptive-seed
  convention, 94% HDI default, save-to-disk discipline, xarray-first idiom.
- `bambi-workflow` adds: formula DSL, family selection, auto-prior inspection
  + override, distributional models, splines / HSGP, marginal effects via
  `interpret`.

**Internal structural choice (relevant to `hssm-workflow`):** this skill's
`references/` is deliberately split into DSL-level files (reusable by
downstream skills) vs Bambi-API-level files (not transferable):

- DSL-level (reusable): `formula-syntax.md`, `priors-in-bambi.md`.
- Bambi-API-level (not reused): `families-and-links.md`,
  `interpret-marginal-effects.md`, `splines-and-hsgp.md`.

`hssm-workflow` is designed to consume only the DSL-level files. See
[`hssm-workflow-skill.md`](./hssm-workflow-skill.md) for the scoped-reference
pattern that exploits this split.

## Workflow overview (numbered, ~9 steps)

1. **Formulate** — Generative story; which family / link suits the outcome
   (binary→Bernoulli, count→Poisson/NegBin, ordinal→Cumulative, etc.).
2. **Specify the formula** — Plain `y ~ x`, group-specific `(x|g)`,
   distributional (`Formula("y ~ x", "sigma ~ z")`), splines/HSGP if needed.
   See `references/formula-syntax.md`.
3. **Pick the family + link** — Built-in families and links; defaults vs
   explicit override. See `references/families-and-links.md`.
4. **Build + inspect auto-priors** — `model.build()` then inspect
   `model.components` to see the auto-scaled priors. Decide whether to
   override per-term via `priors={...}`. See `references/priors-in-bambi.md`.
5. **Prior predictive check** — `model.prior_predictive()` — same mandatory
   step as `bayesian-workflow`, different API surface.
6. **Fit** — `model.fit(inference_method=...)`. Pick backend deliberately
   (defaults to `"pymc"` NUTS; prefer `"nuts_numpyro"` or `"nutpie"` for
   speed when JAX is available).
7. **Diagnose** — Delegate to `bayesian-workflow` (`azs.diagnose`, posterior-
   predictive, LOO, calibration, sensitivity).
8. **Predict and interpret** — `model.predict(idata, kind=...)` for posterior-
   predictive; `bmb.interpret.predictions/comparisons/slopes` for the
   user-facing answer. See `references/interpret-marginal-effects.md`.
9. **Report** — Delegate report template to `bayesian-workflow`, with a
   Bambi-specific addendum table (formula, family/link, priors table,
   `interpret`-derived contrasts).

## Hard rules (categories; iteration 2 fills in rationales + examples)

Style: `**MUST**` / `**NEVER**` prefixes (matches `amortized-workflow`; see
D5).

- **MUST `model.build()` before inspecting priors.** Components / priors are
  not populated until build runs.
- **MUST inspect auto-priors before fit** and document any overrides.
- **MUST run `model.prior_predictive()`** before fit.
- **For distributional models, MUST explicitly model every auxiliary
  parameter.** Auxiliary parameters with no formula silently fall back to
  default flat priors.
- **For categorical predictors, MUST use `C(x)` or pass `categorical=[...]`**
  to the Model constructor; do not rely on auto-detection from string dtypes.
- **For multinomial / categorical families, MUST set `family="categorical"`
  or `family="multinomial"` explicitly**; do not let `family=` auto-detect.
- **NEVER report posterior summaries alone for predictor effects on
  non-Gaussian families.** Use `bmb.interpret.comparisons` or `slopes` so the
  effect is reported on the response (probability / count / etc.) scale, not
  the linear-predictor scale.
- **NEVER mix `priors=` keys for `"common"` / `"group_specific"` aliases with
  term-specific keys** without verifying the resolution order.
- **MUST delegate** convergence / calibration / sensitivity / LOO to
  `bayesian-workflow`.

## References (5 files; one-liner each)

- `formula-syntax.md` — Wilkinson DSL: `y ~ x`, group-specific terms,
  interactions, `C()`, intercept removal, `offset()`, distributional
  formulas. DSL-level, reusable.
- `families-and-links.md` — Full family table with the natural link, when to
  use Hurdle vs ZeroInflated, ordinal families' multi-formula restriction.
- `priors-in-bambi.md` — How `auto_scale` works (the 2.5×SD rule for slopes,
  the y-centred intercept rule), how to inspect and override, group-level
  scale priors. Mostly reusable.
- `interpret-marginal-effects.md` — `predictions`, `comparisons`, `slopes`
  and their `plot_*` variants; conditional vs marginal vs average effects.
- `splines-and-hsgp.md` — `bs()`, `hsgp(x, m=..., c=...)`, prior on
  marginal-σ, when each is appropriate.

## Scripts (2 files; one-liner each)

- `inspect_priors.py` — Loads a built model (`.pkl` or rebuilt from
  formula + data), dumps the full auto-prior table (term → distribution +
  params) to JSON, flags terms whose auto-priors look suspect by simple
  heuristics (e.g. σ much larger than response SD).
- `marginal_effects_report.py` — Wraps `interpret.predictions /
  comparisons / slopes` calls into a structured JSON + markdown table for
  the report, with 94% HDI by default and named contrasts.

## Evals

### Design notes — four-layer anatomy

baygent-skills evals decompose into four checkable layers — *trigger*,
*workflow*, *hard rules*, *scripts* — each measured separately. Applied
to `bambi-workflow`:

| Layer | Artifact | This skill's specifics |
|---|---|---|
| Trigger | `description:` keywords + `trigger_eval_set.json` | Fires on Wilkinson formulas / GLMM / `bmb.interpret` requests; does NOT fire on raw-PyMC custom-likelihood asks or HSSM RT models |
| Workflow | Numbered steps + `references/` | Scenario assertions cite which workflow steps executed (`model.build()`, `model.prior_predictive()`, `fit`, `interpret`) |
| Hard rules | `MUST` / `NEVER` bullets | One scenario assertion per rule, line-cited evidence in `grading.json` |
| Scripts | `inspect_priors.py`, `marginal_effects_report.py` | Assertions check the script was *used*, not just present |

`bambi-workflow` is primarily a **quality and ergonomics** layer (in
contrast with `hssm-workflow`'s silent-failure prevention emphasis).
Assertions therefore mostly check *idiomatic API use*: picking the right
family, declaring distributional models explicitly, reporting marginal
effects on the response scale via `interpret` rather than raw posterior
summaries on the linear-predictor scale.

### Runner / grader externality

The eval runner and LLM grader live **outside this repo**. Only authored
fixtures (`eval_metadata.json`, `trigger_eval_set.json`) and committed
results (`grading.json`, `timing.json`, `outputs/`, `benchmark.json`) are
versioned. Consequence: every assertion MUST be checkable from the static
output artifact — code, comments, generated markdown — *without re-running
the code*. Assertions like *"recovers parameters within 5%"* are out;
assertions like *"Calls `model.prior_predictive(...)` before
`model.fit(...)`"* are in.

### Per-scenario file layout

```text
evals/bambi-workflow/iteration-N/<scenario>/
  eval_metadata.json
  with_skill/{grading.json, timing.json, outputs/}
  without_skill/{grading.json, timing.json, outputs/}
```

### Benchmark schema target

Match the **iteration-3** `benchmark.json` shape (the most recent in the
repo, used by `review.html`):

```jsonc
{
  "skill_name": "bambi-workflow",
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
anatomy. Each scenario pressure-tests one or two hard rules.

1. `glm-poisson-injuries` — Poisson regression with offset and one
   covariate. *Tests family choice, `offset()` term, prior inspection
   workflow.*
2. `glmm-bernoulli-survey` — Hierarchical Bernoulli with `(1|region)`.
   *Tests group-specific term, auto-prior on group-level scale,
   non-Gaussian-family-requires-`interpret`-rule.*
3. `distributional-gaussian-heteroskedastic` —
   `Formula("y ~ x", "sigma ~ z")`. *Pressure-tests the "must model every
   auxiliary parameter" rule.*
4. `ordinal-cumulative-likert` — Cumulative family on Likert data.
   *Tests ordinal-family-forbids-multi-formula handling and intercept
   auto-removal.*
5. `hsgp-time-trend` — `y ~ hsgp(t, m=15, c=1.5) + group`. *Tests
   HSGP-prior-pred check and basis-size choice.*
6. `marginal-effects-comparison` — Logistic GLMM where the user asks
   for an effect. *Tests `interpret.comparisons` on probability scale,
   not raw coefficient reporting.*

### Trigger-eval seed queries (~10 true / ~15 false)

**True triggers (samples)**

- *"I have survey responses (yes/no) from 12 regions, want hierarchical
  regression on age and income. Python."*
- *"I'm fitting a heteroskedastic regression — predictors affect both the
  mean and the standard deviation."*
- *"What's the average treatment effect of `condition` on the probability
  of conversion?"* (cue for `interpret.comparisons`)

**False triggers — adjacent-domain adversaries**

Drawn from carve-out boundaries so the `description:` keywords don't
drift into neighbour skills' territory:

- *"Write a custom PyMC model for a non-standard likelihood — I need
  fine control of the log-prob and the gradient."* → `bayesian-workflow`,
  not Bambi.
- *"Fit a hierarchical drift-diffusion model on my RT data with subject
  random effects."* → `hssm-workflow`, not Bambi (superficially looks
  like a GLMM ask).
- *"Train an amortized posterior estimator for my custom simulator."* →
  `amortized-workflow`.
- *"Build a Gaussian process regression on spatial data using sklearn's
  `GaussianProcessRegressor`."* → out of scope (not Bayesian-stack).
- (~10 more drafted in iteration 2 — must cover XGBoost, statsmodels
  MixedLM, Prophet, sklearn pipelines, scipy.stats hypothesis tests)

## Decisions carried in this skill

- **D2 (resolved):** `interpret` (`predictions`, `comparisons`, `slopes` +
  `plot_*` variants) is in-scope as a core feature. Splines / HSGP get one
  reference file (light coverage).
- **D5 (resolved):** Hard rules use `**MUST**` / `**NEVER**` prefixes,
  matching `amortized-workflow`.

## Iteration-2 follow-ups

- Flesh out each Hard Rule with a one-paragraph rationale + example.
- Draft each reference file's TOC + first prose paragraph.
- Draft each script's docstring + argparse signature.
- Write `eval_metadata.json` for at least 1 scenario.
- Sketch the SKILL.md frontmatter `description:` paragraph in full.
- Confirm the references/ DSL-vs-API split is implemented as planned so
  `hssm-workflow` can lean on it.
