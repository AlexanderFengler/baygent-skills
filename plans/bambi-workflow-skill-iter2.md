# `bambi-workflow` — Iteration-2 Content Plan

**Status:** WIP — iteration 2, content drafted at concrete-artifact level.
The plan still lives in `plans/`; turning these drafts into shippable files
under `bambi-workflow/` is iteration 3. Implementation and validation remain
pending.

**Reads with:**
[`bambi-workflow-skill.md`](./bambi-workflow-skill.md) — iteration-1
architecture (scope, composition, hard-rule categories).

## What's in this file

1. Full `SKILL.md` frontmatter `description:` paragraph.
2. Every hard rule with rationale + one-line code example.
3. Reference-file TOCs + opening prose.
4. Script docstrings + argparse signatures.
5. One fully-drafted `eval_metadata.json` (`glmm-bernoulli-survey`).

## 1. SKILL.md frontmatter — full draft

```yaml
---
name: bambi-workflow
description: >
  Opinionated workflow for fitting Bayesian generalised linear and mixed
  models with Bambi on top of PyMC and ArviZ. Enforces guardrails an agent
  won't apply unprompted: model.build() before inspecting auto-priors,
  mandatory prior predictive checks, explicit family / link choice,
  distributional formulas for heteroscedastic data, C() / categorical=
  declaration for string predictors, response-scale reporting via
  bmb.interpret rather than raw posterior summaries on the
  linear-predictor scale, and explicit handling of HSGP / spline bases.
  Delegates convergence diagnostics, LOO-CV, calibration (LOO-PIT),
  prior sensitivity (psense), and the analysis-report template to the
  bayesian-workflow skill. Trigger on: Bambi, bmb.Model, Wilkinson formula
  syntax (`y ~ x + (1|g)`), GLM, GLMM, hierarchical regression, multilevel
  regression, mixed-effects, distributional model, location-scale model,
  HSGP, spline regression, bs(), hsgp(), marginal effects, average
  marginal effect, conditional effects, bmb.interpret, predict, prior
  auto-scaling, Bambi families, Bambi priors.
license: MIT
metadata:
  author: [Alexander Fengler](https://www.alexanderfengler.com)
  version: "1.0"
---
```

Notes on the description:

- Lead sentence states *what* and *which libraries*, matching the
  bayesian-workflow / amortized-workflow opener style.
- Second sentence enumerates guardrails — these are the agent-neutral
  cue for *why* the skill exists.
- Third sentence names the upstream delegation explicitly — bambi-workflow
  is incomplete without bayesian-workflow loaded too.
- "Trigger on:" enumeration mirrors the convention from the existing
  three skills. Bambi-API symbols (`bmb.Model`, `model.build`, `predict`,
  `bmb.interpret`) are included verbatim because trigger-eval false
  prompts will lean on overlapping wording.

## 2. Hard rules — rationale + example

Each rule below maps 1:1 to a `MUST` / `NEVER` bullet in the iter-1 plan.
The rationale paragraph + example will land in the SKILL.md "## Hard
rules" block. **Style note:** the rationale leads with *why*, not *what*
— the rule statement itself already says what.

### R1. MUST `model.build()` before inspecting priors

**Why.** Bambi defers prior assembly until `build()` is called, which is
also where `auto_scale` rescales default priors to the data. Inspecting
`model.priors` or `model.components` before `build()` returns either
empty dicts or the un-rescaled defaults — the agent will reason about
priors that aren't actually the priors used at sampling time.

```python
model = bmb.Model("y ~ x + (1|g)", df)
# WRONG: model.priors is incomplete here
model.build()
priors_now = {
    name: term.prior
    for name, term in model.components["mu"].common_terms.items()
}
```

### R2. MUST inspect auto-priors before fit and document overrides

**Why.** Bambi's `auto_scale=True` is convenient but applies a single
2.5×SD heuristic to all slope priors regardless of unit, scale, or
domain knowledge. Multi-modal data and skewed predictors silently yield
implausible priors. The fix is cheap (inspect once, override where
needed) and the cost of missing it is hidden — the model fits fine but
the prior-predictive distribution lies far from plausible data.

```python
model.build()
print(model.components["mu"].common_terms["x"].prior)
# Decide: keep, narrow, widen, or replace
```

### R3. MUST run `model.prior_predictive()` before fit

**Why.** Same rule as `bayesian-workflow`'s prior-predictive mandate,
but the Bambi API differs and the agent will often skip it because Bambi
"feels lighter" than raw PyMC. The check is mandatory regardless of
which package writes the model.

```python
idata_prior = model.prior_predictive(draws=500)
az.plot_ppc(idata_prior, group="prior")
```

### R4. For distributional models, MUST explicitly model every auxiliary parameter

**Why.** `Formula("y ~ x", "sigma ~ z")` declares `sigma` as a regressed
parameter, but auxiliary parameters left off the formula list fall back
to silently-applied default priors. For `Gaussian`, `sigma` defaults to
`HalfStudentT(4, 1)`; for `NegativeBinomial`, `alpha` defaults to
`HalfNormal(1)`; etc. Users who think they've written a heteroscedastic
model but omitted `sigma ~` get a homoscedastic model with a flat tail.

```python
# WRONG — sigma silently homoscedastic
bmb.Model(bmb.Formula("y ~ x"), df, family="gaussian")

# RIGHT
bmb.Model(bmb.Formula("y ~ x", "sigma ~ z"), df, family="gaussian")
```

### R5. For categorical predictors, MUST use `C(x)` or pass `categorical=`

**Why.** Bambi auto-detects pandas `Categorical` columns but treats
plain string columns as object-typed predictors, which dummy-encodes
based on `sorted(df["x"].unique())` order — alphabetical, not
domain-meaningful. A "control" / "treatment" column will encode
"control" as 0, "treatment" as 1 (intended); but
"high" / "low" / "medium" will encode "high" as 0, "low" as 1,
"medium" as 2 — a wrong baseline.

```python
# WRONG: string column → alphabetical baseline
bmb.Model("y ~ dose", df)

# RIGHT (explicit C() — preferred)
bmb.Model("y ~ C(dose)", df)

# RIGHT (cast at construction)
bmb.Model("y ~ dose", df, categorical=["dose"])
```

### R6. For multinomial / categorical families, MUST set `family=` explicitly

**Why.** Bambi auto-detects family from the response dtype — but
multi-level outcomes (e.g. integer-coded responses or string-coded
choices) often look like ordinal or count data to the auto-detector.
Without `family="categorical"` or `"multinomial"`, the agent silently
fits the wrong likelihood.

```python
# WRONG: ints → may auto-detect as Poisson
bmb.Model("y ~ x", df)

# RIGHT
bmb.Model("y ~ x", df, family="categorical")
```

### R7. NEVER report posterior summaries alone for non-Gaussian families

**Why.** A Bernoulli logit coefficient β has interpretation in
log-odds space, which is uninformative to a non-statistician. The
substantive question — *how much does X change Pr(Y=1)?* — requires
mapping back through the link, which depends on the values of the other
covariates. `bmb.interpret.comparisons` and `slopes` handle this
correctly and report 94% HDIs on the response scale.

```python
# WRONG: log-odds table, hard to interpret
az.summary(idata, var_names=["x"])

# RIGHT: probability-scale comparison
contrasts = bmb.interpret.comparisons(model, idata, "x", value=[0, 1])
```

### R8. NEVER mix `priors=` aliases (`"common"` / `"group_specific"`) with term-specific keys without verifying resolution order

**Why.** Bambi resolves `priors` dict keys in a documented but
non-obvious order: term-specific keys override `"common"` /
`"group_specific"` aliases, which override the auto-prior defaults.
Mixing aliases and specific keys without intent silently overrides
auto-priors on terms the agent didn't intend to touch.

```python
# RISKY — alias hits all common terms; term-specific only if order known
priors = {
    "common": bmb.Prior("Normal", mu=0, sigma=2),
    "x": bmb.Prior("Normal", mu=0, sigma=1),
}
```

Recommended pattern: prefer term-specific keys; use aliases only when
deliberately blanketing every term of a kind.

### R9. MUST delegate convergence / calibration / sensitivity / LOO to `bayesian-workflow`

**Why.** These rules live upstream and apply identically regardless of
whether the model came from raw PyMC or Bambi. Re-deriving them in this
skill would drift over time. The composition contract is: `bambi-workflow`
hands `bayesian-workflow` an `InferenceData` object and the latter takes
over from there.

```python
idata = model.fit(inference_method="nuts_numpyro")

# Hand off to bayesian-workflow:
import arviz_stats as azs
azs.diagnose(idata)
pm.compute_log_likelihood(idata, model=model.backend.model)
pm.compute_log_prior(idata, model=model.backend.model)
loo = az.loo(idata, pointwise=True)
azp.plot_ppc_pit(idata, loo_pit=True)
```

(`model.backend.model` exposes the underlying PyMC model — needed for
`compute_log_likelihood` / `compute_log_prior` which work at the PyMC
layer.)

## 3. References — TOCs + opening prose

### `references/formula-syntax.md` (DSL-reusable; ~120 lines target)

```text
# Formula Syntax in Bambi

## Contents
- Basic Wilkinson DSL
- Group-specific (random) terms
- Interactions and intercept handling
- Categorical predictors via C()
- Offset terms
- Splines and HSGP (cross-link to splines-and-hsgp.md)
- Distributional formulas
- Common syntax errors
```

**Opening paragraph.**

> Bambi accepts Wilkinson-style formulas via the `formulae` package. The
> DSL is intentionally similar to R's `lme4` and `brms` syntax so that
> users porting models across ecosystems can keep the same structure.
> This document is the canonical reference for the subset of DSL features
> Bambi supports — both for users authoring `bmb.Model(...)` calls
> directly and for downstream skills that consume Bambi formulas via
> their own constructors (e.g., `hssm.HSSM` accepts the same syntax in
> `Param(formula=...)` entries). The downstream consumers MUST NOT rely
> on Bambi-API semantics beyond formula parsing — see each downstream
> skill's carve-out rules.

### `references/families-and-links.md` (Bambi-API-specific; ~140 lines target)

```text
# Families and Links in Bambi

## Contents
- The family decision table (29 univariate + 2 multivariate)
- Default link per family
- When to override the link
- Hurdle vs ZeroInflated — choosing between them
- Ordinal families (Cumulative, StoppingRatio) and their constraints
- Custom families via bmb.Family
```

**Opening paragraph.**

> The choice of family is more consequential than the choice of priors
> in most applied work. This document gives the explicit mapping from
> outcome type to family, then walks through the cases where the
> default link is the wrong default (count data with very low rates,
> bounded continuous outcomes, etc.). The ordinal-family section is
> emphasised because Cumulative and StoppingRatio interact badly with
> distributional formulas (they auto-remove the intercept and forbid
> multi-formula spec) — a silent failure source.

### `references/priors-in-bambi.md` (mostly DSL-reusable; ~110 lines target)

```text
# Priors in Bambi

## Contents
- The auto-scale machinery: what 2.5×SD means and when it's wrong
- Inspecting priors after model.build()
- Overriding per-term via priors= dict
- Common vs group-specific priors and the alias trap
- Hierarchical scale priors (group-level sigma)
- Hand-off to bayesian-workflow for prior predictive + sensitivity
```

**Opening paragraph.**

> Bambi's `auto_scale=True` is one of its biggest ergonomic wins and one
> of its biggest hidden traps. This document is split into "what the
> auto-scaler does" (so you can predict its output before calling
> `model.build()`) and "when to override it" (so you don't end up with
> weakly-informative priors on every coefficient regardless of physical
> scale). The override patterns extend to downstream skills that
> instantiate Bambi-style priors as nested dicts inside their own
> constructors — see `hssm.Param(prior=...)`.

### `references/interpret-marginal-effects.md` (Bambi-API-specific; ~150 lines target)

```text
# Marginal Effects via bmb.interpret

## Contents
- The three interpret functions: predictions, comparisons, slopes
- Conditional vs marginal vs average effects
- predict(kind="response_params") vs predict(kind="response")
- Choosing the right contrast for the question
- Plotting variants (plot_predictions / plot_comparisons / plot_slopes)
- When to fall back to raw posterior summaries
```

**Opening paragraph.**

> Coefficients of non-Gaussian models live on the linear-predictor
> scale; the substantive answers users actually want — *how much does X
> change Pr(Y=1)?*, *what's the average treatment effect of T on a
> count?* — live on the response scale. `bmb.interpret` makes the
> linear-to-response mapping explicit, with proper uncertainty
> propagation. This document is the canonical reference for picking
> which `interpret` function answers a given question.

### `references/splines-and-hsgp.md` (Bambi-API-specific; ~100 lines target)

```text
# Splines and HSGP in Bambi

## Contents
- bs() / poly() for splines
- hsgp(x, m=20, c=1.5) — what m and c control
- Prior on the marginal-σ and ℓ (length-scale)
- Prior-predictive check for HSGP basis adequacy
- When to prefer hsgp() over bs() and vice versa
```

**Opening paragraph.**

> Splines and HSGPs are smooth-term tools with very different default
> behaviours. `bs(x)` is a natural cubic spline with a default knot
> count derived from the data; `hsgp(x, m=..., c=...)` is a Hilbert-
> space approximation to a Gaussian process whose smoothness is
> controlled by the prior on its hyperparameters. This document covers
> the two main hyperparameter knobs you actually need to set — basis
> count and the length-scale prior — and the prior-predictive check
> that proves the basis is wide enough for your data range.

## 4. Scripts — docstrings + argparse signatures

### `scripts/inspect_priors.py`

```python
"""
Inspect a built Bambi model's prior table and flag suspect auto-priors.

Reads a Bambi model from a saved pickle OR reconstructs it from a
formula + data CSV. After `model.build()`, walks every common and
group-specific term and dumps:
  - the auto-scaled prior name and parameters
  - a `suspect: True/False` flag from simple heuristics
  - a `recommendation` string when suspect

Heuristics (kept conservative — false positives are cheap, false
negatives are not):
  - slope-prior sigma > 5× response SD       → "auto-prior too wide"
  - slope-prior sigma < 0.05× response SD    → "auto-prior too narrow"
  - hierarchical group-scale Exponential(0.5) on a small group count
    (<3 groups) → "consider a tighter prior — partial pooling not
    well-identified"
  - intercept-prior centered far from response mean → "consider
    re-centering"

Usage:
    python inspect_priors.py --model model.pkl
    python inspect_priors.py --formula "y ~ x + (1|g)" --data df.csv \\
        --family bernoulli
    python inspect_priors.py --model model.pkl --output priors.json

Output: JSON with keys
  - parameters:   {term_name: {prior: {...}, suspect: bool,
                               recommendation: str | None}}
  - summary:      {n_terms: int, n_suspect: int}
  - next_steps:   list[str]   (e.g., "override `x` prior" if suspect)
"""

import argparse, json, pickle, sys
import bambi as bmb
import pandas as pd


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Path to pickled bmb.Model (built)")
    src.add_argument("--formula", help="Wilkinson formula string")
    p.add_argument("--data", help="CSV of data (required with --formula)")
    p.add_argument("--family", default="gaussian",
                   help="Bambi family name (default: gaussian)")
    p.add_argument("--output", help="JSON output path (default: stdout)")
    p.add_argument("--auto-scale", default=True, type=bool,
                   help="Pass through to bmb.Model auto_scale")
    return p


def inspect_priors(model: bmb.Model) -> dict: ...
def main() -> int: ...
```

### `scripts/marginal_effects_report.py`

```python
"""
Run bmb.interpret.{predictions, comparisons, slopes} and emit a
structured JSON + markdown table for the analysis report.

This is the canonical post-fit reporting helper for any non-Gaussian
Bambi model. It computes:
  - predictions:   posterior-predictive means on the response scale at a
                   grid of predictor values (1 row per grid point)
  - comparisons:   pairwise contrasts between user-specified predictor
                   values, on the response scale
  - slopes:        finite-difference marginal effects, on the response
                   scale, with 94% HDIs

The 94% HDI default matches the bayesian-workflow convention. Contrasts
are named explicitly so the report can refer to them by name.

Usage:
    python marginal_effects_report.py --model model.pkl --idata idata.nc \\
        --variable x \\
        --predictions-grid 10 \\
        --comparisons 0,1 \\
        --output-md effects.md \\
        --output-json effects.json

Output:
  - JSON: {predictions: [...], comparisons: [...], slopes: [...]}
    with per-row {value, mean, hdi_low, hdi_high}
  - Markdown: human-readable table for the report, ready to paste
"""

import argparse, json, sys
from pathlib import Path
import arviz as az
import bambi as bmb
import pandas as pd


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Path to pickled bmb.Model")
    p.add_argument("--idata", required=True, help="Path to .nc idata")
    p.add_argument("--variable", required=True,
                   help="Predictor variable to compute effects on")
    p.add_argument("--predictions-grid", type=int, default=10,
                   help="Number of grid points for predictions")
    p.add_argument("--comparisons",
                   help="Comma-separated values for pairwise comparisons "
                        "(e.g. '0,1' or 'low,high')")
    p.add_argument("--hdi-prob", type=float, default=0.94,
                   help="HDI probability (default 0.94 per repo convention)")
    p.add_argument("--output-md", help="Markdown output path")
    p.add_argument("--output-json", help="JSON output path")
    return p


def run_effects(...) -> dict: ...
def render_markdown_table(report: dict) -> str: ...
def main() -> int: ...
```

## 5. Fully-drafted `eval_metadata.json` — `glmm-bernoulli-survey`

Chosen because this scenario stresses the most rules at once: hierarchical
auto-prior inspection (R1, R2), prior predictive (R3), categorical
declaration (R5), response-scale reporting via `interpret` (R7), and the
delegation hand-off to `bayesian-workflow` (R9).

```json
{
  "eval_id": 2,
  "eval_name": "glmm-bernoulli-survey",
  "prompt": "I have survey responses (yes/no, column called `support`) from 3000 people across 12 regions, with predictors `age` (continuous), `income_bracket` (low/medium/high), and `urban` (binary). Some regions only have 30 respondents. Build a hierarchical Bayesian logistic regression so I can borrow strength across regions, and tell me the predicted probability of support at age=40, income=medium, in an urban region — with uncertainty.",
  "assertions": [
    "Uses bmb.Model with a Wilkinson formula including a group-specific intercept `(1|region)` for partial pooling across regions",
    "Specifies family=\"bernoulli\" explicitly in the bmb.Model constructor",
    "Uses C(income_bracket) or passes categorical=[\"income_bracket\"] to handle the string-encoded predictor explicitly",
    "Calls model.build() before inspecting priors or fitting",
    "Inspects auto-priors after build() (e.g., prints model.components[...].common_terms or model.priors) before fitting",
    "Calls model.prior_predictive(...) and visualises the result before fitting",
    "Calls model.fit(...) without hardcoding chains (no chains=4) — uses default chain selection",
    "Uses inference_method=\"nuts_numpyro\" or \"nutpie\" rather than the slower default \"pymc\" when JAX is available, OR explains why the default was kept",
    "Uses bmb.interpret.predictions (or model.predict with appropriate grid) to compute the posterior-predictive probability at age=40, income=medium, urban=1 — NOT a raw posterior summary of the logit coefficients",
    "Reports the predicted probability with a 94% HDI (or explicitly justifies a different interval)",
    "Hands off to bayesian-workflow patterns for diagnostics — e.g. uses arviz_stats.diagnose(idata) or az.summary + az.plot_trace; calls pm.compute_log_likelihood for LOO-CV",
    "Generates a companion analysis_notes.md file (markdown) interpreting the regional shrinkage, the income effect on the probability scale, and the diagnostic results"
  ]
}
```

**Notes on this scenario:**

- 12 assertions — within the 8–12 target band.
- 9 of 12 assertions are layered into the four-layer anatomy:
  - Workflow (assertions 1–3, 7): which steps executed.
  - Hard rules (assertions 4, 5, 6, 8, 9, 10): one rule per assertion.
  - Scripts (none in this scenario — scripts come in later scenarios
    where `inspect_priors.py` is the natural call).
  - Companion (assertion 12): cross-skill convention from
    `bayesian-workflow`.
- Adversarial dimension: the `without_skill` baseline will probably
  produce a raw-PyMC model or a Bambi model with default `family=`
  auto-detection, missing prior inspection, and a logit-coefficient
  summary. The with-skill delta is highest on assertions 2, 3, 4, 5, 6,
  9, 10.

## What iteration 3 turns this into

The artefacts below get written under `bambi-workflow/` (the actual
skill directory, sibling to `bayesian-workflow/`):

- `bambi-workflow/SKILL.md` (frontmatter from §1 + workflow + hard rules
  from §2 + cross-links to references)
- `bambi-workflow/references/{formula-syntax,families-and-links,priors-in-bambi,interpret-marginal-effects,splines-and-hsgp}.md`
- `bambi-workflow/scripts/{inspect_priors,marginal_effects_report}.py`
- `bambi-workflow/README.md` (short, mirrors `bayesian-workflow/README.md`)
- `evals/bambi-workflow/iteration-1/glmm-bernoulli-survey/eval_metadata.json`
  (from §5; 5 more scenarios drafted before submitting iteration-1
  baseline)
- `evals/bambi-workflow/trigger_eval_set.json`
- One-line edit to `environment.yml` if any pins change (likely none —
  Bambi already declared).
- One-line edit to top-level `README.md` listing the new skill.

## Iteration-3 follow-ups (pre-PR checklist)

- Audit references/ split — confirm `formula-syntax.md` and
  `priors-in-bambi.md` are genuinely DSL-only so `hssm-workflow` can
  link to them without dragging in Bambi-API rules.
- Pin Bambi version in environment.yml if iter-3 testing reveals
  incompatibilities with the latest minor release.
- Draft remaining 5 `eval_metadata.json` scenarios (one per remaining
  scenario seed).
- Draft `trigger_eval_set.json` (~10 true + ~15 false prompts).
- Self-grade the SKILL.md against the iter-3 hard rules section before
  PR submission.
