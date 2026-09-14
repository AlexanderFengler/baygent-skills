# `bambi-workflow` — Architecture

**Status:** refreshed planning baseline, 2026-09-14. No skill implementation.
This replaces iteration 1 at `ee52405`; the original remains in Git history.
See the [upstream assessment](README.md) and [M1 plan](bambi-workflow-skill-iter2.md).

## Purpose and first supported scope

Help an agent turn a regression question into a Bambi model, check its priors
and predictions, and answer the question on an appropriate outcome scale.
`bayesian-workflow` continues to own general Bayesian diagnostics, calibration,
prior sensitivity, and reporting.

The first milestone supports Gaussian regression and Bernoulli regression with
a varying intercept. Target released Bambi **0.21.0** at
[`3f795e07`](https://github.com/bambinos/bambi/tree/3f795e07c8ec87d26f48e476b98e5579e0eb1a42),
using Python 3.12 and the PyMC 6 / ArviZ 1.x stack. Its
[dependency declarations](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/pyproject.toml)
include PyMC `>=6,<7`, PyTensor `>=3,<4`, and Formulae `>=0.6,<0.7`.
Do not build released examples from a contributor's newer `dev` checkout.

Distributional regression, additional families, splines/HSGP, advanced new-group
prediction, custom families, and extra inference backends are later increments.
A count-regression request should be recognized as adjacent Bambi work without
implying that its full workflow has already been validated. Explicit raw-PyMC,
HSSM/SSM, causal-design, and BayesFlow-training requests keep their own tools.

## Match the existing skill architecture

The examples are the shipped
[Bayesian](../bayesian-workflow/SKILL.md),
[causal](../causal-inference/SKILL.md), and
[amortized](../amortized-workflow/SKILL.md) workflows, not their incidental file
counts. Follow [CLAUDE.md](../CLAUDE.md):

- A standalone `bambi-workflow/` folder, copied into an agent's skill directory.
- `SKILL.md` frontmatter with `name`, a focused agent-neutral `description`,
  `license`, and author/version metadata. Quote Markdown author strings and
  keep the description within 1,024 characters.
- A direct `## Dependencies` section, numbered workflow, a few evidence-based
  rules, reference links where needed, and a short troubleshooting section.
- Detailed examples in `references/`; small deterministic helpers in `scripts/`
  only when they add value beyond native Bambi/shared Bayesian functions.
- Scenario metadata and real evaluation results under root `evals/`.
- A short skill README and root README listing/install guidance, as in the
  existing repo. Do not add Codex-specific metadata or a new packaging layer.

Keep the entrypoint as short as the supported workflow permits. The target is
roughly the existing Bayesian/causal entrypoint size, not a line-count gate.
Installed skill instructions must work without the repository's `plans/`,
`evals/`, environment files, or a particular agent's hardcoded installation path.
Resolve the installed dependency's location before referring to its resources.

## Composition and ownership

Require `bayesian-workflow` directly, like `causal-inference` does. Its presence
is checked before dependent analysis steps; install into the user's selected
skill location when installation is authorized. Loading the dependency is not
permission to replace an explicitly requested Bambi model with raw PyMC code.

| Owner | Responsibility |
|---|---|
| `bambi-workflow` | Formula and outcome encoding; family/link choice; resolved Bambi priors; Bambi model construction, fit, prior/posterior predictions, and likelihood/prior computation; native interpretation. |
| `bayesian-workflow` | Convergence and calibration interpretation; LOO/model comparison and sensitivity guidance; reusable diagnostic helpers; canonical report structure and assessment language. |
| The analysis | Scientific question, predictor scales, intended contrasts, prior justification, and limitations supported by actual evidence. |

The handoff is a saved, xarray-backed sampling artifact with the groups and
observed-variable names required by the selected shared helpers. It is not a
promise that every package version or arbitrary likelihood can be processed
without adaptation. Bambi 0.21 uses DataTree; shared helper compatibility must
be exercised with a real Bambi output, not inferred from a raw PyMC example.

A future HSSM skill may consult a precise formula-syntax section. The Bambi
prior reference is package-specific: its auto-scaling, components, and
construction behavior are not a reusable HSSM prior policy.

## Workflow

1. **Frame the question.** Identify outcome, observational unit, grouping, and
   the prediction or contrast the user wants. Honor choices already supplied.
2. **Check the data and formula.** Verify response event, category levels and
   reference, predictor scales, group support, and the intended design matrix.
3. **Select family/link and priors.** Specify the intended likelihood explicitly;
   inspect resolved priors and justify any overrides on the relevant scale.
4. **Build and check prior predictions.** Build the PyMC graph through Bambi,
   call Bambi's prior-predictive API, and assess plausibility before fitting.
5. **Fit and save.** Select an available supported backend, record resolved
   versions/seeds, and save sampling output before downstream processing.
6. **Generate predictions and diagnostic inputs.** Use Bambi's own prediction,
   log-likelihood, and log-prior methods. Save the resulting artifact with
   consistent observed/predictive names and observation coordinates.
7. **Diagnose and criticize.** Follow the shared Bayesian workflow and helpers;
   preserve failed/missing checks as limitations rather than inventing ratings.
8. **Answer the question.** Use native `bmb.interpret`/prediction surfaces for
   response-scale summaries and contrasts, with uncertainty and an explicit
   conditioning/averaging target. Report coefficients too when useful.
9. **Report.** Fill the shared `<slug>/report.md`, using shared assessments.
   Add formula, family/link, priors, event/reference coding, and contrast/grid
   information within the existing sections.

## Corrections to the original hard rules

These are source-based boundaries, not new universal modeling requirements.

| Retired draft claim | Current rule |
|---|---|
| Priors do not exist until `build()` | Bambi assembles/scales component priors in construction. Inspect before fit; build for graph inspection/prior predictive. |
| Family is auto-detected from response dtype | The constructor defaults to Gaussian. Explicitly verify the intended family and response event. |
| `C(x)`/`categorical=` ensures a meaningful baseline | Object columns are already converted to categorical. Verify levels/reference and encoded columns; the wrapper alone does not choose a scientific baseline. |
| Every auxiliary parameter needs a formula | Constant auxiliary parameters are legitimate. Add a distributional formula when the scientific model calls for one; inspect its defaults either way. |
| Every slope follows one 2.5×response-SD rule | Scaling depends on family, link, and predictors. Inspect resolved priors and prior predictions; do not rate them against generic raw-scale cutoffs. |
| Use `nuts_numpyro` and raw PyMC log-prior computation | Verify release sampler names (`nutpie`, `numpyro`) and use Bambi's likelihood/prior bridge, including its omitted-offset handling. |
| `comparisons(..., value=...)` | Use the released `contrast` interface; test the exact native call in M1. |
| An artifact with the right API names proves correctness | Static agent evaluation and executed package smoke checks are different evidence; require both. |

Source anchors: released
[model implementation](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/models.py),
[log-prior handling](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/models.py#L1191),
and [comparison API](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/interpret/effects.py#L592).

Do not duplicate the upstream calibration implementation or old ArviZ plotting
recipes. In particular, `plot_ppc_pit(..., loo_pit=True)` was fixed upstream;
follow the current Bayesian calibration reference/helper. Report ELPD differences
with their uncertainty and predictive evidence, not a new fixed cutoff.

## Resource and helper boundaries

The initial references cover formula/encoding, supported families/links,
Bambi priors, and prediction/interpretation. A short reporting section links to
the shared canonical template instead of copying it. Detailed rationales live
there rather than expanding every entrypoint rule into a tutorial.

Use native Bambi interpretation outputs initially. The original pickle-based
model transport, general formula/CSV reconstruction CLI, and marginal-effects
wrapper are deferred. An `inspect_priors(model)` exporter is justified only if
M1 demonstrates repeated useful work; it should expose resolved priors without
labeling them plausible/implausible using arbitrary thresholds.

Bambi M1 is complete only when the installable folder and two supported analyses
pass the [acceptance gates](bambi-workflow-skill-iter2.md). Broader coverage is a
subsequent evaluated increment, not an obligation to fill six references or two
script placeholders before the first useful contribution.
