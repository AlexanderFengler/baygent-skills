# `bambi-workflow` — M1 implementation and acceptance plan

**Status:** implementation and runtime gates validated, 2026-09-14; paired
agent and trigger evaluations remain pending. See the
[validation record](../evals/bambi-workflow/iteration-1/README.md). This replaces
the WIP content draft at `73de41c`, retained in Git history. Read with the [architecture](bambi-workflow-skill.md).

## Milestone outcome

An independently installable Bambi skill can complete a Gaussian regression and
a hierarchical Bernoulli survey analysis, including meaningful prior checks,
predictions with uncertainty, shared diagnostic assessments, and a canonical
report. Both analyses run against released Bambi 0.21.0 on the documented modern
stack. This is the first useful contribution; a full Bambi feature catalog is
not a prerequisite.

## Entrypoint draft

```yaml
---
name: bambi-workflow
description: >
  Fit and interpret Bayesian regressions with Bambi. Use when the user asks
  for Bambi, bmb.Model, formula-based GLMs or GLMMs, partial pooling,
  Bambi prior inspection, or response-scale predictions and contrasts.
  Guides formula and outcome encoding, family/link choice, prior predictive
  checks, and native Bambi interpretation; delegates shared diagnostics
  and reporting to bayesian-workflow. Preserve explicit requests to use
  raw PyMC, HSSM sequential-sampling models, or BayesFlow training.
license: MIT
metadata:
  author: "[Alexander Fengler](https://www.alexanderfengler.com)"
  version: "0.1"
---
```

Use the numbered workflow and ownership split from the architecture. Keep only
consequential package-specific constraints in the entrypoint. Explain
conditional details in references rather than requiring every possible modeling
step for every request.

## Files to produce

```text
bambi-workflow/
  SKILL.md
  README.md
  references/
    formula-syntax.md
    families-and-links.md
    priors-in-bambi.md
    interpretation-and-reporting.md
evals/bambi-workflow/
  trigger_eval_set.json
  iteration-1/
    gaussian-regression/eval_metadata.json
    glmm-bernoulli-survey/eval_metadata.json
evals/smoke/
  test_bambi_workflow.py
```

These files now exist, along with two marimo examples under
`examples/bambi-workflow/` and executed report snapshots in the validation record.
Keep developer setup and validation notes in root documentation/evaluation records. No new executable helper is mandatory:
reuse Bambi and the installed Bayesian skill first. If a prior exporter is
needed, give it an in-process function and a narrowly justified input contract;
do not require pickled models or infer prior quality from generic thresholds.

Reference responsibilities:

- `formula-syntax.md`: common/group terms and interactions used by the two
  scenarios; outcome and predictor coding, category reference verification.
  Separate formula notation from Bambi constructor/API behavior.
- `families-and-links.md`: Gaussian/Bernoulli decision and event/link semantics;
  constant auxiliary parameters are valid. Mark untested families as extensions.
- `priors-in-bambi.md`: resolved-prior inspection, overrides and appropriate
  scales, graph construction, prior-predictive judgment. Entirely Bambi-owned.
- `interpretation-and-reporting.md`: native predictions/contrasts, known-group
  conditioning and averaging target, saved diagnostic handoff, and a link to
  the installed Bayesian report template. Use posterior-mean probabilities
  when answering predictive-probability questions and propagate uncertainty.

## Runtime and artifact contract

Use Bambi's released APIs for construction, `fit`, prior/predictive generation,
`compute_log_likelihood`, and `compute_log_prior`. Verify native `bmb.interpret`
arguments against 0.21.0. Shared raw-PyMC modeling examples are not replacement
implementations for these Bambi operations.

Save the DataTree immediately after sampling, then update the saved artifact
after adding the predictive/log-density groups needed by the selected checks.
Check observed/predictive variable names, coordinates, and group presence. Use
the installed `bayesian-workflow` utilities on that artifact:

1. `diagnose_model.py` → `diagnostics.json`.
2. `calibration_check.py` → `calibration.json` and calibration figures.
3. `check_diagnostics.py` → `check_report.json` with ratings/next steps.
4. The canonical `<slug>/report.md`, with the Bayesian template's figure and
   artifact names and Bambi model/contrast details in the existing sections.

Supply the actual outcome name where required. Exercise log-likelihood and
prior-sensitivity inputs, including retained Bambi offsets and consistent intercept
coordinates; a missing prerequisite must remain visible. Do not claim a calibration or sensitivity
assessment merely because a JSON file was written. A numerical check that is
unstable on a small fixture is recorded and diagnosed, not graded as healthy.

## Developer environment

The modern package set in `environment-pymc6.yml` now includes
`bambi==0.21.0` and marimo. Resolved dependency versions are recorded with the run.
Check the dependency declarations at the frozen release used in the
[architecture](bambi-workflow-skill.md). The existing PyMC 5 environment does not
establish modern Bambi support.

Keep the shared workflow's legacy compatibility guarantee separate. M1 supports
the tested Bambi/PyMC 6 path; older Bambi support requires another explicit run.
If shared diagnostics scripts change, run their existing two-environment smoke
and equivalence gates as documented in root CLAUDE.md. If they do not change,
the focused Bambi integration smoke is the new required evidence. Do not publish
an environment change until it has resolved and the supported workflow has run.

## Evaluation fixtures

Use the repo's existing `eval_id`, `eval_name`, `prompt`, `assertions` schema.
Author assertions about observed decisions and output, not exact wording or
mandatory incidental syntax such as `C()` when category coding is already sound.
The survey prompt below explicitly requests Bambi so the test assesses skill
quality rather than penalizing a legitimate raw-PyMC tool choice.

```json
{
  "eval_id": 2,
  "eval_name": "glmm-bernoulli-survey",
  "prompt": "Use Bambi for a hierarchical logistic regression of survey support. The data contain support (0=no, 1=yes), age, income_bracket (low/medium/high), urban (0/1), and region; some regions have few respondents. Use low income as the reference. Check the priors, fit the model, and report the predicted probability of support at age 40, medium income, and urban=1 in a specified observed region, with uncertainty and diagnostic limitations.",
  "assertions": [
    "Uses Bambi with an explicit Bernoulli likelihood and a varying intercept for region",
    "Verifies the modeled support event, low-income reference category, and predictor coding",
    "Inspects resolved priors and assesses prior predictions before fitting",
    "Uses supported Bambi fit and prediction APIs and saves sampling output before downstream processing",
    "Defines the observed-region prediction target and reports posterior-mean support probability with an explicit uncertainty interval",
    "Uses Bambi's log-density bridge and provides compatible observed/predictive groups to the shared Bayesian diagnostic helpers",
    "Reports convergence and calibration evidence honestly, including missing or failed checks",
    "Writes the canonical results-folder report.md using shared check_report.json assessments and adds the formula, family/link, event coding, and prediction target"
  ]
}
```

Create a small deterministic synthetic fixture separately for execution; do not
invent numerical conclusions or grades in this planning file. The Gaussian
scenario checks continuous outcomes, legitimate constant residual scale,
prior-predictive assessment, and a supported native predictor contrast.

Trigger fixtures use the existing `query`/`should_trigger` JSON shape. Include
explicit Bambi requests and formula-based partial pooling; negatives include an
explicit raw-PyMC model, HSSM choice/RT inference, and BayesFlow training. Do not
force exclusivity with `bayesian-workflow`, which is the required dependency.
Unsupported advanced Bambi requests should route to scoped guidance rather
than falsely imply that every requested feature is validated.

## Acceptance gates

1. **Installed structure:** parse frontmatter, check description length and
   relative links, then test with only the copied skill and its declared
   dependency. No reliance on root plans/evals or the developer checkout.
2. **Actual execution:** run both small analyses on the declared stack. Verify
   prior/posterior predictive operations, actual interpretation calls, trace
   save/load, likelihood/prior groups, helper outputs, and report artifact links.
   Small smoke budgets may use explicit chains for repeatability; they do not
   establish posterior adequacy for a substantive analysis.
3. **Agent behavior:** independently execute paired with-skill/without-skill
   evaluations and the trigger set. Store generated output, grading evidence,
   timing, and a truthful benchmark summary in the existing eval layout.
   If runner/grader access is unavailable, record that gate as pending; static
   schema checks are not a substitute and no scores are fabricated.
4. **Review:** inspect the installed workflow for package/API ownership,
   unsupported claims, plausible scientific reporting, and scope creep.
   Record tested versions and any unresolved limitations with the milestone.

Implement and verify these as small commits: skill/reference surface; execution
fixtures and any justified helper; evaluation evidence and discovery/install
documentation. Do not advertise M1 as completed until all gates have evidence.
Distributional models, HSGP/splines, generic model serialization, comprehensive
family tables, and six-scenario coverage remain later evaluated increments.

## Implementation progress

- ~~Standalone skill and four references, direct Bayesian dependency, discovery/install docs.~~
- ~~Two narrated marimo examples with prior/PPC checks, native targets and complete saved artifacts.~~
- ~~Thirteen execution checks, independent analytic likelihood/prior/contrast verification.~~
- ~~Full 4-chain exports, canonical reports, diagnostic JSON, figures and resolved environment record.~~
- ~~Copied-folder link/resource validation and independent scientific/API review.~~
- Paired agent behavior and trigger evaluation: pending authorization for model-service transmission.

Runtime review refined the initial bridge assumption: Bambi 0.21 does not repair
saved intercept coordinates or reconstruct omitted offsets for log-prior evaluation.
The examples explicitly scale predictors, disable internal centering, retain offsets
and verify every free-variable prior density. This is release-scoped evidence, not
a generic prohibition on centered models. The survey's region-scale sensitivity
flag remains visible and motivates a justified alternative-prior comparison.
