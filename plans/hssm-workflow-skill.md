# `hssm-workflow` — Architecture and first milestone

**Status:** refreshed planning baseline, 2026-09-14. No skill implementation.
This replaces iteration 1 at `ee52405`, retained in Git history. See the
[upstream assessment and milestone order](README.md).

## Purpose and scope

Guide choice/reaction-time inference with HSSM: select an SSM suited to the
question, verify data and likelihood assumptions, express parameter priors and
links, fit through HSSM, and judge predictions in the choice/RT domain.
General Bayesian diagnostics and report assessment remain shared.

M2 begins with a **flat analytical DDM** on clean, positive reaction times and
two declared choices. Use released HSSM **0.5.0** at
[`fefed57d`](https://github.com/lnccbrown/HSSM/tree/fefed57d2142637af503b0e92cefe799715c0f46)
as the source baseline. This is a deliberate narrow validation target, not a
claim that HSSM supports only this model or data regime.

The first Bambi milestone precedes M2 to establish the shared diagnostics and
evaluation pattern. HSSM's installed skill does not require Bambi's skill.
Hierarchical effects, LAN likelihoods, RLSSMs, missing RTs, deadlines, lapse
regressions, and other SSM families are later evaluated increments.

## Structure and dependencies

Use the same independent folder convention as the existing skills:

```text
hssm-workflow/
  SKILL.md
  README.md
  references/
    data-and-ddm.md
    priors-and-links.md
    prediction-and-reporting.md
evals/hssm-workflow/
  trigger_eval_set.json
  iteration-1/ddm-flat/eval_metadata.json
evals/smoke/
  test_hssm_workflow.py
```

These are planned paths. Keep `SKILL.md` focused on triggers, dependencies,
numbered workflow, essential domain constraints, and links. Use the repo's
`name`/`description`/`license`/quoted author/version frontmatter. Details go in
references, while a `scripts/` helper is added only for demonstrated repeated
work. Do not create a new registry, plugin, CLI framework, or diagnostic engine.

**Require `bayesian-workflow` directly.** Resolve its installed location and
use its diagnostics/reporting resources. Do not depend on it transitively through
an optional Bambi skill. Installed instructions must work without root plans,
evals, developer environments, or a full repository checkout.

**Bambi is optional reference enrichment.** A specific formula-notation section
can help when a later HSSM milestone adds parameter regressions. HSSM must still
provide enough guidance to construct its supported examples without that skill.
Do not import Bambi's prior reference as generic syntax: its auto-scaling,
component internals, `fit`, and interpretation rules belong to Bambi.

| Owner | Responsibility |
|---|---|
| `hssm-workflow` | SSM/data/likelihood choice; parameter priors, support, links and parameterization; HSSM sampling and predictive surfaces; RT/choice checks and scientific interpretation. |
| `bayesian-workflow` | Shared convergence assessments, applicable LOO/sensitivity/calibration guidance, diagnostic ratings, canonical report structure. |
| `bambi-workflow` | Its own regression workflow; optional precisely scoped formula reference only. |
| `amortized-workflow` | BayesFlow training and amortized inference; not an HSSM runtime dependency. |

## M2 workflow

1. **Define the question.** Establish RT units, task choices, and why a DDM
   addresses the requested inference. Honor explicit choices; ask only for
   missing decisions that affect the model.
2. **Validate the supported data regime.** For this milestone, verify positive
   finite RTs, declared choice labels, units, and the absence of missing/deadline
   cases. Preserve a record of intentional recoding and excluded observations.
3. **Choose the likelihood explicitly.** Use the released analytical DDM path.
   Inspect the selected configuration rather than teaching that every HSSM model
   defaults to a LAN or that analytical likelihoods exist only for DDMs.
4. **Specify priors and links.** Use HSSM's parameter interface and inspect the
   effective priors, support and links. Parameter bounds, a regression link, and
   a transformed coefficient prior are different objects.
5. **Check prior predictions.** Use `model.sample_prior_predictive(...)` and
   HSSM plotting to examine RT ranges, tails, and choice proportions before fit.
6. **Fit and save.** Use `model.sample(...)` with a supported backend; record
   installed versions and seeds, and save sampling output before post-processing.
7. **Check posterior predictions.** Use `model.sample_posterior_predictive(...)`
   and HSSM RT/choice plots. Compare choice proportions and conditional RT
   distributions; retain domain-specific discrepancies alongside convergence.
8. **Assess and report.** Use the shared diagnostic/report contract where its
   input assumptions hold. Add the model/likelihood, data coding, effective
   parameter priors/links, and choice/RT predictive evidence in the canonical
   `<slug>/report.md` sections.

The preferred HSSM APIs keep the workflow on its owning package surface; do not
justify them by claiming raw PyMC can only simulate a linear predictor. HSSM's
observed random variable itself has simulator-backed behavior. See the released
[predictive implementation](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/base.py#L1296).

## Data and diagnostics handoff

HSSM 0.5.0 uses DataTree. Validate actual groups, coordinates and observed-variable
names when saving/reloading an HSSM artifact. The Bayesian workflow's
InferenceData/DataTree compatibility is useful but does not prove an HSSM output
can be passed through every generic helper unchanged.

In particular, an observed RT/choice vector is not automatically a scalar
outcome for generic PIT/calibration. M2 must establish a valid representation and
check the intended conditional/marginal quantity before applying the existing
calibration helper. Reuse shared code when its contract fits; if a small
HSSM-specific adapter is needed, test it and document its input/output semantics.
Do not flatten the RT and choice axes, silently ignore one component, or invent
"healthy calibration" from a successfully written JSON file.

The report uses shared convergence ratings and next steps for applicable checks,
plus explicit RT/choice predictive evidence. Preserve unsupported checks as
unassessed with a reason; a valid domain assessment is a milestone requirement,
not something that an unassessed label alone completes. Establish pointwise
likelihood units before LOO and the actual prior-density inputs before sensitivity.

## Corrected prior and likelihood boundaries

The old universal non-centering rule is retired. HSSM 0.5.0 deliberately centers
some safe generated unmatched group-only locations, can reject explicit priors
whose requested non-centered form cannot be represented faithfully, and rejects
some generic bounded identity-link group-only defaults. Honor the package's
policy; do not force non-centering or rewrite explicit priors to bypass an error.
These are future hierarchical-milestone concerns, not extra complexity in M2.
See the released
[safe-prior implementation](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/param/regression_param.py#L241).

Formula notation may transfer from Bambi; prior scales, supports, default
parameterization, and prediction APIs do not. An unreleased Bambi fork fix must
not silently become an assumption of the HSSM skill. Freeze a compatible tested
HSSM dependency set independently of Bambi M1; current broad dependency floors
are not proof that an arbitrary latest Bambi checkout is supported.

Wrong declared choice labels are checked by HSSM and can raise; do not label
all coding mistakes silent failures. Nevertheless, verify the scientific meaning
of choices and units because a valid encoding can still represent the wrong
event. See released
[data validation](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/data_validator.py#L77).

A future LAN-domain check must use documented artifact training bounds and
trial-wise likelihood parameter values, not regression coefficients. Widening
an HSSM configuration bound does not expand a network's training domain.
Missing provenance is uncertainty to report, not permission to fabricate
coverage. Defer the original `check_lan_coverage.py` proposal until this contract
has concrete data and tests.

## Acceptance and later milestones

For M2, require:

- An installable skill with valid frontmatter, resolvable dependency/resources,
  and a clear narrow support statement.
- A reproducible small analytical DDM run exercising data validation,
  prior/posterior predictive APIs, save/load, diagnostics handoff, and a report
  whose figures and conclusions come from that run.
- A meaningful RT/choice model check with documented calibration applicability;
  no generic-PIT success claim based solely on container compatibility.
- Agent evaluations in the existing scenario/trigger layout, plus an executed
  smoke test. Paired output/grading/timing artifacts are produced by actual
  runs; metadata or code inspection alone does not establish performance.
- Recorded exact runtime dependencies and review of the statistical/API
  boundaries above. No claim that shared helper tests establish HSSM-wide
  compatibility.

Then add a hierarchical condition contrast as a separate milestone, including
safe-prior/link policy and parameter-level prediction checks. LAN models follow
only after artifact-domain provenance and trial-wise coverage are testable.
RLSSMs, alternate response regimes, custom likelihoods, deadline/missing-data
modes, and lapse regressions need their own scoped evaluations. Training a new
LAN remains separate from fitting an HSSM model.
